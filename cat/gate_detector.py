"""
gate_detector.py — Gate-Signature Adversarial Detection (GSAD).

Detects adversarial / out-of-distribution inputs by reading the internal
adaptive-gate activations of AAM-TRANS. The classifier is clean-trained;
the detector is built ONLY from the clean-data distribution of gate
signatures. No adversarial examples are used for training anything —
they are used only at evaluation time.

Components
----------
GateSignatureExtractor : pulls the L*H gate vector from a forward pass
CleanProfile           : per-class Gaussian of clean gate vectors
                         -> Mahalanobis "outlierness" score (Signal A)
MCInstability          : gate variance + prediction-flip under MC noise
                         -> "fragility" score (Signal B)
GateSignatureDetector  : fuses A + B into one adversarial score and a
                         two-dimensional decision
                         (TRUSTED / NOVEL-OOD / ADVERSARIAL)

See DESIGN.md for the full rationale.
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# 1. Gate-Signature Extraction
# ──────────────────────────────────────────────────────────────────────────────

class GateSignatureExtractor:
    """
    Extracts the flattened adaptive-gate vector g(x) in R^{L*H} from an
    AAM-TRANS forward pass, plus auxiliary clean-derived features
    (softmax confidence, entropy, anomaly score).
    """

    def __init__(self, model, device):
        self.model  = model
        self.device = device

    @torch.no_grad()
    def extract(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:  x : [B, L, F]
        Returns dict with:
            gate     : [B, L*H]   flattened gate signature
            logits   : [B, C]
            preds    : [B]
            conf     : [B]        softmax max-probability
            entropy  : [B]        softmax entropy
            anomaly  : [B]        anomaly-head score (0 if absent)
        """
        self.model.eval()
        x = x.to(self.device)
        out = self.model(x)

        gates = out["adaptive_gates"]            # list of [B, H], len = L
        gate  = torch.cat([g for g in gates], dim=-1)   # [B, L*H]

        logits = out["logits"]
        probs  = F.softmax(logits, dim=-1)
        conf   = probs.max(dim=-1).values
        entropy = -(probs.clamp_min(1e-12) * probs.clamp_min(1e-12).log()).sum(-1)
        preds  = logits.argmax(dim=-1)
        anomaly = out.get("anomaly_score", torch.zeros_like(conf.unsqueeze(-1)))
        anomaly = anomaly.squeeze(-1)

        return {"gate": gate.cpu(), "logits": logits.cpu(),
                "preds": preds.cpu(), "conf": conf.cpu(),
                "entropy": entropy.cpu(), "anomaly": anomaly.cpu()}

    @torch.no_grad()
    def extract_loader(self, loader) -> Dict[str, np.ndarray]:
        """Extract over a whole DataLoader. Returns numpy arrays."""
        G, L, P, C, E, A, Y = [], [], [], [], [], [], []
        for x, y in loader:
            d = self.extract(x)
            G.append(d["gate"].numpy())
            P.append(d["preds"].numpy())
            C.append(d["conf"].numpy())
            E.append(d["entropy"].numpy())
            A.append(d["anomaly"].numpy())
            Y.append(np.asarray(y))
        return {"gate": np.vstack(G), "preds": np.concatenate(P),
                "conf": np.concatenate(C), "entropy": np.concatenate(E),
                "anomaly": np.concatenate(A), "labels": np.concatenate(Y)}


# ──────────────────────────────────────────────────────────────────────────────
# 2. Clean Profile  — Signal A (outlierness)
# ──────────────────────────────────────────────────────────────────────────────

class CleanProfile:
    """
    Per-class Gaussian model of clean gate signatures. Provides the
    Mahalanobis distance of a new gate vector to the clean distribution
    of its predicted class. Classes with too few samples fall back to a
    global Gaussian.
    """

    def __init__(self, shrinkage: float = 1e-3, min_per_class: int = 30):
        self.shrinkage     = shrinkage
        self.min_per_class = min_per_class
        self.mu:      Dict[int, np.ndarray] = {}
        self.inv_cov: Dict[int, np.ndarray] = {}
        self.global_mu:      Optional[np.ndarray] = None
        self.global_inv_cov: Optional[np.ndarray] = None

    def _fit_gaussian(self, X: np.ndarray):
        mu  = X.mean(axis=0)
        d   = X.shape[1]
        cov = np.cov(X, rowvar=False) if X.shape[0] > 1 else np.eye(d)
        cov = np.atleast_2d(cov)
        cov = cov + self.shrinkage * np.eye(d)         # regularise
        inv = np.linalg.pinv(cov)
        return mu, inv

    def fit(self, gate: np.ndarray, labels: np.ndarray):
        """Fit per-class + global Gaussians on CLEAN gate vectors."""
        gate   = np.asarray(gate, dtype=np.float64)
        labels = np.asarray(labels)
        self.global_mu, self.global_inv_cov = self._fit_gaussian(gate)
        for c in np.unique(labels):
            Xc = gate[labels == c]
            if len(Xc) >= self.min_per_class:
                self.mu[int(c)], self.inv_cov[int(c)] = self._fit_gaussian(Xc)
        logger.info(
            f"  CleanProfile fitted: {len(self.mu)} per-class Gaussians "
            f"+ 1 global (dim={gate.shape[1]})"
        )
        return self

    def mahalanobis(self, gate: np.ndarray, pred_class: np.ndarray) -> np.ndarray:
        """
        Mahalanobis distance of each gate vector to the clean Gaussian of
        its predicted class (global fallback if class profile missing).
        """
        gate       = np.asarray(gate, dtype=np.float64)
        pred_class = np.asarray(pred_class)
        out = np.zeros(len(gate))
        for i in range(len(gate)):
            c = int(pred_class[i])
            if c in self.mu:
                mu, inv = self.mu[c], self.inv_cov[c]
            else:
                mu, inv = self.global_mu, self.global_inv_cov
            diff = gate[i] - mu
            out[i] = float(np.sqrt(max(0.0, diff @ inv @ diff)))
        return out


# ──────────────────────────────────────────────────────────────────────────────
# 3. MC Instability  — Signal B (fragility)
# ──────────────────────────────────────────────────────────────────────────────

class MCInstability:
    """
    Measures how unstable an input's gate signature and prediction are
    under small stochastic perturbation (Bayesian-gate noise). Adversarial
    inputs sit near decision boundaries and are unstable; clean inputs
    (even novel ones) tend to be more stable.
    """

    def __init__(self, model, device, n_mc: int = 16,
                 gate_noise: float = 0.15):
        self.model      = model
        self.device     = device
        self.n_mc       = n_mc
        self.gate_noise = gate_noise

    @torch.no_grad()
    def measure(self, x: torch.Tensor) -> Dict[str, np.ndarray]:
        """
        Returns dict with:
            inst_gate : [B]  mean variance of the gate vector over n_mc passes
            inst_pred : [B]  prediction-flip rate over n_mc passes
        """
        self.model.eval()
        x = x.to(self.device)

        # Enable inference-time gate noise if the model supports it
        if hasattr(self.model, "set_inference_gate_noise"):
            self.model.set_inference_gate_noise(self.gate_noise)

        gate_runs: List[torch.Tensor] = []
        pred_runs: List[torch.Tensor] = []
        for _ in range(self.n_mc):
            out = self.model(x)
            g   = torch.cat([gg for gg in out["adaptive_gates"]], dim=-1)
            gate_runs.append(g.unsqueeze(0))                  # [1, B, L*H]
            pred_runs.append(out["logits"].argmax(-1).unsqueeze(0))  # [1, B]

        if hasattr(self.model, "set_inference_gate_noise"):
            self.model.set_inference_gate_noise(0.0)          # reset

        gates = torch.cat(gate_runs, dim=0)                   # [n_mc, B, L*H]
        preds = torch.cat(pred_runs, dim=0)                   # [n_mc, B]

        inst_gate = gates.var(dim=0).mean(dim=-1)             # [B]
        # prediction-flip rate vs the per-sample majority vote
        maj = torch.mode(preds, dim=0).values                # [B]
        inst_pred = (preds != maj.unsqueeze(0)).float().mean(dim=0)  # [B]

        return {"inst_gate": inst_gate.cpu().numpy(),
                "inst_pred": inst_pred.cpu().numpy()}

    @torch.no_grad()
    def measure_loader(self, loader) -> Dict[str, np.ndarray]:
        IG, IP = [], []
        for x, _ in loader:
            d = self.measure(x)
            IG.append(d["inst_gate"]); IP.append(d["inst_pred"])
        return {"inst_gate": np.concatenate(IG),
                "inst_pred": np.concatenate(IP)}


# ──────────────────────────────────────────────────────────────────────────────
# 4. Gate-Signature Detector  — fusion + 2-D decision
# ──────────────────────────────────────────────────────────────────────────────

class GateSignatureDetector:
    """
    Fuses Signal A (Mahalanobis outlierness) and Signal B (MC instability)
    into a single adversarial score, and assigns a two-dimensional
    decision label.

    Calibration uses CLEAN data only:
      * standardisation (z-score) stats from clean validation
      * decision threshold at a target false-positive rate
    """

    def __init__(self, profile: CleanProfile,
                 w_maha: float = 1.0, w_inst_gate: float = 1.0,
                 w_inst_pred: float = 1.0, target_fpr: float = 0.05):
        self.profile      = profile
        self.w            = np.array([w_maha, w_inst_gate, w_inst_pred],
                                     dtype=np.float64)
        self.target_fpr   = target_fpr
        # calibration state (filled by .calibrate)
        self.mean_: Optional[np.ndarray] = None
        self.std_:  Optional[np.ndarray] = None
        self.threshold_:    Optional[float] = None
        self.maha_thr_:     Optional[float] = None   # outlier axis cutoff
        self.inst_thr_:     Optional[float] = None   # instability axis cutoff

    # ── raw 3-feature matrix ────────────────────────────────────────────────
    @staticmethod
    def _features(maha, inst_gate, inst_pred) -> np.ndarray:
        return np.stack([np.asarray(maha),
                         np.asarray(inst_gate),
                         np.asarray(inst_pred)], axis=1)   # [N, 3]

    # ── calibration on CLEAN validation data ────────────────────────────────
    def calibrate(self, maha_clean, inst_gate_clean, inst_pred_clean):
        """Fit z-score stats and the decision threshold on clean data."""
        feats = self._features(maha_clean, inst_gate_clean, inst_pred_clean)
        self.mean_ = feats.mean(axis=0)
        self.std_  = feats.std(axis=0) + 1e-8

        s_clean = self._fused_score(feats)
        # threshold = (1 - target_fpr) quantile of the clean score
        self.threshold_ = float(np.quantile(s_clean, 1.0 - self.target_fpr))
        # per-axis cutoffs for the 2-D decision
        self.maha_thr_ = float(np.quantile(maha_clean,      1.0 - self.target_fpr))
        self.inst_thr_ = float(np.quantile(inst_gate_clean, 1.0 - self.target_fpr))
        logger.info(
            f"  Detector calibrated: score_thr={self.threshold_:.3f}, "
            f"maha_thr={self.maha_thr_:.3f}, inst_thr={self.inst_thr_:.5f}"
        )
        return self

    def _fused_score(self, feats: np.ndarray) -> np.ndarray:
        z = (feats - self.mean_) / self.std_
        return z @ self.w

    # ── scoring ─────────────────────────────────────────────────────────────
    def score(self, maha, inst_gate, inst_pred) -> np.ndarray:
        """Fused adversarial score (higher = more adversarial)."""
        assert self.mean_ is not None, "call calibrate() first"
        feats = self._features(maha, inst_gate, inst_pred)
        return self._fused_score(feats)

    def decide(self, maha, inst_gate, inst_pred) -> List[str]:
        """
        Two-dimensional decision per sample:
          TRUSTED      — gate signature in-distribution
          NOVEL_OOD    — outlier but stable  (likely a new network attack)
          ADVERSARIAL  — outlier and unstable
        """
        assert self.maha_thr_ is not None, "call calibrate() first"
        maha      = np.asarray(maha)
        inst_gate = np.asarray(inst_gate)
        labels = []
        for m, ig in zip(maha, inst_gate):
            if m <= self.maha_thr_:
                labels.append("TRUSTED")
            elif ig > self.inst_thr_:
                labels.append("ADVERSARIAL")
            else:
                labels.append("NOVEL_OOD")
        return labels

    def is_flagged(self, maha, inst_gate, inst_pred) -> np.ndarray:
        """Boolean array: True if the input should be rejected/flagged."""
        return self.score(maha, inst_gate, inst_pred) > self.threshold_
