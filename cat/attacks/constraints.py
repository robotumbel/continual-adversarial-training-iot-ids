"""
constraints.py — Feature Constraint Classification (FCC) method.

Novel contribution supporting Objective 3 (protocol-constrained adversarial evaluation).

FCC automatically classifies each feature into one of three types based on
column name analysis and training-data statistics, then computes valid bounds
for projecting adversarial perturbations back into feasible domains.

Feature types:
  INTEGER_PROTOCOL  — known protocol fields (port, TTL, flags, protocol IDs)
                      constraint: rounded to int, clipped to known protocol range
  INTEGER_STAT      — integer-valued statistical counters (e.g., ack_count)
                      constraint: rounded to int, clipped to [min_train, max_train]
  CONTINUOUS        — float statistical features (rate, IAT, variance, etc.)
                      constraint: clipped to [Q1-1.5*IQR, Q3+1.5*IQR] from training

Validity Rate (VR) is then computed as the proportion of adversarial samples
that fall within all feature bounds (satisfying protocol constraints).

Academic rationale:
  Alhussien et al. (2024) identify four domain constraint categories for network
  adversarial attacks. FCC operationalises the "feature values" and
  "feature modification capability" categories in a data-driven, dataset-agnostic way.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

# ── Protocol-related keyword triggers ────────────────────────────────────────
_PROTOCOL_KEYWORDS = (
    "port", "ttl", "time_to_live", "protocol", "flag",
    "fin_flag", "syn_flag", "rst_flag", "psh_flag", "ack_flag",
    "ece_flag", "cwr_flag", "http", "https", "dns", "telnet",
    "smtp", "ssh", "irc", "tcp", "udp", "dhcp", "arp", "icmp",
    "igmp", "ipv", "llc", "header_length",
    "id",        # CAN bus message IDs  (CICIoV)
    "data_",     # CAN bus data bytes   (CICIoV)
)

# Hard protocol bounds (when keyword is matched)
_PROTOCOL_BOUNDS: Dict[str, Tuple[float, float]] = {
    "port":           (0, 65535),
    "ttl":            (0, 255),
    "time_to_live":   (0, 255),
    "protocol":       (0, 255),
    "flag":           (0, 1),
    "fin_flag":       (0, 1),
    "syn_flag":       (0, 1),
    "rst_flag":       (0, 1),
    "psh_flag":       (0, 1),
    "ack_flag":       (0, 1),
    "ece_flag":       (0, 1),
    "cwr_flag":       (0, 1),
    "http":           (0, 1),
    "https":          (0, 1),
    "dns":            (0, 1),
    "telnet":         (0, 1),
    "smtp":           (0, 1),
    "ssh":            (0, 1),
    "irc":            (0, 1),
    "tcp":            (0, 1),
    "udp":            (0, 1),
    "dhcp":           (0, 1),
    "arp":            (0, 1),
    "icmp":           (0, 1),
    "igmp":           (0, 1),
    "ipv":            (0, 1),
    "llc":            (0, 1),
    "header_length":  (0, 9000),   # max Ethernet jumbo frame
    "id":             (0, 2047),   # CAN bus standard ID (11-bit)
    "data_":          (0, 255),    # CAN bus data byte
}


class FeatureConstraintClassifier:
    """
    Fits on training data and classifies each feature into a type,
    then projects adversarial samples back into valid bounds.

    Usage:
        fcc = FeatureConstraintClassifier()
        fcc.fit(X_train_raw, feature_names)
        x_constrained = fcc.project(x_adv_numpy)
        vr = fcc.validity_rate(x_orig_numpy, x_adv_numpy)
    """

    TYPES = ("INTEGER_PROTOCOL", "INTEGER_STAT", "CONTINUOUS")

    def __init__(self):
        self.feature_names: List[str] = []
        self.feature_types: List[str] = []
        self.lower_bounds:  np.ndarray = None
        self.upper_bounds:  np.ndarray = None
        self.is_fitted:     bool = False

    # ──────────────────────────────────────────────────────────────────────────
    def fit(self, X_raw: np.ndarray, feature_names: List[str]) -> "FeatureConstraintClassifier":
        """
        Analyse training data and store per-feature type and bounds.

        Args:
            X_raw:         raw (un-normalised) feature matrix [N, F]
            feature_names: list of column names, length F
        """
        assert X_raw.shape[1] == len(feature_names), \
            "X_raw columns must match feature_names length"

        self.feature_names = feature_names
        n_features = len(feature_names)

        ftypes  = []
        lowers  = np.zeros(n_features)
        uppers  = np.zeros(n_features)

        for i, name in enumerate(feature_names):
            col = X_raw[:, i]
            name_lower = name.lower()

            # ── Check for protocol keyword match ─────────────────────────────
            matched_key = None
            for kw in _PROTOCOL_KEYWORDS:
                if kw in name_lower:
                    matched_key = kw
                    break

            if matched_key is not None:
                ftype = "INTEGER_PROTOCOL"
                # Use hard-coded bounds if available, else data-driven int range
                if matched_key in _PROTOCOL_BOUNDS:
                    lo, hi = _PROTOCOL_BOUNDS[matched_key]
                else:
                    lo = float(np.floor(np.nanmin(col)))
                    hi = float(np.ceil(np.nanmax(col)))

            elif _is_integer_column(col):
                ftype = "INTEGER_STAT"
                lo = float(np.floor(np.nanmin(col)))
                hi = float(np.ceil(np.nanmax(col)))

            else:
                ftype = "CONTINUOUS"
                # Robust range: [Q1 - 1.5*IQR, Q3 + 1.5*IQR]
                q1, q3 = np.nanpercentile(col, [25, 75])
                iqr    = q3 - q1
                lo     = q1 - 1.5 * iqr
                hi     = q3 + 1.5 * iqr

            ftypes.append(ftype)
            lowers[i] = lo
            uppers[i] = hi

        self.feature_types = ftypes
        self.lower_bounds  = lowers
        self.upper_bounds  = uppers
        self.is_fitted     = True

        counts = {t: ftypes.count(t) for t in self.TYPES}
        logger.info(f"FCC fitted: {counts}")
        return self

    # ──────────────────────────────────────────────────────────────────────────
    def project(self, x: np.ndarray) -> np.ndarray:
        """
        Project adversarial samples back into feasible feature domain.

        Args:
            x: adversarial feature matrix [N, F] (raw, un-normalised)
        Returns:
            constrained x_adv [N, F]
        """
        assert self.is_fitted, "Call fit() before project()"
        x_proj = np.clip(x, self.lower_bounds, self.upper_bounds)

        # Round integer-type features
        for i, ftype in enumerate(self.feature_types):
            if ftype in ("INTEGER_PROTOCOL", "INTEGER_STAT"):
                x_proj[:, i] = np.round(x_proj[:, i])

        return x_proj

    # ──────────────────────────────────────────────────────────────────────────
    def project_tensor(self, x_tensor: "torch.Tensor",
                       scaler=None) -> "torch.Tensor":
        """
        Project a (normalised) tensor adversarial sample.
        Inverse-transforms, projects in raw space, re-normalises.

        Args:
            x_tensor: [B, seq_len, F] tensor (normalised)
            scaler:   sklearn StandardScaler used during data loading
        Returns:
            constrained [B, seq_len, F] tensor
        """
        import torch
        device = x_tensor.device
        B, L, F = x_tensor.shape
        x_np = x_tensor.detach().cpu().numpy().reshape(-1, F)  # [B*L, F]

        if scaler is not None:
            x_raw = scaler.inverse_transform(x_np)
        else:
            x_raw = x_np

        x_proj = self.project(x_raw)

        if scaler is not None:
            x_out = scaler.transform(x_proj)
        else:
            x_out = x_proj

        return torch.FloatTensor(x_out.reshape(B, L, F)).to(device)

    # ──────────────────────────────────────────────────────────────────────────
    def validity_rate(self, x_orig: np.ndarray,
                      x_adv:  np.ndarray,
                      rtol: float = 1e-5) -> float:
        """
        Compute Validity Rate (VR) — proportion of adversarial samples
        whose features all lie within the feasible bounds.

        VR = (1/N) Σ 1[isValid(x_adv_i)]

        `rtol` is not cosmetic. A projected sample is normalised, stored
        as float32, and inverse-transformed again before it reaches this
        check, and a value clipped exactly onto a bound comes back a few
        ULPs outside it. Comparing exactly then marks it invalid. Because
        a sample counts as valid only if EVERY feature passes, a
        part-per-million excursion on a handful of features drives the
        whole rate to zero — which is what produced the implausible
        "0-3% validity" this project reported before the tolerance was
        added. The tolerance scales with each feature's own magnitude.
        """
        assert self.is_fitted, "Call fit() before validity_rate()"
        lo = self.lower_bounds[None, :]
        hi = self.upper_bounds[None, :]
        scale = np.maximum(np.maximum(np.abs(lo), np.abs(hi)), 1.0)
        tol = rtol * scale
        within_lower = (x_adv >= lo - tol)
        within_upper = (x_adv <= hi + tol)
        all_valid    = (within_lower & within_upper).all(axis=1)
        return float(all_valid.mean())

    # ──────────────────────────────────────────────────────────────────────────
    def feature_validity_rate(self, x_adv: np.ndarray,
                              rtol: float = 1e-5) -> float:
        """Fraction of individual feature values inside their bounds.

        Reported alongside the sample-level rate because the two answer
        different questions: this one degrades gracefully, while the
        sample-level rate is an AND over every feature and so collapses
        as soon as any single feature drifts.
        """
        assert self.is_fitted, "Call fit() before feature_validity_rate()"
        lo = self.lower_bounds[None, :]
        hi = self.upper_bounds[None, :]
        scale = np.maximum(np.maximum(np.abs(lo), np.abs(hi)), 1.0)
        tol = rtol * scale
        ok = (x_adv >= lo - tol) & (x_adv <= hi + tol)
        return float(ok.mean())

    # ──────────────────────────────────────────────────────────────────────────
    def summary(self) -> pd.DataFrame:
        """Return a DataFrame summarising the feature type classification."""
        assert self.is_fitted
        return pd.DataFrame({
            "feature":    self.feature_names,
            "type":       self.feature_types,
            "lower":      self.lower_bounds,
            "upper":      self.upper_bounds,
        })


# ── Helper ────────────────────────────────────────────────────────────────────
def _is_integer_column(col: np.ndarray, tol: float = 1e-6) -> bool:
    """Return True if all non-NaN values are effectively integers."""
    valid = col[np.isfinite(col)]
    if len(valid) == 0:
        return False
    return np.all(np.abs(valid - np.round(valid)) < tol)
