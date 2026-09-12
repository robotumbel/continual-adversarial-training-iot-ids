"""
robust_eval.py — Inference-time defensive wrapper for AAM-TRANS.

V3 strategy: instead of trying to make the training loop produce a robust
model (which broke clean accuracy in V2), we keep training relatively clean
and apply *defensive* mechanisms at evaluation time only:

  1. Bayesian gates — gate logits inside each AAM-TRANS attention block
     get N(0, σ²) noise at every forward pass, so each forward produces a
     slightly different decision surface.

  2. MC averaging — n forward passes are averaged to produce the final
     softmax probability. This creates an implicit ensemble of n models
     and obfuscates the gradient signal that white/black-box gradient-
     estimating attackers would otherwise exploit.

  3. Transparent interface — `RobustEvalWrapper` exposes the same
     `forward(x)` returning {"logits": ..., "anomaly_score": ...} that the
     base AAMTransIDS model returns, so it is a drop-in replacement for
     evaluation in `run_aam_trans.py` and the evaluation pipeline.

Important: this wrapper is for EVALUATION ONLY. It is not used at training
time. Training continues to use the deterministic gates.
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


class RobustEvalWrapper(nn.Module):
    """
    Inference-time wrapper that turns an AAMTransIDS model into a stochastic
    smoothed classifier via Bayesian gate noise + MC averaging.

    Args
    ----
    model         : AAMTransIDS instance (the trained model)
    gate_noise    : std of N(0, std²) added to gate logits in every block
    mc_samples    : number of stochastic forward passes to average over
    return_var    : if True, also returns the variance of softmax probs
                    across MC samples (useful for uncertainty estimation)
    """

    def __init__(
        self,
        model:        nn.Module,
        gate_noise:   float = 0.15,
        mc_samples:   int   = 8,
        return_var:   bool  = False,
    ):
        super().__init__()
        self.model      = model
        self.gate_noise = gate_noise
        self.mc_samples = mc_samples
        self.return_var = return_var
        # Activate Bayesian gates on the wrapped model
        if hasattr(model, "set_inference_gate_noise"):
            model.set_inference_gate_noise(gate_noise)

    def train(self, mode: bool = True):
        # Always keep the underlying model in eval mode for robustness
        # eval — but allow flipping the wrapper itself.
        super().train(mode)
        self.model.eval()
        return self

    def forward(self, x: torch.Tensor):
        """
        Returns a dict matching AAMTransIDS forward output, with `logits`
        being log-of-averaged-probabilities so that downstream argmax /
        cross-entropy behave correctly.

        Grad-handling policy:
          - This method INTENTIONALLY does *not* wrap with @torch.no_grad().
            FGSM-style attacks need gradients to flow back to the input
            tensor, and standard evaluation callers (evaluate_epoch) already
            wrap their loops in `with torch.no_grad():`, so memory is still
            saved during pure evaluation.
          - When MC sampling under enabled grad, we accumulate n graphs;
            this is intentional so adversarial attackers see the *same*
            stochastic surface the defender uses at inference.
        """
        probs_sum   = None
        probs_sqsum = None
        anom_sum    = None
        n = max(1, self.mc_samples)

        for _ in range(n):
            out    = self.model(x)
            logits = out["logits"]
            probs  = F.softmax(logits, dim=-1)
            if probs_sum is None:
                probs_sum   = probs
                probs_sqsum = probs.pow(2)
                anom_sum    = out.get("anomaly_score", torch.zeros_like(probs[:, :1]))
            else:
                probs_sum   = probs_sum + probs
                probs_sqsum = probs_sqsum + probs.pow(2)
                if "anomaly_score" in out:
                    anom_sum = anom_sum + out["anomaly_score"]

        mean_probs = (probs_sum / n).clamp(min=1e-12)
        mean_anom  = anom_sum  / n
        # Return log(mean_probs) so that argmax / NLL on these `logits` work
        # exactly as if the wrapped model had returned them directly.
        out_dict = {
            "logits":        mean_probs.log(),
            "anomaly_score": mean_anom,
        }
        if self.return_var:
            var = (probs_sqsum / n) - mean_probs.pow(2)
            out_dict["mc_variance"] = var.clamp(min=0.0)
        return out_dict

    # ── Compatibility shims — some code paths poke attributes directly ───────
    @property
    def classification_head(self):
        return self.model.classification_head

    @property
    def embedding(self):
        return self.model.embedding

    def parameters(self, recurse: bool = True):
        return self.model.parameters(recurse=recurse)

    def state_dict(self, *a, **kw):
        return self.model.state_dict(*a, **kw)

    def load_state_dict(self, *a, **kw):
        return self.model.load_state_dict(*a, **kw)
