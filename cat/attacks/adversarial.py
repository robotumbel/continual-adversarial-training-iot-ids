"""
adversarial.py — FGSM, PGD, and Gaussian Noise adversarial attackers.

All attackers optionally apply FCC protocol-constraint projection after
each perturbation step, implementing Objective 3 of the dissertation:
"realistically constrained adversarial perturbations that preserve
network protocol validity."

Attack configurations follow dissertation Table (Ch.3):
  FGSM: ε ∈ {0.01, 0.05, 0.10, 0.15, 0.20}
  PGD:  ε=0.10, α=0.01, iterations=10
  Noise: σ ∈ {0.01, 0.05, 0.10, 0.15, 0.20}
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Optional
from .constraints import FeatureConstraintClassifier
import logging

logger = logging.getLogger(__name__)


class AdversarialAttacker:
    """
    Unified adversarial attack generator.

    Args:
        model:       PyTorch model (must output dict with 'logits' key)
        device:      torch device
        fcc:         FeatureConstraintClassifier (optional; for protocol projection)
        scaler:      sklearn scaler for inverse/forward transform during projection
    """

    def __init__(
        self,
        model,
        device:     torch.device,
        fcc:        Optional[FeatureConstraintClassifier] = None,
        scaler      = None,
    ):
        self.model  = model
        self.device = device
        self.fcc    = fcc
        self.scaler = scaler

    # ──────────────────────────────────────────────────────────────────────────
    # Fast Gradient Sign Method (FGSM)
    # x_adv = x + ε · sign(∇_x L(θ, x, y))
    # ──────────────────────────────────────────────────────────────────────────

    def fgsm(
        self,
        x:       torch.Tensor,
        y:       torch.Tensor,
        epsilon: float = 0.10,
        apply_constraints: bool = True,
        batch:   int   = 256,
    ) -> torch.Tensor:
        """
        Single-step FGSM — processed in mini-batches to avoid GPU OOM.

        Args:
            x:       [N, seq_len, F] normalised input on CPU or GPU
            y:       [N] integer labels
            epsilon: perturbation magnitude
            batch:   mini-batch size for gradient computation
        Returns:
            x_adv: [N, seq_len, F] on CPU
        """
        device = next(self.model.parameters()).device
        x = x.to(device)
        y = y.to(device)

        chunks = []
        # train() required for cuDNN RNN backward
        self.model.train()
        for i in range(0, len(x), batch):
            x_b = x[i: i + batch].clone().detach().requires_grad_(True)
            y_b = y[i: i + batch]

            out  = self.model(x_b)
            # Clamp labels to model's output range — handles class-space mismatch
            # when evaluating old tasks after head adaptation (e.g. 34→19 classes).
            n_cls = out["logits"].size(1)
            y_clamped = y_b.clamp(0, n_cls - 1)
            loss = F.cross_entropy(out["logits"], y_clamped)
            self.model.zero_grad()
            loss.backward()

            with torch.no_grad():
                grad  = x_b.grad.data
                x_adv = x[i: i + batch] + epsilon * grad.sign()
            chunks.append(x_adv.detach().cpu())
            del out, loss, x_b, grad

        self.model.eval()
        torch.cuda.empty_cache()

        x_adv = torch.cat(chunks, dim=0)
        if apply_constraints and self.fcc is not None:
            x_adv = self.fcc.project_tensor(x_adv.to(device), self.scaler).cpu()

        return x_adv

    # ──────────────────────────────────────────────────────────────────────────
    # Projected Gradient Descent (PGD)
    # x_{t+1} = Clip_{x,ε}(x_t + α · sign(∇_x L(x_t, y)))
    # ──────────────────────────────────────────────────────────────────────────

    def pgd(
        self,
        x:          torch.Tensor,
        y:          torch.Tensor,
        epsilon:    float = 0.10,
        alpha:      float = 0.01,
        n_iter:     int   = 10,
        apply_constraints: bool = True,
        batch:      int   = 256,
    ) -> torch.Tensor:
        """
        Iterative PGD attack — processed in mini-batches to avoid GPU OOM.
        """
        device = next(self.model.parameters()).device
        x = x.to(device)
        y = y.to(device)

        chunks = []
        self.model.train()
        for i in range(0, len(x), batch):
            x_b    = x[i: i + batch].clone().detach()
            y_b    = y[i: i + batch]
            x_orig = x_b.clone()

            for _ in range(n_iter):
                x_b = x_b.requires_grad_(True)
                out  = self.model(x_b)
                n_cls = out["logits"].size(1)
                y_clamped = y_b.clamp(0, n_cls - 1)
                loss = F.cross_entropy(out["logits"], y_clamped)
                self.model.zero_grad()
                loss.backward()

                with torch.no_grad():
                    grad  = x_b.grad.data
                    x_b   = x_b + alpha * grad.sign()
                    delta = torch.clamp(x_b - x_orig, -epsilon, epsilon)
                    x_b   = (x_orig + delta).detach()
                del out, loss

            if apply_constraints and self.fcc is not None:
                x_b = self.fcc.project_tensor(x_b, self.scaler)
            chunks.append(x_b.detach().cpu())
            torch.cuda.empty_cache()

        self.model.eval()
        return torch.cat(chunks, dim=0)

    # ──────────────────────────────────────────────────────────────────────────
    # Square Attack (Andriushchenko et al., ECCV 2020) — Score-based BLACKBOX
    # Attacker only needs probabilistic outputs (no gradient).
    # ──────────────────────────────────────────────────────────────────────────

    @torch.no_grad()
    def square_attack(
        self,
        x:           torch.Tensor,
        y:           torch.Tensor,
        epsilon:     float = 0.10,
        n_queries:   int   = 1000,
        p_init:      float = 0.05,
        apply_constraints: bool = True,
        batch:       int   = 256,
    ) -> torch.Tensor:
        """
        Square Attack — state-of-the-art **score-based blackbox** attack.

        The attacker only observes the model's output probability (NO gradient).
        At each step it samples a random rectangular "square" patch of one
        contiguous feature region per sample and tries flipping the sign of
        the perturbation in that patch. Accepts the flip iff loss increases.

        Reference: Andriushchenko et al., "Square Attack: a query-efficient
        black-box adversarial attack via random search", ECCV 2020.

        Args
        ----
        x         : [N, L, F] inputs
        y         : [N] integer labels
        epsilon   : L-infinity perturbation budget
        n_queries : maximum #queries (more queries → stronger attack)
        p_init    : initial fraction of features perturbed per sample
        batch     : process in mini-batches to bound GPU memory

        Returns
        -------
        x_adv : [N, L, F] perturbed tensor on CPU
        """
        device = next(self.model.parameters()).device
        x = x.to(device)
        y = y.to(device)
        self.model.eval()

        chunks = []
        n_batches = (len(x) + batch - 1) // batch
        for bii, bi in enumerate(range(0, len(x), batch)):
            if bii == 0 or (bii + 1) % 5 == 0 or bii == n_batches - 1:
                logger.info(f"    Square Attack: batch {bii + 1}/{n_batches}")
            x_b = x[bi: bi + batch].clone()
            y_b = y[bi: bi + batch]
            B, L, F_ = x_b.shape

            # Initialize: vertical stripe of +/-eps across full sequence length
            init_sign = torch.sign(torch.empty(B, 1, F_, device=device).uniform_(-1, 1))
            init_sign[init_sign == 0] = 1.0
            delta = init_sign * epsilon
            x_adv = (x_b + delta).clone()

            out  = self.model(x_adv)
            n_cls = out["logits"].size(1)
            y_c  = y_b.clamp(0, n_cls - 1)
            log_p = F.log_softmax(out["logits"], dim=-1)
            # Margin loss: log P(y_true) - max log P(y_other) → minimize this
            true_lp  = log_p.gather(1, y_c.unsqueeze(1)).squeeze(1)
            log_p_o  = log_p.clone()
            log_p_o.scatter_(1, y_c.unsqueeze(1), float("-inf"))
            other_lp = log_p_o.max(dim=1).values
            best_loss = -(true_lp - other_lp)   # higher = more attacked

            for q in range(n_queries):
                # Decay perturbation patch size
                p = p_init * max(0.1, 1.0 - q / n_queries)
                patch_w = max(1, int(round(np.sqrt(p * F_))))
                # Random rectangular region in feature axis (sequence-wide)
                start = np.random.randint(0, max(1, F_ - patch_w + 1))
                # Generate candidate delta: flip sign in patch only
                sign = torch.sign(torch.empty(B, 1, patch_w, device=device).uniform_(-1, 1))
                sign[sign == 0] = 1.0
                cand_delta = delta.clone()
                cand_delta[:, :, start: start + patch_w] = sign * epsilon

                x_cand = x_b + cand_delta
                out = self.model(x_cand)
                log_p = F.log_softmax(out["logits"], dim=-1)
                true_lp  = log_p.gather(1, y_c.unsqueeze(1)).squeeze(1)
                log_p_o  = log_p.clone()
                log_p_o.scatter_(1, y_c.unsqueeze(1), float("-inf"))
                other_lp = log_p_o.max(dim=1).values
                cand_loss = -(true_lp - other_lp)

                # Accept candidate per-sample where it improves the attack
                improved = cand_loss > best_loss
                if improved.any():
                    delta[improved] = cand_delta[improved]
                    best_loss[improved] = cand_loss[improved]
                    x_adv = (x_b + delta).clone()

            if apply_constraints and self.fcc is not None:
                x_adv = self.fcc.project_tensor(x_adv, self.scaler)
            chunks.append(x_adv.detach().cpu())
            torch.cuda.empty_cache()

        return torch.cat(chunks, dim=0)

    # ──────────────────────────────────────────────────────────────────────────
    # Transfer Attack — Most realistic BLACKBOX threat model for IDS
    # Attacker trains a surrogate DNN with public data, generates FGSM on
    # that surrogate, then evaluates the same adversarial examples on target.
    # Requires ZERO queries to the target model.
    # ──────────────────────────────────────────────────────────────────────────

    def set_surrogate(self, surrogate_model):
        """Attach a separately-trained surrogate model for transfer attacks."""
        self._surrogate = surrogate_model.to(self.device)
        self._surrogate.eval()

    def transfer_attack(
        self,
        x:                 torch.Tensor,
        y:                 torch.Tensor,
        epsilon:           float = 0.10,
        apply_constraints: bool  = True,
        batch:             int   = 256,
    ) -> torch.Tensor:
        """
        Transfer Attack — FGSM on a surrogate model, evaluated on target.

        Threat model: attacker has NO access to the target AAM-TRANS model
        (no gradients, no scores, no queries). They train their own surrogate
        on public data, generate adversarial examples there, and submit those
        to the target. This is the *most realistic* black-box scenario for a
        deployed IDS — there is no actual query budget required.

        If no surrogate has been registered (via set_surrogate), we fall
        back to a fresh untrained DNN; the resulting attack is weak but
        provides a lower-bound robustness signal.
        """
        if not hasattr(self, "_surrogate") or self._surrogate is None:
            logger.warning(
                "transfer_attack: no surrogate registered; using untrained "
                "fallback. Call AdversarialAttacker.set_surrogate(model) "
                "before evaluation for meaningful results."
            )
            from models.baselines import DNNModel
            n_cls_target = self.model.classification_head[-1].out_features \
                if hasattr(self.model, "classification_head") else 2
            self._surrogate = DNNModel(
                input_dim=x.shape[-1],
                n_classes=n_cls_target,
            ).to(self.device)
            self._surrogate.eval()

        # FGSM on surrogate
        device = self.device
        x = x.to(device)
        y = y.to(device)

        chunks = []
        self._surrogate.train()
        for i in range(0, len(x), batch):
            x_b = x[i: i + batch].clone().detach().requires_grad_(True)
            y_b = y[i: i + batch]

            out  = self._surrogate(x_b)
            n_cls = out["logits"].size(1)
            y_c   = y_b.clamp(0, n_cls - 1)
            loss  = F.cross_entropy(out["logits"], y_c)
            self._surrogate.zero_grad()
            loss.backward()
            with torch.no_grad():
                grad = x_b.grad.data
                x_adv = x[i: i + batch] + epsilon * grad.sign()
            chunks.append(x_adv.detach().cpu())
            del out, loss, x_b, grad

        self._surrogate.eval()
        torch.cuda.empty_cache()

        x_adv = torch.cat(chunks, dim=0)
        if apply_constraints and self.fcc is not None:
            x_adv = self.fcc.project_tensor(x_adv.to(device), self.scaler).cpu()
        return x_adv

    # ──────────────────────────────────────────────────────────────────────────
    # Gaussian Noise Attack
    # x_adv = x + N(0, σ²)
    # ──────────────────────────────────────────────────────────────────────────

    def gaussian_noise(
        self,
        x:     torch.Tensor,
        sigma: float = 0.10,
        apply_constraints: bool = True,
    ) -> torch.Tensor:
        """
        Additive Gaussian noise (non-gradient baseline).

        Used to differentiate vulnerability to random disturbances vs
        adversarially optimised perturbations (dissertation Ch.3).
        """
        with torch.no_grad():
            noise = torch.randn_like(x) * sigma
            x_adv = x + noise

        if apply_constraints and self.fcc is not None:
            x_adv = self.fcc.project_tensor(x_adv, self.scaler)

        return x_adv.detach()

    # ──────────────────────────────────────────────────────────────────────────
    # Convenience: generate all configured attacks
    # ──────────────────────────────────────────────────────────────────────────

    def generate_all(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        fgsm_epsilons:   list = None,
        pgd_epsilon:     float = 0.10,
        pgd_alpha:       float = 0.01,
        pgd_iterations:  int   = 10,
        noise_sigmas:    list  = None,
        apply_constraints: bool = True,
    ) -> dict:
        """
        Generate all attack variants and return as dict keyed by attack name.
        Each value is a [B, seq_len, F] tensor.
        """
        if fgsm_epsilons is None:
            fgsm_epsilons = [0.01, 0.05, 0.10, 0.15, 0.20]
        if noise_sigmas is None:
            noise_sigmas  = [0.01, 0.05, 0.10, 0.15, 0.20]

        results = {"Clean": x.detach()}

        for eps in fgsm_epsilons:
            key = f"FGSM_eps{eps}"
            results[key] = self.fgsm(x, y, epsilon=eps,
                                     apply_constraints=apply_constraints)

        results[f"PGD_eps{pgd_epsilon}"] = self.pgd(
            x, y,
            epsilon=pgd_epsilon,
            alpha=pgd_alpha,
            n_iter=pgd_iterations,
            apply_constraints=apply_constraints,
        )

        for sigma in noise_sigmas:
            key = f"Gaussian_std{sigma}"
            results[key] = self.gaussian_noise(x, sigma=sigma,
                                               apply_constraints=apply_constraints)

        return results
