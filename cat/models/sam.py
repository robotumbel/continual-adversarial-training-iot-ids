"""
sam.py — Sharpness-Aware Minimization (Foret et al., ICLR 2021)

Minimizes max_{||ε||≤ρ} L(w + ε), encouraging flat-minima solutions which
empirically generalize better AND are more robust to adversarial attacks.

Cost: 2× forward+backward per step. Worth it for the robustness gain.

Usage (drop-in for a regular optimizer):

    base_optim = torch.optim.AdamW(model.parameters(), lr=1e-3)
    optim     = SAM(model.parameters(), base_optim, rho=0.05, adaptive=False)

    for x, y in loader:
        # 1st step — compute "adversarial weight perturbation"
        loss = criterion(model(x), y)
        loss.backward()
        optim.first_step(zero_grad=True)

        # 2nd step — compute gradient at the perturbed weights, then update
        loss2 = criterion(model(x), y)
        loss2.backward()
        optim.second_step(zero_grad=True)

If you can't run twice (e.g. inside ZeRO), use ASAM with `adaptive=True`
which often works with rho=0.5.
"""
from __future__ import annotations
from typing import Iterable
import torch


class SAM(torch.optim.Optimizer):
    """Sharpness-Aware Minimization wrapping a base optimizer."""

    def __init__(
        self,
        params:         Iterable[torch.nn.Parameter],
        base_optimizer: torch.optim.Optimizer,
        rho:            float = 0.05,
        adaptive:       bool  = False,
        **kwargs,
    ):
        assert rho >= 0, "rho must be non-negative"
        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super().__init__(params, defaults)

        self.base_optimizer = base_optimizer
        # Sync the param_groups with the base optimizer so any LR / WD
        # scheduler attached to the base also affects SAM seamlessly.
        self.param_groups   = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)
        # Inject SAM-specific keys into each param group (base optimizer
        # groups don't have them).
        for group in self.param_groups:
            group.setdefault("rho",      rho)
            group.setdefault("adaptive", adaptive)

    @torch.no_grad()
    def first_step(self, zero_grad: bool = False):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)
            for p in group["params"]:
                if p.grad is None:
                    continue
                e_w = p.grad
                if group["adaptive"]:
                    e_w = (torch.pow(p, 2)) * e_w
                e_w = e_w * scale
                p.add_(e_w)
                self.state[p]["e_w"] = e_w
        if zero_grad:
            self.zero_grad(set_to_none=True)

    @torch.no_grad()
    def second_step(self, zero_grad: bool = False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None or "e_w" not in self.state[p]:
                    continue
                p.sub_(self.state[p]["e_w"])
        # ── Actual parameter update via base optimizer ──
        self.base_optimizer.step()
        if zero_grad:
            self.zero_grad(set_to_none=True)

    def step(self, closure=None):
        """Two-step SAM; closure must perform forward + backward once."""
        assert closure is not None, "SAM requires a closure for one-call use."
        closure = torch.enable_grad()(closure)
        loss = closure()
        self.first_step(zero_grad=True)
        closure()
        self.second_step(zero_grad=True)
        return loss

    def _grad_norm(self) -> torch.Tensor:
        shared_device = self.param_groups[0]["params"][0].device
        flat_norms = []
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                if group["adaptive"]:
                    g = torch.abs(p) * g
                flat_norms.append(g.to(shared_device).norm(p=2))
        if not flat_norms:
            return torch.tensor(0.0, device=shared_device)
        return torch.stack(flat_norms).norm(p=2)

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups
