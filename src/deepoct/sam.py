"""Sharpness-Aware Minimization (SAM) optimizer wrapper.

Foret et al., "Sharpness-Aware Minimization for Efficiently Improving
Generalization" (ICLR 2021, arXiv:2010.01412). Instead of minimizing the loss
at the current weights ``w``, SAM minimizes the WORST-CASE loss in a small
neighborhood of ``w``, which biases training toward FLAT minima. Flat minima
generalize markedly better when the training set is small — exactly the
DeepOCT-Ultra bottleneck (few / unseen patients -> sharp minima -> poor val
generalization).

WHY THIS IS A WRAPPER (not a new optimizer): SAM does not change how the base
optimizer (AdamW here) applies an update. It changes WHICH gradient the base
optimizer sees — the gradient measured at a nearby "adversarial" point ``w_adv``
that maximizes the loss inside an L2 ball of radius ``rho``. So SAM wraps an
already-constructed base optimizer and drives it in two phases per training step:

    1. ``first_step``  — using the gradient ``g`` at ``w``, climb to the sharpest
       nearby point ``w_adv = w + rho * g / ||g||`` (the ascent step). Weights are
       moved IN PLACE; the perturbation ``e_w`` is remembered per parameter.
    2. (caller recomputes loss+grad on the SAME batch at ``w_adv``)
    3. ``second_step`` — restore ``w`` (subtract the remembered ``e_w``), then let
       the base optimizer apply its normal update using the ``w_adv`` gradient.

ASAM (Kwon et al., 2021, arXiv:2102.11600) is supported via ``adaptive=True``:
the perturbation is scaled per-parameter by ``|w|`` so the neighborhood is
invariant to weight rescaling. Default is standard SAM (``adaptive=False``).

This wrapper is DELIBERATELY minimal and does NOT subclass ``torch.optim.
Optimizer``; it delegates ``zero_grad`` / ``state_dict`` / ``load_state_dict`` to
the wrapped base optimizer, so training-loop checkpointing (which saves/loads the
base optimizer state) is unchanged. It shares ``param_groups`` with the base
optimizer so the two always agree on the parameter set.
"""

from __future__ import annotations

import torch


class SAM:
    """Sharpness-Aware Minimization wrapper around an existing base optimizer.

    Args:
        base_optimizer: an already-constructed ``torch.optim.Optimizer`` (e.g.
            AdamW). SAM never replaces its update rule — it only feeds it the
            ``w_adv`` gradient.
        rho: neighborhood radius for the ascent step (default 0.05, the standard
            SAM value). Larger ``rho`` = flatter-minimum bias but more aggressive.
        adaptive: if True, use ASAM (per-parameter ``|w|``-scaled neighborhood);
            default False = standard SAM.
        eps: numerical floor added to the gradient norm before dividing.
    """

    def __init__(self, base_optimizer: torch.optim.Optimizer,
                 rho: float = 0.05, adaptive: bool = False,
                 eps: float = 1e-12) -> None:
        if rho < 0.0:
            raise ValueError(f"rho must be non-negative, got {rho}")
        self.base_optimizer = base_optimizer
        self.rho = float(rho)
        self.adaptive = bool(adaptive)
        self.eps = float(eps)
        # Share the base optimizer's param_groups so both iterate the SAME params.
        self.param_groups = base_optimizer.param_groups
        # Per-parameter perturbation cache (kept OUT of the base optimizer state so
        # it never leaks into the checkpoint). Keyed by the Parameter object.
        self._e_w: dict = {}

    # -- gradient norm across ALL param groups (the ||g|| in w + rho*g/||g||) -----
    def _grad_norm(self) -> torch.Tensor:
        # Put the norm on the device of the first parameter that has a grad.
        shared_device = self.param_groups[0]["params"][0].device
        norms = []
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                # ASAM scales the direction by |w| per parameter; standard SAM does
                # not (the (|w| if adaptive else 1) factor collapses to 1).
                g = (torch.abs(p) * p.grad) if self.adaptive else p.grad
                norms.append(g.norm(p=2).to(shared_device))
        if not norms:
            # No gradients at all — return a zero norm so callers see ||g||==0.
            return torch.tensor(0.0, device=shared_device)
        return torch.norm(torch.stack(norms), p=2)

    @torch.no_grad()
    def first_step(self, zero_grad: bool = False) -> None:
        """Ascent step: move weights to ``w_adv = w + rho * g / ||g||`` in place.

        The per-parameter perturbation ``e_w`` is cached so ``second_step`` can
        restore ``w`` exactly. Call AFTER a ``loss.backward()`` at ``w``.
        """
        grad_norm = self._grad_norm()
        scale = self.rho / (grad_norm + self.eps)
        self._e_w.clear()
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                # ASAM: e_w = rho * (w^2 * g) / ||...||  (per-param |w|^2 weighting);
                # standard SAM: e_w = rho * g / ||g||.
                e_w = p.grad * scale.to(p)
                if self.adaptive:
                    e_w = e_w * torch.pow(p, 2)
                p.add_(e_w)            # climb to the local sharpest point
                self._e_w[p] = e_w
        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad: bool = False, apply_step: bool = True) -> None:
        """Restore ``w`` (undo the ascent), then apply the base optimizer step.

        Call AFTER recomputing ``loss.backward()`` at ``w_adv`` on the SAME batch,
        so ``p.grad`` now holds the ``w_adv`` gradient that the base optimizer will
        consume.

        ``apply_step=False`` restores the weights but SKIPS the base update — used
        by the training loop to bail out safely on a non-finite ``w_adv`` gradient
        WITHOUT leaving the weights stranded at the perturbed point.
        """
        for group in self.param_groups:
            for p in group["params"]:
                e_w = self._e_w.get(p)
                if e_w is None:
                    continue
                p.sub_(e_w)            # back to the original w
        self._e_w.clear()
        if apply_step:
            self.base_optimizer.step()   # normal AdamW update using w_adv grads
        if zero_grad:
            self.zero_grad()

    # -- delegation so the training loop can treat SAM like an optimizer ---------
    def zero_grad(self, set_to_none: bool = True) -> None:
        self.base_optimizer.zero_grad(set_to_none=set_to_none)

    def state_dict(self) -> dict:
        # Only the base optimizer carries persistent state (moments etc.); the
        # e_w cache is transient within a single step and never checkpointed.
        return self.base_optimizer.state_dict()

    def load_state_dict(self, state_dict: dict) -> None:
        self.base_optimizer.load_state_dict(state_dict)
        # Re-sync param_groups reference after the base optimizer rebuilds it.
        self.param_groups = self.base_optimizer.param_groups
