import math
from typing import Optional

import torch
from torch import nn, Tensor
from torch.autograd import grad
from torch.nn import functional as F

from adv_lib.utils.visdom_logger import VisdomLogger


def ddn(model: nn.Module,
        inputs: Tensor,
        labels: Tensor,
        targeted: bool = False,
        steps: int = 100,
        γ: float = 0.05,
        init_norm: float = 1.,
        levels: Optional[int] = 256,
        callback: Optional[VisdomLogger] = None) -> Tensor:
    """
    Decoupled Direction and Norm attack from https://arxiv.org/abs/1811.09600.

    Parameters
    ----------
    model : nn.Module
        Model to attack.
    inputs : Tensor
        Inputs to attack. Should be in [0, 1].
    labels : Tensor
        Labels corresponding to the inputs if untargeted, else target labels.
    targeted : bool
        Whether to perform a targeted attack or not.
    steps : int
        Number of optimization steps.
    γ : float
        Factor by which the norm will be modified. new_norm = norm * (1 + or - γ).
    init_norm : float
        Initial value for the norm of the attack.
    levels : int
        If not None, the returned adversarials will have quantized values to the specified number of levels.
    callback : Optional

    Returns
    -------
    adv_inputs : Tensor
        Modified inputs to be adversarial to the model.

    """
    if inputs.min() < 0 or inputs.max() > 1: raise ValueError('Input values should be in the [0, 1] range.')
    device = inputs.device
    batch_size = len(inputs)
    batch_view = lambda tensor: tensor.view(batch_size, *[1] * (inputs.ndim - 1))

    # Init variables
    multiplier = -1 if targeted else 1
    δ = torch.zeros_like(inputs, requires_grad=True)
    ε = torch.full((batch_size,), init_norm, device=device, dtype=torch.float)
    worst_norm = torch.max(inputs, 1 - inputs).flatten(1).norm(p=2, dim=1)

    # Init trackers
    best_l2 = worst_norm.clone()
    best_δ = torch.zeros_like(inputs)
    adv_found = torch.zeros(batch_size, dtype=torch.bool, device=device)

    for i in range(steps):
        α = torch.tensor(0.01 + (1 - 0.01) * (1 + math.cos(math.pi * i / steps)) / 2, device=device)

        l2 = δ.data.flatten(1).norm(p=2, dim=1)
        adv_inputs = inputs + δ
        logits = model(adv_inputs)
        pred_labels = logits.argmax(1)
        ce_loss = F.cross_entropy(logits, labels, reduction='none')
        loss = multiplier * ce_loss

        is_adv = (pred_labels == labels) if targeted else (pred_labels != labels)
        is_smaller = l2 < best_l2
        is_both = is_adv & is_smaller
        adv_found.logical_or_(is_adv)
        best_l2 = torch.where(is_both, l2, best_l2)
        best_δ = torch.where(batch_view(is_both), δ.detach(), best_δ)

        δ_grad = grad(loss.sum(), δ, only_inputs=True)[0]
        # renorming gradient
        grad_norms = δ_grad.flatten(1).norm(p=2, dim=1)
        δ_grad.div_(batch_view(grad_norms))
        # avoid nan or inf if gradient is 0
        if (zero_grad := (grad_norms < 1e-12)).any():
            δ_grad[zero_grad] = torch.randn_like(δ_grad[zero_grad])

        if callback is not None:
            cosine = F.cosine_similarity(δ_grad.flatten(1), δ.data.flatten(1), dim=1).mean()
            callback.accumulate_line('ce', i, ce_loss.mean())
            callback_best = best_l2.masked_select(adv_found).mean()
            callback.accumulate_line(['ε', 'l2', 'best_l2'], i, [ε.mean(), l2.mean(), callback_best])
            callback.accumulate_line(['cosine', 'α', 'success'], i, [cosine, α, adv_found.float().mean()])

            if (i + 1) % (steps // 20) == 0 or (i + 1) == steps:
                callback.update_lines()

        δ.data.add_(δ_grad, alpha=α)

        ε = torch.where(is_adv, (1 - γ) * ε, (1 + γ) * ε)
        ε = torch.minimum(ε, worst_norm)

        δ.data.mul_(batch_view(ε / δ.data.flatten(1).norm(p=2, dim=1)))
        δ.data.add_(inputs).clamp_(min=0, max=1)
        if levels is not None:
            δ.data.mul_(levels - 1).round_().div_(levels - 1)
        δ.data.sub_(inputs)

    return inputs + best_δ
