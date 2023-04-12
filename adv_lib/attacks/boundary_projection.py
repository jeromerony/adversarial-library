from typing import Optional

import torch
from torch import Tensor, nn
from torch.autograd import grad
from torch.nn import functional as F

from adv_lib.utils.visdom_logger import VisdomLogger


def bp(model: nn.Module,
       inputs: Tensor,
       labels: Tensor,
       targeted: bool = False,
       num_steps: int = 100,
       η: float = 0.4,
       γ_min: float = 0.7,
       γ_max: float = 1,
       α: float = 2,
       β_min: float = 0.1,
       levels: Optional[int] = 256,
       callback: Optional[VisdomLogger] = None) -> Tensor:
    """
    Boundary Projection (BP) attack from https://arxiv.org/abs/1912.02153.

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
    num_steps : int
        Number of optimization steps.
    γ_min : float
        Factor by which the norm will be modified. new_norm = norm * (1 + or - γ).
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

    # Init trackers
    best_l2 = torch.full((batch_size,), float('inf'), device=device)
    best_adv = inputs.clone()
    adv_found = torch.zeros(batch_size, dtype=torch.bool, device=device)

    for i in range(num_steps):
        adv_inputs = inputs + δ
        logits = model(adv_inputs)

        if i == 0:
            num_classes = logits.shape[1]
            one_hot_labels = F.one_hot(labels, num_classes=num_classes)

        # "softmax_cross_entropy_better" loss
        tmp = one_hot_labels * logits
        logits_1 = logits - tmp
        j_best = logits_1.amax(dim=1)
        logits_2 = logits_1 - j_best.unsqueeze(1) + one_hot_labels * j_best.unsqueeze(1)
        tmp_s = tmp.amax(dim=1)
        up = tmp_s - j_best
        down = logits_2.exp().add(1).sum(dim=1).log()
        loss = up - down

        l2 = δ.data.flatten(1).norm(p=2, dim=1)

        δ_grad = grad(multiplier * loss.sum(), δ, only_inputs=True)[0]
        δ_grad_l2 = δ_grad.flatten(1).norm(p=2, dim=1)
        δ_grad_normalized = δ_grad / batch_view(δ_grad_l2.clamp(min=1e-6))

        pred_labels = logits.argmax(1)
        is_adv = (pred_labels == labels) if targeted else (pred_labels != labels)
        is_smaller = l2 < best_l2
        is_both = is_adv & is_smaller
        adv_found.logical_or_(is_adv)
        best_l2 = torch.where(is_both, l2, best_l2)
        best_adv = torch.where(batch_view(is_both), adv_inputs.detach(), best_adv)

        if callback is not None:
            callback.accumulate_line('loss', i, loss.mean(), title='BP - Loss')
            callback_best = best_l2.masked_select(adv_found).mean()
            callback.accumulate_line(['l2', 'best_l2'], i, [l2.mean(), callback_best])
            callback.accumulate_line(['success'], i, [adv_found.float().mean()], title='BP - ASR')

            if (i + 1) % (num_steps // 20) == 0 or (i + 1) == num_steps:
                callback.update_lines()

        # step-size
        γ = γ_min + i / (num_steps + 1) * (γ_max - γ_min)

        # stage 1
        y_stage_1 = adv_inputs.data - (α * γ) * δ_grad_normalized

        # stage 2
        r = (δ.data / (batch_view(l2.clamp(min=1e-6))) * δ_grad_normalized).flatten(1).sum(dim=1)

        # case OUT
        ε = γ * l2
        v_star = inputs + batch_view(r) * δ_grad_normalized
        z = v_star + η * (adv_inputs.data - v_star) * batch_view((ε ** 2 - r ** 2).clamp_(min=0).sqrt())
        # Q_OUT: find β with binary search
        z_norm = z.flatten(1).norm(p=2, dim=1)
        β_low, β_up = torch.zeros_like(l2), torch.full_like(l2, 2)
        while ((β_up - β_low).abs() > 1e-3).any():
            β_mid = (β_up + β_low) / 2
            quantized = adv_inputs.data + batch_view(β_mid) * (z - adv_inputs.data)
            quantized.mul_(levels - 1).round_().div_(levels - 1)
            quantized_norm = quantized.flatten(1).norm(p=2, dim=1)
            condition = quantized_norm < z_norm
            β_low = torch.where(condition, β_mid, β_low)
            β_up = torch.where(condition, β_up, β_mid)
        β = (β_up + β_low) / 2
        y_out = adv_inputs.data + batch_view(β) * (z - adv_inputs.data)
        y_out.mul_(levels - 1).round_().div_(levels - 1)

        # case IN
        ε = l2 / γ
        z = adv_inputs.data - batch_view(r + (ε ** 2 - l2 ** 2 + r ** 2).sqrt()) * δ_grad_normalized
        β = (β_min / (z - adv_inputs.data).flatten(1).norm(p=2, dim=1)).clamp_(min=1)
        y_in = adv_inputs.data + batch_view(β) * (z - adv_inputs.data)
        y_in.mul_(levels - 1).round_().div_(levels - 1)

        # compose cases
        y_new = torch.where(batch_view(adv_found), torch.where(batch_view(is_adv), y_out, y_in), y_stage_1)
        # update optimization variable and constrain to box
        δ.data = y_new
        δ.data.clamp_(min=0, max=1).sub_(inputs)

    return best_adv
