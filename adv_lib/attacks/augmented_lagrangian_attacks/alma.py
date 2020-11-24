from functools import partial
from typing import Optional, Callable

import torch
from torch import nn, Tensor
from torch.autograd import grad

from adv_lib.distances.color_difference import ciede2000_loss
from adv_lib.distances.lp_norms import l2_distances, l1_distances, linf_distances
from adv_lib.distances.lpips import LPIPS
from adv_lib.distances.structural_similarity import ssim_loss, ms_ssim_loss
from adv_lib.utils.losses import difference_of_logits_ratio
from adv_lib.utils.optimizers import RMSprop
from adv_lib.utils.visdom_logger import VisdomLogger


def init_lr_finder(inputs: Tensor, grad: Tensor, distance_function: Callable, target_distance: float):
    batch_size = len(inputs)
    batch_view = lambda tensor: tensor.view(batch_size, *[1] * (inputs.ndim - 1))
    lr = torch.ones(batch_size, device=inputs.device)
    lower = torch.zeros_like(lr)

    found_upper = distance_function((inputs - grad).clamp(0, 1)) > target_distance
    while (~found_upper).any():
        lower = torch.where(found_upper, lower, lr)
        lr = torch.where(found_upper, lr, lr * 2)
        found_upper = distance_function((inputs - batch_view(lr) * grad).clamp(0, 1)) > target_distance

    for i in range(20):
        new_lr = (lower + lr) / 2
        larger = distance_function((inputs - batch_view(new_lr) * grad).clamp(0, 1)) > target_distance
        lower, lr = torch.where(larger, lower, new_lr), torch.where(larger, new_lr, lr)

    return (lr + lower) / 2


_distances = {
    'ssim': ssim_loss,
    'msssim': ms_ssim_loss,
    'ciede2000': partial(ciede2000_loss, ε=1e-12),
    'lpips': LPIPS,
    'l2': l2_distances,
    'l1': l1_distances,
}


def alma(model: nn.Module,
         inputs: Tensor,
         labels: Tensor,
         penalty: Callable,
         targeted: bool = False,
         num_steps: int = 1000,
         lr_init: float = 0.1,
         lr_reduction: float = 0.01,
         distance: str = 'l2',
         init_lr_distance: Optional[float] = None,
         μ_init: float = 1,
         ρ_init: float = 1,
         check_steps: int = 10,
         τ: float = 0.95,
         γ: float = 1.2,
         α: float = 0.9,
         α_rms: Optional[float] = None,
         momentum: Optional[float] = None,
         logit_tolerance: float = 1e-4,
         levels: Optional[int] = None,
         callback: Optional[VisdomLogger] = None) -> Tensor:
    device = inputs.device
    batch_size = len(inputs)
    batch_view = lambda tensor: tensor.view(batch_size, *[1] * (inputs.ndim - 1))
    multiplier = -1 if targeted else 1

    # Setup variables
    δ = torch.zeros_like(inputs, requires_grad=True)
    optimizer = RMSprop([δ], lr=1, alpha=α_rms or α, momentum=momentum or α)
    lr = torch.full((batch_size,), lr_init, device=device, dtype=torch.float)

    # Init rho and mu
    μ = torch.full((batch_size,), μ_init, device=device, dtype=torch.float)
    ρ = torch.full((batch_size,), ρ_init, device=device, dtype=torch.float)

    # Init similarity metric
    if distance in ['lpips']:
        dist_func = _distances[distance](target=inputs)
    else:
        dist_func = partial(_distances[distance], inputs)

    # Init trackers
    best_dist = torch.full((batch_size,), float('inf'), device=device)
    best_adv = inputs.clone()
    adv_found = torch.zeros_like(best_dist, dtype=torch.bool)
    step_found = torch.full_like(best_dist, num_steps + 1)

    for i in range(num_steps):

        adv_inputs = inputs + δ
        logits = model(adv_inputs)
        dist = dist_func(adv_inputs)

        if i == 0:
            labels_infhot = torch.zeros_like(logits.detach()).scatter(1, labels.unsqueeze(1), float('inf'))
            dlr_func = partial(difference_of_logits_ratio, labels=labels, labels_infhot=labels_infhot,
                               targeted=targeted, ε=logit_tolerance)

        dlr = multiplier * dlr_func(logits)

        if i == 0:
            prev_dlr = dlr.detach()
        elif (i + 1) % check_steps == 0:
            improved_dlr = (dlr.detach() < τ * prev_dlr)
            ρ = torch.where(~(adv_found | improved_dlr), γ * ρ, ρ)
            prev_dlr = dlr.detach()

        if i:
            new_μ = grad(penalty(dlr, ρ, μ).sum(), dlr, only_inputs=True)[0]
            μ.mul_(α).add_(new_μ, alpha=1 - α).clamp_(1e-6, 1e12)

        is_adv = dlr < 0
        is_smaller = dist < best_dist
        is_both = is_adv & is_smaller
        step_found.masked_fill_((~adv_found) & is_adv, i)
        adv_found.masked_fill_(is_adv, True)
        best_dist = torch.where(is_both, dist.detach(), best_dist)
        best_adv = torch.where(batch_view(is_both), adv_inputs.detach(), best_adv)

        if i == 0:
            loss = penalty(dlr, ρ, μ)
        else:
            loss = dist + penalty(dlr, ρ, μ)
        δ_grad = grad(loss.sum(), δ, only_inputs=True)[0]

        grad_norm = δ_grad.flatten(1).norm(p=2, dim=1)
        if init_lr_distance is not None and i == 0:
            randn_grad = torch.randn_like(δ_grad).renorm(dim=0, p=2, maxnorm=1)
            δ_grad = torch.where(batch_view(grad_norm <= 1e-6), randn_grad, δ_grad)
            lr = init_lr_finder(inputs, δ_grad, dist_func, target_distance=init_lr_distance)

        δ.grad = δ_grad

        exp_decay = lr_reduction ** ((i - step_found).clamp_min(0) / (num_steps - step_found))
        optimizer.param_groups[0]['lr'] = batch_view(lr * exp_decay)
        optimizer.step()
        δ.data.add_(inputs).clamp_(0, 1)
        if levels is not None:
            δ.data.mul_(levels - 1).round_().div_(levels - 1)
        δ.data.sub_(inputs)

        if callback:
            cb_best_dist = best_dist.masked_select(adv_found).mean()
            cb_lr = torch.as_tensor(optimizer.param_groups[0]['lr'], device=device).mean()
            callback.accumulate_line([distance, 'dlr'], i, [dist.mean(), dlr.mean()])
            callback.accumulate_line(['μ_c', 'ρ_c'], i, [μ.mean(), ρ.mean()])
            callback.accumulate_line('grad_norm', i, grad_norm.mean())
            callback.accumulate_line(['best_{}'.format(distance), 'success', 'lr'], i,
                                     [cb_best_dist, adv_found.float().mean(), cb_lr])

            if (i + 1) % (num_steps // 20) == 0 or (i + 1) == num_steps:
                callback.update_lines()

    return best_adv
