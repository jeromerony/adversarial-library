import math
from functools import partial
from typing import Callable, Optional

import torch
from torch import Tensor, nn
from torch.autograd import Function, grad

from adv_lib.utils.losses import difference_of_logits_ratio
from adv_lib.utils.visdom_logger import VisdomLogger


def prox_l1_indicator(δ: Tensor, λ: Tensor, lower: Tensor, upper: Tensor) -> Tensor:
    prox = δ.flatten(1).abs().sub_(λ.unsqueeze(1)).clamp_(min=0).view_as(δ).copysign_(δ)
    prox.clamp_(min=lower, max=upper)
    return prox


def prox_l2_square_indicator(δ: Tensor, λ: Tensor, lower: Tensor, upper: Tensor) -> Tensor:
    prox = δ.flatten(1).div(λ.unsqueeze(1).mul(2).add_(1)).view_as(δ)
    prox.clamp_(min=lower, max=upper)
    return prox


def prox_linf_indicator(δ: Tensor, λ: Tensor, lower: Tensor, upper: Tensor, ε: float = 1e-6,
                        section: float = 1 / 3) -> Tensor:
    δ_, λ_ = δ.flatten(1), λ.unsqueeze(1)
    δ_proj = δ_.clamp(min=lower.flatten(1), max=upper.flatten(1))
    right = δ_proj.abs().amax(dim=1, keepdim=True)
    left = torch.zeros_like(right)
    steps = (ε / right.max()).log_().mul_(math.log(math.e, 1 - section)).ceil_().long()
    prox, Δ, left_third, right_third, f_left, f_right, cond = (None,) * 7
    for _ in range(steps):
        Δ = torch.sub(right, left, out=Δ)
        Δ.mul_(section)
        left_third = torch.add(left, Δ, out=left_third)
        right_third = torch.sub(right, Δ, out=right_third)

        prox = torch.clamp(δ_proj, min=-left_third, max=left_third, out=prox)
        f_left = torch.sum(prox.sub_(δ_).square_(), dim=1, keepdim=True, out=f_left)
        f_left.div_(2).addcmul_(left_third, λ_)

        prox = torch.clamp(δ_proj, min=-right_third, max=right_third, out=prox)
        f_right = torch.sum(prox.sub_(δ_).square_(), dim=1, keepdim=True, out=f_right)
        f_right.div_(2).addcmul_(right_third, λ_)

        cond = torch.ge(f_left, f_right, out=cond)
        left = torch.where(cond, left_third, left)
        right = torch.where(cond, right, right_third)
    left.add_(right).div_(2)
    prox = torch.clamp(δ_proj, min=-left, max=left, out=prox).view_as(δ)
    return prox


def init_lr_finder(grad: Tensor, norm: float, target_distance: float, lower: Tensor, upper: Tensor) -> Tensor:
    batch_size = len(grad)
    batch_view = lambda tensor: tensor.view(batch_size, *[1] * (grad.ndim - 1))
    lr = torch.ones(batch_size, device=grad.device)
    low = torch.zeros_like(lr)

    found_upper = grad.neg().clamp_(lower, upper).flatten(1).norm(p=norm, dim=1) > target_distance
    while (~found_upper).any():
        low = torch.where(found_upper, low, lr)
        lr = torch.where(found_upper, lr, lr * 2)
        found_upper = grad.mul(-batch_view(lr)).clamp_(lower, upper).flatten(1).norm(p=norm, dim=1) > target_distance

    for i in range(20):
        new_lr = (low + lr) / 2
        larger = grad.mul(-batch_view(new_lr)).clamp_(lower, upper).flatten(1).norm(p=norm, dim=1) > target_distance
        low, lr = torch.where(larger, low, new_lr), torch.where(larger, new_lr, lr)

    return (lr + low) / 2


_prox = {
    1: prox_l1_indicator,
    2: prox_l2_square_indicator,
    float('inf'): partial(prox_linf_indicator, ε=1e-5),
}


class P(Function):
    @staticmethod
    def forward(ctx, y: Tensor, ρ: Tensor, μ: Tensor) -> Tensor:
        y_sup = μ * y + μ * ρ * y ** 2 + 1 / 6 * ρ ** 2 * y ** 3
        y_inf = μ * y / (1 - ρ.clamp(min=1) * y.clamp(max=0))
        sup = y >= 0
        ctx.save_for_backward(y, ρ, μ, sup)
        return torch.where(sup, y_sup, y_inf)

    @staticmethod
    def backward(ctx, grad_output):
        y, ρ, μ, sup = ctx.saved_tensors
        grad_y_sup = μ * y + 2 * μ * ρ * y + 1 / 2 * ρ ** 2 * y ** 2
        grad_y_inf = μ / (1 - ρ.clamp(min=1) * y.clamp(max=0)).square_()
        return grad_output * torch.where(sup, grad_y_sup, grad_y_inf), None, None, None


def alma_prox(model: nn.Module,
              inputs: Tensor,
              labels: Tensor,
              masks: Tensor = None,
              targeted: bool = False,
              adv_threshold: float = 0.99,
              penalty: Callable = P.apply,
              norm: float = float('inf'),
              num_steps: int = 500,
              lr_init: float = 0.1,
              lr_reduction: float = 0.1,
              init_lr_distance: Optional[float] = None,
              μ_init: float = 1e-4,
              ρ_init: float = 1,
              check_steps: int = 10,
              τ: float = 0.95,
              γ: float = 2,
              α: float = 0.8,
              α_rms: float = None,
              scale_min: float = 0.05,
              scale_max: float = 1,
              scale_init: Optional[float] = None,
              scale_γ: float = 0.02,
              logit_tolerance: float = 1e-4,
              constraint_masking: bool = True,
              mask_decay: bool = True,
              callback: Optional[VisdomLogger] = None) -> Tensor:
    attack_name = f'ALMA L{norm}'
    device = inputs.device
    batch_size = len(inputs)
    batch_view = lambda tensor: tensor.view(batch_size, *[1] * (inputs.ndim - 1))
    multiplier = -1 if targeted else 1
    α_rms = α if α_rms is None else α_rms

    # Setup variables
    δ = torch.zeros_like(inputs, requires_grad=True)
    lr = torch.full((batch_size,), lr_init, device=device, dtype=torch.float)
    s = torch.zeros_like(lr)
    lower, upper = -inputs, 1 - inputs

    prox_func = partial(_prox[float(norm)], lower=lower, upper=upper)

    # Init constraint parameters
    μ = torch.full_like(labels, μ_init, device=device, dtype=torch.float)
    ρ = torch.full_like(labels, ρ_init, device=device, dtype=torch.float)
    if scale_init is None:
        scale_init = math.exp(math.log(scale_min * scale_max) / 2)
    w = torch.full_like(lr, scale_init)  # constraint scale

    # Init trackers
    best_dist = torch.full_like(lr, float('inf'))
    best_adv_percent = torch.zeros_like(lr)
    adv_found = torch.zeros_like(lr, dtype=torch.bool)
    best_adv = inputs.clone()
    pixel_adv_found = torch.zeros_like(labels, dtype=torch.bool)
    step_found = torch.full_like(lr, num_steps // 2)

    for i in range(num_steps):

        adv_inputs = inputs + δ
        logits = model(adv_inputs)
        dist = δ.data.flatten(1).norm(p=norm, dim=1)

        if i == 0:
            # initialize variables based on model's output
            num_classes = logits.size(1)
            if masks is None:
                masks = labels < num_classes
            masks_sum = masks.flatten(1).sum(dim=1)
            labels_ = labels * masks
            masks_inf = torch.zeros_like(masks, dtype=torch.float).masked_fill_(~masks, float('inf'))
            labels_infhot = torch.zeros_like(logits.detach()).scatter_(1, labels_.unsqueeze(1), float('inf'))
            diff_func = partial(difference_of_logits_ratio, labels=labels_, labels_infhot=labels_infhot,
                                targeted=targeted, ε=logit_tolerance)
            k = ((1 - adv_threshold) * masks_sum).long()  # number of constraints that can be violated
            constraint_mask = masks

        # track progress
        pred = logits.argmax(dim=1)
        pixel_is_adv = (pred == labels) if targeted else (pred != labels)
        pixel_adv_found.logical_or_(pixel_is_adv)
        adv_percent = (pixel_is_adv & masks).flatten(1).sum(dim=1) / masks_sum
        is_adv = adv_percent >= adv_threshold
        is_smaller = dist <= best_dist
        improves_constraints = adv_percent >= best_adv_percent.clamp_max(adv_threshold)
        is_better_adv = (is_smaller & is_adv) | (~adv_found & improves_constraints)
        if i < num_steps // 2: step_found.masked_fill_((~adv_found) & is_adv, i)  # reduce lr before num_steps // 2
        adv_found.logical_or_(is_adv)
        best_dist = torch.where(is_better_adv, dist.detach(), best_dist)
        best_adv_percent = torch.where(is_better_adv, adv_percent, best_adv_percent)
        best_adv = torch.where(batch_view(is_better_adv), adv_inputs.detach(), best_adv)

        # adjust constraint scale
        w = torch.where(is_adv, w * (1 / (1 + scale_γ)), w * (1 / (1 - scale_γ)))
        w.clamp_(min=scale_min, max=scale_max)

        dlr = multiplier * diff_func(logits)
        constraints = w.view(-1, 1, 1) * dlr

        if constraint_masking:
            if mask_decay:
                k = ((1 - adv_threshold) * masks_sum).mul_(i / (num_steps - 1)).long()
            if k.any():
                top_constraints = constraints.detach().sub(masks_inf).flatten(1).topk(k=k.max()).values
                ξ = top_constraints.gather(1, k.unsqueeze(1) - 1).squeeze(1)
                constraint_mask = masks & (constraints <= ξ.view(-1, 1, 1))

        # adjust constraint parameters
        if i == 0:
            prev_constraints = constraints.detach()
        elif (i + 1) % check_steps == 0:
            improved_constraint = (constraints.detach() * constraint_mask <= τ * prev_constraints)
            ρ = torch.where(~(pixel_adv_found | improved_constraint), γ * ρ, ρ)
            prev_constraints = constraints.detach()
            pixel_adv_found.fill_(False)

        if i:
            new_μ = grad(penalty(constraints, ρ, μ)[constraint_mask].sum(), constraints, only_inputs=True)[0]
            μ.mul_(α).add_(new_μ, alpha=1 - α).clamp_(1e-12, 1)

        loss = penalty(constraints, ρ, μ).mul(constraint_mask).flatten(1).sum(dim=1)
        δ_grad = grad(loss.sum(), δ, only_inputs=True)[0]

        # decay and estimate step size
        grad_norm = δ_grad.flatten(1).norm(p=2, dim=1)
        if i == 0 and init_lr_distance is not None:
            if (zero_grad := (grad_norm <= 1e-8)).any():
                δ_grad[zero_grad] = torch.randn_like(δ_grad[zero_grad]).renorm_(dim=0, p=2, maxnorm=1)
                grad_norm[zero_grad] = 1
            lr = init_lr_finder(δ_grad / batch_view(grad_norm), norm=norm,
                                target_distance=init_lr_distance, lower=lower, upper=upper)

        if lr_reduction != 1:
            tangent = lr_reduction / (1 - lr_reduction) * (num_steps - step_found).clamp_(min=1)
            decay = tangent / ((i - step_found).clamp_(min=0) + tangent)
            step_lr = lr * decay.clamp_(min=lr_reduction)
        else:
            step_lr = lr

        s.mul_(α_rms).add_(δ_grad.flatten(1).square().sum(dim=1), alpha=1 - α_rms)
        λ = step_lr / (s.div(1 - α_rms ** (i + 1)).sqrt_() + 1e-6)

        # gradient step
        δ.data.addcmul_(δ_grad, batch_view(λ), value=-1)

        # proximal step
        δ.data = prox_func(δ=δ.data, λ=λ)

        if callback:
            callback.accumulate_line('const', i, constraints[masks].mean(), title=attack_name + ' - Constraints')
            callback.accumulate_line(['μ', 'ρ', 'scale'], i, [μ[masks].mean(), ρ[masks].mean(), w.mean()],
                                     title=attack_name + ' - Penalty parameters', ytype='log')
            callback.accumulate_line(['||g||₂', '√s'], i, [grad_norm.mean(), s.sqrt().mean()],
                                     title=attack_name + ' - Grad norm', ytype='log')
            callback.accumulate_line('λ', i, λ.mean(), title=attack_name + ' - Step size', ytype='log')
            callback.accumulate_line(['adv%', 'best_adv%'], i, [adv_percent.mean(), best_adv_percent.mean()],
                                     title=attack_name + ' - APSR')
            callback.accumulate_line([f'ℓ{norm}', f'best ℓ{norm}'], i,
                                     [dist.mean(), best_dist.mean()], title=attack_name + ' - Norms')

            if (i + 1) % (num_steps // 20) == 0 or (i + 1) == num_steps:
                callback.update_lines()

    return best_adv
