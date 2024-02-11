import math
from functools import partial
from typing import Callable, Optional

import torch
from torch import Tensor, nn
from torch.autograd import Function, grad

from adv_lib.utils.losses import difference_of_logits_ratio
from adv_lib.utils.visdom_logger import VisdomLogger


def prox_linf_indicator(δ: Tensor, λ: Tensor, lower: Tensor, upper: Tensor, H: Optional[Tensor] = None,
                        ε: float = 1e-6, section: float = 1 / 3) -> Tensor:
    """Proximity operator of λ||·||_∞ + \iota_Λ in the diagonal metric H. The lower and upper tensors correspond to
    the bounds of Λ. The problem is solved using a ternary search with section 1/3 up to an absolute error of ε on the
    prox. Using a section of 1 - 1/φ (with φ the golden ratio) yields the Golden-section search, which is a bit faster,
    but less numerically stable."""
    δ_, λ_ = δ.flatten(1), 2 * λ.unsqueeze(1)
    H_ = H.flatten(1) if H is not None else None
    δ_proj = δ_.clamp(min=lower.flatten(1), max=upper.flatten(1))
    right = δ_proj.norm(p=float('inf'), dim=1, keepdim=True)
    left = torch.zeros_like(right)
    steps = (ε / right.max()).log_().mul_(math.log(math.e, 1 - section)).ceil_().long()
    prox, left_third, right_third, f_left, f_right, cond = (None,) * 6
    for _ in range(steps):
        left_third = torch.lerp(left, right, weight=section, out=left_third)
        right_third = torch.lerp(left, right, weight=1 - section, out=right_third)

        prox = torch.clamp(δ_proj, min=-left_third, max=left_third, out=prox).sub_(δ_).square_()
        if H_ is not None:
            prox.mul_(H_)
        f_left = torch.sum(prox, dim=1, keepdim=True, out=f_left)
        f_left.addcmul_(left_third, λ_)

        prox = torch.clamp(δ_proj, min=-right_third, max=right_third, out=prox).sub_(δ_).square_()
        if H_ is not None:
            prox.mul_(H_)
        f_right = torch.sum(prox, dim=1, keepdim=True, out=f_right)
        f_right.addcmul_(right_third, λ_)

        cond = torch.ge(f_left, f_right, out=cond)
        left = torch.where(cond, left_third, left, out=left)
        right = torch.where(cond, right, right_third, out=right)
    left.lerp_(right, weight=0.5)
    return δ_proj.clamp_(min=-left, max=left).view_as(δ)


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
              num_steps: int = 500,
              lr_init: float = 0.001,
              lr_reduction: float = 0.1,
              μ_init: float = 1,
              ρ_init: float = 0.01,
              check_steps: int = 10,
              τ: float = 0.95,
              γ: float = 2,
              α: float = 0.8,
              α_rms: float = None,
              scale_min: float = 0.1,
              scale_max: float = 1,
              scale_init: float = 1,
              scale_γ: float = 0.02,
              logit_tolerance: float = 1e-4,
              constraint_masking: bool = True,
              mask_decay: bool = True,
              callback: Optional[VisdomLogger] = None) -> Tensor:
    """
    ALMA prox attack from https://arxiv.org/abs/2206.07179 to find $\ell_\infty$ perturbations.

    Parameters
    ----------
    model : nn.Module
        Model to attack.
    inputs : Tensor
        Inputs to attack. Should be in [0, 1].
    labels : Tensor
        Labels corresponding to the inputs if untargeted, else target labels.
    masks : Tensor
        Binary mask indicating which pixels to attack, to account for unlabeled pixels (e.g. void in Pascal VOC)
    targeted : bool
        Whether to perform a targeted attack or not.
    adv_threshold : float
        Fraction of pixels required to consider an attack successful.
    penalty : Callable
        Penalty-Lagrangian function to use. A good default choice is P2 (see the original article).
    num_steps : int
        Number of optimization steps. Corresponds to the number of forward and backward propagations.
    lr_init : float
        Initial learning rate.
    lr_reduction : float
        Reduction factor for the learning rate. The final learning rate is lr_init * lr_reduction
    μ_init : float
        Initial value of the penalty multiplier.
    ρ_init : float
        Initial value of the penalty parameter.
    check_steps : int
        Number of steps between checks for the improvement of the constraint. This corresponds to M in the original
        article.
    τ : float
        Constraint improvement rate.
    γ : float
        Penalty parameter increase rate.
    α : float
        Weight for the exponential moving average.
    α_rms : float
        Smoothing constant for gradient normalization. If none is provided, defaults to α.
    scale_min : float
        Minimum constraint scale, corresponding to w_min in the paper.
    scale_max : float
        Maximum constraint scale.
    scale_init : float
        Initial constraint scale w^{(0)}.
    scale_γ : float
        Constraint scale adjustment rate.
    logit_tolerance : float
        Small quantity added to the difference of logits to avoid solutions where the difference of logits is 0, which
        can results in inconsistent class prediction (using argmax) on GPU. This can also be used as a confidence
        parameter κ as in https://arxiv.org/abs/1608.04644, however, a confidence parameter on logits is not robust to
        scaling of the logits.
    constraint_masking : bool
        Discard (1 - adv_threshold) fraction of the largest constraints, which are less likely to be satisfied.
    mask_decay : bool
        Linearly decrease the number of discarded constraints.
    callback : VisdomLogger
        Callback to visualize the progress of the algorithm.

    Returns
    -------
    best_adv : Tensor
        Perturbed inputs (inputs + perturbation) that are adversarial and have smallest perturbation norm.

    """
    attack_name = f'ALMA prox'
    device = inputs.device
    batch_size = len(inputs)
    batch_view = lambda tensor: tensor.view(batch_size, *[1] * (inputs.ndim - 1))
    multiplier = -1 if targeted else 1
    α_rms = α if α_rms is None else α_rms

    # Setup variables
    δ = torch.zeros_like(inputs, requires_grad=True)
    lr = torch.full((batch_size,), lr_init, device=device, dtype=torch.float)
    s = torch.zeros_like(δ)
    lower, upper = -inputs, 1 - inputs
    prox_func = partial(prox_linf_indicator, lower=lower, upper=upper)

    # Init constraint parameters
    μ = torch.full_like(labels, μ_init, device=device, dtype=torch.double)
    ρ = torch.full_like(labels, ρ_init, device=device, dtype=torch.double)
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
        dist = δ.data.flatten(1).norm(p=float('inf'), dim=1)

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
        w.div_(torch.where(is_adv, 1 + scale_γ, 1 - scale_γ)).clamp_(min=scale_min, max=scale_max)

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
            c = constraints.to(dtype=μ.dtype)
            new_μ = grad(penalty(c, ρ, μ)[constraint_mask].sum(), c, only_inputs=True)[0]
            μ.lerp_(new_μ, weight=1 - α).clamp_(1e-12, 1)

        loss = penalty(constraints, ρ, μ).mul(constraint_mask).flatten(1).sum(dim=1)
        δ_grad = grad(loss.sum(), δ, only_inputs=True)[0]

        if lr_reduction != 1:
            tangent = lr_reduction / (1 - lr_reduction) * (num_steps - step_found).clamp_(min=1)
            decay = tangent / ((i - step_found).clamp_(min=0) + tangent)
            λ = lr * decay
        else:
            λ = lr

        s.mul_(α_rms).addcmul_(δ_grad, δ_grad, value=1 - α_rms)
        H = s.div(1 - α_rms ** (i + 1)).sqrt_().clamp_(min=1e-8)

        # gradient step
        δ.data.addcmul_(δ_grad, batch_view(λ) / H, value=-1)

        # proximal step
        δ.data = prox_func(δ=δ.data, λ=λ, H=H)

        if isinstance(callback, VisdomLogger):
            callback.accumulate_line('const', i, constraints[masks].mean(), title=attack_name + ' - Constraints')
            callback.accumulate_line(['μ', 'ρ', 'scale'], i, [μ[masks].mean(), ρ[masks].mean(), w.mean()],
                                     title=attack_name + ' - Penalty parameters', ytype='log')
            callback.accumulate_line(['mean(H)', 'min(H)', 'max(H)'], i, [H.mean(), H.min(), H.max()],
                                     title=attack_name + ' - Metric', ytype='log')
            callback.accumulate_line('λ', i, λ.mean(), title=attack_name + ' - Step size', ytype='log')
            callback.accumulate_line(['adv%', 'best_adv%'], i, [adv_percent.mean(), best_adv_percent.mean()],
                                     title=attack_name + ' - APSR')
            callback.accumulate_line([f'ℓ∞', f'best ℓ∞'], i,
                                     [dist.mean(), best_dist.mean()], title=attack_name + ' - Norms')

            if (i + 1) % (num_steps // 20) == 0 or (i + 1) == num_steps:
                callback.update_lines()

    return best_adv
