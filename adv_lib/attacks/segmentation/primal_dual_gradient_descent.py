import math
from functools import partial
from typing import Optional

import torch
from torch import Tensor, nn
from torch.autograd import grad
from torch.nn import functional as F

from adv_lib.attacks.primal_dual_gradient_descent import l0_proximal_, l1_proximal, l23_proximal, l2_proximal_, \
    linf_proximal_
from adv_lib.distances.lp_norms import l0_distances, l1_distances, l2_distances, linf_distances
from adv_lib.utils.losses import difference_of_logits
from adv_lib.utils.visdom_logger import VisdomLogger


def softmax_plus_one(tensor: torch.Tensor) -> torch.Tensor:
    zero_pad = F.pad(tensor.flatten(1), pad=(1, 0), mode='constant', value=0)
    return zero_pad.softmax(dim=1)[:, 1:].view_as(tensor)


def pdgd(model: nn.Module,
         inputs: Tensor,
         labels: Tensor,
         masks: Tensor = None,
         targeted: bool = False,
         adv_threshold: float = 0.99,
         num_steps: int = 500,
         random_init: float = 0,
         primal_lr: float = 0.1,
         primal_lr_decrease: float = 0.01,
         dual_ratio_init: float = 1,
         dual_lr: float = 0.1,
         dual_lr_decrease: float = 0.1,
         dual_ema: float = 0.9,
         dual_min_ratio: float = 1e-6,
         constraint_masking: bool = False,
         mask_decay: bool = False,
         callback: Optional[VisdomLogger] = None) -> Tensor:
    """Primal-Dual Gradient Descent (PDGD) attack from https://arxiv.org/abs/2106.01538 adapted to semantic
    segmentation. This version is only suitable for the L2-norm."""
    attack_name = 'PDGD L2'
    device = inputs.device
    batch_size = len(inputs)
    batch_view = lambda tensor: tensor.view(batch_size, *[1] * (inputs.ndim - 1))
    multiplier = -1 if targeted else 1
    log_min_dual_ratio = math.log(dual_min_ratio)

    # Setup variables
    r = torch.zeros_like(inputs, requires_grad=True)
    if random_init:
        nn.init.uniform_(r, -random_init, random_init)
        r.data.add_(inputs).clamp_(min=0, max=1).sub_(inputs)

    # Adam variables
    exp_avg = torch.zeros_like(inputs)
    exp_avg_sq = torch.zeros_like(inputs)
    β_1, β_2 = 0.9, 0.999

    # dual variables
    λ = torch.zeros_like(labels, dtype=torch.double)

    # Init trackers
    best_l2 = torch.full((batch_size,), float('inf'), device=device)
    best_adv = inputs.clone()
    adv_found = torch.zeros_like(best_l2, dtype=torch.bool)
    best_adv_percent = torch.zeros_like(best_l2)

    for i in range(num_steps):

        adv_inputs = inputs + r
        logits = model(adv_inputs)
        l2 = r.flatten(1).norm(p=2, dim=1)

        if i == 0:
            num_classes = logits.size(1)
            if masks is None:
                masks = labels < num_classes
            masks_sum = masks.flatten(1).sum(dim=1)
            labels_ = labels.clone()
            labels_[~masks] = 0

            masks_inf = torch.zeros_like(masks, dtype=torch.float).masked_fill_(~masks, float('inf'))
            labels_infhot = torch.zeros_like(logits.detach()).scatter(1, labels_.unsqueeze(1), float('inf'))
            dl_func = partial(difference_of_logits, labels=labels_, labels_infhot=labels_infhot)

            # init dual variables with masks
            λ.add_(masks_sum.float().mul_(dual_ratio_init).log_().neg_().view(-1, 1, 1))
            λ[~masks] = -float('inf')
            λ_ema = softmax_plus_one(λ)

            # init constraint masking
            k = ((1 - adv_threshold) * masks_sum).long()  # number of constraints that can be violated
            constraint_mask = masks
            constraint_inf_mask = torch.zeros_like(constraint_mask, dtype=torch.float)
            constraint_inf_mask.masked_fill_(~constraint_mask, float('inf'))

        # track progress
        pred = logits.argmax(dim=1)
        pixel_is_adv = (pred == labels) if targeted else (pred != labels)
        adv_percent = (pixel_is_adv & masks).flatten(1).sum(dim=1) / masks_sum
        is_adv = adv_percent >= adv_threshold
        is_smaller = l2 <= best_l2
        improves_constraints = adv_percent >= best_adv_percent.clamp_max(adv_threshold)
        is_better_adv = (is_smaller & is_adv) | (~adv_found & improves_constraints)
        adv_found.logical_or_(is_adv)
        best_l2 = torch.where(is_better_adv, l2.detach(), best_l2)
        best_adv_percent = torch.where(is_better_adv, adv_percent, best_adv_percent)
        best_adv = torch.where(batch_view(is_better_adv), adv_inputs.detach(), best_adv)

        m_y = multiplier * dl_func(logits)

        if constraint_masking:
            if mask_decay:
                k = ((1 - adv_threshold) * masks_sum).mul_(i / (num_steps - 1)).long()
            if k.any():
                top_constraints = m_y.detach().sub(masks_inf).flatten(1).topk(k=k.max()).values
                ξ = top_constraints.gather(1, k.unsqueeze(1) - 1).squeeze(1)
                constraint_mask = masks & (m_y <= ξ.view(-1, 1, 1))
                constraint_inf_mask.fill_(0).masked_fill_(~constraint_mask, float('inf'))

        if i:
            λ_ema.mul_(dual_ema).add_(softmax_plus_one(λ - constraint_inf_mask), alpha=1 - dual_ema)
        λ_ema_masked = λ_ema * constraint_mask
        λ_1 = 1 - λ_ema_masked.flatten(1).sum(dim=1)

        L_r = λ_1 * l2 + F.softplus(m_y).mul(λ_ema_masked).flatten(1).sum(dim=1)

        grad_r = grad(L_r.sum(), inputs=r, only_inputs=True)[0]
        grad_λ = m_y.detach().sign().mul_(masks)

        # Adam algorithm
        exp_avg.mul_(β_1).add_(grad_r, alpha=1 - β_1)
        exp_avg_sq.mul_(β_2).addcmul_(grad_r, grad_r, value=1 - β_2)
        bias_correction1 = 1 - β_1 ** (i + 1)
        bias_correction2 = 1 - β_2 ** (i + 1)
        denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(1e-8)
        # primal step size exponential decay
        step_size = primal_lr * primal_lr_decrease ** (i / num_steps)
        # gradient descent on primal variables
        r.data.addcdiv_(exp_avg, denom, value=-step_size / bias_correction1)

        # projection on feasible set
        r.data.add_(inputs).clamp_(min=0, max=1).sub_(inputs)

        # gradient ascent on dual variables and exponential moving average
        θ_λ = dual_lr * ((num_steps - 1 - i) / (num_steps - 1) * (1 - dual_lr_decrease) + dual_lr_decrease)
        λ.add_(grad_λ, alpha=θ_λ).clamp_(min=log_min_dual_ratio, max=-log_min_dual_ratio)
        λ[~masks] = -float('inf')

        if callback is not None:
            callback.accumulate_line('m_y', i, m_y.mean(), title=f'{attack_name} - Logit difference')
            callback.accumulate_line('1 - sum(λ)', i, λ_1.mean(), title=f'{attack_name} - Dual variables')
            callback.accumulate_line(['θ_r', 'θ_λ'], i, [step_size, θ_λ], title=f'{attack_name} - Learning rates')
            callback.accumulate_line(['l2', 'best_l2'], i, [l2.mean(), best_l2.mean()],
                                     title=f'{attack_name} - L2 norms')
            callback.accumulate_line(['adv%', 'best_adv%'], i, [adv_percent.mean(), best_adv_percent.mean()],
                                     title=f'{attack_name} - APSRs')

            if (i + 1) % (num_steps // 20) == 0 or (i + 1) == num_steps:
                callback.update_lines()

    return best_adv


def pdpgd(model: nn.Module,
          inputs: Tensor,
          labels: Tensor,
          norm: float,
          masks: Tensor = None,
          targeted: bool = False,
          adv_threshold: float = 0.99,
          num_steps: int = 500,
          random_init: float = 0,
          proximal_operator: Optional[float] = None,
          primal_lr: float = 0.1,
          primal_lr_decrease: float = 0.01,
          dual_ratio_init: float = 1,
          dual_lr: float = 0.1,
          dual_lr_decrease: float = 0.1,
          dual_ema: float = 0.9,
          dual_min_ratio: float = 1e-12,
          proximal_steps: int = 5,
          ε_threshold: float = 1e-2,
          constraint_masking: bool = False,
          mask_decay: bool = False,
          callback: Optional[VisdomLogger] = None) -> Tensor:
    """Primal-Dual Proximal Gradient Descent (PDPGD) attacks from https://arxiv.org/abs/2106.01538 adapted to semantic
    segmentation."""
    attack_name = f'PDPGD L{norm}'
    _distance = {
        0: l0_distances,
        1: l1_distances,
        2: l2_distances,
        float('inf'): linf_distances,
    }
    _proximal_operator = {
        0: l0_proximal_,
        1: l1_proximal,
        2: l2_proximal_,
        float('inf'): linf_proximal_,
        23: l23_proximal,
    }
    device = inputs.device
    batch_size = len(inputs)
    batch_view = lambda tensor: tensor.view(batch_size, *[1] * (inputs.ndim - 1))
    multiplier = -1 if targeted else 1
    distance = _distance[norm]
    proximity_operator = _proximal_operator[norm if proximal_operator is None else proximal_operator]
    log_min_dual_ratio = math.log(dual_min_ratio)

    # Setup variables
    r = torch.zeros_like(inputs, requires_grad=True)
    if random_init:
        nn.init.uniform_(r, -random_init, random_init)
        r.data.add_(inputs).clamp_(min=0, max=1).sub_(inputs)

    # Adam variables
    exp_avg = torch.zeros_like(inputs)
    exp_avg_sq = torch.zeros_like(inputs)
    β_1, β_2 = 0.9, 0.999

    # dual variables
    λ = torch.zeros_like(labels, dtype=torch.double)

    # Init trackers
    best_dist = torch.full((batch_size,), float('inf'), device=device)
    best_adv = inputs.clone()
    adv_found = torch.zeros_like(best_dist, dtype=torch.bool)
    best_adv_percent = torch.zeros_like(best_dist)

    for i in range(num_steps):

        adv_inputs = inputs + r
        logits = model(adv_inputs)
        dist = distance(adv_inputs.detach(), inputs)

        if i == 0:
            num_classes = logits.size(1)
            if masks is None:
                masks = labels < num_classes
            masks_sum = masks.flatten(1).sum(dim=1)
            labels_ = labels.clone()
            labels_[~masks] = 0

            masks_inf = torch.zeros_like(masks, dtype=torch.float).masked_fill_(~masks, float('inf'))
            labels_infhot = torch.zeros_like(logits.detach()).scatter(1, labels_.unsqueeze(1), float('inf'))
            dl_func = partial(difference_of_logits, labels=labels_, labels_infhot=labels_infhot)

            # init dual variables with masks
            λ.add_(masks_sum.float().mul_(dual_ratio_init).log_().neg_().view(-1, 1, 1))
            λ[~masks] = -float('inf')
            λ_ema = softmax_plus_one(λ)

            # init constraint masking
            k = ((1 - adv_threshold) * masks_sum).long()  # number of constraints that can be violated
            constraint_mask = masks
            constraint_inf_mask = torch.zeros_like(constraint_mask, dtype=torch.float)
            constraint_inf_mask.masked_fill_(~constraint_mask, float('inf'))

        # track progress
        pred = logits.argmax(dim=1)
        pixel_is_adv = (pred == labels) if targeted else (pred != labels)
        adv_percent = (pixel_is_adv & masks).flatten(1).sum(dim=1) / masks_sum
        is_adv = adv_percent >= adv_threshold
        is_smaller = dist <= best_dist
        improves_constraints = adv_percent >= best_adv_percent.clamp_max(adv_threshold)
        is_better_adv = (is_smaller & is_adv) | (~adv_found & improves_constraints)
        adv_found.logical_or_(is_adv)
        best_dist = torch.where(is_better_adv, dist.detach(), best_dist)
        best_adv_percent = torch.where(is_better_adv, adv_percent, best_adv_percent)
        best_adv = torch.where(batch_view(is_better_adv), adv_inputs.detach(), best_adv)

        m_y = multiplier * dl_func(logits)

        if constraint_masking:
            if mask_decay:
                k = ((1 - adv_threshold) * masks_sum).mul_(i / (num_steps - 1)).long()
            if k.any():
                top_constraints = m_y.detach().sub(masks_inf).flatten(1).topk(k=k.max()).values
                ξ = top_constraints.gather(1, k.unsqueeze(1) - 1).squeeze(1)
                constraint_mask = masks & (m_y <= ξ.view(-1, 1, 1))
                constraint_inf_mask.fill_(0).masked_fill_(~constraint_mask, float('inf'))

        if i:
            λ_ema.mul_(dual_ema).add_(softmax_plus_one(λ - constraint_inf_mask), alpha=1 - dual_ema)
        λ_ema_masked = λ_ema * constraint_mask

        cls_loss = F.softplus(m_y).mul(λ_ema_masked).flatten(1).sum(dim=1)

        grad_r = grad(cls_loss.sum(), inputs=r, only_inputs=True)[0]
        grad_λ = m_y.detach().sign().mul_(constraint_mask)

        # Adam algorithm
        exp_avg.mul_(β_1).add_(grad_r, alpha=1 - β_1)
        exp_avg_sq.mul_(β_2).addcmul_(grad_r, grad_r, value=1 - β_2)
        bias_correction1 = 1 - β_1 ** (i + 1)
        bias_correction2 = 1 - β_2 ** (i + 1)
        denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(1e-8)
        # primal step size exponential decay
        step_size = primal_lr * primal_lr_decrease ** (i / num_steps)
        # gradient descent on primal variables
        r.data.addcdiv_(exp_avg, denom, value=-step_size / bias_correction1)

        # projection on feasible set
        r.data.add_(inputs).clamp_(min=0, max=1).sub_(inputs)

        # proximal adam https://arxiv.org/abs/1910.10094
        ψ_max = denom.flatten(1).amax(dim=1)
        effective_lr = step_size / ψ_max

        # proximal sub-iterations variables
        z_curr = r.detach()
        ε = torch.ones_like(best_dist)
        λ_sum = λ_ema_masked.flatten(1).sum(dim=1)
        μ = ((1 - λ_sum) / λ_sum).to(dtype=torch.float).mul_(effective_lr)
        H_div = denom / batch_view(ψ_max)
        for _ in range(proximal_steps):
            z_prev = z_curr

            z_new = proximity_operator(z_curr.addcmul(H_div, z_curr - r.detach(), value=-1), batch_view(μ))
            z_new.add_(inputs).clamp_(min=0, max=1).sub_(inputs)

            z_curr = torch.where(batch_view(ε > ε_threshold), z_new, z_prev)
            ε = torch.norm((z_curr - z_prev).flatten(1), p=2, dim=1, out=ε).div_(z_curr.flatten(1).norm(p=2, dim=1))

            if (ε < ε_threshold).all():
                break

        r.data = z_curr

        # gradient ascent on dual variables and exponential moving average
        θ_λ = dual_lr * ((num_steps - 1 - i) / (num_steps - 1) * (1 - dual_lr_decrease) + dual_lr_decrease)
        λ.add_(grad_λ, alpha=θ_λ).clamp_(min=log_min_dual_ratio, max=-log_min_dual_ratio)
        λ[~masks] = -float('inf')

        if callback is not None:
            callback.accumulate_line('m_y', i, m_y.mean(), title=f'{attack_name} - Logit difference')
            callback.accumulate_line('1 - sum(λ)', i, (1 - λ_sum).mean(), title=f'{attack_name} - Dual variables')
            callback.accumulate_line(['θ_r', 'θ_λ'], i, [step_size, θ_λ], title=f'{attack_name} - Learning rates')
            callback.accumulate_line([f'l{norm}', f'best_l{norm}'], i, [dist.mean(), best_dist.mean()],
                                     title=f'{attack_name} - L{norm} norms')
            callback.accumulate_line(['adv%', 'best_adv%'], i, [adv_percent.mean(), best_adv_percent.mean()],
                                     title=f'{attack_name} - APSRs')

            if (i + 1) % (num_steps // 20) == 0 or (i + 1) == num_steps:
                callback.update_lines()

    return best_adv
