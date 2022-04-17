# Adapted from https://github.com/aam-at/cpgd
import math
from functools import partial
from typing import Optional

import torch
from torch import Tensor, nn
from torch.autograd import grad
from torch.nn import functional as F

from adv_lib.distances.lp_norms import l0_distances, l1_distances, l2_distances, linf_distances
from adv_lib.utils.losses import difference_of_logits
from adv_lib.utils.projections import l1_ball_euclidean_projection
from adv_lib.utils.visdom_logger import VisdomLogger


def pdgd(model: nn.Module,
         inputs: Tensor,
         labels: Tensor,
         targeted: bool = False,
         num_steps: int = 500,
         random_init: float = 0,
         primal_lr: float = 0.1,
         primal_lr_decrease: float = 0.01,
         dual_ratio_init: float = 0.01,
         dual_lr: float = 0.1,
         dual_lr_decrease: float = 0.1,
         dual_ema: float = 0.9,
         dual_min_ratio: float = 1e-6,
         callback: Optional[VisdomLogger] = None) -> Tensor:
    """
    Primal-Dual Gradient Descent (PDGD) attack from https://arxiv.org/abs/2106.01538. This version is only suitable for
    the L2-norm.

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
        Number of optimization steps. Corresponds to the number of forward and backward propagations.
    random_init : float
        If random_init != 0, will start from a  random perturbation drawn from U(-random_init, random_init).
    primal_lr : float
        Learning rate for primal variables.
    primal_lr_decrease : float
        Final learning rate multiplier for primal variables.
    dual_ratio_init : float
        Initial ratio λ_0 / λ_1. A smaller value corresponds to a larger weight on the (mis)classification constraint.
    dual_lr : float
        Learning rate for dual variables.
    dual_lr_decrease : float
        Final learning rate multiplier for dual variables.
    dual_ema : float
        Coefficient for exponential moving average. Equivalent to no EMA if dual_ema == 0.
    dual_min_ratio : float
        Minimum ratio λ_0 / λ_1 and λ_1 / λ_0 to avoid having too large absolute values of λ_1.
    callback : VisdomLogger
        Callback to visualize the progress of the algorithm.

    Returns
    -------
    best_adv : Tensor
        Perturbed inputs (inputs + perturbation) that are adversarial and have smallest distance with the original
        inputs.

    """
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
    λ = torch.zeros(batch_size, 2, dtype=torch.float, device=device)
    λ[:, 1] = -math.log(dual_ratio_init)
    λ_ema = λ.softmax(dim=1)

    # Init trackers
    best_l2 = torch.full((batch_size,), float('inf'), device=device)
    best_adv = inputs.clone()
    adv_found = torch.zeros_like(best_l2, dtype=torch.bool)

    for i in range(num_steps):

        adv_inputs = inputs + r
        logits = model(adv_inputs)
        l2 = r.flatten(1).norm(p=2, dim=1)

        if i == 0:
            labels_infhot = torch.zeros_like(logits).scatter_(1, labels.unsqueeze(1), float('inf'))
            dl_func = partial(difference_of_logits, labels=labels, labels_infhot=labels_infhot)

        m_y = multiplier * dl_func(logits)

        is_adv = m_y < 0
        is_smaller = l2 < best_l2
        is_both = is_adv & is_smaller
        adv_found.logical_or_(is_adv)
        best_l2 = torch.where(is_both, l2.detach(), best_l2)
        best_adv = torch.where(batch_view(is_both), adv_inputs.detach(), best_adv)

        L_r = λ_ema[:, 0] * l2 + λ_ema[:, 1] * F.softplus(m_y.clamp(min=0))

        grad_r = grad(L_r.sum(), inputs=r, only_inputs=True)[0]
        grad_λ = m_y.detach().sign()

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
        λ[:, 1].add_(grad_λ, alpha=θ_λ).clamp_(min=log_min_dual_ratio, max=-log_min_dual_ratio)
        λ_ema.mul_(dual_ema).add_(λ.softmax(dim=1), alpha=1 - dual_ema)

        if callback is not None:
            callback.accumulate_line('m_y', i, m_y.mean(), title=f'{attack_name} - Logit difference')
            callback_best = best_l2.masked_select(adv_found).mean()
            callback.accumulate_line(['l2', 'best_l2'], i, [l2.mean(), callback_best],
                                     title=f'{attack_name} - L2 norms')
            callback.accumulate_line(['λ_1', 'λ_2'], i, [λ_ema[:, 0].mean(), λ_ema[:, 1].mean()],
                                     title=f'{attack_name} - Dual variables')
            callback.accumulate_line(['θ_r', 'θ_λ'], i, [step_size, θ_λ], title=f'{attack_name} - Learning rates')
            callback.accumulate_line('success', i, adv_found.float().mean(), title=f'{attack_name} - Success')

            if (i + 1) % (num_steps // 20) == 0 or (i + 1) == num_steps:
                callback.update_lines()

    return best_adv


def l0_proximal(x: Tensor, λ: Tensor) -> Tensor:
    thresholding = x.abs().amax(dim=1, keepdim=True) >= (2 * λ).sqrt_()
    return thresholding.float() * x


def l1_proximal(x: Tensor, λ: Tensor) -> Tensor:
    return x.abs().sub_(λ).clamp_(min=0).copysign_(x)


def l2_proximal(x: Tensor, λ: Tensor) -> Tensor:
    norms = x.flatten(1).norm(p=2, dim=1, keepdim=True)
    return (x.flatten(1) * (λ.flatten(1) / norms.neg_()).add_(1).clamp_(min=0)).view_as(x)


def linf_proximal(x: Tensor, λ: Tensor) -> Tensor:
    l1_projection = l1_ball_euclidean_projection(x=(x / λ).flatten(1), ε=1, inplace=True).view_as(x)
    return x - l1_projection.mul_(λ)


def l23_proximal(x: Tensor, λ: Tensor) -> Tensor:
    """Proximal operator for L_2/3 norm."""
    a = x.pow(4).mul_(1 / 256).sub_(λ.pow(3).mul_(8 / 729)).sqrt_()
    x_square_16 = x.square().mul_(1 / 16)
    b = x_square_16.add(a).pow_(1 / 3).add_(x_square_16.sub_(a).pow_(1 / 3)).mul_(2)
    b_sqrt = b.sqrt()
    z = x.abs().mul_(2).div_(b_sqrt).sub_(b).sqrt_().add_(b_sqrt).pow_(3).mul_(1 / 8).copysign_(x).nan_to_num_(nan=0)
    return z


def pdpgd(model: nn.Module,
          inputs: Tensor,
          labels: Tensor,
          norm: float,
          targeted: bool = False,
          num_steps: int = 500,
          random_init: float = 0,
          proximal_operator: Optional[float] = None,
          primal_lr: float = 0.1,
          primal_lr_decrease: float = 0.01,
          dual_ratio_init: float = 0.01,
          dual_lr: float = 0.1,
          dual_lr_decrease: float = 0.1,
          dual_ema: float = 0.9,
          dual_min_ratio: float = 1e-6,
          proximal_steps: int = 5,
          ε_threshold: float = 1e-2,
          callback: Optional[VisdomLogger] = None) -> Tensor:
    """
    Primal-Dual Proximal Gradient Descent (PDPGD) attacks from https://arxiv.org/abs/2106.01538.

    Parameters
    ----------
    model : nn.Module
        Model to attack.
    inputs : Tensor
        Inputs to attack. Should be in [0, 1].
    labels : Tensor
        Labels corresponding to the inputs if untargeted, else target labels.
    norm: float
        Norm to minimize.
    targeted : bool
        Whether to perform a targeted attack or not.
    num_steps : int
        Number of optimization steps. Corresponds to the number of forward and backward propagations.
    random_init : float
        If random_init != 0, will start from a  random perturbation drawn from U(-random_init, random_init).
    proximal_operator : float
        If not None, uses the corresponding proximal operator in [0, 23, 1, 2, float('inf')]. 23 corresponds to the
        L-2/3 proximal operator and is preferred to minimze the L0-norm instead of the L0 proximal operator.
    primal_lr : float
        Learning rate for primal variables.
    primal_lr_decrease : float
        Final learning rate multiplier for primal variables.
    dual_ratio_init : float
        Initial ratio λ_0 / λ_1. A smaller value corresponds to a larger weight on the (mis)classification constraint.
    dual_lr : float
        Learning rate for dual variables.
    dual_lr_decrease : float
        Final learning rate multiplier for dual variables.
    dual_ema : float
        Coefficient for exponential moving average. Equivalent to no EMA if dual_ema == 0.
    dual_min_ratio : float
        Minimum ratio λ_0 / λ_1 and λ_1 / λ_0 to avoid having too large absolute values of λ_1.
    proximal_steps : int
        Number of steps for proximal Adam (https://arxiv.org/abs/1910.10094).
    ε_threshold : float
        Convergence criterion for proximal Adam.
    callback : VisdomLogger
        Callback to visualize the progress of the algorithm.

    Returns
    -------
    best_adv : Tensor
        Perturbed inputs (inputs + perturbation) that are adversarial and have smallest distance with the original
        inputs.

    """
    attack_name = f'PDPGD L{norm}'
    _distance = {
        0: l0_distances,
        1: l1_distances,
        2: l2_distances,
        float('inf'): linf_distances,
    }
    _proximal_operator = {
        0: l0_proximal,
        1: l1_proximal,
        2: l2_proximal,
        float('inf'): linf_proximal,
        23: l23_proximal,
    }
    device = inputs.device
    batch_size = len(inputs)
    batch_view = lambda tensor: tensor.view(batch_size, *[1] * (inputs.ndim - 1))
    multiplier = -1 if targeted else 1
    distance = _distance[norm]
    proximity_operator = _proximal_operator[proximal_operator or norm]
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
    λ = torch.zeros(batch_size, 2, dtype=torch.float, device=device)
    λ[:, 1] = -math.log(dual_ratio_init)
    λ_ema = λ.softmax(dim=1)

    # Init trackers
    best_dist = torch.full((batch_size,), float('inf'), device=device)
    best_adv = inputs.clone()
    adv_found = torch.zeros_like(best_dist, dtype=torch.bool)

    for i in range(num_steps):

        adv_inputs = inputs + r
        logits = model(adv_inputs)
        dist = distance(adv_inputs.detach(), inputs)

        if i == 0:
            labels_infhot = torch.zeros_like(logits).scatter_(1, labels.unsqueeze(1), float('inf'))
            dl_func = partial(difference_of_logits, labels=labels, labels_infhot=labels_infhot)

        m_y = multiplier * dl_func(logits)

        is_adv = m_y < 0
        is_smaller = dist < best_dist
        is_both = is_adv & is_smaller
        adv_found.logical_or_(is_adv)
        best_dist = torch.where(is_both, dist.detach(), best_dist)
        best_adv = torch.where(batch_view(is_both), adv_inputs.detach(), best_adv)

        cls_loss = F.softplus(m_y.clamp(min=0))

        grad_r = grad(cls_loss.sum(), inputs=r, only_inputs=True)[0]
        grad_λ = m_y.detach().sign()

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
        μ = (λ_ema[:, 0] / λ_ema[:, 1]).mul_(effective_lr)
        H_div = denom / batch_view(ψ_max)
        for _ in range(proximal_steps):
            z_prev = z_curr

            z_new = proximity_operator((r.detach() - z_curr).mul_(H_div).add_(z_curr), batch_view(μ))
            z_new.add_(inputs).clamp_(min=0, max=1).sub_(inputs)

            z_curr = torch.where(batch_view(ε > ε_threshold), z_new, z_prev)
            ε = (z_curr - z_prev).flatten(1).norm(p=2, dim=1).div_(z_curr.flatten(1).norm(p=2, dim=1))

            if (ε < ε_threshold).all():
                break

        r.data = z_curr

        # gradient ascent on dual variables and exponential moving average
        θ_λ = dual_lr * ((num_steps - 1 - i) / (num_steps - 1) * (1 - dual_lr_decrease) + dual_lr_decrease)
        λ[:, 1].add_(grad_λ, alpha=θ_λ).clamp_(min=log_min_dual_ratio, max=-log_min_dual_ratio)
        λ_ema.mul_(dual_ema).add_(λ.softmax(dim=1), alpha=1 - dual_ema)

        if callback is not None:
            callback.accumulate_line('m_y', i, m_y.mean(), title=f'{attack_name} - Logit difference')
            callback_best = best_dist.masked_select(adv_found).mean()
            callback.accumulate_line([f'l{norm}', f'best_l{norm}'], i, [dist.mean(), callback_best],
                                     title=f'{attack_name} - L{norm} norms')
            callback.accumulate_line(['λ_1', 'λ_2'], i, [λ_ema[:, 0].mean(), λ_ema[:, 1].mean()],
                                     title=f'{attack_name} - Dual variables')
            callback.accumulate_line(['θ_r', 'θ_λ'], i, [step_size, θ_λ], title=f'{attack_name} - Learning rates')
            callback.accumulate_line('success', i, adv_found.float().mean(), title=f'{attack_name} - Success')

            if (i + 1) % (num_steps // 20) == 0 or (i + 1) == num_steps:
                callback.update_lines()

    return best_adv
