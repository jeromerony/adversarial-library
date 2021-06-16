from functools import partial
from typing import Optional

import torch
from torch import Tensor, nn, optim
from torch.autograd import grad
from torch.nn import functional as F

from adv_lib.utils.losses import difference_of_logits
from adv_lib.utils.visdom_logger import VisdomLogger


def pdgd(model: nn.Module,
         inputs: Tensor,
         labels: Tensor,
         targeted: bool = False,
         num_steps: int = 500,
         random_init: float = 0,
         primal_lr: float = 0.1,
         primal_lr_decrease: float = 0.01,
         dual_lr: float = 0.1,
         dual_lr_decrease: float = 0.1,
         dual_ema: float = 0.9,
         callback: Optional[VisdomLogger] = None) -> Tensor:
    """
    Primal-Dual Gradient Descent (PDGD) attack from https://arxiv.org/abs/2106.01538.

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
    dual_lr : float
        Learning rate for dual variables.
    dual_lr_decrease : float
        Final learning rate multiplier for dual variables.
    dual_ema : float
        Coefficient for exponential moving average. Equivalent to no EMA if dual_ema == 0.
    callback : VisdomLogger
        Callback to visualize the progress of the algorithm.

    Returns
    -------
    best_adv : Tensor
        Perturbed inputs (inputs + perturbation) that are adversarial and have smallest distance with the original
        inputs.

    """
    device = inputs.device
    batch_size = len(inputs)
    batch_view = lambda tensor: tensor.view(batch_size, *[1] * (inputs.ndim - 1))
    multiplier = -1 if targeted else 1

    # Setup variables
    r = torch.zeros_like(inputs, requires_grad=True)
    if random_init:
        nn.init.uniform_(r, -random_init, random_init)
        r.data.add_(inputs).clamp_(min=0, max=1).sub_(inputs)
    optimizer = optim.Adam([r], lr=primal_lr)
    lr_lambda = lambda i: primal_lr_decrease ** (i / num_steps)
    scheduler = optim.lr_scheduler.LambdaLR(optimizer=optimizer, lr_lambda=lr_lambda)
    λ = torch.full((batch_size, 2), 0, dtype=torch.float, device=device)
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
            labels_infhot = torch.zeros_like(logits.detach()).scatter(1, labels.unsqueeze(1), float('inf'))
            dl_func = partial(difference_of_logits, labels=labels, labels_infhot=labels_infhot)

        m_y = multiplier * dl_func(logits)

        is_adv = m_y < 0
        is_smaller = l2 < best_l2
        is_both = is_adv & is_smaller
        adv_found.logical_or_(is_adv)
        best_l2 = torch.where(is_both, l2.detach(), best_l2)
        best_adv = torch.where(batch_view(is_both), adv_inputs.detach(), best_adv)

        L_r = λ_ema[:, 0] * l2 + λ_ema[:, 1] * F.softplus(m_y.clamp_min(0))

        grad_r = grad(L_r.sum(), inputs=r, only_inputs=True)[0]
        grad_λ = m_y.detach().sign()

        # gradient descent on primal variables
        r.grad = grad_r
        optimizer.step()
        scheduler.step()
        r.data.add_(inputs).clamp_(min=0, max=1).sub_(inputs)

        # gradient ascent on dual variables and exponential moving average
        α_λ = dual_lr * ((num_steps - 1 - i) / (num_steps - 1) * (1 - dual_lr_decrease) + dual_lr_decrease)
        λ[:, 1].add_(grad_λ, alpha=α_λ)
        λ_ema.mul_(dual_ema).add_(λ.softmax(dim=1), alpha=1 - dual_ema)

        if callback is not None:
            callback.accumulate_line('m_y', i, m_y.mean())
            callback_best = best_l2.masked_select(adv_found).mean()
            callback.accumulate_line(['l2', 'best_l2'], i, [l2.mean(), callback_best])
            callback.accumulate_line(['λ_1', 'λ_2'], i, [λ_ema[:, 0].mean(), λ_ema[:, 1].mean()])
            callback.accumulate_line(['lr', 'λ_lr'], i, [optimizer.param_groups[0]['lr'], α_λ])
            callback.accumulate_line('success', i, adv_found.float().mean())

            if (i + 1) % (num_steps // 20) == 0 or (i + 1) == num_steps:
                callback.update_lines()

    return best_adv
