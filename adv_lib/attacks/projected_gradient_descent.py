from functools import partial
from typing import Optional, Tuple

import torch
from torch import nn, Tensor
from torch.autograd import grad
from torch.nn import functional as F

from adv_lib.utils.losses import difference_of_logits, difference_of_logits_ratio
from adv_lib.utils.visdom_logger import VisdomLogger


def pgd_linf(model: nn.Module,
             inputs: Tensor,
             labels: Tensor,
             ε: float,
             targeted: bool = False,
             steps: int = 40,
             random_init: bool = True,
             restarts: int = 1,
             loss_function: str = 'ce',
             relative_step_size: float = 0.01 / 0.3,
             absolute_step_size: Optional[float] = None,
             callback: Optional[VisdomLogger] = None) -> Tensor:
    device = inputs.device
    batch_size = len(inputs)

    adv_inputs = inputs.clone()
    adv_found = torch.zeros(batch_size, dtype=torch.bool, device=device)
    pgd_attack = partial(_pgd_linf, model=model, ε=ε, targeted=targeted, steps=steps, random_init=random_init,
                         loss_function=loss_function, relative_step_size=relative_step_size,
                         absolute_step_size=absolute_step_size)

    for i in range(restarts):

        adv_found_run, adv_inputs_run = pgd_attack(inputs=inputs[~adv_found], labels=labels[~adv_found])
        adv_inputs[~adv_found] = adv_inputs_run
        adv_found[~adv_found] = adv_found_run

        if callback:
            callback.line('success', i + 1, adv_found.float().mean())

        if adv_found.all():
            break

    return adv_inputs


def _pgd_linf(model: nn.Module,
              inputs: Tensor,
              labels: Tensor,
              ε: float,
              targeted: bool = False,
              steps: int = 40,
              random_init: bool = True,
              loss_function: str = 'ce',
              relative_step_size: float = 0.01 / 0.3,
              absolute_step_size: Optional[float] = None) -> Tuple[Tensor, Tensor]:
    _loss_functions = {
        'ce': (partial(F.cross_entropy, reduction='none'), 1),
        'dl': (difference_of_logits, -1),
        'dlr': (partial(difference_of_logits_ratio, targeted=targeted), -1),
    }

    device = inputs.device
    batch_size = len(inputs)
    batch_view = lambda tensor: tensor.view(batch_size, *[1] * (inputs.ndim - 1))
    clamp = lambda tensor: tensor.data.clamp_(min=-ε, max=ε).add_(inputs).clamp_(min=0, max=1).sub_(inputs)

    loss_func, multiplier = _loss_functions[loss_function.lower()]

    if absolute_step_size is not None:
        step_size = absolute_step_size
    else:
        step_size = ε * relative_step_size

    if targeted:
        step_size *= -1

    δ = torch.zeros_like(inputs, requires_grad=True)
    δ_adv = torch.zeros_like(inputs)
    adv_found = torch.zeros(batch_size, dtype=torch.bool, device=device)

    if random_init:
        δ.data.uniform_(-ε, ε)
        clamp(δ)
    else:
        δ.data.zero_()

    for i in range(steps):
        logits = model(inputs + δ)

        if i == 0 and loss_function.lower() in ['dl', 'dlr']:
            labels_infhot = torch.zeros_like(logits).scatter_(1, labels.unsqueeze(1), float('inf'))
            loss_func = partial(loss_func, labels_infhot=labels_infhot)

        loss = multiplier * loss_func(logits, labels)
        δ_grad = grad(loss.sum(), δ, only_inputs=True)[0]

        is_adv = (logits.argmax(1) == labels) if targeted else (logits.argmax(1) != labels)
        δ_adv = torch.where(batch_view(is_adv), δ.detach(), δ_adv)
        adv_found.logical_or_(is_adv)

        δ.data.add_(δ_grad.sign(), alpha=step_size)
        clamp(δ)

    return adv_found, inputs + δ_adv
