from functools import partial
from typing import Optional, Tuple, Union

import torch
from torch import Tensor, nn
from torch.autograd import grad
from torch.nn import functional as F

from adv_lib.utils.losses import difference_of_logits, difference_of_logits_ratio
from adv_lib.utils.projections import clamp_
from adv_lib.utils.visdom_logger import VisdomLogger


def pgd_linf(model: nn.Module,
             inputs: Tensor,
             labels: Tensor,
             ε: Union[float, Tensor],
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

    if isinstance(ε, (int, float)):
        ε = torch.full_like(adv_found, ε, dtype=inputs.dtype)

    pgd_attack = partial(_pgd_linf, model=model, targeted=targeted, steps=steps, random_init=random_init,
                         loss_function=loss_function, relative_step_size=relative_step_size,
                         absolute_step_size=absolute_step_size)

    for i in range(restarts):

        adv_found_run, adv_inputs_run = pgd_attack(inputs=inputs[~adv_found], labels=labels[~adv_found],
                                                   ε=ε[~adv_found])
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
              ε: Tensor,
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
    lower, upper = torch.maximum(-inputs, -batch_view(ε)), torch.minimum(1 - inputs, batch_view(ε))

    loss_func, multiplier = _loss_functions[loss_function.lower()]

    if absolute_step_size is not None:
        step_size = absolute_step_size
    else:
        step_size = ε * relative_step_size

    if targeted:
        step_size *= -1

    δ = torch.zeros_like(inputs, requires_grad=True)
    best_adv = inputs.clone()
    adv_found = torch.zeros(batch_size, dtype=torch.bool, device=device)

    if random_init:
        δ.data.uniform_(-1, 1).mul_(batch_view(ε))
        clamp_(δ, lower=lower, upper=upper)

    for i in range(steps):
        adv_inputs = inputs + δ
        logits = model(adv_inputs)

        if i == 0 and loss_function.lower() in ['dl', 'dlr']:
            labels_infhot = torch.zeros_like(logits).scatter_(1, labels.unsqueeze(1), float('inf'))
            loss_func = partial(loss_func, labels_infhot=labels_infhot)

        loss = multiplier * loss_func(logits, labels)
        δ_grad = grad(loss.sum(), δ, only_inputs=True)[0].sign_().mul_(batch_view(step_size))

        is_adv = (logits.argmax(1) == labels) if targeted else (logits.argmax(1) != labels)
        best_adv = torch.where(batch_view(is_adv), adv_inputs.detach(), best_adv)
        adv_found.logical_or_(is_adv)

        δ.data.add_(δ_grad)
        clamp_(δ, lower=lower, upper=upper)

    return adv_found, best_adv
