# Adapted from https://github.com/fra31/auto-attack

from functools import partial
from typing import Tuple, Optional, Union

import torch
import torch.nn as nn
from torch import Tensor

from adv_lib.utils.losses import difference_of_logits_ratio


def check_oscillation(loss_steps: Tensor, j: int, k: int, k3: float = 0.75) -> Tensor:
    t = torch.zeros_like(loss_steps[0])
    for counter5 in range(k):
        t.add_(loss_steps[j - counter5] > loss_steps[j - counter5 - 1])

    return t <= k * k3


def apgd(model: nn.Module,
         inputs: Tensor,
         labels: Tensor,
         eps: Union[float, Tensor],
         norm: float,
         targeted: bool = False,
         n_iter: int = 100,
         n_restarts: int = 1,
         loss_function: str = 'ce',
         eot_iter: int = 1,
         rho: float = 0.75,
         best_loss: bool = False) -> Tensor:
    device = inputs.device
    batch_size = len(inputs)

    adv_inputs = inputs.clone()
    adv_found = torch.zeros(batch_size, device=device, dtype=torch.bool)
    if isinstance(eps, (int, float)):
        eps = torch.full_like(adv_found, eps, dtype=torch.float)

    apgd_attack = partial(_apgd, model=model, norm=norm, targeted=targeted, n_iter=n_iter, loss_function=loss_function,
                          eot_iter=eot_iter, rho=rho)

    if not best_loss:
        for _ in range(n_restarts):
            if adv_found.all():
                break
            to_attack = ~adv_found
            _, adv_found_run, _, adv_inputs_run = apgd_attack(inputs=inputs[to_attack], labels=labels[to_attack],
                                                              eps=eps[to_attack])
            adv_inputs[to_attack] = adv_inputs_run
            adv_found[to_attack] = adv_found_run

    else:
        loss = torch.full_like(adv_found, -float('inf'), dtype=torch.float)

        for _ in range(n_restarts):
            adv_inputs_run, adv_found_run, loss_run, _ = apgd_attack(inputs=inputs, labels=labels, eps=eps)

            better_loss = loss_run > loss
            adv_inputs[better_loss] = adv_inputs_run[better_loss]
            loss[better_loss] = loss_run[better_loss]

    return adv_inputs


def apgd_targeted(model: nn.Module,
                  inputs: Tensor,
                  labels: Tensor,
                  eps: Union[float, Tensor],
                  norm: float,
                  n_iter: int = 100,
                  n_restarts: int = 1,
                  eot_iter: int = 1,
                  rho: float = 0.75,
                  num_targets: Optional[int] = None,
                  seed: Optional[int] = None,
                  **kwargs) -> Tensor:
    device = inputs.device
    batch_size = len(inputs)
    if seed is not None:
        torch.manual_seed(seed)

    adv_inputs = inputs.clone()
    adv_found = torch.zeros(batch_size, device=device, dtype=torch.bool)
    if isinstance(eps, (int, float)):
        eps = torch.full_like(adv_found, eps, dtype=torch.float)

    apgd_attack = partial(_apgd, model=model, norm=norm, targeted=True, n_iter=n_iter, loss_function='dlr',
                          eot_iter=eot_iter, rho=rho)

    #  determine the number of classes based on the size of the model's output
    most_likely_classes = model(inputs).argsort(dim=1, descending=True)[:, 1:]
    num_classes_to_attack = num_targets or most_likely_classes.size(1)

    for i in range(num_classes_to_attack):
        targets = most_likely_classes[:, i]

        for counter in range(n_restarts):
            if adv_found.all():
                break
            to_attack = ~adv_found

            _, adv_found_run, _, adv_inputs_run = apgd_attack(inputs=inputs[to_attack], labels=targets[to_attack],
                                                              eps=eps[to_attack])
            adv_inputs[to_attack] = adv_inputs_run
            adv_found[to_attack] = adv_found_run

    return adv_inputs


def minimal_apgd(model: nn.Module,
                 inputs: Tensor,
                 labels: Tensor,
                 norm: float,
                 max_eps: float,
                 binary_search_steps: int = 20,
                 targeted_version: bool = False,
                 n_iter: int = 100,
                 n_restarts: int = 1,
                 eot_iter: int = 1,
                 rho: float = 0.75,
                 num_targets: Optional[int] = None,
                 seed: Optional[int] = None,
                 **kwargs) -> Tensor:
    device = inputs.device
    batch_size = len(inputs)
    if seed is not None:
        torch.manual_seed(seed)

    adv_inputs = inputs.clone()
    best_eps = torch.full((batch_size,), 2 * max_eps, dtype=torch.float, device=device)
    eps_low = torch.zeros_like(best_eps)

    if targeted_version:
        attack = partial(apgd_targeted, model=model, norm=norm, n_iter=n_iter, n_restarts=n_restarts, eot_iter=eot_iter,
                         rho=rho, num_targets=num_targets)
    else:
        attack = partial(apgd, model=model, norm=norm, n_iter=n_iter, n_restarts=n_restarts, eot_iter=eot_iter, rho=rho)

    for _ in range(binary_search_steps):
        eps = (eps_low + best_eps) / 2

        adv_inputs_run = attack(inputs=inputs, labels=labels, eps=eps)
        adv_found_run = model(adv_inputs_run).argmax(1) != labels

        better_adv = adv_found_run & (eps < best_eps)
        adv_inputs[better_adv] = adv_inputs_run[better_adv]

        eps_low = torch.where(better_adv, eps_low, eps)
        best_eps = torch.where(better_adv, eps, best_eps)

    return adv_inputs


def _apgd(model: nn.Module,
          inputs: Tensor,
          labels: Tensor,
          eps: Tensor,
          norm: float,
          targeted: bool = False,
          n_iter: int = 100,
          loss_function: str = 'ce',
          eot_iter: int = 1,
          rho: float = 0.75) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    _loss_functions = {
        'ce': (nn.CrossEntropyLoss(reduction='none'), -1 if targeted else 1),
        'dlr': (partial(difference_of_logits_ratio, targeted=targeted), 1 if targeted else -1),
    }

    device = inputs.device
    batch_size = len(inputs)
    batch_view = lambda tensor: tensor.view(-1, *[1] * (inputs.ndim - 1))
    criterion_indiv, multiplier = _loss_functions[loss_function.lower()]

    lower, upper = (inputs - batch_view(eps)).clamp(0, 1), (inputs + batch_view(eps)).clamp(0, 1)

    n_iter_2, n_iter_min, size_decr = max(int(0.22 * n_iter), 1), max(int(0.06 * n_iter), 1), max(int(0.03 * n_iter), 1)

    if norm == float('inf'):
        t = 2 * torch.rand_like(inputs) - 1
        x_adv = inputs + t * batch_view(eps / t.flatten(1).norm(p=float('inf'), dim=1))
    elif norm == 2:
        t = torch.randn_like(inputs)
        x_adv = inputs + t * batch_view(eps / t.flatten(1).norm(p=2, dim=1) + 1e-12)

    x_adv.clamp_(0., 1.)
    x_best = inputs.clone()
    x_best_adv = inputs.clone()
    loss_steps = torch.zeros(n_iter, batch_size, device=device)
    loss_best_steps = torch.zeros(n_iter + 1, batch_size, device=device)
    adv_found_steps = torch.zeros_like(loss_best_steps)

    x_adv.requires_grad_()
    grad = torch.zeros_like(inputs)
    for _ in range(eot_iter):
        logits = model(x_adv)
        loss_indiv = multiplier * criterion_indiv(logits, labels)
        grad.add_(torch.autograd.grad(loss_indiv.sum(), x_adv, only_inputs=True)[0])

    grad.div_(eot_iter)
    grad_best = grad.clone()
    x_adv.detach_()

    adv_found = (logits.argmax(1) == labels) if targeted else (logits.argmax(1) != labels)
    adv_found_steps[0] = adv_found
    loss_best = loss_indiv.detach().clone()

    step_size = eps * 2
    x_adv_old = x_adv.clone()
    k = n_iter_2
    counter3 = 0

    loss_best_last_check = loss_best.clone()
    reduced_last_check = torch.zeros_like(loss_best, dtype=torch.bool)

    for i in range(n_iter):
        ### gradient step
        grad2 = x_adv - x_adv_old
        x_adv_old = x_adv

        a = 0.75 if i else 1.0

        if norm == float('inf'):
            x_adv_1 = x_adv + batch_view(step_size) * torch.sign(grad)
            x_adv_1 = torch.min(torch.max(x_adv_1, lower), upper)

            # momentum
            x_adv_1 = x_adv + (x_adv_1 - x_adv) * a + grad2 * (1 - a)
            x_adv_1 = torch.min(torch.max(x_adv_1, lower), upper)

        elif norm == 2:
            x_adv_1 = x_adv + batch_view(step_size) * grad / batch_view(grad.flatten(1).norm(p=2, dim=1) + 1e-12)
            delta = x_adv_1 - inputs
            delta_norm = delta.flatten(1).norm(p=2, dim=1)
            x_adv_1 = (inputs + delta * batch_view(torch.min(delta_norm, eps) / (delta_norm + 1e-12))).clamp(0.0, 1.0)

            # momentum
            x_adv_1 = x_adv + (x_adv_1 - x_adv) * a + grad2 * (1 - a)
            delta = x_adv_1 - inputs
            delta_norm = delta.flatten(1).norm(p=2, dim=1)
            x_adv_1 = (inputs + delta * batch_view(torch.min(delta_norm, eps) / (delta_norm + 1e-12))).clamp(0.0, 1.0)

        x_adv = x_adv_1

        ### get gradient
        x_adv.requires_grad_(True)
        grad.zero_()
        for _ in range(eot_iter):
            logits = model(x_adv)
            loss_indiv = multiplier * criterion_indiv(logits, labels)
            grad.add_(torch.autograd.grad(loss_indiv.sum(), x_adv, only_inputs=True)[0])

        grad.div_(eot_iter)
        x_adv.detach_(), loss_indiv.detach_()

        is_adv = (logits.argmax(1) == labels) if targeted else (logits.argmax(1) != labels)
        adv_found.logical_or_(is_adv)
        adv_found_steps[i + 1] = adv_found
        x_best_adv[is_adv] = x_adv[is_adv]

        ### check step size
        loss_steps[i] = loss_indiv
        ind = loss_indiv > loss_best
        x_best[ind] = x_adv[ind]
        grad_best[ind] = grad[ind]
        loss_best[ind] = loss_indiv[ind]
        loss_best_steps[i + 1] = loss_best

        counter3 += 1

        if counter3 == k:
            fl_reduce_no_impr = (~reduced_last_check) & (loss_best_last_check >= loss_best)
            reduced_last_check = check_oscillation(loss_steps, i, k, k3=rho) | fl_reduce_no_impr
            loss_best_last_check = loss_best

            if reduced_last_check.any():
                step_size[reduced_last_check] /= 2.0
                x_adv[reduced_last_check] = x_best[reduced_last_check]
                grad[reduced_last_check] = grad_best[reduced_last_check]

            counter3 = 0
            k = max(k - size_decr, n_iter_min)

    return x_best, adv_found, loss_best, x_best_adv
