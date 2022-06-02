# Adapted from https://github.com/amirgholami/trattack

import warnings
from functools import partial
from typing import Tuple

import torch
from torch import Tensor, nn
from torch.autograd import grad


def select_index(model: nn.Module,
                 inputs: Tensor,
                 p: float = 2,
                 worst_case: bool = False) -> Tensor:
    """Select the attack target class."""
    _duals = {2: 2, float('inf'): 1}
    dual = _duals[p]

    inputs.requires_grad_(True)
    logits = model(inputs)
    logits, indices = torch.sort(logits, descending=True)

    top_logits = logits[:, 0]
    top_grad = grad(top_logits.sum(), inputs, only_inputs=True, retain_graph=True)[0]
    pers = []

    num_classes = logits.size(1)
    for i in range(num_classes):
        other_logits = logits[:, i + 1]
        other_grad = grad(other_logits.sum(), inputs, only_inputs=True, retain_graph=i + 1 != num_classes)[0]
        grad_dual_norm = (top_grad - other_grad).flatten(1).norm(p=dual, dim=1)
        pers.append((top_logits.detach() - other_logits.detach()).div_(grad_dual_norm))

    pers = torch.stack(pers, dim=1)
    inputs.detach_()

    if worst_case:
        index = pers.argmax(dim=1, keepdim=True)
    else:
        index = pers.clamp_(min=0).argmin(dim=1, keepdim=True)

    return indices.gather(1, index + 1).squeeze(1)


def _step(model: nn.Module,
          inputs: Tensor,
          labels: Tensor,
          target_labels: Tensor,
          eps: Tensor,
          p: float = 2) -> Tuple[Tensor, Tensor]:
    _duals = {2: 2, float('inf'): 1}
    dual = _duals[p]

    inputs.requires_grad_(True)
    logits = model(inputs)

    logit_diff = (logits.gather(1, target_labels.unsqueeze(1)) - logits.gather(1, labels.unsqueeze(1))).squeeze(1)

    grad_inputs = grad(logit_diff.sum(), inputs, only_inputs=True)[0].flatten(1)
    inputs.detach_()
    per = logit_diff.detach().neg_().div_(grad_inputs.norm(p=dual, dim=1).clamp_(min=1e-6))

    if p == float('inf'):
        grad_inputs.sign_()
    elif p == 2:
        grad_inputs.div_(grad_inputs.norm(p=2, dim=1, keepdim=True).clamp_(min=1e-6))

    per = torch.min(per, eps)
    adv_inputs = grad_inputs.mul_(per.add_(1e-4).mul_(1.02).unsqueeze(1)).view_as(inputs).add_(inputs)
    adv_inputs.clamp_(min=0, max=1)
    return adv_inputs, eps


def _adaptive_step(model: nn.Module,
                   inputs: Tensor,
                   labels: Tensor,
                   target_labels: Tensor,
                   eps: Tensor,
                   p: float = 2) -> Tuple[Tensor, Tensor]:
    _duals = {2: 2, float('inf'): 1}
    dual = _duals[p]

    inputs.requires_grad_(True)
    logits = model(inputs)

    class_logits = logits.gather(1, labels.unsqueeze(1)).squeeze(1)
    target_logits = logits.gather(1, target_labels.unsqueeze(1)).squeeze(1)
    logit_diff = target_logits - class_logits

    grad_inputs = grad(logit_diff.sum(), inputs, only_inputs=True)[0].flatten(1)
    inputs.detach_()
    per = logit_diff.detach().neg_().div_(grad_inputs.norm(p=dual, dim=1).clamp_(min=1e-6))

    if p == float('inf'):
        grad_inputs.sign_()
    elif p == 2:
        grad_inputs.div_(grad_inputs.norm(p=2, dim=1, keepdim=True).clamp_(min=1e-6))

    new_eps = torch.min(per, eps)

    adv_inputs = grad_inputs.mul_(new_eps.add(1e-4).mul_(1.02).unsqueeze_(1)).view_as(inputs).add_(inputs)
    adv_inputs.clamp_(min=0, max=1)

    adv_logits = model(adv_inputs)
    class_adv_logits = adv_logits.gather(1, labels.unsqueeze(1)).squeeze(1)

    obj_diff = (class_logits - class_adv_logits).div_(new_eps)
    increase = obj_diff > 0.9
    decrease = obj_diff < 0.5
    new_eps = torch.where(increase, new_eps * 1.2, torch.where(decrease, new_eps / 1.2, new_eps))
    if p == 2:
        new_eps.clamp_(min=0.0005, max=0.05)
    elif p == float('inf'):
        new_eps.clamp_(min=0.0001, max=0.01)

    return adv_inputs, new_eps


def tr(model: nn.Module,
       inputs: Tensor,
       labels: Tensor,
       iter: int = 100,
       adaptive: bool = False,
       p: float = 2,
       eps: float = 0.001,
       worst_case: bool = False,
       targeted: bool = False) -> Tensor:
    if targeted:
        warnings.warn('TR attack is untargeted only. Returning inputs.')
        return inputs

    adv_inputs = inputs.clone()
    target_labels = select_index(model, inputs, p=p, worst_case=worst_case)
    attack_step = partial(_adaptive_step if adaptive else _step, model=model, p=p)

    to_attack = torch.ones(len(inputs), dtype=torch.bool, device=inputs.device)
    eps = torch.full_like(to_attack, eps, dtype=torch.float, device=inputs.device)

    for _ in range(iter):

        logits = model(adv_inputs[to_attack])
        to_attack.masked_scatter_(to_attack, logits.argmax(dim=1) == labels[to_attack])
        if (~to_attack).all():
            break
        adv_inputs[to_attack], eps[to_attack] = attack_step(inputs=adv_inputs[to_attack], labels=labels[to_attack],
                                                            target_labels=target_labels[to_attack], eps=eps[to_attack])

    return adv_inputs
