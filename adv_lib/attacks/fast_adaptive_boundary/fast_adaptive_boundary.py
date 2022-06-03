# Adapted from https://github.com/fra31/auto-attack

import warnings
from functools import partial
from typing import Optional, Tuple

import torch
from torch import Tensor, nn
from torch.autograd import grad

from .projections import projection_l1, projection_l2, projection_linf


def fab(model: nn.Module,
        inputs: Tensor,
        labels: Tensor,
        norm: float,
        n_iter: int = 100,
        ε: Optional[float] = None,
        α_max: float = 0.1,
        β: float = 0.9,
        η: float = 1.05,
        restarts: Optional[int] = None,
        targeted_restarts: bool = False,
        targeted: bool = False) -> Tensor:
    """
    Fast Adaptive Boundary (FAB) attack from https://arxiv.org/abs/1907.02044

    Parameters
    ----------
    model : nn.Module
        Model to attack.
    inputs : Tensor
        Inputs to attack. Should be in [0, 1].
    labels : Tensor
        Labels corresponding to the inputs if untargeted, else target labels.
    norm : float
        Norm to minimize in {1, 2 ,float('inf')}.
    n_iter : int
        Number of optimization steps. This does not correspond to the number of forward / backward propagations for this
        attack. For a more comprehensive discussion on complexity, see section 4 of https://arxiv.org/abs/1907.02044 and
        for a comparison of complexities, see https://arxiv.org/abs/2011.11857.
        TL;DR: FAB performs 2 forwards and K - 1 (i.e. number of classes - 1) backwards per step in default mode. If
        `targeted_restarts` is `True`, performs `2 * restarts` forwards and `restarts` or (K - 1) backwards per step.
    ε : float
        Maximum norm of the random initialization for restarts.
    α_max : float
        Maximum weight for the biased  gradient step. α = 0 corresponds to taking the projection of the `adv_inputs` on
        the decision hyperplane, while α = 1 corresponds to taking the projection of the `inputs` on the decision
        hyperplane.
    β : float
        Weight for the biased backward step, i.e. a linear interpolation between `inputs` and `adv_inputs` at step i.
        β = 0 corresponds to taking the original `inputs` and β = 1 corresponds to taking the `adv_inputs`.
    η : float
        Extrapolation for the optimization step. η = 1 corresponds to projecting the `adv_inputs` on the decision
        hyperplane. η > 1 corresponds to overshooting to increase the probability of crossing the decision hyperplane.
    restarts : int
        Number of random restarts in default mode; starts from the inputs in the first run and then add random noise for
        the consecutive restarts. Number of classes to attack if `targeted_restarts` is `True`.
    targeted_restarts : bool
        If `True`, performs targeted attack towards the most likely classes for the unperturbed `inputs`. If `restarts`
        is not given, this will attack each class (except the original class). If `restarts` is given, the `restarts`
        most likely classes will be attacked. If `restarts` is larger than K - 1, this will re-attack the most likely
        classes with random noise.
    targeted : bool
        Placeholder argument for library. FAB is only for untargeted attacks, so setting this to True will raise a
        warning and return the inputs.

    Returns
    -------
    adv_inputs : Tensor
        Modified inputs to be adversarial to the model.

    """
    if targeted:
        warnings.warn('FAB attack is untargeted only. Returning inputs.')
        return inputs

    best_adv = inputs.clone()
    best_norm = torch.full_like(labels, float('inf'), dtype=torch.float)

    fab_attack = partial(_fab, model=model, norm=norm, n_iter=n_iter, ε=ε, α_max=α_max, β=β, η=η)

    if targeted_restarts:
        logits = model(inputs)
        n_target_classes = logits.size(1) - 1
        labels_infhot = torch.zeros_like(logits).scatter_(1, labels.unsqueeze(1), float('inf'))
        k = min(restarts or n_target_classes, n_target_classes)
        topk_labels = (logits - labels_infhot).topk(k=k, dim=1).indices

    n_restarts = restarts or (n_target_classes if targeted_restarts else 1)
    for i in range(n_restarts):

        if targeted_restarts:
            target_labels = topk_labels[:, i % n_target_classes]
            adv_inputs_run, adv_found_run, norm_run = fab_attack(
                inputs=inputs, labels=labels, random_start=i >= n_target_classes, targets=target_labels, u=best_norm)
        else:
            adv_inputs_run, adv_found_run, norm_run = fab_attack(inputs=inputs, labels=labels, random_start=i != 0,
                                                                 u=best_norm)

        is_better_adv = adv_found_run & (norm_run < best_norm)
        best_norm[is_better_adv] = norm_run[is_better_adv]
        best_adv[is_better_adv] = adv_inputs_run[is_better_adv]

    return best_adv


def get_best_diff_logits_grads(model: nn.Module,
                               inputs: Tensor,
                               labels: Tensor,
                               other_labels: Tensor,
                               q: float) -> Tuple[Tensor, Tensor]:
    batch_view = lambda tensor: tensor.view(-1, *[1] * (inputs.ndim - 1))
    min_ratio = torch.full_like(labels, float('inf'), dtype=torch.float)
    best_logit_diff, best_grad_diff = torch.zeros_like(labels, dtype=torch.float), torch.zeros_like(inputs)

    inputs.requires_grad_(True)
    logits = model(inputs)
    class_logits = logits.gather(1, labels.unsqueeze(1)).squeeze(1)

    n_other_labels = other_labels.size(1)
    for i, o_labels in enumerate(other_labels.transpose(0, 1)):
        other_logits = logits.gather(1, o_labels.unsqueeze(1)).squeeze(1)
        logits_diff = other_logits - class_logits
        grad_diff = grad(logits_diff.sum(), inputs, only_inputs=True, retain_graph=i + 1 != n_other_labels)[0]
        ratio = logits_diff.abs().div_(grad_diff.flatten(1).norm(p=q, dim=1).clamp_(min=1e-12))

        smaller_ratio = ratio < min_ratio
        min_ratio = torch.min(ratio, min_ratio)
        best_logit_diff = torch.where(smaller_ratio, logits_diff.detach(), best_logit_diff)
        best_grad_diff = torch.where(batch_view(smaller_ratio), grad_diff.detach(), best_grad_diff)

    inputs.detach_()
    return best_logit_diff, best_grad_diff


def _fab(model: nn.Module,
         inputs: Tensor,
         labels: Tensor,
         norm: float,
         n_iter: int = 100,
         ε: Optional[float] = None,
         α_max: float = 0.1,
         β: float = 0.9,
         η: float = 1.05,
         random_start: bool = False,
         u: Optional[Tensor] = None,
         targets: Optional[Tensor] = None) -> Tuple[Tensor, Tensor, Tensor]:
    _projection_dual_default_ε = {
        1: (projection_l1, float('inf'), 5),
        2: (projection_l2, 2, 1),
        float('inf'): (projection_linf, 1, 0.3)
    }

    device = inputs.device
    batch_size = len(inputs)
    batch_view = lambda tensor: tensor.view(-1, *[1] * (inputs.ndim - 1))
    projection, dual_norm, default_ε = _projection_dual_default_ε[norm]
    ε = ε or default_ε

    logits = model(inputs)
    if targets is not None:
        other_labels = targets.unsqueeze(1)
    else:
        # generate all other labels
        n_classes = logits.size(1)
        other_labels = torch.zeros(len(labels), n_classes - 1, dtype=torch.long, device=device)
        all_classes = set(range(n_classes))
        for i in range(len(labels)):
            diff_labels = list(all_classes.difference({labels[i].item()}))
            other_labels[i] = torch.tensor(diff_labels, device=device)

    get_df_dg = partial(get_best_diff_logits_grads, model=model, labels=labels, other_labels=other_labels, q=dual_norm)

    adv_inputs = inputs.clone()
    adv_found = torch.zeros(batch_size, device=device, dtype=torch.bool)
    best_norm = u if u is not None else torch.full((batch_size,), float('inf'), device=device, dtype=torch.float)
    best_adv = inputs.clone()

    if random_start:
        if norm == float('inf'):
            t = 2 * torch.rand_like(inputs) - 1
        elif norm in [1, 2]:
            t = torch.randn_like(inputs)

        adv_inputs = inputs + 0.5 * t * batch_view(best_norm.clamp(max=ε) / t.flatten(1).norm(p=norm, dim=1))
        adv_inputs.clamp_(min=0.0, max=1.0)

    for i in range(n_iter):
        df, dg = get_df_dg(inputs=adv_inputs)
        b = (-df + (dg * adv_inputs).flatten(1).sum(dim=1))
        w = dg.flatten(1)

        d3 = projection(torch.cat((adv_inputs.flatten(1), inputs.flatten(1)), 0), w.repeat(2, 1), b.repeat(2))
        d1, d2 = map(lambda t: t.view_as(adv_inputs), torch.chunk(d3, 2, dim=0))

        a0 = batch_view(d3.flatten(1).norm(p=norm, dim=1).clamp_(min=1e-8))
        a1, a2 = torch.chunk(a0, 2, dim=0)

        α = (a1 / (a1 + a2)).clamp_(min=0, max=α_max)
        adv_inputs = ((adv_inputs + η * d1) * (1 - α) + (inputs + d2 * η) * α).clamp_(min=0, max=1)

        is_adv = model(adv_inputs).argmax(1) != labels
        adv_found.logical_or_(is_adv)
        adv_norm = (adv_inputs - inputs).flatten(1).norm(p=norm, dim=1)
        is_smaller = adv_norm < best_norm
        is_both = is_adv & is_smaller
        best_norm = torch.where(is_both, adv_norm, best_norm)
        best_adv = torch.where(batch_view(is_both), adv_inputs, best_adv)

        adv_inputs = torch.where(batch_view(is_adv), inputs + (adv_inputs - inputs) * β, adv_inputs)

    return best_adv, adv_found, best_norm
