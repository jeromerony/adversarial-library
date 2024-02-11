# Adapted from https://github.com/fra31/auto-attack
import math
import numbers
from functools import partial
from typing import Tuple, Optional, Union

import torch
from torch import nn, Tensor
from torch.nn import functional as F

from adv_lib.utils.losses import difference_of_logits_ratio


def apgd(model: nn.Module,
         inputs: Tensor,
         labels: Tensor,
         eps: Union[float, Tensor],
         norm: float,
         targeted: bool = False,
         n_iter: int = 100,
         n_restarts: int = 1,
         loss_function: str = 'dlr',
         eot_iter: int = 1,
         rho: float = 0.75,
         use_large_reps: bool = False,
         use_rs: bool = True,
         best_loss: bool = False) -> Tensor:
    """
    Auto-PGD (APGD) attack from https://arxiv.org/abs/2003.01690 with L1 variant from https://arxiv.org/abs/2103.01208.

    Parameters
    ----------
    model : nn.Module
        Model to attack.
    inputs : Tensor
        Inputs to attack. Should be in [0, 1].
    labels : Tensor
        Labels corresponding to the inputs if untargeted, else target labels.
    eps : float or Tensor
        Maximum norm for the adversarial perturbation. Can be a float used for all samples or a Tensor containing the
        distance for each corresponding sample.
    norm : float
        Norm corresponding to eps in {1, 2, float('inf')}.
    targeted : bool
        Whether to perform a targeted attack or not.
    n_iter : int
        Number of optimization steps.
    n_restarts : int
        Number of random restarts for the attack.
    loss_function : str
        Loss to optimize in ['ce', 'dlr'].
    eot_iter : int
        Number of iterations for expectation over transformation.
    rho : float
        Parameters for decreasing the step size.
    use_large_reps : bool
        Split iterations in three phases starting with larger eps (see section 3.2 of https://arxiv.org/abs/2103.01208).
    use_rs : bool
        Use a random start when using large reps.
    best_loss : bool
        If True, search for the strongest adversarial perturbation within the distance budget instead of stopping as
        soon as it finds one.

    Returns
    -------
    adv_inputs : Tensor
        Modified inputs to be adversarial to the model.

    """
    assert norm in [1, 2, float('inf')]
    device = inputs.device
    batch_size = len(inputs)

    adv_inputs = inputs.clone()
    adv_found = torch.zeros(batch_size, device=device, dtype=torch.bool)
    if isinstance(eps, numbers.Real):
        eps = torch.full_like(adv_found, eps, dtype=torch.float)

    if use_large_reps:
        epss = [3 * eps, 2 * eps, eps]
        iters = [0.3 * n_iter, 0.3 * n_iter, 0.4 * n_iter]
        iters = [math.ceil(i) for i in iters]
        iters[-1] = n_iter - sum(iters[:-1])

    apgd_attack = partial(_apgd, model=model, norm=norm, targeted=targeted, loss_function=loss_function,
                          eot_iter=eot_iter, rho=rho)

    if best_loss:
        loss = torch.full_like(adv_found, -float('inf'), dtype=torch.float)

        for _ in range(n_restarts):
            adv_inputs_run, adv_found_run, loss_run, _ = apgd_attack(inputs=inputs, labels=labels, eps=eps)

            better_loss = loss_run > loss
            adv_inputs[better_loss] = adv_inputs_run[better_loss]
            loss[better_loss] = loss_run[better_loss]

    else:
        for _ in range(n_restarts):
            if adv_found.all():
                break
            to_attack = ~adv_found

            inputs_to_attack = inputs[to_attack]
            labels_to_attack = labels[to_attack]

            if use_large_reps:
                assert norm == 1
                if use_rs:
                    x_init = inputs_to_attack + torch.randn_like(inputs_to_attack)
                    x_init += l1_projection(inputs_to_attack, x_init - inputs_to_attack, epss[0][to_attack])
                else:
                    x_init = None

                for eps_, iter in zip(epss, iters):
                    eps_to_attack = eps_[to_attack]
                    if x_init is not None:
                        x_init += l1_projection(inputs_to_attack, x_init - inputs_to_attack, eps_to_attack)

                    x_init, adv_found_run, _, adv_inputs_run = apgd_attack(
                        inputs=inputs_to_attack, labels=labels_to_attack, eps=eps_to_attack, x_init=x_init, n_iter=iter)

            else:
                _, adv_found_run, _, adv_inputs_run = apgd_attack(inputs=inputs_to_attack, labels=labels_to_attack,
                                                                  eps=eps[to_attack], n_iter=n_iter)
            adv_inputs[to_attack] = adv_inputs_run
            adv_found[to_attack] = adv_found_run

    return adv_inputs


def apgd_targeted(model: nn.Module,
                  inputs: Tensor,
                  labels: Tensor,
                  eps: Union[float, Tensor],
                  norm: float,
                  targeted: bool = False,
                  n_iter: int = 100,
                  n_restarts: int = 1,
                  loss_function: str = 'dlr',
                  eot_iter: int = 1,
                  rho: float = 0.75,
                  use_large_reps: bool = False,
                  use_rs: bool = True,
                  num_targets: Optional[int] = None) -> Tensor:
    """
    Targeted variant of the Auto-PGD (APGD) attack from https://arxiv.org/abs/2003.01690 with L1 variant from
    https://arxiv.org/abs/2103.01208. This attack is not a targeted one: it tries to find an adversarial perturbation by
    attacking each class, starting with the most likely one (different from the original class).

    Parameters
    ----------
    model : nn.Module
        Model to attack.
    inputs : Tensor
        Inputs to attack. Should be in [0, 1].
    labels : Tensor
        Labels corresponding to the inputs if untargeted, else target labels.
    eps : float or Tensor
        Maximum norm for the adversarial perturbation. Can be a float used for all samples or a Tensor containing the
        distance for each corresponding sample.
    norm : float
        Norm corresponding to eps in {1, 2, float('inf')}.
    targeted : bool
        Required argument for the library. Will raise an assertion error if True (will be ignored if the -O flag is
        used).
    n_iter : int
        Number of optimization steps.
    n_restarts : int
        Number of random restarts for the attack for each class attacked.
    loss_function : str
        Loss to optimize in ['ce', 'dlr'].
    eot_iter : int
        Number of iterations for expectation over transformation.
    rho : float
        Parameters for decreasing the step size.
    use_large_reps : bool
        Split iterations in three phases starting with larger eps (see section 3.2 of https://arxiv.org/abs/2103.01208).
    use_rs : bool
        Use a random start when using large reps.
    num_targets : int or None
        Number of classes to attack. If None, it will attack every class (except the original class).

    Returns
    -------
    adv_inputs : Tensor
        Modified inputs to be adversarial to the model.

    """
    assert targeted == False
    device = inputs.device
    batch_size = len(inputs)

    adv_inputs = inputs.clone()
    adv_found = torch.zeros(batch_size, device=device, dtype=torch.bool)
    if isinstance(eps, numbers.Real):
        eps = torch.full_like(adv_found, eps, dtype=torch.float)

    if use_large_reps:
        epss = [3 * eps, 2 * eps, eps]
        iters = [0.3 * n_iter, 0.3 * n_iter, 0.4 * n_iter]
        iters = [math.ceil(i) for i in iters]
        iters[-1] = n_iter - sum(iters[:-1])

    apgd_attack = partial(_apgd, model=model, norm=norm, targeted=True, loss_function=loss_function,
                          eot_iter=eot_iter, rho=rho)

    #  determine the number of classes based on the size of the model's output
    most_likely_classes = model(inputs).argsort(dim=1, descending=True)[:, 1:]
    num_classes_to_attack = most_likely_classes.size(1) if num_targets is None else num_targets

    for i in range(num_classes_to_attack):
        targets = most_likely_classes[:, i]

        for counter in range(n_restarts):
            if adv_found.all():
                break
            to_attack = ~adv_found

            inputs_to_attack = inputs[to_attack]
            targets_to_attack = targets[to_attack]

            if use_large_reps:
                assert norm == 1
                if use_rs:
                    x_init = inputs_to_attack + torch.randn_like(inputs_to_attack)
                    x_init += l1_projection(inputs_to_attack, x_init - inputs_to_attack, epss[0][to_attack])
                else:
                    x_init = None

                for eps_, iter in zip(epss, iters):
                    eps_to_attack = eps_[to_attack]
                    if x_init is not None:
                        x_init += l1_projection(inputs_to_attack, x_init - inputs_to_attack, eps_to_attack)

                    x_init, adv_found_run, _, adv_inputs_run = apgd_attack(
                        inputs=inputs_to_attack, labels=targets_to_attack, eps=eps_to_attack, x_init=x_init,
                        n_iter=iter)

            else:
                _, adv_found_run, _, adv_inputs_run = apgd_attack(inputs=inputs_to_attack, labels=targets_to_attack,
                                                                  eps=eps[to_attack], n_iter=n_iter)

            adv_inputs[to_attack] = adv_inputs_run
            adv_found[to_attack] = adv_found_run

    return adv_inputs


def minimal_apgd(model: nn.Module,
                 inputs: Tensor,
                 labels: Tensor,
                 norm: float,
                 max_eps: float,
                 targeted: bool = False,
                 binary_search_steps: int = 20,
                 targeted_version: bool = False,
                 n_iter: int = 100,
                 n_restarts: int = 1,
                 loss_function: str = 'dlr',
                 eot_iter: int = 1,
                 rho: float = 0.75,
                 use_large_reps: bool = False,
                 use_rs: bool = True,
                 num_targets: Optional[int] = None) -> Tensor:
    device = inputs.device
    batch_size = len(inputs)

    adv_inputs = inputs.clone()
    best_eps = torch.full((batch_size,), 2 * max_eps, dtype=torch.float, device=device)
    eps_low = torch.zeros_like(best_eps)

    if targeted_version:
        attack = partial(apgd_targeted, model=model, norm=norm, n_iter=n_iter, n_restarts=n_restarts,
                         loss_function=loss_function, eot_iter=eot_iter, rho=rho, use_large_reps=use_large_reps,
                         use_rs=use_rs, num_targets=num_targets)
    else:
        attack = partial(apgd, model=model, norm=norm, targeted=targeted, n_iter=n_iter, n_restarts=n_restarts,
                         loss_function=loss_function, eot_iter=eot_iter, rho=rho, use_large_reps=use_large_reps,
                         use_rs=use_rs)

    for _ in range(binary_search_steps):
        eps = (eps_low + best_eps) / 2

        adv_inputs_run = attack(inputs=inputs, labels=labels, eps=eps)
        adv_found_run = model(adv_inputs_run).argmax(1) != labels

        better_adv = adv_found_run & (eps < best_eps)
        adv_inputs[better_adv] = adv_inputs_run[better_adv]

        eps_low = torch.where(better_adv, eps_low, eps)
        best_eps = torch.where(better_adv, eps, best_eps)

    return adv_inputs


def l1_projection(x: Tensor, y: Tensor, eps: Tensor) -> Tensor:
    device = x.device
    shape = x.shape
    x, y = x.flatten(1), y.flatten(1)
    u = torch.min(1 - x - y, x + y).clamp_(max=0)
    l = y.abs().neg_()
    d = u.clone()

    bs, indbs = torch.sort(torch.cat((u, l), dim=1).neg_(), dim=1)
    bs2 = F.pad(bs[:, 1:], (0, 1))

    inu = (indbs < u.shape[1]).float().mul_(2).sub_(1).cumsum_(dim=1)

    s1 = u.sum(dim=1).neg_()

    c = l.sum(dim=1).add_(eps)
    c5 = s1 + c < 0

    s = (bs2 - bs).mul_(inu).cumsum_(dim=1).add_(s1.unsqueeze(-1))

    if c5.any():
        lb = torch.zeros(c5.sum(), device=device)
        ub = torch.full_like(lb, bs.shape[1] - 1)

        nitermax = math.ceil(math.log2(bs.shape[1]))
        counter = 0

        while counter < nitermax:
            counter4 = torch.floor((lb + ub) / 2)
            counter2 = counter4.long()

            c8 = s[c5, counter2] + c[c5] < 0
            lb[c8] = counter4[c8]
            ub[~c8] = counter4[~c8]

            counter += 1

        lb2 = lb.long()
        alpha = (-s[c5, lb2] - c[c5]) / inu[c5, lb2 + 1] + bs2[c5, lb2]
        d[c5] = -torch.min(torch.max(-u[c5], alpha.unsqueeze(-1)), -l[c5])

    return d.mul_(y.sign()).view(shape)


def check_oscillation(loss_steps: Tensor, j: int, k: int, k3: float = 0.75) -> Tensor:
    t = torch.zeros_like(loss_steps[0])
    for counter5 in range(k):
        t.add_(loss_steps[j - counter5] > loss_steps[j - counter5 - 1])
    return t <= k * k3


def _apgd(model: nn.Module,
          inputs: Tensor,
          labels: Tensor,
          eps: Tensor,
          norm: float,
          x_init: Optional[Tensor] = None,
          targeted: bool = False,
          n_iter: int = 100,
          loss_function: str = 'dlr',
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

    lower, upper = (inputs - batch_view(eps)).clamp_(min=0, max=1), (inputs + batch_view(eps)).clamp_(min=0, max=1)

    n_iter_2, n_iter_min, size_decr = max(int(0.22 * n_iter), 1), max(int(0.06 * n_iter), 1), max(int(0.03 * n_iter), 1)

    if x_init is not None:
        x_adv = x_init.clone()
    elif norm == float('inf'):
        t = 2 * torch.rand_like(inputs) - 1
        x_adv = inputs + t * batch_view(eps / t.flatten(1).norm(p=float('inf'), dim=1))
    elif norm == 2:
        t = torch.randn_like(inputs)
        x_adv = inputs + t * batch_view(eps / t.flatten(1).norm(p=2, dim=1))
    elif norm == 1:
        t = torch.randn_like(inputs)
        delta = l1_projection(inputs, t, eps)
        x_adv = inputs + t + delta

    x_adv.clamp_(min=0, max=1)
    x_best = x_adv.clone()
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

    alpha = 2 if norm in [2, float('inf')] else 1 if norm == 1 else 2e-2
    step_size = alpha * eps
    x_adv_old = x_adv.clone()
    k = n_iter_2
    counter3 = 0

    if norm == 1:
        k = max(int(0.04 * n_iter), 1)
        n_fts = inputs[0].numel()
        if x_init is None:
            topk = torch.ones(len(inputs), device=device).mul_(0.2)
            sp_old = torch.full_like(topk, n_fts, dtype=torch.float)
        else:
            sp_old = (x_adv - inputs).flatten(1).norm(p=0, dim=1)
            topk = sp_old / n_fts / 1.5
        adasp_redstep = 1.5
        adasp_minstep = 10.

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
            delta = x_adv + grad.mul_(batch_view(step_size / grad.flatten(1).norm(p=2, dim=1).add_(1e-12)))
            delta.sub_(inputs)
            delta_norm = delta.flatten(1).norm(p=2, dim=1).add_(1e-12)
            x_adv_1 = delta.mul_(batch_view(torch.min(delta_norm, eps).div_(delta_norm))).add_(inputs).clamp_(min=0,
                                                                                                              max=1)

            # momentum
            delta = x_adv.lerp(x_adv_1, weight=a).add_(grad2, alpha=1 - a)
            delta.sub_(inputs)
            delta_norm = delta.flatten(1).norm(p=2, dim=1).add_(1e-12)
            x_adv_1 = delta.mul_(batch_view(torch.min(delta_norm, eps).div_(delta_norm))).add_(inputs).clamp_(min=0,
                                                                                                              max=1)

        elif norm == 1:
            grad_abs = grad.abs()
            grad_topk = grad_abs.flatten(1).sort(dim=1).values
            topk_curr = (1 - topk).mul_(n_fts).clamp_(min=0, max=n_fts - 1).long()
            grad_topk = grad_topk.gather(1, topk_curr.unsqueeze(1))
            grad.mul_(grad_abs >= batch_view(grad_topk))
            grad_sign = grad.sign()

            x_adv_1 = x_adv + grad_sign * batch_view(step_size / grad_sign.flatten(1).norm(p=1, dim=1).add_(1e-10))

            x_adv_1.sub_(inputs)
            delta_p = l1_projection(inputs, x_adv_1, eps)
            x_adv_1.add_(inputs).add_(delta_p)

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
            if norm in [2, float('inf')]:
                fl_reduce_no_impr = (~reduced_last_check) & (loss_best_last_check >= loss_best)
                reduced_last_check = check_oscillation(loss_steps, i, k, k3=rho) | fl_reduce_no_impr
                loss_best_last_check = loss_best

                if reduced_last_check.any():
                    step_size[reduced_last_check] /= 2.0
                    x_adv[reduced_last_check] = x_best[reduced_last_check]
                    grad[reduced_last_check] = grad_best[reduced_last_check]

                k = max(k - size_decr, n_iter_min)

            elif norm == 1:
                sp_curr = (x_best - inputs).flatten(1).norm(p=0, dim=1)
                fl_redtopk = (sp_curr / sp_old) < 0.95
                topk = sp_curr / n_fts / 1.5
                step_size = torch.where(fl_redtopk, alpha * eps, step_size / adasp_redstep)
                step_size = torch.min(torch.max(step_size, alpha * eps / adasp_minstep), alpha * eps)
                sp_old = sp_curr

                x_adv[fl_redtopk] = x_best[fl_redtopk]
                grad[fl_redtopk] = grad_best[fl_redtopk]

            counter3 = 0

    return x_best, adv_found, loss_best, x_best_adv
