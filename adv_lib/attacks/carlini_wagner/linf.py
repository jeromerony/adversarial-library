# Adapted from https://github.com/carlini/nn_robust_attacks

from typing import Tuple, Optional

import torch
from torch import nn, optim, Tensor

from adv_lib.utils.losses import difference_of_logits
from adv_lib.utils.visdom_logger import VisdomLogger


def carlini_wagner_linf(model: nn.Module,
                        inputs: Tensor,
                        labels: Tensor,
                        targeted: bool = False,
                        learning_rate: float = 0.01,
                        max_iterations: int = 1000,
                        initial_const: float = 1e-5,
                        largest_const: float = 2e+1,
                        const_factor: float = 2,
                        reduce_const: bool = False,
                        decrease_factor: float = 0.9,
                        abort_early: bool = True,
                        image_constraints: Tuple[float, float] = (0, 1),
                        callback: Optional[VisdomLogger] = None) -> Tensor:
    """
    Carlini and Wagner Linf attack from https://arxiv.org/abs/1608.04644.

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
    learning_rate: float
        The learning rate for the attack algorithm. Smaller values produce better results but are slower to converge.
    max_iterations : int
        The maximum number of iterations. Larger values are more accurate; setting too small will require a large
        learning rate and will produce poor results.
    initial_const : float
        The initial tradeoff-constant to use to tune the relative importance of distance and classification objective.
    largest_const : float
        The maximum tradeoff-constant to use to tune the relative importance of distance and classification objective.
    const_factor : float
        The multiplicative factor by which the constant is increased if the search failed.
    reduce_const : float
        If true, after each successful attack, make the constant smaller.
    decrease_factor : float
        Rate at which τ is decreased. Larger produces better quality results.
    abort_early : bool
        If true, allows early aborts if gradient descent gets stuck.
    image_constraints : Tuple[float, float]
        Minimum and maximum pixel values.
    callback : Optional

    Returns
    -------
    adv_inputs : Tensor
        Modified inputs to be adversarial to the model.

    """
    device = inputs.device
    batch_size = len(inputs)
    boxmin, boxmax = image_constraints
    boxmul, boxplus = (boxmax - boxmin) / 2, (boxmin + boxmax) / 2
    t_inputs = ((inputs - boxplus) / boxmul).mul_(1 - 1e-6).atanh()
    multiplier = -1 if targeted else 1

    # set modifier and the parameters used in the optimization
    modifier = torch.zeros_like(inputs)
    c = torch.full((batch_size,), initial_const, device=device, dtype=torch.float)
    τ = torch.ones(batch_size, device=device)

    o_adv_found = torch.zeros_like(c, dtype=torch.bool)
    o_best_linf = torch.ones_like(c)
    o_best_adv = inputs.clone()

    outer_loops = 0
    total_iters = 0
    while (to_optimize := (τ > 1 / 255) & (c < largest_const)).any():

        inputs_, t_inputs_, labels_ = inputs[to_optimize], t_inputs[to_optimize], labels[to_optimize]
        batch_view = lambda tensor: tensor.view(len(inputs_), *[1] * (inputs_.ndim - 1))

        if callback:
            callback.line(['const', 'τ'], outer_loops, [c[to_optimize].mean(), τ[to_optimize].mean()])
            callback.line(['success', 'best_linf'], outer_loops, [o_adv_found.float().mean(), best_linf.mean()])

        # setup the optimizer
        modifier_ = modifier[to_optimize].requires_grad_(True)
        optimizer = optim.Adam([modifier_], lr=learning_rate)
        c_, τ_ = c[to_optimize], τ[to_optimize]

        adv_found = torch.zeros(len(modifier_), device=device, dtype=torch.bool)
        best_linf = o_best_linf[to_optimize]
        best_adv = inputs_.clone()

        if callback:
            callback.line(['const', 'τ'], outer_loops, [c_.mean(), τ_.mean()])
            callback.line(['success', 'best_linf'], outer_loops, [o_adv_found.float().mean(), o_best_linf.mean()])

        for i in range(max_iterations):

            adv_inputs = torch.tanh(t_inputs_ + modifier_) * boxmul + boxplus
            linf = (adv_inputs.detach() - inputs_).flatten(1).norm(p=float('inf'), dim=1)
            logits = model(adv_inputs)

            if i == 0:
                labels_infhot = torch.zeros_like(logits).scatter_(1, labels[to_optimize].unsqueeze(1), float('inf'))

            # adjust the best result found so far
            predicted_classes = logits.argmax(1)

            is_adv = (predicted_classes == labels_) if targeted else (predicted_classes != labels_)
            is_smaller = linf < best_linf
            is_both = is_adv & is_smaller
            adv_found.logical_or_(is_both)
            best_linf = torch.where(is_both, linf, best_linf)
            best_adv = torch.where(batch_view(is_both), adv_inputs.detach(), best_adv)

            logit_dists = multiplier * difference_of_logits(logits, labels_, labels_infhot=labels_infhot)
            linf_loss = (adv_inputs - inputs_).abs_().sub_(batch_view(τ_)).clamp_(min=0).flatten(1).sum(1)
            loss = linf_loss + c_ * logit_dists.clamp_(min=0)

            # check if we should abort search
            if abort_early and (loss < 0.0001 * c_).all():
                break

            optimizer.zero_grad()
            loss.sum().backward()
            optimizer.step()

            if callback:
                callback.accumulate_line('logit_dist', total_iters, logit_dists.mean())
                callback.accumulate_line('linf_norm', total_iters, linf.mean())

                if (i + 1) % (max_iterations // 10) == 0 or (i + 1) == max_iterations:
                    callback.update_lines()

            total_iters += 1

        o_adv_found[to_optimize] = adv_found | o_adv_found[to_optimize]
        o_best_linf[to_optimize] = torch.where(adv_found, best_linf, o_best_linf[to_optimize])
        o_best_adv[to_optimize] = torch.where(batch_view(adv_found), best_adv, o_best_adv[to_optimize])
        modifier[to_optimize] = modifier_.detach()

        smaller_τ_ = adv_found & (best_linf < τ_)
        τ_ = torch.where(smaller_τ_, best_linf, τ_)
        τ[to_optimize] = torch.where(adv_found, decrease_factor * τ_, τ_)
        c[to_optimize] = torch.where(~adv_found, const_factor * c_, c_)
        if reduce_const:
            c[to_optimize] = torch.where(adv_found, c[to_optimize] / 2, c[to_optimize])

        outer_loops += 1

    # return the best solution found
    return o_best_adv
