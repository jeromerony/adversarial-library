from typing import Tuple, Optional

import torch
from torch import nn, optim, Tensor
from torch.autograd import grad

from adv_lib.utils.losses import difference_of_logits
from adv_lib.utils.visdom_logger import VisdomLogger


def carlini_wagner_l2(model: nn.Module,
                      inputs: Tensor,
                      labels: Tensor,
                      targeted: bool = False,
                      confidence: float = 0,
                      learning_rate: float = 0.01,
                      initial_const: float = 0.001,
                      binary_search_steps: int = 9,
                      max_iterations: int = 10000,
                      abort_early: bool = True,
                      image_constraints: Tuple[float, float] = (0, 1),
                      callback: Optional[VisdomLogger] = None) -> Tensor:
    """
    Carlini and Wagner L2 attack from https://arxiv.org/abs/1608.04644.

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
    confidence : float
        Confidence of adversarial examples: higher produces examples that are farther away, but more strongly classified
        as adversarial.
    learning_rate: float
        The learning rate for the attack algorithm. Smaller values produce better results but are slower to converge.
    initial_const : float
        The initial tradeoff-constant to use to tune the relative importance of distance and confidence. If
        binary_search_steps is large, the initial constant is not important.
    binary_search_steps : int
        The number of times we perform binary search to find the optimal tradeoff-constant between distance and
        confidence.
    max_iterations : int
        The maximum number of iterations. Larger values are more accurate; setting too small will require a large
        learning rate and will produce poor results.
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
    batch_view = lambda tensor: tensor.view(batch_size, *[1] * (inputs.ndim - 1))
    boxmin, boxmax = image_constraints
    boxmul, boxplus = (boxmax - boxmin) / 2, (boxmin + boxmax) / 2
    t_inputs = ((inputs - boxplus) / boxmul).mul_(1 - 1e-6).atanh()
    multiplier = -1 if targeted else 1

    # set the lower and upper bounds accordingly
    c = torch.full((batch_size,), initial_const, device=device)
    lower_bound = torch.zeros_like(c)
    upper_bound = torch.full_like(c, 1e10)

    o_best_l2 = torch.full_like(c, float('inf'))
    o_best_adv = inputs.clone()
    o_adv_found = torch.zeros(batch_size, device=device, dtype=torch.bool)

    i_total = 0
    for outer_step in range(binary_search_steps):

        # setup the modifier variable and the optimizer
        modifier = torch.zeros_like(inputs, requires_grad=True)
        optimizer = optim.Adam([modifier], lr=learning_rate)
        best_l2 = torch.full_like(c, float('inf'))
        adv_found = torch.zeros(batch_size, device=device, dtype=torch.bool)

        # The last iteration (if we run many steps) repeat the search once.
        if (binary_search_steps >= 10) and outer_step == (binary_search_steps - 1):
            c = upper_bound

        prev = float('inf')
        for i in range(max_iterations):

            adv_inputs = torch.tanh(t_inputs + modifier) * boxmul + boxplus
            l2_squared = (adv_inputs - inputs).flatten(1).pow(2).sum(1)
            l2 = l2_squared.detach().sqrt()
            logits = model(adv_inputs)

            if outer_step == 0 and i == 0:
                # setup the target variable, we need it to be in one-hot form for the loss function
                labels_onehot = torch.zeros_like(logits).scatter(1, labels.unsqueeze(1), 1)
                labels_infhot = torch.zeros_like(logits).scatter(1, labels.unsqueeze(1), float('inf'))

            # adjust the best result found so far
            predicted_classes = (logits - labels_onehot * confidence).argmax(1) if targeted else \
                (logits + labels_onehot * confidence).argmax(1)

            is_adv = (predicted_classes == labels) if targeted else (predicted_classes != labels)
            is_smaller = l2 < best_l2
            o_is_smaller = l2 < o_best_l2
            is_both = is_adv & is_smaller
            o_is_both = is_adv & o_is_smaller

            best_l2 = torch.where(is_both, l2, best_l2)
            adv_found.logical_or_(is_both)
            o_best_l2 = torch.where(o_is_both, l2, o_best_l2)
            o_adv_found.logical_or_(is_both)
            o_best_adv = torch.where(batch_view(o_is_both), adv_inputs.detach(), o_best_adv)

            logit_dists = multiplier * difference_of_logits(logits, labels, labels_infhot=labels_infhot)
            loss = l2_squared + c * (logit_dists + confidence).clamp_min(0)

            # check if we should abort search if we're getting nowhere.
            if abort_early and i % (max_iterations // 10) == 0:
                if (loss > prev * 0.9999).all():
                    break
                prev = loss.detach()

            optimizer.zero_grad()
            modifier.grad = grad(loss.sum(), modifier, only_inputs=True)[0]
            optimizer.step()

            if callback:
                i_total += 1
                callback.accumulate_line('logit_dist', i_total, logit_dists.mean())
                callback.accumulate_line('l2_norm', i_total, l2.mean())
                if i_total % (max_iterations // 20) == 0:
                    callback.update_lines()

        if callback:
            best_l2 = o_best_l2[o_adv_found].mean() if o_adv_found.any() else torch.tensor(float('nan'), device=device)
            callback.line(['success', 'best_l2', 'c'], outer_step, [o_adv_found.float().mean(), best_l2, c.mean()])

        # adjust the constant as needed
        upper_bound[adv_found] = torch.min(upper_bound[adv_found], c[adv_found])
        adv_not_found = ~adv_found
        lower_bound[adv_not_found] = torch.max(lower_bound[adv_not_found], c[adv_not_found])
        is_smaller = upper_bound < 1e9
        c[is_smaller] = (lower_bound[is_smaller] + upper_bound[is_smaller]) / 2
        c[(~is_smaller) & adv_not_found] *= 10

    # return the best solution found
    return o_best_adv
