import warnings

import torch
from torch import Tensor, nn
from torch.autograd import grad

from adv_lib.utils.attack_utils import get_all_targets


def df(model: nn.Module,
       inputs: Tensor,
       labels: Tensor,
       targeted: bool = False,
       steps: int = 100,
       overshoot: float = 0.02,
       norm: float = 2,
       return_unsuccessful: bool = False,
       return_targets: bool = False) -> Tensor:
    """
    DeepFool attack from https://arxiv.org/abs/1511.04599. Properly implement parallel sample-wise early-stopping.

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
    steps : int
        Maixmum number of attack steps.
    overshoot : float
        Ratio by which to overshoot the boundary estimated from linear model.
    norm : float
        Norm to minimize in {2, float('inf')}.
    return_unsuccessful : bool
        Whether to return unsuccessful adversarial inputs ; used by SuperDeepFool.
    return_unsuccessful : bool
        Whether to return last target labels ; used by SuperDeepFool.

    Returns
    -------
    adv_inputs : Tensor
        Modified inputs to be adversarial to the model.

    """
    if targeted:
        warnings.warn('DeepFool attack is untargeted only. Returning inputs.')
        return inputs

    if inputs.min() < 0 or inputs.max() > 1: raise ValueError('Input values should be in the [0, 1] range.')
    device = inputs.device
    batch_size = len(inputs)
    batch_view = lambda tensor: tensor.view(-1, *[1] * (inputs.ndim - 1))

    # Setup variables
    adv_inputs = inputs.clone()
    adv_inputs.requires_grad_(True)

    adv_out = inputs.clone()
    adv_found = torch.zeros(batch_size, dtype=torch.bool, device=device)
    if return_targets:
        targets = labels.clone()

    arange = torch.arange(batch_size, device=device)
    for i in range(steps):

        logits = model(adv_inputs)

        if i == 0:
            other_labels = get_all_targets(labels=labels, num_classes=logits.shape[1])

        pred_labels = logits.argmax(dim=1)
        is_adv = (pred_labels == labels) if targeted else (pred_labels != labels)

        if is_adv.any():
            adv_not_found = ~adv_found
            adv_out[adv_not_found] = torch.where(batch_view(is_adv), adv_inputs.detach(), adv_out[adv_not_found])
            adv_found.masked_scatter_(adv_not_found, is_adv)
            if is_adv.all():
                break

            not_adv = ~is_adv
            logits, labels, other_labels = logits[not_adv], labels[not_adv], other_labels[not_adv]
            arange = torch.arange(not_adv.sum(), device=device)

        f_prime = logits.gather(dim=1, index=other_labels) - logits.gather(dim=1, index=labels.unsqueeze(1))
        w_prime = []
        for j, f_prime_k in enumerate(f_prime.unbind(dim=1)):
            w_prime_k = grad(f_prime_k.sum(), inputs=adv_inputs, retain_graph=(j + 1) < f_prime.shape[1],
                             only_inputs=True)[0]
            w_prime.append(w_prime_k)
        w_prime = torch.stack(w_prime, dim=1)  # batch_size × num_classes × ...

        if is_adv.any():
            not_adv = ~is_adv
            adv_inputs, w_prime = adv_inputs[not_adv], w_prime[not_adv]

        if norm == 2:
            w_prime_norms = w_prime.flatten(2).norm(p=2, dim=2).clamp_(min=1e-6)
        elif norm == float('inf'):
            w_prime_norms = w_prime.flatten(2).norm(p=1, dim=2).clamp_(min=1e-6)

        distance = f_prime.detach().abs_().div_(w_prime_norms).add_(1e-4)
        l_hat = distance.argmin(dim=1)

        if return_targets:
            targets[~adv_found] = torch.where(l_hat >= labels, l_hat + 1, l_hat)

        if norm == 2:
            # 1e-4 added in original implementation
            scale = distance[arange, l_hat] / w_prime_norms[arange, l_hat]
            adv_inputs.data.addcmul_(batch_view(scale), w_prime[arange, l_hat], value=1 + overshoot)
        elif norm == float('inf'):
            adv_inputs.data.addcmul_(batch_view(distance[arange, l_hat]), w_prime[arange, l_hat].sign(),
                                     value=1 + overshoot)
        adv_inputs.data.clamp_(min=0, max=1)

    if return_unsuccessful and not adv_found.all():
        adv_out[~adv_found] = adv_inputs.detach()

    if return_targets:
        return adv_out, targets

    return adv_out
