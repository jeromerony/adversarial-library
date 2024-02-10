# Adapted from https://github.com/Cinofix/sigma-zero-adversarial-attack
import math
import warnings

import torch
from torch import Tensor, nn
from torch.autograd import grad

from adv_lib.utils.losses import difference_of_logits


def sigma_zero(model: nn.Module,
               inputs: Tensor,
               labels: Tensor,
               num_steps: int = 1000,
               η_0: float = 1.0,
               σ: float = 0.001,
               τ_0: float = 0.3,
               τ_factor: float = 0.01,
               grad_norm: float = float('inf'),
               targeted: bool = False) -> Tensor:
    """
    σ-zero attack from https://arxiv.org/abs/2402.01879.

    Parameters
    ----------
    model : nn.Module
        Model to attack.
    inputs : Tensor
        Inputs to attack. Should be in [0, 1].
    labels : Tensor
        Labels corresponding to the inputs if untargeted, else target labels.
    num_steps : int
        Number of optimization steps. Corresponds to the number of forward and backward propagations.
    η_0 : float
        Initial step size.
    σ : float
        \ell_0 approximation parameter: smaller values produce sharper approximations while larger values produce a
        smoother approximation.
    τ_0 : float
        Initial sparsity threshold.
    τ_factor : float
        Threshold adjustment factor w.r.t. step size η.
    grad_norm: float
        Norm to use for gradient normalization.
    targeted : bool
        Attack is untargeted only: will raise a warning and return inputs if targeted is True.

    Returns
    -------
    best_adv : Tensor
        Perturbed inputs (inputs + perturbation) that are adversarial and have smallest distance with the original
        inputs.

    """
    if targeted:
        warnings.warn('σ-zero attack is untargeted only. Returning inputs.')
        return inputs

    batch_size, numel = len(inputs), inputs[0].numel()
    batch_view = lambda tensor: tensor.view(batch_size, *[1] * (inputs.ndim - 1))

    δ = torch.zeros_like(inputs, requires_grad=True)
    # Adam variables
    exp_avg = torch.zeros_like(inputs)
    exp_avg_sq = torch.zeros_like(inputs)
    β_1, β_2 = 0.9, 0.999

    best_l0 = inputs.new_full((batch_size,), numel)
    best_adv = inputs.clone()
    τ = torch.full_like(best_l0, τ_0)

    η = η_0
    for i in range(num_steps):
        adv_inputs = inputs + δ

        # compute loss
        logits = model(adv_inputs)
        dl_loss = difference_of_logits(logits, labels).clamp_(min=0)
        δ_square = δ.square()
        l0_approx_normalized = (δ_square / (δ_square + σ)).flatten(1).mean(dim=1)

        # keep best solutions
        predicted_classes = logits.argmax(dim=1)
        l0_norm = δ.data.flatten(1).norm(p=0, dim=1)
        is_adv = (predicted_classes == labels) if targeted else (predicted_classes != labels)
        is_smaller = l0_norm < best_l0
        is_both = is_adv & is_smaller
        best_l0 = torch.where(is_both, l0_norm, best_l0)
        best_adv = torch.where(batch_view(is_both), adv_inputs.detach(), best_adv)

        # compute loss and gradient
        adv_loss = (dl_loss + l0_approx_normalized).sum()
        δ_grad = grad(adv_loss, inputs=δ, only_inputs=True)[0]

        # normalize gradient based on grad_norm type
        δ_inf_norm = δ_grad.flatten(1).norm(p=grad_norm, dim=1).clamp_(min=1e-12)
        δ_grad.div_(batch_view(δ_inf_norm))

        # adam computations
        exp_avg.mul_(β_1).add_(δ_grad, alpha=1 - β_1)
        exp_avg_sq.mul_(β_2).addcmul_(δ_grad, δ_grad, value=1 - β_2)
        bias_correction1 = 1 - β_1 ** (i + 1)
        bias_correction2 = 1 - β_2 ** (i + 1)
        denom = exp_avg_sq.sqrt().div_(bias_correction2 ** 0.5).add_(1e-8)

        # step and clamp
        δ.data.addcdiv_(exp_avg, denom, value=-η / bias_correction1)
        δ.data.add_(inputs).clamp_(min=0, max=1).sub_(inputs)

        # update step size with cosine annealing
        η = 0.1 * η_0 + 0.9 * η_0 * (1 + math.cos(math.pi * i / num_steps)) / 2
        # dynamic thresholding
        τ.add_(torch.where(is_adv, τ_factor * η, -τ_factor * η)).clamp_(min=0, max=1)

        # filter components
        δ.data[δ.data.abs() < batch_view(τ)] = 0

    return best_adv
