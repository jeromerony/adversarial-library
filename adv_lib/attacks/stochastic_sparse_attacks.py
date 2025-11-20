from typing import Optional

import torch
from torch import Tensor, nn
from torch.autograd import grad


def fga(model: nn.Module,
        inputs: Tensor,
        labels: Tensor,
        targeted: bool,
        increasing: bool = True,
        max_iter: Optional[int] = None,
        n_samples: int = 10,
        large_memory: bool = False) -> Tensor:
    """Folded Gaussian Attack (FGA) attack from https://arxiv.org/abs/2011.12423.

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
    increasing : bool
        Whether to add positive or negative perturbations.
    max_iter : int
        Maximum number of iterations for the attack. If None is provided, the attack will run as long as adversarial
        examples are not found and non-modified pixels are left.
    n_samples : int
        Number of random samples to draw in each iteration.
    large_memory : bool
        If True, performs forward propagations on all randomly perturbed inputs in one batch. This is slightly faster
        for small models but also uses `n_samples` times more memory. This should only be used when working on small
        models. For larger models, the speed gain is negligible, so this option should be left to False.

    Returns
    -------
    adv_inputs : Tensor
        Modified inputs to be adversarial to the model.

    """
    batch_size, *input_shape = inputs.shape
    batch_view = lambda tensor: tensor.view(-1, *[1] * (inputs.ndim - 1))
    input_view = lambda tensor: tensor.view(-1, *input_shape)
    device = inputs.device
    model_ = lambda t: model(t).softmax(dim=1)
    multiplier = 1 if targeted else -1

    adv_inputs = inputs.clone()
    best_adv = inputs.clone()
    adv_found = torch.zeros(batch_size, device=device, dtype=torch.bool)

    max_iter = inputs[0].numel() if max_iter is None else max_iter
    for i in range(max_iter):
        Γ = adv_inputs == inputs
        Γ_empty = ~Γ.flatten(1).any(1)
        if (Γ_empty | adv_found).all():
            break
        to_attack = ~(Γ_empty | adv_found)

        inputs_, labels_, Γ_ = adv_inputs[to_attack], labels[to_attack], Γ[to_attack]
        batch_size_ = len(inputs_)
        inputs_.requires_grad_(True)
        Γ_inf = torch.zeros_like(Γ_, dtype=torch.float).masked_fill_(~Γ_, float('inf'))

        probs = model_(inputs_)
        label_probs = probs.gather(1, labels_.unsqueeze(1)).squeeze(1)
        grad_label_probs = grad(multiplier * label_probs.sum(), inputs=inputs_, only_inputs=True)[0]
        inputs_.detach_()

        # find index of most relevant feature
        if increasing:
            i_0 = grad_label_probs.mul_(1 - inputs_).sub_(Γ_inf).flatten(1).argmax(dim=1, keepdim=True)
        else:
            i_0 = grad_label_probs.mul_(inputs_).add_(Γ_inf).flatten(1).argmin(dim=1, keepdim=True)

        # compute variance of gaussian noise
        θ = inputs_.flatten(1).gather(1, i_0).neg_()
        if increasing:
            θ.add_(1)
        # generate random perturbation from folded Gaussian noise
        S = torch.randn(batch_size_, n_samples, device=device).abs_().mul_(θ)

        # add perturbation to inputs
        perturbed_inputs = inputs_.flatten(1).unsqueeze(1).repeat(1, n_samples, 1)
        perturbed_inputs.scatter_add_(
            2, i_0.repeat_interleave(n_samples, dim=1, output_size=n_samples).unsqueeze(2), S.unsqueeze(2)
        )
        perturbed_inputs.clamp_(min=0, max=1)

        # get probabilities for perturbed inputs
        if large_memory:
            new_probs = model_(input_view(perturbed_inputs))
        else:
            new_probs = []
            for chunk in torch.chunk(input_view(perturbed_inputs), chunks=n_samples):
                new_probs.append(model_(chunk))
            new_probs = torch.cat(new_probs, dim=0)
        new_probs = new_probs.view(batch_size_, n_samples, -1)
        new_preds = new_probs.argmax(dim=2)

        new_label_probs = new_probs.gather(2, labels_.view(-1, 1, 1).expand(-1, n_samples, 1)).squeeze(2)
        if targeted:
            # finding the index of max probability for target class. If a sample is adv, it will be prioritized. If
            # several are adversarial, taking the index of the adv sample with max probability.
            adv_found_ = new_preds == labels_.unsqueeze(1)
            best_sample_index = (new_label_probs + adv_found_.float()).argmax(dim=1)
        else:
            # finding the index of min probability for original class. If a sample is adv, it will be prioritized. If
            # several are adversarial, taking the index of the adv sample with min probability.
            adv_found_ = new_preds != labels_.unsqueeze(1)
            best_sample_index = (new_label_probs - adv_found_.float()).argmin(dim=1)

        # update trackers
        adv_inputs[to_attack] = input_view(perturbed_inputs[range(batch_size_), best_sample_index])
        preds = new_preds.gather(1, best_sample_index.unsqueeze(1)).squeeze(1)
        is_adv = (preds == labels_) if targeted else (preds != labels_)
        adv_found[to_attack] = is_adv
        best_adv[to_attack] = torch.where(batch_view(is_adv), adv_inputs[to_attack], best_adv[to_attack])

    return best_adv


def vfga(model: nn.Module,
         inputs: Tensor,
         labels: Tensor,
         targeted: bool,
         max_iter: Optional[int] = None,
         n_samples: int = 10,
         large_memory: bool = False) -> Tensor:
    """Voting Folded Gaussian Attack (VFGA) attack from https://arxiv.org/abs/2011.12423.

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
    max_iter : int
        Maximum number of iterations for the attack. If None is provided, the attack will run as long as adversarial
        examples are not found and non-modified pixels are left.
    n_samples : int
        Number of random samples to draw in each iteration.
    large_memory : bool
        If True, performs forward propagations on all randomly perturbed inputs in one batch. This is slightly faster
        for small models but also uses `n_samples` times more memory. This should only be used when working on small
        models. For larger models, the speed gain is negligible, so this option should be left to False.

    Returns
    -------
    adv_inputs : Tensor
        Modified inputs to be adversarial to the model.

    """
    batch_size, *input_shape = inputs.shape
    batch_view = lambda tensor: tensor.view(-1, *[1] * (inputs.ndim - 1))
    input_view = lambda tensor: tensor.view(-1, *input_shape)
    device = inputs.device
    model_ = lambda t: model(t).softmax(dim=1)
    multiplier = 1 if targeted else -1

    adv_inputs = inputs.clone()
    best_adv = inputs.clone()
    adv_found = torch.zeros(batch_size, device=device, dtype=torch.bool)

    max_iter = inputs[0].numel() if max_iter is None else max_iter
    for i in range(max_iter):
        Γ = adv_inputs == inputs
        Γ_empty = ~Γ.flatten(1).any(1)
        if (Γ_empty | adv_found).all():
            break
        to_attack = ~(Γ_empty | adv_found)

        inputs_, labels_, Γ_ = adv_inputs[to_attack], labels[to_attack], Γ[to_attack]
        batch_size_ = len(inputs_)
        inputs_.requires_grad_(True)
        Γ_inf = torch.zeros_like(Γ_, dtype=torch.float).masked_fill_(~Γ_, float('inf'))

        probs = model_(inputs_)
        label_probs = probs.gather(1, labels_.unsqueeze(1)).squeeze(1)
        grad_label_probs = grad(multiplier * label_probs.sum(), inputs=inputs_, only_inputs=True)[0]
        inputs_.detach_()

        # find index of most relevant feature
        i_plus = (1 - inputs_).mul_(grad_label_probs).sub_(Γ_inf).flatten(1).argmax(dim=1, keepdim=True)
        i_minus = grad_label_probs.mul_(inputs_).add_(Γ_inf).flatten(1).argmin(dim=1, keepdim=True)
        # compute variance of gaussian noise
        θ_plus = 1 - inputs_.flatten(1).gather(1, i_plus)
        θ_minus = inputs_.flatten(1).gather(1, i_minus)
        # generate random perturbation from folded Gaussian noise
        S_plus = torch.randn(batch_size_, n_samples, device=device).abs_().mul_(θ_plus)
        S_minus = torch.randn(batch_size_, n_samples, device=device).abs_().neg_().mul_(θ_minus)

        # add perturbation to inputs
        perturbed_inputs = inputs_.flatten(1).unsqueeze(1).repeat(1, 2 * n_samples, 1)
        i_plus_minus = torch.cat([i_plus, i_minus], dim=1).repeat_interleave(
            n_samples, dim=1, output_size=2 * n_samples
        )
        S_plus_minus = torch.cat([S_plus, S_minus], dim=1)
        perturbed_inputs.scatter_add_(2, i_plus_minus.unsqueeze(2), S_plus_minus.unsqueeze(2))
        perturbed_inputs.clamp_(min=0, max=1)

        # get probabilities for perturbed inputs
        if large_memory:
            new_probs = model_(input_view(perturbed_inputs))
        else:
            new_probs = []
            for chunk in torch.chunk(input_view(perturbed_inputs), chunks=n_samples):
                new_probs.append(model_(chunk))
            new_probs = torch.cat(new_probs, dim=0)
        new_probs = new_probs.view(batch_size_, 2 * n_samples, -1)
        new_preds = new_probs.argmax(dim=2)

        new_label_probs = new_probs.gather(2, labels_.view(-1, 1, 1).expand(-1, 2 * n_samples, 1)).squeeze(2)
        if targeted:
            # finding the index of max probability for target class. If a sample is adv, it will be prioritized. If
            # several are adversarial, taking the index of the adv sample with max probability.
            adv_found_ = new_preds == labels_.unsqueeze(1)
            best_sample_index = (new_label_probs + adv_found_.float()).argmax(dim=1)
        else:
            # finding the index of min probability for original class. If a sample is adv, it will be prioritized. If
            # several are adversarial, taking the index of the adv sample with min probability.
            adv_found_ = new_preds != labels_.unsqueeze(1)
            best_sample_index = (new_label_probs - adv_found_.float()).argmin(dim=1)

        # update trackers
        adv_inputs[to_attack] = input_view(perturbed_inputs[range(batch_size_), best_sample_index])
        preds = new_preds.gather(1, best_sample_index.unsqueeze(1)).squeeze(1)
        is_adv = (preds == labels_) if targeted else (preds != labels_)
        adv_found[to_attack] = is_adv
        best_adv[to_attack] = torch.where(batch_view(is_adv), adv_inputs[to_attack], best_adv[to_attack])

    return best_adv
