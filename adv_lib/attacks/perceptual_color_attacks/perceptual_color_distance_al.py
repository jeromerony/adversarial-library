# Adapted from https://github.com/ZhengyuZhao/PerC-Adversarial

from math import pi, cos

import torch
from torch import nn, Tensor
from torch.autograd import grad

from .differential_color_functions import rgb2lab_diff, ciede2000_diff


def quantization(x):
    """quantize the continus image tensors into 255 levels (8 bit encoding)"""
    x_quan = torch.round(x * 255) / 255
    return x_quan


def perc_al(model: nn.Module,
            images: Tensor,
            labels: Tensor,
            num_classes: int,
            targeted: bool = False,
            max_iterations: int = 1000,
            alpha_l_init: float = 1.,
            alpha_c_init: float = 0.5,
            confidence: float = 0, **kwargs) -> Tensor:
    """
    PerC_AL: Alternating Loss of Classification and Color Differences to achieve imperceptibile perturbations with few
    iterations. Adapted from https://github.com/ZhengyuZhao/PerC-Adversarial.

    Parameters
    ----------
    model : nn.Module
        Model to fool.
    images : Tensor
        Batch of image examples in the range of [0,1].
    labels : Tensor
        Original labels if untargeted, else labels of targets.
    targeted : bool, optional
        Whether to perform a targeted adversary or not.
    max_iterations : int
        Number of iterations for the optimization.
    alpha_l_init: float
        step size for updating perturbations with respect to classification loss
    alpha_c_init: float
        step size for updating perturbations with respect to perceptual color differences. for relatively easy
        untargeted case, alpha_c_init is adjusted to a smaller value (e.g., 0.1 is used in the paper)
    confidence : float, optional
        Confidence of the adversary for Carlini's loss, in term of distance between logits.
        Note that this approach only supports confidence setting in an untargeted case

    Returns
    -------
    Tensor
        Batch of image samples modified to be adversarial
    """

    if images.min() < 0 or images.max() > 1: raise ValueError('Input values should be in the [0, 1] range.')
    device = images.device

    alpha_l_min = alpha_l_init / 100
    alpha_c_min = alpha_c_init / 10
    multiplier = -1 if targeted else 1

    X_adv_round_best = images.clone()
    inputs_LAB = rgb2lab_diff(images)
    batch_size = images.shape[0]
    delta = torch.zeros_like(images, requires_grad=True)
    mask_isadv = torch.zeros(batch_size, dtype=torch.bool, device=device)
    color_l2_delta_bound_best = torch.full((batch_size,), 100000, dtype=torch.float, device=device)

    if (targeted == False) and confidence != 0:
        labels_onehot = torch.zeros(labels.size(0), num_classes, device=device)
        labels_onehot.scatter_(1, labels.unsqueeze(1), 1)
        labels_infhot = torch.zeros_like(labels_onehot).scatter_(1, labels.unsqueeze(1), float('inf'))
    if (targeted == True) and confidence != 0:
        print('Only support setting confidence in untargeted case!')
        return

    # check if some images are already adversarial
    if (targeted == False) and confidence != 0:
        logits = model(images)
        real = logits.gather(1, labels.unsqueeze(1)).squeeze(1)
        other = (logits - labels_infhot).max(1).values
        mask_isadv = (real - other) <= -40
    elif confidence == 0:
        if targeted:
            mask_isadv = model(images).argmax(1) == labels
        else:
            mask_isadv = model(images).argmax(1) != labels
    color_l2_delta_bound_best[mask_isadv] = 0
    X_adv_round_best[mask_isadv] = images[mask_isadv]

    for i in range(max_iterations):
        # cosine annealing for alpha_l_init and alpha_c_init
        alpha_c = alpha_c_min + 0.5 * (alpha_c_init - alpha_c_min) * (1 + cos(i / max_iterations * pi))
        alpha_l = alpha_l_min + 0.5 * (alpha_l_init - alpha_l_min) * (1 + cos(i / max_iterations * pi))

        loss = multiplier * nn.CrossEntropyLoss(reduction='sum')(model(images + delta), labels)
        grad_a = grad(loss, delta, only_inputs=True)[0]
        delta.data[~mask_isadv] = delta.data[~mask_isadv] + alpha_l * (grad_a.permute(1, 2, 3, 0) / torch.norm(
            grad_a.flatten(1), dim=1)).permute(3, 0, 1, 2)[~mask_isadv]

        d_map = ciede2000_diff(inputs_LAB, rgb2lab_diff(images + delta)).unsqueeze(1)
        color_dis = torch.norm(d_map.flatten(1), dim=1)
        grad_color = grad(color_dis.sum(), delta, only_inputs=True)[0]
        delta.data[mask_isadv] = delta.data[mask_isadv] - alpha_c * (grad_color.permute(1, 2, 3, 0) / torch.norm(
            grad_color.flatten(1), dim=1)).permute(3, 0, 1, 2)[mask_isadv]

        delta.data = (images + delta.data).clamp_(min=0, max=1) - images
        X_adv_round = quantization(images + delta.data)

        if (targeted == False) and confidence != 0:
            logits = model(X_adv_round)
            real = logits.gather(1, labels.unsqueeze(1)).squeeze(1)
            other = (logits - labels_infhot).max(1).values
            mask_isadv = (real - other) <= -40
        elif confidence == 0:
            if targeted:
                mask_isadv = model(X_adv_round).argmax(1) == labels
            else:
                mask_isadv = model(X_adv_round).argmax(1) != labels
        mask_best = (color_dis.data < color_l2_delta_bound_best)
        mask = mask_best * mask_isadv
        color_l2_delta_bound_best[mask] = color_dis.data[mask]
        X_adv_round_best[mask] = X_adv_round[mask]

    return X_adv_round_best
