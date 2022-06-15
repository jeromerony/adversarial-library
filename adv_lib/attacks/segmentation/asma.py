from typing import Optional

import torch
from adv_lib.utils.visdom_logger import VisdomLogger
from torch import Tensor, nn
from torch.autograd import grad


def iou_masks(mask1: Tensor, mask2: Tensor, n: int):
    k = (mask1 >= 0) & (mask1 < n)
    inds = n * mask1[k].to(torch.int64) + mask2[k]
    mat = torch.bincount(inds, minlength=n ** 2).reshape(n, n)
    iu = torch.diag(mat) / (mat.sum(1) + mat.sum(0) - torch.diag(mat) + 1e-6)
    return iu.mean().item()


def asma(model: nn.Module,
         inputs: Tensor,
         labels: Tensor,
         masks: Tensor = None,
         targeted: bool = False,
         adv_threshold: float = 0.99,
         num_steps: int = 1000,
         τ: float = 1e-7,
         β: float = 1e-6,
         callback: Optional[VisdomLogger] = None) -> Tensor:
    "ASMA attack from https://arxiv.org/abs/1907.13124"
    attack_name = 'ASMA'
    device = inputs.device
    batch_size = len(inputs)
    batch_view = lambda tensor: tensor.view(batch_size, *[1] * (inputs.ndim - 1))

    # Setup variables
    δ = torch.zeros_like(inputs, requires_grad=True)
    lower, upper = -inputs, 1 - inputs
    pert_mul = τ

    # Init trackers
    best_dist = torch.full((batch_size,), float('inf'), device=device)
    best_adv_percent = torch.zeros_like(best_dist)
    adv_found = torch.zeros_like(best_dist, dtype=torch.bool)
    best_adv = inputs.clone()

    for i in range(num_steps):

        adv_inputs = inputs + δ
        logits = model(adv_inputs)
        l2_squared = δ.flatten(1).square().sum(dim=1)

        if i == 0:
            # initialize variables based on model's output
            num_classes = logits.size(1)
            if masks is None:
                masks = labels < num_classes
            masks_sum = masks.flatten(1).sum(dim=1)
            labels_ = labels * masks

        # track progress
        pred = logits.argmax(dim=1)
        pixel_is_adv = (pred == labels) if targeted else (pred != labels)
        adv_percent = (pixel_is_adv & masks).flatten(1).sum(dim=1) / masks_sum
        is_adv = adv_percent >= adv_threshold
        is_smaller = l2_squared <= best_dist
        improves_constraints = adv_percent >= best_adv_percent.clamp_max(adv_threshold)
        is_better_adv = (is_smaller & is_adv) | (~adv_found & improves_constraints)
        adv_found.logical_or_(is_adv)
        best_dist = torch.where(is_better_adv, l2_squared.detach(), best_dist)
        best_adv_percent = torch.where(is_better_adv, adv_percent, best_adv_percent)
        best_adv = torch.where(batch_view(is_better_adv), adv_inputs.detach(), best_adv)

        iou = iou_masks(labels, pred, n=num_classes)
        if i:
            pert_mul = β * iou + τ

        logit_loss = logits.gather(1, labels_.unsqueeze(1)).squeeze(1).mul(masks & (pred != labels_)).sum()
        loss = logit_loss - l2_squared
        δ_grad = grad(loss.sum(), δ, only_inputs=True)[0]

        δ.data.add_(δ_grad, alpha=pert_mul).clamp_(min=lower, max=upper)

        if callback:
            callback.accumulate_line('logit_loss', i, logit_loss.mean(), title=attack_name + ' - Logit loss')
            callback.accumulate_line(['adv%', 'best_adv%'], i, [adv_percent.mean(), best_adv_percent.mean()],
                                     title=attack_name + ' - APSR')
            callback.accumulate_line(['ℓ2', 'best ℓ2'], i, [l2_squared.detach().sqrt().mean(), best_dist.sqrt().mean()],
                                     title=attack_name + ' - L2 Norms')
            callback.accumulate_line('lr', i, pert_mul, title=attack_name + ' - Step size')
            callback.accumulate_line('IoU', i, iou, title=attack_name + ' - IoU')
            if (i + 1) % (num_steps // 20) == 0 or (i + 1) == num_steps:
                callback.update_lines()

    return best_adv
