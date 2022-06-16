from typing import Optional

import torch
from adv_lib.utils.losses import difference_of_logits
from adv_lib.utils.visdom_logger import VisdomLogger
from torch import Tensor, nn
from torch.autograd import grad


def dag(model: nn.Module,
        inputs: Tensor,
        labels: Tensor,
        masks: Tensor = None,
        targeted: bool = False,
        adv_threshold: float = 0.99,
        max_iter: int = 200,
        γ: float = 0.5,
        p: float = float('inf'),
        callback: Optional[VisdomLogger] = None) -> Tensor:
    """DAG attack from https://arxiv.org/abs/1703.08603"""
    device = inputs.device
    batch_size = len(inputs)
    batch_view = lambda tensor: tensor.view(-1, *[1] * (inputs.ndim - 1))
    multiplier = -1 if targeted else 1

    # Setup variables
    r = torch.zeros_like(inputs)

    # Init trackers
    best_adv_percent = torch.zeros(batch_size, device=device)
    adv_found = torch.zeros_like(best_adv_percent, dtype=torch.bool)
    best_adv = inputs.clone()

    for i in range(max_iter):

        active_inputs = ~adv_found
        inputs_ = inputs[active_inputs]
        r_ = r[active_inputs]
        r_.requires_grad_(True)

        adv_inputs_ = (inputs_ + r_).clamp(0, 1)
        logits = model(adv_inputs_)

        if i == 0:
            num_classes = logits.size(1)
            if masks is None:
                masks = labels < num_classes
            masks_sum = masks.flatten(1).sum(dim=1)
            masked_labels = labels * masks
            labels_infhot = torch.zeros_like(logits.detach()).scatter(1, masked_labels.unsqueeze(1), float('inf'))

        dl = multiplier * difference_of_logits(logits, labels=masked_labels[active_inputs],
                                               labels_infhot=labels_infhot[active_inputs])
        pixel_is_adv = dl < 0

        active_masks = masks[active_inputs]
        adv_percent = (pixel_is_adv & active_masks).flatten(1).sum(dim=1) / masks_sum[active_inputs]
        is_adv = adv_percent >= adv_threshold
        adv_found[active_inputs] = is_adv
        best_adv[active_inputs] = torch.where(batch_view(is_adv), adv_inputs_.detach(), best_adv[active_inputs])

        if callback:
            callback.accumulate_line('dl', i, dl[active_masks].mean(), title=f'DAG (p={p}, γ={γ}) - DL')
            callback.accumulate_line(f'L{p}', i, r.flatten(1).norm(p=p, dim=1).mean(), title=f'DAG (p={p}, γ={γ}) - Norm')
            callback.accumulate_line('adv%', i, adv_percent.mean(), title=f'DAG (p={p}, γ={γ}) - Adv percent')

            if (i + 1) % (max_iter // 20) == 0 or (i + 1) == max_iter:
                callback.update_lines()

        if is_adv.all():
            break

        loss = (dl[~is_adv] * active_masks[~is_adv]).relu()
        r_grad = grad(loss.sum(), r_, only_inputs=True)[0]
        r_grad.div_(batch_view(r_grad.flatten(1).norm(p=p, dim=1).clamp_min_(1e-8)))
        r_.data.sub_(r_grad, alpha=γ)

        r[active_inputs] = r_

    if callback:
        callback.update_lines()

    return best_adv
