import warnings

import torch
from torch import Tensor, nn
from torch.autograd import grad

from .deepfool import df


def sdf(model: nn.Module,
        inputs: Tensor,
        labels: Tensor,
        targeted: bool = False,
        steps: int = 100,
        df_steps: int = 100,
        overshoot: float = 0.02,
        search_iter: int = 10) -> Tensor:
    """
    SuperDeepFool attack from https://arxiv.org/abs/2303.12481.

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
        Number of steps.
    df_steps : int
        Maximum number of steps for DeepFool attack at each iteration of SuperDeepFool.
    overshoot : float
        overshoot parameter in DeepFool.
    search_iter : int
        Number of binary search steps at the end of the attack.

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
    adv_inputs = inputs_ = inputs
    labels_ = labels
    adv_out = inputs.clone()
    adv_found = torch.zeros(batch_size, dtype=torch.bool, device=device)

    for i in range(steps):
        logits = model(adv_inputs)
        pred_labels = logits.argmax(dim=1)

        is_adv = pred_labels != labels_
        if is_adv.any():
            adv_not_found = ~adv_found
            adv_out[adv_not_found] = torch.where(batch_view(is_adv), adv_inputs, adv_out[adv_not_found])
            adv_found.masked_scatter_(adv_not_found, is_adv)
            if is_adv.all():
                break

            not_adv = ~is_adv
            inputs_, adv_inputs, labels_ = inputs_[not_adv], adv_inputs[not_adv], labels_[not_adv]

        # start by doing deepfool -> need to return adv_inputs even for unsuccessful attacks
        df_adv_inputs, df_targets = df(model=model, inputs=adv_inputs, labels=labels_, steps=df_steps, norm=2,
                                       overshoot=overshoot, return_unsuccessful=True, return_targets=True)

        r_df = df_adv_inputs - inputs_
        df_adv_inputs.requires_grad_(True)
        logits = model(df_adv_inputs)
        pred_labels = logits.argmax(dim=1)
        pred_labels = torch.where(pred_labels != labels_, pred_labels, df_targets)

        logit_diff = logits.gather(1, pred_labels.unsqueeze(1)) - logits.gather(1, labels_.unsqueeze(1))
        w = grad(logit_diff.sum(), inputs=df_adv_inputs, only_inputs=True)[0]
        w.div_(batch_view(w.flatten(1).norm(p=2, dim=1).clamp_(min=1e-6)))  # w / ||w||_2
        scale = torch.linalg.vecdot(r_df.flatten(1), w.flatten(1), dim=1)  # (\tilde{x} - x_0)^T w / ||w||_2

        adv_inputs = adv_inputs.addcmul(batch_view(scale), w)
        adv_inputs.clamp_(min=0, max=1)  # added compared to original implementation to produce valid adv

    if search_iter:  # binary search to bring perturbation as close to the decision boundary as possible
        low, high = torch.zeros(batch_size, device=device), torch.ones(batch_size, device=device)
        for i in range(search_iter):
            mid = (low + high) / 2
            logits = torch.lerp(inputs, adv_out, weight=batch_view(mid))
            pred_labels = model(logits).argmax(dim=1)
            is_adv = pred_labels != labels
            high = torch.where(is_adv, mid, high)
            low = torch.where(is_adv, low, mid)
        adv_out = torch.lerp(inputs, adv_out, weight=batch_view(high))

    return adv_out
