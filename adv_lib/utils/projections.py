from typing import Union

import torch
from torch import Tensor


def l1_ball_euclidean_projection(x: Tensor, ε: Union[float, Tensor], inplace: bool = False) -> Tensor:
    """
    Compute Euclidean projection onto the L1 ball for a batch.

      min ||x - u||_2 s.t. ||u||_1 <= eps

    Inspired by the corresponding numpy version by Adrien Gaidon.
    Adapted from Tony Duan's implementation https://gist.github.com/tonyduan/1329998205d88c566588e57e3e2c0c55

    Parameters
    ----------
    x: Tensor
        Batch of tensors to project.
    ε: float or Tensor
        Radius of L1-ball to project onto. Can be a single value for all tensors in the batch or a batch of values.
    inplace : bool
        Can optionally do the operation in-place.

    Returns
    -------
    projected_x: Tensor
        Batch of projected tensors with the same shape as x.

    Notes
    -----
    The complexity of this algorithm is in O(dlogd) as it involves sorting x.

    References
    ----------
    [1] Efficient Projections onto the l1-Ball for Learning in High Dimensions
        John Duchi, Shai Shalev-Shwartz, Yoram Singer, and Tushar Chandra.
        International Conference on Machine Learning (ICML 2008)
    """
    if (to_project := x.norm(p=1, dim=1) > ε).any():
        x_to_project = x[to_project].abs()
        ε_ = ε[to_project] if isinstance(ε, Tensor) else torch.tensor([ε], device=x.device)
        if not inplace:
            x = x.clone()
        μ = x_to_project.sort(dim=1, descending=True)[0]
        cumsum = μ.cumsum(dim=1)
        j_s = torch.arange(1, x.shape[-1] + 1, device=x.device).unsqueeze(0)
        ρ = (((μ * j_s) > (cumsum - ε_.unsqueeze(1))) * j_s).argmax(dim=1, keepdim=True)
        θ = (cumsum.gather(dim=1, index=ρ) - ε_.unsqueeze(1)) / (ρ + 1)
        x[to_project] = (x_to_project - θ).clamp_min(0) * x[to_project].sign()
        return x
    else:
        return x
