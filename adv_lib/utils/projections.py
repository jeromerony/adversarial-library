from distutils.version import LooseVersion
from typing import Union

import torch
from torch import Tensor

use_tensors_in_clamp = False
if LooseVersion(torch.__version__) >= LooseVersion('1.9'):
    use_tensors_in_clamp = True


@torch.no_grad()
def clamp(x: Tensor, lower: Tensor, upper: Tensor, inplace: bool = False) -> Tensor:
    """Clamp based on lower and upper Tensor bounds. Clamping method depends on torch version: clamping with tensors was
    introduced in torch 1.9."""
    δ_clamped = x if inplace else None
    if use_tensors_in_clamp:
        δ_clamped = torch.clamp(x, min=lower, max=upper, out=δ_clamped)
    else:
        δ_clamped = torch.maximum(x, lower, out=δ_clamped)
        δ_clamped = torch.minimum(δ_clamped, upper, out=δ_clamped)
    return δ_clamped


def clamp_(x: Tensor, lower: Tensor, upper: Tensor) -> Tensor:
    """In-place alias for clamp."""
    return clamp(x=x, lower=lower, upper=upper, inplace=True)


def simplex_projection(x: Tensor, ε: Union[float, Tensor] = 1, inplace: bool = False) -> Tensor:
    """
    Simplex projection based on sorting.

    Parameters
    ----------
    x : Tensor
        Batch of vectors to project on the simplex.
    ε : float or Tensor
        Size of the simplex, default to 1 for the probability simplex.
    inplace : bool
        Can optionally do the operation in-place.

    Returns
    -------
    projected_x : Tensor
        Batch of projected vectors on the simplex.
    """
    u = x.sort(dim=1, descending=True)[0]
    ε = ε.unsqueeze(1) if isinstance(ε, Tensor) else torch.tensor(ε, device=x.device)
    indices = torch.arange(x.size(1), device=x.device)
    cumsum = torch.cumsum(u, dim=1).sub_(ε).div_(indices + 1)
    K = (cumsum < u).long().mul_(indices).amax(dim=1, keepdim=True)
    τ = cumsum.gather(1, K)
    x = x.sub_(τ) if inplace else x - τ
    return x.clamp_(min=0)


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
        x_to_project = x[to_project]
        ε_ = ε[to_project] if isinstance(ε, Tensor) else torch.tensor([ε], device=x.device)
        if not inplace:
            x = x.clone()
        simplex_proj = simplex_projection(x_to_project.abs(), ε=ε_, inplace=True)
        x[to_project] = simplex_proj.copysign_(x_to_project)
        return x
    else:
        return x
