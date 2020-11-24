from functools import partial
from typing import Union

from torch import Tensor


def lp_distances(x1: Tensor, x2: Tensor, p: Union[float, int] = 2, dim: int = 1) -> Tensor:
    return (x1 - x2).flatten(dim).norm(p=p, dim=dim)


l0_distances = partial(lp_distances, p=0)
l1_distances = partial(lp_distances, p=1)
l2_distances = partial(lp_distances, p=2)
linf_distances = partial(lp_distances, p=float('inf'))


def squared_l2_distances(x1: Tensor, x2: Tensor, dim: int = 1) -> Tensor:
    return (x1 - x2).pow(2).flatten(dim).sum(1)
