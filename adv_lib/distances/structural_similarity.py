# Adapted from https://github.com/pytorch/pytorch/pull/22289
from functools import lru_cache
from typing import Tuple

import torch
from torch import Tensor
from torch.nn import _reduction as _Reduction
from torch.nn.functional import conv2d, avg_pool2d


@lru_cache
def _fspecial_gaussian(size: int, channel: int, sigma: float, device: torch.device,
                       max_size: Tuple[int, int]) -> Tensor:
    coords = -(torch.arange(size, device=device) - (size - 1) / 2) ** 2 / (2. * sigma ** 2)
    if max(max_size) <= size:
        coords_x, coords_y = torch.zeros(max_size[0], device=device), torch.zeros(max_size[1], device=device)
    elif max_size[0] <= size:
        coords_x, coords_y = torch.zeros(max_size[0], device=device), coords
    elif max_size[1] <= size:
        coords_x, coords_y = coords, torch.zeros(max_size[1], device=device)
    else:
        coords_x = coords_y = coords
    final_size = (min(max_size[0], size), min(max_size[1], size))

    grid = coords_x.view(-1, 1) + coords_y.view(1, -1)
    kernel = grid.view(1, -1).softmax(-1).view(1, 1, *final_size).expand(channel, 1, -1, -1).contiguous()
    return kernel


def _ssim(input: Tensor, target: Tensor, max_val: float, k1: float, k2: float, channel: int,
          kernel: Tensor) -> Tuple[Tensor, Tensor]:
    c1 = (k1 * max_val) ** 2
    c2 = (k2 * max_val) ** 2

    mu1 = conv2d(input, kernel, groups=channel)
    mu2 = conv2d(target, kernel, groups=channel)

    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = conv2d(input * input, kernel, groups=channel) - mu1_sq
    sigma2_sq = conv2d(target * target, kernel, groups=channel) - mu2_sq
    sigma12 = conv2d(input * target, kernel, groups=channel) - mu1_mu2

    v1 = 2 * sigma12 + c2
    v2 = sigma1_sq + sigma2_sq + c2

    ssim = ((2 * mu1_mu2 + c1) * v1) / ((mu1_sq + mu2_sq + c1) * v2)
    return ssim, v1 / v2


def ssim(input: Tensor, target: Tensor, max_val: float, filter_size: int = 11, k1: float = 0.01, k2: float = 0.03,
         sigma: float = 1.5, size_average=None, reduce=None, reduction: str = 'mean') -> Tensor:
    """Measures the structural similarity index (SSIM) error."""
    dim = input.dim()
    if dim != 4:
        raise ValueError('Expected 4 dimensions (got {})'.format(dim))

    if input.size() != target.size():
        raise ValueError('Expected input size ({}) to match target size ({}).'
                         .format(input.size(0), target.size(0)))

    if size_average is not None or reduce is not None:
        reduction = _Reduction.legacy_get_string(size_average, reduce)

    _, channel, _, _ = input.size()
    kernel = _fspecial_gaussian(filter_size, channel, sigma, device=input.device, max_size=input.shape[-2:])
    ret, _ = _ssim(input, target, max_val, k1, k2, channel, kernel)

    if reduction != 'none':
        ret = torch.mean(ret) if reduction == 'mean' else torch.sum(ret)
    return ret


def compute_ssim(input: Tensor, target: Tensor, **kwargs) -> Tensor:
    c_ssim = ssim(input=input, target=target, max_val=1, reduction='none', **kwargs).mean([2, 3])
    return c_ssim.mean(1)


def ssim_loss(*args, **kwargs) -> Tensor:
    return 1 - compute_ssim(*args, **kwargs)


def ms_ssim(input: Tensor, target: Tensor, max_val: float, filter_size: int = 11, k1: float = 0.01, k2: float = 0.03,
            sigma: float = 1.5, size_average=None, reduce=None, reduction: str = 'mean') -> Tensor:
    """Measures the multi-scale structural similarity index (MS-SSIM) error."""
    dim = input.dim()
    if dim != 4:
        raise ValueError('Expected 4 dimensions (got {}) from input'.format(dim))

    if input.size() != target.size():
        raise ValueError('Expected input size ({}) to match target size ({}).'
                         .format(input.size(0), target.size(0)))

    if size_average is not None or reduce is not None:
        reduction = _Reduction.legacy_get_string(size_average, reduce)

    _, channel, _, _ = input.size()
    kernel = _fspecial_gaussian(filter_size, channel, sigma, device=input.device, max_size=input.shape[-2:])

    weights = torch.tensor([0.0448, 0.2856, 0.3001, 0.2363, 0.1333], device=input.device)
    weights = weights.unsqueeze(-1).unsqueeze(-1)
    levels = weights.size(0)
    mssim = []
    mcs = []
    for i in range(levels):

        if i:
            input = avg_pool2d(input, kernel_size=2, ceil_mode=True)
            target = avg_pool2d(target, kernel_size=2, ceil_mode=True)

        if min(size := input.shape[-2:]) <= filter_size:
            kernel = _fspecial_gaussian(filter_size, channel, sigma, device=input.device, max_size=size)

        ssim, cs = _ssim(input, target, max_val, k1, k2, channel, kernel)
        ssim = ssim.mean((2, 3))
        cs = cs.mean((2, 3))
        mssim.append(ssim)
        mcs.append(cs)

    mssim = torch.stack(mssim)
    mcs = torch.stack(mcs)
    p1 = mcs ** weights
    p2 = mssim ** weights

    ret = torch.prod(p1[:-1], 0) * p2[-1]

    if reduction != 'none':
        ret = torch.mean(ret) if reduction == 'mean' else torch.sum(ret)
    return ret


def compute_ms_ssim(input: Tensor, target: Tensor, **kwargs) -> Tensor:
    channel_ssim = ms_ssim(input=input, target=target, max_val=1, reduction='none', **kwargs)
    return channel_ssim.mean(1)


def ms_ssim_loss(*args, **kwargs) -> Tensor:
    return 1 - compute_ms_ssim(*args, **kwargs)
