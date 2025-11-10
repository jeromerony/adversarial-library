from functools import lru_cache
from typing import Tuple

import torch
from torch import Tensor
from torch.nn import _reduction as _Reduction, functional as F


def _reflect_pad(input: Tensor, pad_size: int, dim: int) -> Tensor:
    shape = list(input.shape)
    shape[dim] += 2 * pad_size
    out = input.new_empty(*shape)
    out.narrow(dim=dim, start=pad_size, length=input.size(dim)).copy_(input)
    out.narrow(dim=dim, start=0, length=pad_size).copy_(
        torch.flip(input.narrow(start=0, dim=dim, length=pad_size), (dim,)))
    out.narrow(dim=dim, start=out.size(dim) - pad_size, length=pad_size).copy_(
        torch.flip(input.narrow(start=input.size(dim) - pad_size, dim=dim, length=pad_size), (dim,)))
    return out


def _gaussian_kernel_2d(input: Tensor, kernel: Tensor) -> Tensor:
    c = input.size(1)
    s = (kernel.size(-1) - 1) // 2
    kernel_ = kernel.expand(c, -1, -1, -1)
    out = F.conv2d(_reflect_pad(input, pad_size=s, dim=2), kernel_.mT, groups=c)
    out = F.conv2d(_reflect_pad(out, pad_size=s, dim=3), kernel_, groups=c)
    return out


def _ssim(input: Tensor, target: Tensor, max_val: float, k1: float, k2: float, kernel: Tensor) -> Tuple[Tensor, Tensor]:
    s = (kernel.size(-1) - 1) // 2
    c1 = (k1 * max_val) ** 2
    c2 = (k2 * max_val) ** 2

    mu1 = _gaussian_kernel_2d(input, kernel=kernel)
    mu2 = _gaussian_kernel_2d(target, kernel=kernel)

    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = _gaussian_kernel_2d(input.square(), kernel=kernel).sub_(mu1_sq)
    sigma2_sq = _gaussian_kernel_2d(target.square(), kernel=kernel).sub_(mu2_sq)
    sigma12 = _gaussian_kernel_2d(input * target, kernel=kernel).sub_(mu1_mu2)

    v1 = (2 * sigma12).add_(c2)
    v2 = sigma1_sq.add_(sigma2_sq).add_(c2)

    ssim = (2 * mu1_mu2).add_(c1).mul(v1) / (mu1_sq + mu2_sq).add_(c1).mul(v2)
    ratio = v1 / v2
    return ssim[..., s:-s, s:-s], ratio[..., s:-s, s:-s]


def ssim(input: Tensor, target: Tensor, max_val: float, filter_size: int = 11, k1: float = 0.01, k2: float = 0.03,
         sigma: float = 1.5, size_average=None, reduce=None, reduction: str = 'mean') -> Tensor:
    """Measures the structural similarity index (SSIM) error."""
    dim = input.dim()
    if dim != 4:
        raise ValueError('Expected 4 dimensions (got {})'.format(dim))

    if input.size() != target.size():
        raise ValueError('Expected input size ({}) to match target size ({}).'.format(input.size(0), target.size(0)))

    if size_average is not None or reduce is not None:
        reduction = _Reduction.legacy_get_string(size_average, reduce)

    if filter_size % 2 != 1:
        raise ValueError(f"{filter_size=} needs to be odd.")

    start = -(filter_size - 1) // 2
    end = (filter_size - 1) // 2 + 1
    coords = torch.arange(start=start, end=end, device=input.device, dtype=input.dtype)
    kernel = coords.square_().div_(2 * sigma ** 2).neg_().softmax(dim=0).view(1, 1, 1, -1)
    ret, _ = _ssim(input, target, max_val, k1, k2, kernel)

    if reduction != 'none':
        ret = torch.mean(ret) if reduction == 'mean' else torch.sum(ret)
    return ret


def compute_ssim(input: Tensor, target: Tensor, **kwargs) -> Tensor:
    c_ssim = ssim(input=input, target=target, max_val=1, reduction='none', **kwargs).mean([2, 3])
    return c_ssim.mean(1)


def ssim_loss(*args, **kwargs) -> Tensor:
    return 1 - compute_ssim(*args, **kwargs)


@lru_cache()
def ms_weights(device: torch.device):
    return torch.tensor([0.0448, 0.2856, 0.3001, 0.2363, 0.1333], device=device)


def ms_ssim(input: Tensor, target: Tensor, max_val: float, filter_size: int = 11, k1: float = 0.01, k2: float = 0.03,
            sigma: float = 1.5, size_average=None, reduce=None, reduction: str = 'mean') -> Tensor:
    """Measures the multi-scale structural similarity index (MS-SSIM) error."""
    dim = input.dim()
    if dim != 4:
        raise ValueError('Expected 4 dimensions (got {}) from input'.format(dim))

    if input.size() != target.size():
        raise ValueError('Expected input size ({}) to match target size ({}).'.format(input.size(0), target.size(0)))

    if size_average is not None or reduce is not None:
        reduction = _Reduction.legacy_get_string(size_average, reduce)

    start = -(filter_size - 1) // 2
    end = (filter_size - 1) // 2 + 1
    coords = torch.arange(start=start, end=end, device=input.device, dtype=input.dtype)
    kernel = coords.square_().div_(2 * sigma ** 2).neg_().softmax(dim=0).view(1, 1, 1, -1)

    weights = ms_weights(input.device).unsqueeze(-1).unsqueeze(-1)
    levels = weights.size(0)
    mssim = []
    mcs = []
    for i in range(levels):

        if i:
            input = F.avg_pool2d(input, kernel_size=2, ceil_mode=True)
            target = F.avg_pool2d(target, kernel_size=2, ceil_mode=True)

        if min(input.shape[-2:]) <= filter_size:
            raise ValueError(
                "Image too small after average pooling, you should probably not use MS-SSIM on a small image."
            )

        ssim, cs = _ssim(input, target, max_val, k1, k2, kernel)
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
