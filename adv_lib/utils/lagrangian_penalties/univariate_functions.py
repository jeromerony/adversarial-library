import math

import torch
from torch import Tensor
from torch.nn import functional as F

__all__ = [
    'Quadratic',
    'FourThirds',
    'Cosh',
    'Exp',
    'LogExp',
    'LogQuad_1',
    'LogQuad_2',
    'HyperExp',
    'HyperQuad',
    'DualLogQuad',
    'CubicQuad',
    'ExpQuad',
    'LogBarrierQuad',
    'HyperBarrierQuad',
    'HyperLogQuad',
    'SmoothPlus',
    'NNSmoothPlus',
    'ExpSmoothPlus',
]


def safe_exp(x: Tensor) -> Tensor:
    return torch.exp(x.clamp_max(87.5))


class Quadratic:
    @staticmethod
    def __call__(t: Tensor) -> Tensor:
        return 0.5 * t ** 2

    @staticmethod
    def min(ρ: Tensor, μ: Tensor) -> Tensor:
        return - μ ** 2 / (2 * ρ)

    @staticmethod
    def sup(t: Tensor, ρ: Tensor, μ: Tensor) -> Tensor:
        return μ + ρ * t >= 0


class FourThirds:
    @staticmethod
    def __call__(t: Tensor) -> Tensor:
        return 0.75 * t.abs().pow(4 / 3)

    @staticmethod
    def min(ρ: Tensor, μ: Tensor) -> Tensor:
        return - μ ** 4 / (4 * ρ)

    @staticmethod
    def sup(t: Tensor, ρ: Tensor, μ: Tensor) -> Tensor:
        return ρ.pow(1 / 3) * t.sign() * t.abs().pow(1 / 3) + μ >= 0


class Cosh:
    @staticmethod
    def __call__(t: Tensor) -> Tensor:
        return t.cosh() - 1

    @staticmethod
    def min(ρ: Tensor, μ: Tensor) -> Tensor:
        return μ / ρ * torch.asinh(-μ) + 1 / ρ * (torch.cosh(torch.asinh(-μ)) - 1)

    @staticmethod
    def sup(t: Tensor, ρ: Tensor, μ: Tensor) -> Tensor:
        return torch.sinh(μ + (ρ * t)) >= 0


class Exp:
    mul = 1

    @staticmethod
    def __call__(t: Tensor) -> Tensor:
        return safe_exp(t) - 1

    @staticmethod
    def tilde(μ: Tensor) -> Tensor:
        return torch.log(μ)


class LogExp:
    mul = 1

    @staticmethod
    def __call__(t: Tensor) -> Tensor:
        y_sup = safe_exp(2 * t - 1) + math.log(2) - 1
        y_inf = -torch.log(1 - t.clamp(max=0.5))
        return torch.where(t >= 0.5, y_sup, y_inf)

    @staticmethod
    def tilde(μ: Tensor) -> Tensor:
        y_sup = 0.5 * (torch.log(μ / 2) + 1)
        y_inf = (μ - 1) / μ
        return torch.where(μ >= 2, y_sup, y_inf)


class LogQuad_1:
    mul = 1

    @staticmethod
    def __call__(t: Tensor) -> Tensor:
        y_sup = 2 * t ** 2 + math.log(2) - 0.5
        y_inf = -torch.log(1 - t.clamp(max=0.5))
        return torch.where(t >= 0.5, y_sup, y_inf)

    @staticmethod
    def tilde(μ: Tensor) -> Tensor:
        y_sup = μ / 4
        y_inf = (μ - 1) / μ
        return torch.where(μ >= 2, y_sup, y_inf)


class LogQuad_2:
    mul = 1

    @staticmethod
    def __call__(t: Tensor) -> Tensor:
        y_sup = 0.5 * t ** 2 + t
        y_inf = -0.25 * torch.log(-2 * t.clamp(max=-0.5)) - 0.375
        return torch.where(t >= -0.5, y_sup, y_inf)

    @staticmethod
    def tilde(μ: Tensor) -> Tensor:
        y_sup = μ - 1
        y_inf = -1 / (4 * μ)
        return torch.where(μ >= 0.5, y_sup, y_inf)


class HyperExp:
    mul = 1

    @staticmethod
    def __call__(t: Tensor) -> Tensor:
        y_sup = safe_exp(4 * t - 2)
        y_inf = t / (1 - t)
        return torch.where(t >= 0.5, y_sup, y_inf)

    @staticmethod
    def tilde(μ: Tensor) -> Tensor:
        y_sup = torch.log(μ / 4) / 4 + 0.5
        y_inf = 1 - 1 / torch.sqrt(μ)
        return torch.where(μ >= 4, y_sup, y_inf)


class HyperQuad:
    mul = 1

    @staticmethod
    def __call__(t: Tensor) -> Tensor:
        y_sup = 8 * t ** 2 - 4 * t + 1
        y_inf = t / (1 - t)
        return torch.where(t >= 0.5, y_sup, y_inf)

    @staticmethod
    def tilde(μ: Tensor) -> Tensor:
        y_sup = (μ + 4) / 16
        y_inf = 1 - 1 / torch.sqrt(μ)
        return torch.where(μ >= 4, y_sup, y_inf)


class DualLogQuad:
    mul = 1

    @staticmethod
    def __call__(t: Tensor) -> Tensor:
        return (1 + t + torch.sqrt((1 + t) ** 2 + 8)) ** 2 / 16 + torch.log(
            0.25 * (1 + t + torch.sqrt((1 + t) ** 2 + 8))) - 1

    @staticmethod
    def tilde(μ: Tensor) -> Tensor:
        return 2 * μ - 1 / μ - 1


class CubicQuad:
    mul = 8

    @staticmethod
    def __call__(t: Tensor) -> Tensor:
        y_sup = 0.5 * t ** 2
        y_inf = 1 / 6 * (t + 0.5).clamp(min=0) ** 3 - 1 / 24
        return torch.where(t >= 0.5, y_sup, y_inf)

    @staticmethod
    def tilde(μ: Tensor) -> Tensor:
        y_sup = μ
        y_inf = torch.sqrt(2 * μ) - 0.5
        return torch.where(μ >= 0.5, y_sup, y_inf)


class ExpQuad:
    mul = 1

    @staticmethod
    def __call__(t: Tensor) -> Tensor:
        y_sup = math.exp(0.5) * (0.5 * t ** 2 + 0.5 * t + 0.625)
        y_inf = safe_exp(t)
        return torch.where(t >= 0.5, y_sup, y_inf)

    @staticmethod
    def tilde(μ: Tensor) -> Tensor:
        y_sup = μ * math.exp(-0.5) - 0.5
        y_inf = torch.log(μ)
        return torch.where(μ >= math.exp(0.5), y_sup, y_inf)


class LogBarrierQuad:
    mul = 0.25

    @staticmethod
    def __call__(t: Tensor) -> Tensor:
        y_sup = 2 * t ** 2 + 4 * t + 0.5 + math.log(2)
        y_inf = -torch.log(-t.clamp(max=-0.5)) - 1
        return torch.where(t >= -0.5, y_sup, y_inf)

    @staticmethod
    def tilde(μ: Tensor) -> Tensor:
        y_sup = (μ - 4) / 4
        y_inf = -1 / μ
        return torch.where(μ >= 2, y_sup, y_inf)


class HyperBarrierQuad:
    mul = 1 / 12

    @staticmethod
    def __call__(t: Tensor) -> Tensor:
        y_sup = 8 * t ** 2 + 12 * t + 6
        y_inf = -1 / t.clamp(max=-0.5)
        return torch.where(t >= -0.5, y_sup, y_inf)

    @staticmethod
    def tilde(μ: Tensor) -> Tensor:
        y_sup = (μ - 12) / 16
        y_inf = -1 / torch.sqrt(μ)
        return torch.where(μ >= 4, y_sup, y_inf)


class HyperLogQuad:
    mul = 0.125

    @staticmethod
    def __call__(t: Tensor) -> Tensor:
        y_sup = 8 * t ** 2 + 8 * t + 1.5 + 2 * math.log(2)
        y_mid = -torch.log(-t.clamp(-1, -0.25))
        y_inf = 4 / (1 - t.clamp(max=-1)) - 2
        return torch.where(t >= -0.25, y_sup, torch.where(t <= -1, y_inf, y_mid))

    @staticmethod
    def tilde(μ: Tensor) -> Tensor:
        y_sup = (μ - 8) / 16
        y_mid = -1 / μ
        y_inf = 1 - 2 / torch.sqrt(μ)
        return torch.where(μ >= 4, y_sup, torch.where(μ <= 1, y_inf, y_mid))


class SmoothPlus:
    mul = 2

    @staticmethod
    def __call__(t: Tensor) -> Tensor:
        return 0.5 * (t + torch.sqrt(t ** 2 + 4))

    @staticmethod
    def tilde(μ: Tensor, ρ: Tensor) -> Tensor:
        return (2 * μ - ρ) / torch.sqrt(μ * ρ - μ ** 2)


class NNSmoothPlus:
    mul = 2

    @staticmethod
    def __call__(t: Tensor) -> Tensor:
        return F.softplus(t)

    @staticmethod
    def tilde(μ: Tensor, ρ: Tensor) -> Tensor:
        return torch.log(μ / (ρ - μ))


class ExpSmoothPlus:
    mul = 2

    @staticmethod
    def __call__(t: Tensor) -> Tensor:
        y_sup = t + 0.5 * safe_exp(-t)
        y_inf = 0.5 * safe_exp(t)
        return torch.where(t >= 0, y_sup, y_inf)

    @staticmethod
    def tilde(μ: Tensor, ρ: Tensor) -> Tensor:
        y_sup = torch.log(ρ / (2 * (ρ - μ)))
        y_inf = torch.log(2 * μ / ρ)
        return torch.where(μ >= 0.5 * ρ, y_sup, y_inf)
