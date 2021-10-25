import torch
from torch import Tensor

__all__ = [
    'PHRQuad',
    'P1',
    'P2',
    'P3',
    'P4',
    'P5',
    'P6',
    'P7',
    'P8',
    'P9'
]


def PHRQuad(y: Tensor, ρ: Tensor, μ: Tensor) -> Tensor:
    return ((μ + ρ * y).relu().pow(2) - μ ** 2).div(2 * ρ)


def P1(y: Tensor, ρ: Tensor, μ: Tensor) -> Tensor:
    y_sup = μ * y + 0.5 * ρ * y ** 2 + ρ ** 2 * y ** 3
    y_mid = μ * y + 0.5 * ρ * y ** 2
    y_inf = - μ ** 2 / (2 * ρ)
    return torch.where(y >= 0, y_sup, torch.where(y <= -μ / ρ, y_inf, y_mid))


class P2(torch.autograd.Function):
    @staticmethod
    def forward(ctx, y: Tensor, ρ: Tensor, μ: Tensor) -> Tensor:
        ctx.save_for_backward(y, ρ, μ)
        y_sup = μ * y + μ * ρ * y ** 2 + 1 / 6 * ρ ** 2 * y ** 3
        y_inf = μ * y / (1 - ρ * y)
        return torch.where(y >= 0, y_sup, y_inf)

    @staticmethod
    def backward(ctx, grad_output):
        y, ρ, μ = ctx.saved_tensors
        grad_y = grad_ρ = grad_μ = None

        sup = y >= 0
        if ctx.needs_input_grad[0]:
            grad_y = grad_output * torch.where(sup, μ + 2 * μ * ρ * y + 0.5 * ρ ** 2 * y ** 2, μ / (1 - ρ * y) ** 2)
        if ctx.needs_input_grad[1]:
            grad_ρ = grad_output * torch.where(sup, 1 / 3 * y ** 2 * (3 * μ + ρ * y), μ * y ** 2 / (1 - ρ * y) ** 2)
        if ctx.needs_input_grad[2]:
            grad_μ = grad_output * torch.where(sup, y * (1 + ρ * y), y / (1 - ρ * y))

        return grad_y, grad_ρ, grad_μ


class P3(torch.autograd.Function):
    @staticmethod
    def forward(ctx, y: Tensor, ρ: Tensor, μ: Tensor) -> Tensor:
        ctx.save_for_backward(y, ρ, μ)
        y_sup = μ * y + μ * ρ * y ** 2
        y_inf = μ * y / (1 - ρ * y)
        return torch.where(y >= 0, y_sup, y_inf)

    @staticmethod
    def backward(ctx, grad_output):
        y, ρ, μ = ctx.saved_tensors
        grad_y = grad_ρ = grad_μ = None

        sup = y >= 0
        if ctx.needs_input_grad[0]:
            grad_y = grad_output * torch.where(sup, μ + 2 * μ * ρ * y, μ / (1 - ρ * y) ** 2)
        if ctx.needs_input_grad[1]:
            grad_ρ = grad_output * torch.where(sup, μ * y ** 2, μ * y ** 2 / (1 - ρ * y) ** 2)
        if ctx.needs_input_grad[2]:
            grad_μ = grad_output * torch.where(sup, y * (1 + ρ * y), y / (1 - ρ * y))

        return grad_y, grad_ρ, grad_μ


class GenericPenaltyLagrangian:
    def __init__(self, θ):
        self.θ = θ  # univariate function


class P4(GenericPenaltyLagrangian):
    def __call__(self, y: Tensor, ρ: Tensor, μ: Tensor) -> Tensor:
        y_sup = μ * y + self.θ(ρ * y) / ρ
        y_inf = self.θ.min(ρ, μ)
        return torch.where(self.θ.sup(y, ρ, μ), y_sup, y_inf)


class P5(GenericPenaltyLagrangian):
    def __call__(self, y: Tensor, ρ: Tensor, μ: Tensor) -> Tensor:
        return self.θ.mul * self.θ(ρ * y) * μ / ρ


class P6(GenericPenaltyLagrangian):
    def __call__(self, y: Tensor, ρ: Tensor, μ: Tensor) -> Tensor:
        return self.θ.mul * self.θ(ρ * μ * y) / ρ


class P7(GenericPenaltyLagrangian):
    def __call__(self, y: Tensor, ρ: Tensor, μ: Tensor) -> Tensor:
        return self.θ.mul * self.θ(ρ * y / μ) * (μ ** 2) / ρ


class P8(GenericPenaltyLagrangian):
    def __call__(self, y: Tensor, ρ: Tensor, μ: Tensor) -> Tensor:
        tilde_x = self.θ.tilde(μ)
        return (self.θ(ρ * y + tilde_x) - self.θ(tilde_x)) / ρ


class P9(GenericPenaltyLagrangian):
    #  Penalty-Lagrangian functions associated with P9 are not well defined for ρ <= μ, so we set ρ = max({ρ, 2μ})
    def __call__(self, y: Tensor, ρ: Tensor, μ: Tensor) -> Tensor:
        ρ_adjusted = torch.max(ρ, 2 * μ)
        tilde_x = self.θ.tilde(μ, ρ_adjusted)
        return self.θ(ρ_adjusted * y + tilde_x) - self.θ(tilde_x)
