import pytest
import torch
from torch.autograd import grad, gradcheck

from adv_lib.utils.lagrangian_penalties import all_penalties


@pytest.mark.parametrize('penalty', list(all_penalties.values()))
def test_grad(penalty) -> None:
    y = torch.randn(512, dtype=torch.double, requires_grad=True)
    ρ = torch.randn(512, dtype=torch.double).abs_().clamp_min_(1e-3)
    μ = torch.randn(512, dtype=torch.double).abs_().clamp_min_(1e-6)
    ρ.requires_grad_(True)
    μ.requires_grad_(True)

    # check if gradients are correct compared to numerical approximations using finite differences
    assert gradcheck(penalty, inputs=(y, ρ, μ))


@pytest.mark.parametrize('penalty,value', [(all_penalties['P2'], 1), (all_penalties['P3'], 1)])
@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_nan_grad(penalty, value, dtype) -> None:
    y = torch.full((1,), value, dtype=dtype, requires_grad=True)
    ρ = torch.full((1,), value, dtype=dtype)
    μ = torch.full((1,), value, dtype=dtype)

    out = penalty(y, ρ, μ)
    g = grad(out, y, only_inputs=True)[0]

    assert torch.isnan(g).any() == False  # check nan in gradients of penalty
