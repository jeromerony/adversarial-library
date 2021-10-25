import pytest
import torch
from torch.autograd import grad, gradcheck

from adv_lib.utils.lagrangian_penalties import univariate_functions


@pytest.mark.parametrize('univariate', univariate_functions.__all__)
def test_grad(univariate) -> None:
    t = torch.randn(512, dtype=torch.double, requires_grad=True)
    # check if gradients are correct compared to numerical approximations using finite differences
    assert gradcheck(univariate_functions.__dict__[univariate](), inputs=t)


@pytest.mark.parametrize('univariate,value', [('LogExp', 1), ('LogQuad_1', 1), ('HyperExp', 1), ('HyperQuad', 1),
                                              ('LogBarrierQuad', 0), ('HyperBarrierQuad', 0), ('HyperLogQuad', 0),
                                              ('HyperLogQuad', 1)])
@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_nan_grad(univariate, value, dtype) -> None:
    t = torch.full((1,), value, dtype=dtype, requires_grad=True)

    univariate_func = univariate_functions.__dict__[univariate]()
    out = univariate_func(t)
    g = grad(out, t, only_inputs=True)[0]

    assert torch.isnan(g).any() == False  # check nan in gradients of penalty
