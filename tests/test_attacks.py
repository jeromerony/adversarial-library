import math
from functools import partial

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from adv_lib import attacks


class Linear(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.weight = nn.Buffer(torch.ones(input_dim) / input_dim)
        self.bias = nn.Buffer(torch.tensor(-0.5))

    def logit(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x.flatten(1), self.weight.unsqueeze(0), self.bias)

    def project(self, x: torch.Tensor) -> torch.Tensor:
        w = self.weight
        projection = x.flatten(1) - self.logit(x) / (w @ w) * w
        return projection.view_as(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logit = self.logit(x)
        return torch.cat([-logit, logit, torch.full_like(logit, -1)], dim=1)  # simulate multi-class


_attacks = {
    'ddn': partial(attacks.ddn, steps=30),
    'alma': partial(attacks.alma, num_steps=50),
    'df': partial(attacks.df, steps=10),
    'c&w': partial(attacks.carlini_wagner_l2, max_iterations=30),
    'tr': partial(attacks.tr, iter=10, eps=0.1),
    'fab': partial(attacks.fab, norm=2, n_iter=10),
    'fmn': partial(attacks.fmn, norm=2, steps=10),
    'pdgd': partial(attacks.pdgd, num_steps=200),
    'pdpgd': partial(attacks.pdpgd, norm=2, num_steps=200),
    'sdf': partial(attacks.sdf, steps=10),
}


@pytest.mark.parametrize('attack', _attacks.keys())
@pytest.mark.parametrize('batch_size', [1, 3, 8])
@pytest.mark.parametrize('dims', ((8,), (4, 6), (5, 7, 7)))
def test_minimal_l2_attack(attack: str, batch_size: int, dims: tuple[int]):
    torch.manual_seed(0)
    model = Linear(input_dim=math.prod(dims))
    inputs = torch.randn(batch_size, *dims).mul_(0.01).add_(0.55).clamp_(min=0, max=1)
    labels = inputs.new_ones(batch_size, dtype=torch.long)

    preds = model(inputs).argmax(dim=1)
    torch.testing.assert_close(preds, labels)

    adv_inputs = _attacks[attack](model=model, inputs=inputs, labels=labels)
    adv_preds = model(adv_inputs).argmax(dim=1)
    assert (adv_preds != labels).all()

    projection = model.project(inputs)
    pred_projection = model(projection)[..., :2]
    torch.testing.assert_close(pred_projection, torch.zeros_like(pred_projection))

    best_norm = (projection - inputs).norm(p=2, dim=1)
    adv_norm = (adv_inputs - inputs).norm(p=2, dim=1)
    assert (adv_norm <= best_norm * 2).all()
