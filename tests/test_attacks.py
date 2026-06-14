import logging
import math
from functools import partial
from typing import Callable

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from adv_lib import attacks

logger = logging.getLogger(__name__)


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
        neg_logit = torch.full_like(logit, -1)
        return torch.cat([-logit, logit, neg_logit, 2*neg_logit], dim=1)  # simulate multi-class


_attacks_untargeted_minimal_l2 = (
    partial(attacks.ddn, steps=50),
    partial(attacks.alma, num_steps=50),
    partial(attacks.df, steps=10),
    partial(attacks.carlini_wagner_l2, max_iterations=20, learning_rate=0.1),
    partial(attacks.tr, iter=50, eps=0.1),
    partial(attacks.fab, norm=2, n_iter=10),
    partial(attacks.fmn, norm=2, steps=10),
    partial(attacks.pdgd, num_steps=200),
    partial(attacks.pdpgd, norm=2, num_steps=200),
    partial(attacks.sdf, steps=10),
)


@pytest.mark.parametrize('attack', _attacks_untargeted_minimal_l2)
@pytest.mark.parametrize('batch_size', [1, 3, 8])
@pytest.mark.parametrize('dims', ((8,), (4, 6), (5, 7, 7)))
def test_untargeted_minimal_l2(attack: Callable, batch_size: int, dims: tuple[int]):
    torch.manual_seed(0)
    model = Linear(input_dim=math.prod(dims))
    inputs = torch.randn(batch_size, *dims).mul_(0.03).add_(0.75).clamp_(min=0, max=1)
    labels = inputs.new_ones(batch_size, dtype=torch.long)

    preds = model(inputs).argmax(dim=1)
    torch.testing.assert_close(preds, labels)

    adv_inputs = attack(model=model, inputs=inputs, labels=labels)
    adv_preds = model(adv_inputs).argmax(dim=1)
    assert (adv_preds != labels).all()

    projection = model.project(inputs)
    pred_projection = model(projection)[..., :2]
    torch.testing.assert_close(pred_projection, torch.zeros_like(pred_projection))

    best_norm = (projection - inputs).flatten(1).norm(p=2, dim=1)
    adv_norm = (adv_inputs - inputs).flatten(1).norm(p=2, dim=1)
    ratio = adv_norm / best_norm
    logger.info(f"Ratio of adv norm over best: {ratio.numpy(force=True)}")
    assert (ratio <= 1.25).all()


_attacks_targeted_minimal_l2 = (
    partial(attacks.ddn, steps=50),
    partial(attacks.alma, num_steps=50),
    partial(attacks.carlini_wagner_l2, max_iterations=20, learning_rate=0.1),
    partial(attacks.fmn, norm=2, steps=10),
    partial(attacks.pdgd, num_steps=200),
    partial(attacks.pdpgd, norm=2, num_steps=200),
)


@pytest.mark.parametrize('attack', _attacks_targeted_minimal_l2)
@pytest.mark.parametrize('batch_size', [1, 3, 8])
@pytest.mark.parametrize('dims', ((8,), (4, 6), (5, 7, 7)))
def test_targeted_minimal_l2(attack: Callable, batch_size: int, dims: tuple[int]):
    torch.manual_seed(0)
    model = Linear(input_dim=math.prod(dims))
    inputs = torch.randn(batch_size, *dims).mul_(0.03).add_(0.75).clamp_(min=0, max=1)
    labels = inputs.new_ones(batch_size, dtype=torch.long)
    targets = inputs.new_zeros(batch_size, dtype=torch.long)

    preds = model(inputs).argmax(dim=1)
    torch.testing.assert_close(preds, labels)

    adv_inputs = attack(model=model, inputs=inputs, labels=targets, targeted=True)
    adv_preds = model(adv_inputs).argmax(dim=1)
    torch.testing.assert_close(adv_preds, targets)

    projection = model.project(inputs)
    pred_projection = model(projection)[..., :2]
    torch.testing.assert_close(pred_projection, torch.zeros_like(pred_projection))

    best_norm = (projection - inputs).flatten(1).norm(p=2, dim=1)
    adv_norm = (adv_inputs - inputs).flatten(1).norm(p=2, dim=1)
    ratio = adv_norm / best_norm
    logger.info(f"Ratio of adv norm over best: {ratio.numpy(force=True)}")
    assert (ratio <= 1.25).all()


_attacks_untargeted_budget_l2 = (
    (partial(attacks.apgd, norm=2, n_iter=10), 'eps'),
)


@pytest.mark.parametrize('attack,budget_kw', _attacks_untargeted_budget_l2)
@pytest.mark.parametrize('batch_size', [1, 3, 8])
@pytest.mark.parametrize('dims', ((8,), (4, 6), (5, 7, 7)))
def test_untargeted_budget_l2(attack: Callable, budget_kw: str, batch_size: int, dims: tuple[int]):
    torch.manual_seed(0)
    model = Linear(input_dim=math.prod(dims))
    inputs = torch.randn(batch_size, *dims).mul_(0.01).add_(0.75).clamp_(min=0, max=1)
    labels = inputs.new_ones(batch_size, dtype=torch.long)

    projection = model.project(inputs)
    pred_projection = model(projection)[..., :2]
    torch.testing.assert_close(pred_projection, torch.zeros_like(pred_projection))

    budget = (projection - inputs).flatten(1).norm(p=2, dim=1).max().item() * 1.25

    preds = model(inputs).argmax(dim=1)
    torch.testing.assert_close(preds, labels)

    kwargs = dict(model=model, inputs=inputs, labels=labels)
    kwargs[budget_kw] = budget
    adv_inputs = attack(**kwargs)
    adv_preds = model(adv_inputs).argmax(dim=1)
    assert (adv_preds != labels).all()
    adv_norm = (adv_inputs - inputs).flatten(1).norm(p=2, dim=1)
    assert (adv_norm <= budget * 1.01).all()


_attacks_targeted_budget_l2 = (
    (partial(attacks.apgd, norm=2, n_iter=10), 'eps'),
)


@pytest.mark.parametrize('attack,budget_kw', _attacks_targeted_budget_l2)
@pytest.mark.parametrize('batch_size', [1, 3, 8])
@pytest.mark.parametrize('dims', ((8,), (4, 6), (5, 7, 7)))
def test_targeted_budget_l2(attack: Callable, budget_kw: str, batch_size: int, dims: tuple[int]):
    torch.manual_seed(0)
    model = Linear(input_dim=math.prod(dims))
    inputs = torch.randn(batch_size, *dims).mul_(0.01).add_(0.75).clamp_(min=0, max=1)
    labels = inputs.new_ones(batch_size, dtype=torch.long)
    targets = inputs.new_zeros(batch_size, dtype=torch.long)

    projection = model.project(inputs)
    pred_projection = model(projection)[..., :2]
    torch.testing.assert_close(pred_projection, torch.zeros_like(pred_projection))

    budget = (projection - inputs).flatten(1).norm(p=2, dim=1).max().item() * 1.25

    preds = model(inputs).argmax(dim=1)
    torch.testing.assert_close(preds, labels)

    kwargs = dict(model=model, inputs=inputs, labels=targets, targeted=True)
    kwargs[budget_kw] = budget
    adv_inputs = attack(**kwargs)
    adv_preds = model(adv_inputs).argmax(dim=1)
    torch.testing.assert_close(adv_preds, targets)
    adv_norm = (adv_inputs - inputs).flatten(1).norm(p=2, dim=1)
    assert (adv_norm <= budget * 1.01).all()
