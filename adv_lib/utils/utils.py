from collections import OrderedDict
from typing import Tuple

import torch
from torch import nn, Tensor


class ForwardCounter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.num_samples_called = 0

    def __call__(self, module, input) -> None:
        self.num_samples_called += len(input[0])


class BackwardCounter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.num_samples_called = 0

    def __call__(self, module, grad_input, grad_output) -> None:
        self.num_samples_called += len(grad_output[0])


class ImageNormalizer(nn.Module):
    def __init__(self, mean: Tuple[float, float, float], std: Tuple[float, float, float]) -> None:
        super(ImageNormalizer, self).__init__()

        self.register_buffer('mean', torch.as_tensor(mean).view(1, 3, 1, 1))
        self.register_buffer('std', torch.as_tensor(std).view(1, 3, 1, 1))

    def forward(self, input: Tensor) -> Tensor:
        return (input - self.mean) / self.std


def normalize_model(model: nn.Module, mean: Tuple[float, float, float], std: Tuple[float, float, float]) -> nn.Module:
    layers = OrderedDict([
        ('normalize', ImageNormalizer(mean, std)),
        ('model', model)
    ])
    return nn.Sequential(layers)


def requires_grad_(model: nn.Module, requires_grad: bool) -> None:
    for param in model.parameters():
        param.requires_grad_(requires_grad)


def predict_inputs(model: nn.Module, inputs: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
    logits = model(inputs)
    probabilities = torch.softmax(logits, 1)
    predictions = logits.argmax(1)
    return logits, probabilities, predictions
