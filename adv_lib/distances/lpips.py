from typing import Tuple, Optional, Union

import torch
from torch import nn, Tensor
from torchvision import models

from adv_lib.utils import ImageNormalizer, requires_grad_


class AlexNetFeatures(nn.Module):
    def __init__(self) -> None:
        super(AlexNetFeatures, self).__init__()
        self.normalize = ImageNormalizer(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        self.model = models.alexnet(pretrained=True)
        self.model.eval()

        self.features_layers = nn.ModuleList([
            self.model.features[:2],
            self.model.features[2:5],
            self.model.features[5:8],
            self.model.features[8:10],
            self.model.features[10:12],
        ])

        requires_grad_(self, False)

    def forward(self, x: Tensor) -> Tuple[Tensor, ...]:
        return self.features(x)

    def features(self, x: Tensor) -> Tuple[Tensor, ...]:
        x = self.normalize(x)

        features = [x]
        for i, layer in enumerate(self.features_layers):
            features.append(layer(features[i]))

        return tuple(features[1:])


def _normalize_features(x: Tensor, ε: float = 1e-12) -> Tensor:
    """Normalize by norm and sqrt of spatial size."""
    norm = torch.norm(x, dim=1, p=2, keepdim=True)
    return x / (norm[0].numel() ** 0.5 * norm.clamp_min(ε))


def _feature_difference(features_1: Tensor, features_2: Tensor, linear_mapping: Optional[nn.Module] = None) -> Tensor:
    features = [map(_normalize_features, feature) for feature in [features_1, features_2]]  # Normalize features
    if linear_mapping is not None:  # Perform linear scaling
        features = [[module(f) for module, f in zip(linear_mapping, feature)] for feature in features]
    features = [torch.cat([f.flatten(1) for f in feature], dim=1) for feature in features]  # Concatenate
    return features[0] - features[1]


class LPIPS(nn.Module):
    _models = {'alexnet': AlexNetFeatures}

    def __init__(self,
                 model: Union[str, nn.Module] = 'alexnet',
                 linear_mapping: Optional[str] = None,
                 target: Optional[Tensor] = None,
                 squared: bool = False) -> None:
        super(LPIPS, self).__init__()

        if isinstance(model, str):
            self.features = self._models[model]()
        else:
            self.features = model

        self.linear_mapping = None
        if linear_mapping is not None:
            convs = []
            sd = torch.load(linear_mapping)
            for k, weight in sd.items():
                out_channels, in_channels = weight.shape[:2]
                conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, bias=False)
                conv.weight.data.copy_(weight)
                convs.append(conv)
            self.linear_mapping = nn.ModuleList(convs)

        self.target_features = None
        if target is not None:
            self.to(target.device)
            self.target_features = self.features(target)

        self.squared = squared

    def forward(self, input: Tensor, target: Optional[Tensor] = None) -> Tensor:
        input_features = self.features(input)
        if target is None and self.target_features is not None:
            target_features = self.target_features
        elif target is not None:
            target_features = self.features(target)
        else:
            raise ValueError('Must provide targets (either in init or in forward).')

        if self.squared:
            return _feature_difference(input_features, target_features).square().sum(dim=1)

        return torch.norm(_feature_difference(input_features, target_features), p=2, dim=1)
