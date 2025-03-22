import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import List, Optional

class _DenseLayer(nn.Module):
    def __init__(self, num_input_features: int, growth_rate: int, bn_size: int, drop_rate: float) -> None:
        super(_DenseLayer, self).__init__()
        self.norm1: nn.BatchNorm2d = nn.BatchNorm2d(num_input_features)
        self.relu1: nn.ReLU = nn.ReLU(inplace=True)
        self.conv1: nn.Conv2d = nn.Conv2d(num_input_features, bn_size * growth_rate, kernel_size=1, stride=1, bias=False)
        self.norm2: nn.BatchNorm2d = nn.BatchNorm2d(bn_size * growth_rate)
        self.relu2: nn.ReLU = nn.ReLU(inplace=True)
        self.conv2: nn.Conv2d = nn.Conv2d(bn_size * growth_rate, growth_rate, kernel_size=3, stride=1, padding=1, bias=False)
        self.drop_rate: float = drop_rate

    def forward(self, x: Tensor) -> Tensor:
        new_features: Tensor = self.conv1(self.relu1(self.norm1(x)))
        new_features = self.conv2(self.relu2(self.norm2(new_features)))
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
        return torch.cat([x, new_features], 1)

class _DenseBlock(nn.Module):
    def __init__(self, num_layers: int, num_input_features: int, growth_rate: int, bn_size: int, drop_rate: float) -> None:
        super(_DenseBlock, self).__init__()
        layers: List[nn.Module] = []
        for i in range(num_layers):
            layer: nn.Module = _DenseLayer(num_input_features + i * growth_rate, growth_rate, bn_size, drop_rate)
            layers.append(layer)
        self.layers: nn.Sequential = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.layers(x)

class _Transition(nn.Module):
    def __init__(self, num_input_features: int, num_output_features: int) -> None:
        super(_Transition, self).__init__()
        self.norm: nn.BatchNorm2d = nn.BatchNorm2d(num_input_features)
        self.relu: nn.ReLU = nn.ReLU(inplace=True)
        self.conv: nn.Conv2d = nn.Conv2d(num_input_features, num_output_features, kernel_size=1, stride=1, bias=False)
        self.pool: nn.AvgPool2d = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(self.relu(self.norm(x)))
        x = self.pool(x)
        return x

class DenseNet(nn.Module):
    def __init__(self, growth_rate: int = 12, block_config: List[int] = [16, 16, 16], num_init_features: int = 24, bn_size: int = 4, drop_rate: float = 0.0, num_classes: int = 10) -> None:
        super(DenseNet, self).__init__()
        self.features: nn.Sequential = nn.Sequential(
            nn.Conv2d(3, num_init_features, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(num_init_features),
            nn.ReLU(inplace=True)
        )
        num_features: int = num_init_features
        blocks: List[nn.Module] = []
        for i, num_layers in enumerate(block_config):
            block: nn.Module = _DenseBlock(num_layers=num_layers, num_input_features=num_features, growth_rate=growth_rate, bn_size=bn_size, drop_rate=drop_rate)
            blocks.append(block)
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                trans: nn.Module = _Transition(num_input_features=num_features, num_output_features=num_features // 2)
                blocks.append(trans)
                num_features = num_features // 2
        self.features.add_module("denseblocks", nn.Sequential(*blocks))
        self.features.add_module("norm_final", nn.BatchNorm2d(num_features))
        self.classifier: nn.Linear = nn.Linear(num_features, num_classes)

    def forward(self, x: Tensor) -> Tensor:
        features: Tensor = self.features(x)
        out: Tensor = F.relu(features, inplace=True)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = torch.flatten(out, 1)
        out = self.classifier(out)
        return out

class DenseNetCreator:
    def __init__(self, growth_rate: int = 12, block_config: Optional[List[int]] = None, num_init_features: int = 24, bn_size: int = 4, drop_rate: float = 0.0, num_classes: int = 10) -> None:
        if block_config is None:
            block_config = [16, 16, 16]
        self.growth_rate: int = growth_rate
        self.block_config: List[int] = block_config
        self.num_init_features: int = num_init_features
        self.bn_size: int = bn_size
        self.drop_rate: float = drop_rate
        self.num_classes: int = num_classes

    def create(self) -> DenseNet:
        return DenseNet(growth_rate=self.growth_rate, block_config=self.block_config, num_init_features=self.num_init_features, bn_size=self.bn_size, drop_rate=self.drop_rate, num_classes=self.num_classes)
