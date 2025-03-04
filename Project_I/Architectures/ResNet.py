import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
from typing import Type, List

class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(self, in_planes: int, planes: int, stride: int = 1) -> None:
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes * self.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes * self.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * self.expansion)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, block: Type[BasicBlock], num_blocks: List[int], num_classes: int = 10, in_channels: int = 1) -> None:
        super(ResNet, self).__init__()
        self.in_planes = 16  # starting number of filters for CIFAR-like data
        self.conv1 = nn.Conv2d(in_channels, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.linear = nn.Linear(64 * block.expansion, num_classes)

    def _make_layer(self, block: Type[BasicBlock], planes: int, num_blocks: int, stride: int) -> nn.Sequential:
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        # Global average pooling: if input is 32x32, the output after pooling is [batch, 64, 1, 1]
        out = F.avg_pool2d(out, out.size(3))
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

    def to_pickle(self, file_path: str) -> None:
        with open(file_path, 'wb') as f:
            pickle.dump(self, f)
        print(f"Model saved to {file_path}")


def ResNet20(num_classes: int = 10, in_channels: int = 1) -> ResNet:
    # For CIFAR, ResNet-20 has 3 blocks in each of the three layers (i.e., [3, 3, 3])
    return ResNet(BasicBlock, [3, 3, 3], num_classes, in_channels)

def load_from_pickle(file_path: str) -> ResNet:
    with open(file_path, 'rb') as f:
        model: ResNet = pickle.load(f)
    print(f"Model loaded from {file_path}")
    return model