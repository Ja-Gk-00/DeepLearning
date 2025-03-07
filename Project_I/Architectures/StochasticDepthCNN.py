import torch
import torch.nn as nn
import pickle


class StochasticDepth(nn.Module):
    def __init__(
        self,
        module: nn.Module,
        p: float = 0.5,
        projection: nn.Module = None,
    ):
        super().__init__()
        if not 0 < p < 1:
            raise ValueError("Stochastic Deepth p must be between 0 and 1 but got {}".format(p))
        self.module = module
        self.p = p
        self.projection = projection
        self._sampler = torch.Tensor(1)

    def forward(self, inputs):
        if self.training and self._sampler.uniform_() < self.p:
            if self.projection is not None:
                return self.projection(inputs)
            return inputs
        return self.module(inputs)


class StochasticDepthCNN(nn.Module):
    def __init__(self, num_classes: int = 10, in_channels: int = 1) -> None:
        super(StochasticDepthCNN, self).__init__()

        # Block 1: No Stochastic Depth
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # Block 2: Apply Stochastic Depth (p=0.2)
        self.block2 = StochasticDepth(
            nn.Sequential(
                nn.Conv2d(64, 128, kernel_size=3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 128, kernel_size=3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2)
            ), p=0.2,
            projection=nn.Conv2d(64, 128, kernel_size=2, stride=2, padding=1),
        )

        # Block 3: Apply Stochastic Depth (p=0.5)
        self.block3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(256 * 4 * 4, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def to_pickle(self, file_path: str) -> None:
        with open(file_path, 'wb') as f:
            pickle.dump(self, f)
        print(f"Model saved to {file_path}")