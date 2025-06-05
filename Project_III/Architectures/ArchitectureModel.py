from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
from DataObjects import DataLoader
import torch


class GeneratorBase(ABC):
    def __init__(self, device: torch.device) -> None:
        self.device = device

    @abstractmethod
    def build_model(self) -> None:
        # initialize network modules
        ...

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # forward pass (training)
        ...

    @abstractmethod
    def sample_latent(self, num_samples: int) -> torch.Tensor:
        # sample latent vectors
        ...

    @abstractmethod
    def generate(self, num_samples: int, **kwargs: Any) -> torch.Tensor:
        # generate images from latent
        ...

    @abstractmethod
    def train_step(self, batch: torch.Tensor) -> Dict[str, torch.Tensor]:
        # one training iteration
        ...

    @abstractmethod
    def train_architecture(self, data:DataLoader) -> None:
        # performs training on the whole of the architecture
        ...


    @abstractmethod
    def evaluate(self, batch: torch.Tensor) -> Dict[str, torch.Tensor]:
        # evaluation metrics
        ...

    @abstractmethod
    def configure_optimizers(self) -> Any:
        # return optimizer(s) and scheduler(s)
        ...

    @abstractmethod
    def save_model(self, path: str) -> None:
        # save model state_dict
        ...

    @abstractmethod
    def load_model(self, path: str, map_location: Optional[str] = None) -> None:
        # load state_dict
        ...

    def set_device(self, device: torch.device) -> None:
        self.device = device
        self._move_to_device()

    @abstractmethod
    def _move_to_device(self) -> None:
        # move modules to self.device
        ...
