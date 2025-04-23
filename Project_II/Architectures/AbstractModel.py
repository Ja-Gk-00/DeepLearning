from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple

from DataObjects.DataLoader import DataLoader
from DataObjects.DataLoader import DataBatch  # For typing


class BaseModel(ABC):

    @abstractmethod
    def train_architecture(self,
              train_loader: DataLoader,
              epochs: int,
              **kwargs: Any) -> None:
        pass

    @abstractmethod
    def evaluate(self,
                 eval_loader: DataLoader) -> Dict[str, float]:
        pass

    @abstractmethod
    def predict(self,
                data_loader: DataLoader) -> Tuple[Any, Any]:
        pass

    @abstractmethod
    def infer(self,
              data_loader: DataLoader) -> Any:
        pass
