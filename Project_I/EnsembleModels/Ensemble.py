import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch import Tensor
from typing import Dict, List, Iterable, Tuple
from DataObjects import DataBatch
from typing import Optional
from tqdm import tqdm

class StackingEnsemble(nn.Module):
    def __init__(self, base_models: Dict[str, nn.Module], num_classes: int) -> None:
        super(StackingEnsemble, self).__init__()
        self.base_models: Dict[str, nn.Module] = base_models
        self.num_models: int = len(base_models)
        self.num_classes: int = num_classes
        for model in self.base_models.values():
            for param in model.parameters():
                param.requires_grad = False
        self.meta_model: nn.Sequential = nn.Sequential(
            nn.Linear(self.num_models * self.num_classes, 32),
            nn.ReLU(),
            nn.Linear(32, self.num_classes)
        )

    def forward(self, x: Tensor) -> Tensor:
        features: List[Tensor] = []
        for model in self.base_models.values():
            with torch.no_grad():
                out: Tensor = model(x)
                out = F.softmax(out, dim=1)
            features.append(out)
        meta_input: Tensor = torch.cat(features, dim=1)
        meta_output: Tensor = self.meta_model(meta_input)
        return meta_output

    def fit(self, dataloader: Iterable[DataBatch], optimizer: optim.Optimizer, criterion: nn.Module, device: torch.device) -> None:
        self.train()
        self.to(device)
        for batch in dataloader:
            inputs: Tensor = batch.data.to(device)
            labels: Tensor = batch.labels.to(device)
            features: List[Tensor] = []
            for model in self.base_models.values():
                with torch.no_grad():
                    out: Tensor = model(inputs)
                    out = F.softmax(out, dim=1)
                features.append(out)
            meta_input: Tensor = torch.cat(features, dim=1)
            outputs: Tensor = self.meta_model(meta_input)
            loss: Tensor = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    def predict(self, dataloader: Iterable[DataBatch], device: torch.device) -> List[int]:
        self.eval()
        predictions: List[int] = []
        with torch.no_grad():
            for batch in dataloader:
                inputs: Tensor = batch.data.to(device)
                features: List[Tensor] = []
                for model in self.base_models.values():
                    out: Tensor = model(inputs)
                    out = F.softmax(out, dim=1)
                    features.append(out)
                meta_input: Tensor = torch.cat(features, dim=1)
                meta_output: Tensor = self.meta_model(meta_input)
                preds: Tensor = meta_output.argmax(dim=1)
                predictions.extend(preds.cpu().tolist())
        return predictions

    def test(self, dataloader: Iterable[DataBatch], criterion: nn.Module, device: torch.device) -> Tuple[float, float]:
        self.eval()
        total_loss: float = 0.0
        total_correct: int = 0
        total_samples: int = 0
        with torch.no_grad():
            for batch in dataloader:
                inputs: Tensor = batch.data.to(device)
                labels: Tensor = batch.labels.to(device)
                features: List[Tensor] = []
                for model in self.base_models.values():
                    out: Tensor = model(inputs)
                    out = F.softmax(out, dim=1)
                    features.append(out)
                meta_input: Tensor = torch.cat(features, dim=1)
                outputs: Tensor = self.meta_model(meta_input)
                loss: Tensor = criterion(outputs, labels)
                total_loss += loss.item() * inputs.size(0)
                preds: Tensor = outputs.argmax(dim=1)
                total_correct += (preds == labels).sum().item()
                total_samples += inputs.size(0)
        avg_loss: float = total_loss / total_samples if total_samples > 0 else 0.0
        accuracy: float = total_correct / total_samples if total_samples > 0 else 0.0
        return avg_loss, accuracy
    
    def train_ensemble(self, dataloader: Iterable[DataBatch], optimizer: optim.Optimizer, criterion: nn.Module, device: torch.device, epochs: Optional[int] = 100) -> None:
        for epoch in tqdm(range(epochs)):
            self.fit(dataloader, optimizer, criterion, device)
            print(f"Epoch {epoch+1}/{epochs} completed.")

    def set_meta_model(self, meta_model: nn.Module) -> None:
        self.meta_model = meta_model

    def load_meta_weights(self, path: str, device: Optional[torch.device] = None) -> None:
        state = torch.load(path, map_location=device) if device else torch.load(path)
        self.meta_model.load_state_dict(state)

class HardVotingEnsemble:
    def __init__(self, base_models: Dict[str, nn.Module]) -> None:
        self.base_models: Dict[str, nn.Module] = base_models

    def predict(self, dataloader: Iterable[DataBatch], device: torch.device) -> List[int]:
        predictions: List[int] = []
        for batch in dataloader:
            inputs: Tensor = batch.data.to(device)
            batch_preds: List[List[int]] = []
            for model in self.base_models.values():
                with torch.no_grad():
                    out: Tensor = model(inputs)
                    pred: Tensor = out.argmax(dim=1)
                    batch_preds.append(pred.cpu().tolist())
            samples_preds = list(zip(*batch_preds))
            batch_final: List[int] = []
            for preds in samples_preds:
                vote: int = max(set(preds), key=preds.count)
                batch_final.append(vote)
            predictions.extend(batch_final)
        return predictions

    def test(self, dataloader: Iterable[DataBatch], device: torch.device) -> float:
        total_correct: int = 0
        total_samples: int = 0
        predictions: List[int] = self.predict(dataloader, device)
        true_labels: List[int] = []
        for batch in dataloader:
            true_labels.extend(batch.labels.cpu().tolist())
            total_samples += batch.labels.size(0)
        for pred, true in zip(predictions, true_labels):
            if pred == true:
                total_correct += 1
        accuracy: float = total_correct / total_samples if total_samples > 0 else 0.0
        return accuracy
    
class SoftVotingEnsemble:
    def __init__(self, base_models: Dict[str, nn.Module]) -> None:
        self.base_models: Dict[str, nn.Module] = base_models

    def predict(self, dataloader: Iterable[DataBatch], device: torch.device) -> List[int]:
        predictions: List[int] = []
        for batch in dataloader:
            inputs: Tensor = batch.data.to(device)
            sum_probs: Optional[Tensor] = None
            for model in self.base_models.values():
                with torch.no_grad():
                    out: Tensor = model(inputs)
                    probs: Tensor = F.softmax(out, dim=1)
                if sum_probs is None:
                    sum_probs = probs
                else:
                    sum_probs += probs
            avg_probs: Tensor = sum_probs / len(self.base_models)
            preds: Tensor = avg_probs.argmax(dim=1)
            predictions.extend(preds.cpu().tolist())
        return predictions

    def test(self, dataloader: Iterable[DataBatch], device: torch.device) -> float:
        total_correct: int = 0
        total_samples: int = 0
        predictions: List[int] = self.predict(dataloader, device)
        true_labels: List[int] = []
        for batch in dataloader:
            true_labels.extend(batch.labels.cpu().tolist())
            total_samples += batch.labels.size(0)
        for pred, true in zip(predictions, true_labels):
            if pred == true:
                total_correct += 1
        accuracy: float = total_correct / total_samples if total_samples > 0 else 0.0
        return accuracy
