import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch import Tensor
from typing import Dict, List, Iterable, Tuple
from DataObjects import DataBatch

# -------------------- Stacking Ensemble -------------------- #
class StackingEnsemble(nn.Module):
    def __init__(self, base_models: Dict[str, nn.Module], num_classes: int) -> None:
        super(StackingEnsemble, self).__init__()
        self.base_models: Dict[str, nn.Module] = base_models
        self.num_models: int = len(base_models)
        self.num_classes: int = num_classes

        # Freeze parameters of base models.
        for model in self.base_models.values():
            for param in model.parameters():
                param.requires_grad = False

        # Meta-model: a simple neural network that takes concatenated softmax outputs.
        self.meta_model: nn.Sequential = nn.Sequential(
            nn.Linear(self.num_models * self.num_classes, 32),
            nn.ReLU(),
            nn.Linear(32, self.num_classes)
        )

    def forward(self, x: Tensor) -> Tensor:
        features: List[Tensor] = []
        for model in self.base_models.values():
            with torch.no_grad():
                out: Tensor = model(x)  # [batch_size, num_classes]
                out = F.softmax(out, dim=1)
            features.append(out)
        meta_input: Tensor = torch.cat(features, dim=1)  # [batch_size, num_models * num_classes]
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

# -------------------- Hard Voting Ensemble -------------------- #
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
            # Group predictions per sample.
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
        for batch in dataloader:
            labels: Tensor = batch.labels
            total_samples += labels.size(0)
        # Flatten true labels from the DataLoader.
        true_labels: List[int] = []
        for batch in dataloader:
            true_labels.extend(batch.labels.cpu().tolist())
        # Calculate accuracy.
        for pred, true in zip(predictions, true_labels):
            if pred == true:
                total_correct += 1
        accuracy: float = total_correct / total_samples if total_samples > 0 else 0.0
        return accuracy

# -------------------- Example Usage -------------------- #
if __name__ == "__main__":
    import os
    from torch.utils.data import Dataset, DataLoader as TorchDataLoader
    from typing import Tuple

    # Dummy dataset that returns (data, label) tuples.
    class DummyDataset(Dataset):
        def __init__(self, num_samples: int, num_classes: int) -> None:
            self.num_samples: int = num_samples
            self.num_classes: int = num_classes

        def __len__(self) -> int:
            return self.num_samples

        def __getitem__(self, idx: int) -> Tuple[Tensor, int]:
            data: Tensor = torch.randn(3, 32, 32)
            label: int = torch.randint(0, self.num_classes, (1,)).item()
            return data, label

    # Collate function that packs a batch into a DataBatch.
    def collate_fn(batch: List[Tuple[Tensor, int]]) -> DataBatch:
        data_list, label_list = zip(*batch)
        data: Tensor = torch.stack(data_list)
        labels: Tensor = torch.tensor(label_list)
        return DataBatch(data, labels)

    num_classes: int = 10
    batch_size: int = 16
    train_dataset = DummyDataset(100, num_classes)
    test_dataset = DummyDataset(30, num_classes)
    # Here, we assume you use your custom DataLoader that yields DataBatch objects.
    train_loader = TorchDataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    test_loader = TorchDataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    # Dummy base models (replace with your actual pre-trained models).
    class DummyModel(nn.Module):
        def __init__(self, num_classes: int) -> None:
            super(DummyModel, self).__init__()
            self.num_classes: int = num_classes

        def forward(self, x: Tensor) -> Tensor:
            batch_size: int = x.size(0)
            return torch.randn(batch_size, self.num_classes, device=x.device)

    base_models: Dict[str, nn.Module] = {
        "model_1": DummyModel(num_classes),
        "model_2": DummyModel(num_classes)
    }

    # Instantiate ensemble objects.
    stacking_ensemble = StackingEnsemble(base_models, num_classes)
    hard_voting_ensemble = HardVotingEnsemble(base_models)

    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    stacking_ensemble.to(device)

    # Train the stacking ensemble meta-model.
    optimizer: optim.Optimizer = optim.Adam(stacking_ensemble.meta_model.parameters(), lr=0.001)
    criterion: nn.Module = nn.CrossEntropyLoss()
    print("Training stacking ensemble meta-model...")
    stacking_ensemble.fit(train_loader, optimizer, criterion, device)

    # Evaluate stacking ensemble.
    print("Evaluating stacking ensemble...")
    stacking_preds: List[int] = stacking_ensemble.predict(test_loader, device)
    test_loss, test_acc = stacking_ensemble.test(test_loader, criterion, device)
    print("Stacking Ensemble Predictions:", stacking_preds)
    print("Stacking Ensemble Test Loss:", test_loss)
    print("Stacking Ensemble Test Accuracy:", test_acc)

    # Evaluate hard voting ensemble.
    print("Evaluating hard voting ensemble...")
    hard_voting_preds: List[int] = hard_voting_ensemble.predict(test_loader, device)
    hv_test_acc: float = hard_voting_ensemble.test(test_loader, device)
    print("Hard Voting Ensemble Predictions:", hard_voting_preds)
    print("Hard Voting Ensemble Test Accuracy:", hv_test_acc)
