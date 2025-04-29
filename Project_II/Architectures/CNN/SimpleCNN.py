import torch
import torch.nn as nn
import torch.optim as optim
from typing import Any, Dict, List, Tuple, Optional, Type

from tqdm import tqdm
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

from ..AbstractModel import BaseModel
from DataObjects.DataLoader import DataLoader


class SimpleCNN(BaseModel, nn.Module):
    """
    SimpleCNN for spectrogram-based classification with optional custom CNN backbone.
    Inherits BaseModel and nn.Module; implements train, evaluate, predict, infer, save, load.

    Args:
        in_channels: Number of input channels (e.g. 1 for spectrograms).
        num_classes: Number of target classes.
        lr: Learning rate for optimizer.
        dropout: Dropout probability for classifier.
        custom_cnn: Optional custom nn.Module to use instead of default shallow CNN.
    """
    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        lr: float = 1e-3,
        dropout: float = 0.3,
        custom_cnn: Optional[nn.Module] = None
    ) -> None:
        nn.Module.__init__(self)
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.lr = lr
        self.num_classes = num_classes

        # CNN backbone: use custom if provided, else default shallow CNN
        if custom_cnn is not None:
            self.cnn = custom_cnn.to(self.device)
        else:
            # default two-layer conv
            self.cnn = nn.Sequential(
                nn.Conv2d(in_channels, 16, kernel_size=3, padding=1),
                nn.BatchNorm2d(16),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2, 2),
                nn.Conv2d(16, 32, kernel_size=3, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2, 2)
            ).to(self.device)

        # Adaptive pooling to fixed-size feature map
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))
        # Fully connected classifier
        self.fc1 = nn.Linear(32 * 4 * 4 if custom_cnn is None else None, 128) if custom_cnn is None else None
        # If custom_cnn, infer feature dimension at runtime in forward
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(128, num_classes)
        self.to(self.device)

    def set_custom_cnn(self, custom_cnn: nn.Module) -> None:
        """
        Replace the default CNN backbone with a custom nn.Module.
        Automatically moves it to the correct device and resets classifier input layer.
        """
        self.cnn = custom_cnn.to(self.device)
        # Reset fc1 so it will be re-initialized based on the new backbone output dims
        self.fc1 = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W)
        features = self.cnn(x)  # (B, F, h, w)
        pooled = self.adaptive_pool(features)  # (B, F, 4, 4)
        B, F, h, w = pooled.shape
        flat = pooled.view(B, F * h * w)
        # if custom_cnn was provided, and fc1 is None, initialize fc1
        if not hasattr(self, 'fc1') or self.fc1 is None:
            self.fc1 = nn.Linear(F * h * w, 128).to(self.device)
        x = torch.relu(self.fc1(flat))
        x = self.dropout(x)
        logits = self.fc2(x)
        return logits

    def train_architecture(
        self,
        train_loader: DataLoader,
        epochs: int,
        val_loader: Optional[DataLoader] = None,
        **kwargs: Any
    ) -> None:
        """
        Train the CNN on spectrogram data; optionally evaluate on val_loader each epoch.
        """
        nn.Module.train(self)
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        criterion = nn.CrossEntropyLoss()

        for epoch in range(1, epochs + 1):
            total_loss = 0.0
            total_correct = 0
            total_samples = 0
            for batch in train_loader:
                inputs = batch.data.to(self.device)
                targets = batch.labels.to(self.device)

                optimizer.zero_grad()
                outputs = self(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

                preds = outputs.argmax(1)
                total_correct += (preds == targets).sum().item()
                total_loss += loss.item() * targets.size(0)
                total_samples += targets.size(0)

            train_loss = total_loss / total_samples
            train_acc = total_correct / total_samples
            print(f"Epoch {epoch}/{epochs} - train loss: {train_loss:.4f} - train acc: {train_acc:.4f}")

            if val_loader is not None:
                res = self.evaluate(val_loader)
                s = res['summary']
                print(
                    f"Epoch {epoch}/{epochs} - val loss: {s['loss']:.4f} "
                    f"- val acc: {s['accuracy']:.4f} "
                    f"- val prec: {s['precision']:.4f} "
                    f"- val rec: {s['recall']:.4f} "
                    f"- val f1: {s['f1']:.4f}"
                )

    def evaluate(
    self,
    eval_loader: DataLoader
) -> Dict[str, Any]:

        nn.Module.eval(self)
        criterion = nn.CrossEntropyLoss()

        metrics_history: List[Dict[str, float]] = []
        preds_all: List[int] = []
        trues_all: List[int] = []
        total_loss = 0.0
        total_samples = 0

        for batch in tqdm(eval_loader, total=len(eval_loader), desc="Evaluating CNN"):
            inputs = batch.data.to(self.device)
            targets = batch.labels.to(self.device)

            with torch.no_grad():
                outputs = self(inputs)
                loss = criterion(outputs, targets).item()

            preds = outputs.argmax(1).cpu().numpy()
            trues = targets.cpu().numpy()

            batch_size = targets.size(0)
            total_loss += loss * batch_size
            total_samples += batch_size
            preds_all.extend(preds.tolist())
            trues_all.extend(trues.tolist())

            acc = (preds == trues).mean()
            pr  = precision_score(trues, preds, average='macro', zero_division=0)
            rc  = recall_score(trues, preds, average='macro', zero_division=0)
            f1  = f1_score(trues, preds, average='macro', zero_division=0)
            metrics_history.append({
                'loss': loss,
                'accuracy': acc,
                'precision': pr,
                'recall': rc,
                'f1': f1
            })


        summary = {
            'loss': total_loss / total_samples,
            'accuracy': (torch.tensor(preds_all) == torch.tensor(trues_all))
                            .float()
                            .mean()
                            .item(),
            'precision': precision_score(trues_all, preds_all, average='macro', zero_division=0),
            'recall':    recall_score(trues_all, preds_all, average='macro', zero_division=0),
            'f1':        f1_score(trues_all, preds_all, average='macro', zero_division=0)
        }

        cm = confusion_matrix(trues_all, preds_all)

        return {
            'metrics_history':    metrics_history,
            'summary':            summary,
            'confusion_matrix':   cm
        }

    def predict(
        self,
        data_loader: DataLoader
    ) -> Tuple[List[int], List[int]]:
        """Return predicted and true labels."""
        nn.Module.eval(self)
        preds_all: List[int] = []
        trues_all: List[int] = []
        with torch.no_grad():
            for batch in data_loader:
                inputs = batch.data.to(self.device)
                targets = batch.labels.to(self.device)
                outputs = self(inputs)
                preds_all.extend(outputs.argmax(1).cpu().tolist())
                trues_all.extend(targets.cpu().tolist())
        return preds_all, trues_all

    def infer(
        self,
        data_loader: DataLoader
    ) -> List[int]:
        """Return predicted labels for unlabeled data."""
        nn.Module.eval(self)
        preds_all: List[int] = []
        with torch.no_grad():
            for batch in data_loader:
                inputs = batch.data.to(self.device)
                outputs = self(inputs)
                preds_all.extend(outputs.argmax(1).cpu().tolist())
        return preds_all

    def save(self, path: str) -> None:
        """Save model weights."""
        torch.save(self.state_dict(), path)

    @classmethod
    def load(
        cls: Type['SimpleCNN'],
        path: str,
        **model_kwargs: Any
    ) -> 'SimpleCNN':
        """Load model weights into a new instance."""
        model = cls(**model_kwargs)
        state = torch.load(path, map_location=model.device)
        model.load_state_dict(state)
        model.to(model.device)
        return model
