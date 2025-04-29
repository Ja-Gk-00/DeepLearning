import torch
import torch.nn as nn
import torch.optim as optim
from typing import Any, Dict, Tuple, List, Type, Optional

from tqdm import tqdm
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

from ..AbstractModel import BaseModel
from DataObjects.DataLoader import DataLoader


class CustomTransformerModel(BaseModel, nn.Module):
    """
    A shallow Transformer encoder model for matrix-shaped audio features.
    Accepts inputs as matrices (C x T or H x W) and outputs class logits.
    Provides methods for training, evaluation, prediction, inference,
    as well as saving and loading model weights.
    """
    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 1,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        lr: float = 1e-3
    ) -> None:
        nn.Module.__init__(self)
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # input projection
        self.input_proj = nn.Linear(input_dim, d_model) if input_dim != d_model else nn.Identity()
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=False
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        # classification head
        self.classifier = nn.Linear(d_model, num_classes)

        self.lr = lr
        self.to(self.device)

    def train_architecture(
        self,
        train_loader: DataLoader,
        epochs: int,
        val_loader: Optional[DataLoader] = None,
        **kwargs: Any
    ) -> None:
        """
        Train the transformer model, tracking loss and accuracy each epoch.
        Optionally evaluate on a validation loader after each epoch.
        """
        # set module to training mode
        nn.Module.train(self)
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        criterion = nn.CrossEntropyLoss()

        for epoch in range(1, epochs + 1):
            total_loss = 0.0
            total_samples = 0
            total_correct = 0
            for batch in train_loader:
                x = batch.data.to(self.device)
                y = batch.labels.to(self.device)
                x_seq = self._prepare_sequence(x)
                x_emb = self.input_proj(x_seq)
                z = self.transformer(x_emb)
                z = z.mean(dim=0)
                logits = self.classifier(z)
                loss = criterion(logits, y)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                preds = logits.argmax(dim=1)
                total_correct += (preds == y).sum().item()
                batch_size = y.size(0)
                total_loss += loss.item() * batch_size
                total_samples += batch_size

            avg_loss = total_loss / total_samples
            accuracy = total_correct / total_samples
            print(f"Epoch {epoch}/{epochs} - train loss: {avg_loss:.4f} - train accuracy: {accuracy:.4f}")

            if val_loader is not None:
                # evaluate on validation set
                val_results = self.evaluate(val_loader)
                val_loss = val_results['summary']['loss']
                val_acc = val_results['summary']['accuracy']
                print(f"Epoch {epoch}/{epochs} - val   loss: {val_loss:.4f} - val   accuracy: {val_acc:.4f}")

    def evaluate(
    self,
    eval_loader: DataLoader
) -> Dict[str, Any]:
        criterion = nn.CrossEntropyLoss()
        self.train(False)

        metrics_history: List[Dict[str, float]] = []
        preds_all: List[int] = []
        trues_all: List[int] = []
        total_loss = 0.0
        total_samples = 0

        for batch in tqdm(eval_loader, total=len(eval_loader), desc="Evaluating"):
            x = batch.data.to(self.device)
            y = batch.labels.to(self.device)

            # forward
            x_seq = self._prepare_sequence(x)
            x_emb = self.input_proj(x_seq)
            z     = self.transformer(x_emb)
            z     = z.mean(dim=0)
            logits = self.classifier(z)

            # loss & preds
            loss_val = criterion(logits, y).item()
            preds    = logits.argmax(dim=1)

            # accumulate
            batch_size   = y.size(0)
            total_loss  += loss_val * batch_size
            total_samples += batch_size

            preds_np = preds.cpu().numpy()
            y_np     = y.cpu().numpy()
            preds_all.extend(preds_np.tolist())
            trues_all.extend(y_np.tolist())

            # per‐batch metrics
            acc = (preds_np == y_np).mean()
            pr  = precision_score(y_np, preds_np, average='macro', zero_division=0)
            rc  = recall_score(y_np, preds_np, average='macro', zero_division=0)
            f1  = f1_score(y_np, preds_np, average='macro', zero_division=0)
            metrics_history.append({
                'loss': loss_val,
                'accuracy': acc,
                'precision': pr,
                'recall': rc,
                'f1': f1
            })

        # overall summary
        avg_loss  = total_loss / total_samples
        accuracy  = (torch.tensor(preds_all) == torch.tensor(trues_all)).float().mean().item()
        precision = precision_score(trues_all, preds_all, average='macro', zero_division=0)
        recall    = recall_score(trues_all, preds_all, average='macro', zero_division=0)
        f1_score_ = f1_score(trues_all, preds_all, average='macro', zero_division=0)
        summary = {
            'loss':      avg_loss,
            'accuracy':  accuracy,
            'precision': precision,
            'recall':    recall,
            'f1':        f1_score_
        }

        cm = confusion_matrix(trues_all, preds_all)

        return {
            'metrics_history':  metrics_history,
            'summary':          summary,
            'confusion_matrix': cm
        }

    def predict(
        self,
        data_loader: DataLoader
    ) -> Tuple[List[int], List[int]]:
        self.train(False)
        preds_all: List[int] = []
        trues_all: List[int] = []
        with torch.no_grad():
            for batch in data_loader:
                x = batch.data.to(self.device)
                y = batch.labels.to(self.device)
                x_seq = self._prepare_sequence(x)
                x_emb = self.input_proj(x_seq)
                z = self.transformer(x_emb)
                z = z.mean(dim=0)
                logits = self.classifier(z)
                preds = logits.argmax(dim=1).cpu().tolist()
                trues = y.cpu().tolist()
                preds_all.extend(preds)
                trues_all.extend(trues)
        return preds_all, trues_all

    def infer(
        self,
        data_loader: DataLoader
    ) -> List[int]:
        self.train(False)
        preds_all: List[int] = []
        with torch.no_grad():
            for batch in data_loader:
                x = batch.data.to(self.device)
                x_seq = self._prepare_sequence(x)
                x_emb = self.input_proj(x_seq)
                z = self.transformer(x_emb)
                z = z.mean(dim=0)
                logits = self.classifier(z)
                preds = logits.argmax(dim=1).cpu().tolist()
                preds_all.extend(preds)
        return preds_all

    def save(self, path: str) -> None:
        """
        Save the model state_dict to the given file path.
        """
        torch.save(self.state_dict(), path)

    @classmethod
    def load(
        cls: Type['CustomTransformerModel'],
        path: str,
        **model_kwargs: Any
    ) -> 'CustomTransformerModel':
        """
        Load a model from the given state_dict file.

        Args:
            path: file path of the saved state_dict
            model_kwargs: kwargs to pass to the constructor (input_dim, num_classes, etc.)
        Returns:
            An instance of CustomTransformerModel with weights loaded.
        """
        model = cls(**model_kwargs)
        state = torch.load(path, map_location=model.device)
        model.load_state_dict(state)
        model.to(model.device)
        return model

    def _prepare_sequence(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            B, C, T = x.shape
            return x.permute(2, 0, 1)
        elif x.dim() == 4:
            B, C, H, W = x.shape
            x_flat = x.reshape(B, C * H, W)
            return x_flat.permute(2, 0, 1)
        else:
            raise ValueError(f"Unsupported input dim {x.dim()}")
