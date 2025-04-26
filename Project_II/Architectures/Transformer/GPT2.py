import torch
import torch.nn as nn
import torch.optim as optim
from typing import Any, Dict, Tuple, List, Optional, Type

from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score
from transformers import GPT2Config, GPT2Model
from ray.air import session

from ..AbstractModel import BaseModel
from DataObjects.DataLoader import DataLoader


class GPT2FineTuner(BaseModel, nn.Module):
    """
    GPT-2 adapted for audio features: projects continuous audio sequences
    into GPT-2 hidden representations via a linear layer and performs classification.
    """
    def __init__(
        self,
        audio_dim: int,
        num_labels: int,
        pretrained: bool = True,
        model_name: str = 'gpt2',
        lr: float = 5e-5
    ) -> None:
        # Initialize base classes
        nn.Module.__init__(self)
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.audio_dim = audio_dim
        self.num_labels = num_labels
        self.lr = lr

        # Load GPT-2 config and model
        if pretrained:
            self.config = GPT2Config.from_pretrained(model_name)
            base_model = GPT2Model.from_pretrained(model_name, config=self.config)
        else:
            self.config = GPT2Config()
            base_model = GPT2Model(self.config)
        # Ensure pad token id is set
        self.config.pad_token_id = self.config.eos_token_id

        # Project audio features to GPT-2 embedding dimension
        self.input_proj = nn.Linear(self.audio_dim, self.config.hidden_size)
        self.gpt2 = base_model

        # Classification head maps GPT-2 hidden state to labels
        self.classifier = nn.Linear(self.config.hidden_size, self.num_labels)

        # Move all to device
        self.to(self.device)

    def train_architecture(
        self,
        train_loader: DataLoader,
        epochs: int,
        val_loader: Optional[DataLoader] = None,
        **kwargs: Any
    ) -> None:
        """
        Train the GPT-2-based classifier. Optionally evaluate on validation set.
        """
        # set model to train mode
        nn.Module.train(self)
        optimizer = optim.AdamW(self.parameters(), lr=self.lr)
        criterion = nn.CrossEntropyLoss()

        for epoch in range(1, epochs + 1):
            total_loss = 0.0
            total_samples = 0
            total_correct = 0
            for batch in train_loader:
                x = batch.data.to(self.device)       # shape (B, C, T) or (B, C, H, W)
                y = batch.labels.to(self.device)

                # reshape to (B, seq_len, audio_dim)
                if x.dim() == 3:
                    B, C, T = x.shape
                    feats = x.permute(0, 2, 1)
                else:
                    B, C, H, W = x.shape
                    feats = x.reshape(B, C * H, W).permute(0, 2, 1)

                # project to GPT-2 hidden size
                # ensure input_proj matches current feature dimension
                feat_dim = feats.size(-1)
                if feat_dim != self.audio_dim:
                    # update audio_dim and recreate projection layer
                    self.audio_dim = feat_dim
                    self.input_proj = nn.Linear(self.audio_dim, self.config.hidden_size).to(self.device)
                embeddings = self.input_proj(feats)  # (B, seq_len, hidden)  # (B, seq_len, hidden)

                # project to GPT-2 hidden size
                embeddings = self.input_proj(feats)  # (B, seq_len, hidden)

                # attention mask (no padding here)
                attention_mask = torch.ones(
                    embeddings.size()[:2], dtype=torch.long, device=self.device
                )

                # forward through GPT-2 with embeddings
                outputs = self.gpt2(
                    inputs_embeds=embeddings,
                    attention_mask=attention_mask
                )
                # take last token's representation
                pooled = outputs.last_hidden_state[:, -1, :]

                # classification
                logits = self.classifier(pooled)    # (B, num_labels)
                loss = criterion(logits, y)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                preds = logits.argmax(dim=1)
                total_correct += (preds == y).sum().item()
                batch_size = y.size(0)
                total_loss += loss.item() * batch_size
                total_samples += batch_size

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

                # Only report to Ray Tune if it is active
                if session.get_session():
                    session.report({"accuracy": s["accuracy"]})

    def evaluate(
        self,
        eval_loader: DataLoader
    ) -> Dict[str, Any]:
        """
        Evaluate model, returning per-batch metrics and summary.
        """
        # set model to eval mode
        nn.Module.eval(self)
        criterion = nn.CrossEntropyLoss()
        metrics_history: List[Dict[str, float]] = []
        preds_all: List[int] = []
        trues_all: List[int] = []
        total_loss = 0.0
        total_samples = 0

        for batch in tqdm(eval_loader, total=len(eval_loader), desc="Evaluating GPT2"):
            x = batch.data.to(self.device)
            y = batch.labels.to(self.device)
            if x.dim() == 3:
                B, C, T = x.shape
                feats = x.permute(0, 2, 1)
            else:
                B, C, H, W = x.shape
                feats = x.reshape(B, C * H, W).permute(0, 2, 1)
            embeddings = self.input_proj(feats)
            attention_mask = torch.ones(
                embeddings.size()[:2], dtype=torch.long, device=self.device
            )
            with torch.no_grad():
                outputs = self.gpt2(
                    inputs_embeds=embeddings,
                    attention_mask=attention_mask
                )
                pooled = outputs.last_hidden_state[:, -1, :]
                logits = self.classifier(pooled)

            loss = criterion(logits, y).item()
            preds = logits.argmax(dim=1).cpu().numpy()
            trues = y.cpu().numpy()

            batch_size = y.size(0)
            total_loss += loss * batch_size
            total_samples += batch_size
            preds_all.extend(preds.tolist())
            trues_all.extend(trues.tolist())

            acc = (preds == trues).mean()
            pr = precision_score(trues, preds, average='macro', zero_division=0)
            rc = recall_score(trues, preds, average='macro', zero_division=0)
            f1 = f1_score(trues, preds, average='macro', zero_division=0)
            metrics_history.append({'loss': loss, 'accuracy': acc, 'precision': pr, 'recall': rc, 'f1': f1})

        summary = {
            'loss': total_loss / total_samples,
            'accuracy': (torch.tensor(preds_all) == torch.tensor(trues_all)).float().mean().item(),
            'precision': precision_score(trues_all, preds_all, average='macro', zero_division=0),
            'recall': recall_score(trues_all, preds_all, average='macro', zero_division=0),
            'f1': f1_score(trues_all, preds_all, average='macro', zero_division=0)
        }
        return {'metrics_history': metrics_history, 'summary': summary}

    def predict(
        self,
        data_loader: DataLoader
    ) -> Tuple[List[int], List[int]]:
        """Return predictions and ground truths."""
        nn.Module.eval(self)
        preds_all: List[int] = []
        trues_all: List[int] = []
        with torch.no_grad():
            for batch in data_loader:
                x = batch.data.to(self.device)
                y = batch.labels.to(self.device)
                if x.dim() == 3:
                    B, C, T = x.shape
                    feats = x.permute(0, 2, 1)
                else:
                    B, C, H, W = x.shape
                    feats = x.reshape(B, C * H, W).permute(0, 2, 1)
                embeddings = self.input_proj(feats)
                attention_mask = torch.ones(
                    embeddings.size()[:2], dtype=torch.long, device=self.device
                )
                outputs = self.gpt2(
                    inputs_embeds=embeddings,
                    attention_mask=attention_mask
                )
                pooled = outputs.last_hidden_state[:, -1, :]
                logits = self.classifier(pooled)
                preds_all.extend(logits.argmax(dim=1).cpu().tolist())
                trues_all.extend(y.cpu().tolist())
        return preds_all, trues_all

    def infer(
        self,
        data_loader: DataLoader
    ) -> List[int]:
        """Return predictions for unlabeled data."""
        nn.Module.eval(self)
        preds_all: List[int] = []
        with torch.no_grad():
            for batch in data_loader:
                x = batch.data.to(self.device)
                if x.dim() == 3:
                    B, C, T = x.shape
                    feats = x.permute(0, 2, 1)
                else:
                    B, C, H, W = x.shape
                    feats = x.reshape(B, C * H, W).permute(0, 2, 1)
                embeddings = self.input_proj(feats)
                attention_mask = torch.ones(
                    embeddings.size()[:2], dtype=torch.long, device=self.device
                )
                outputs = self.gpt2(
                    inputs_embeds=embeddings,
                    attention_mask=attention_mask
                )
                pooled = outputs.last_hidden_state[:, -1, :]
                logits = self.classifier(pooled)
                preds_all.extend(logits.argmax(dim=1).cpu().tolist())
        return preds_all

    def save(self, path: str) -> None:
        """Save the model weights."""
        torch.save(self.state_dict(), path)

    @classmethod
    def load(
        cls: Type['GPT2FineTuner'],
        path: str,
        **model_kwargs: Any
    ) -> 'GPT2FineTuner':
        """Load model weights into a new instance."""
        model = cls(**model_kwargs)
        state = torch.load(path, map_location=model.device)
        model.load_state_dict(state)
        model.to(model.device)
        return model