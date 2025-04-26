import torch
import torch.nn as nn
import torch.optim as optim
from typing import Any, Dict, Tuple, List, Optional, Type

from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score
from transformers import BertModel, BertConfig
from ray.air import session

from ..AbstractModel import BaseModel
from DataObjects.DataLoader import DataLoader


class BertFineTuner(BaseModel, nn.Module):
    def __init__(
        self,
        audio_dim: int,
        num_labels: int,
        pretrained: bool = True,
        model_name: str = 'bert-base-uncased',
        lr: float = 5e-5
    ) -> None:
        nn.Module.__init__(self)
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.audio_dim = audio_dim
        self.num_labels = num_labels
        self.lr = lr

        if pretrained:
            self.config = BertConfig.from_pretrained(model_name)
            base_model = BertModel.from_pretrained(model_name, config=self.config)
        else:
            self.config = BertConfig()
            base_model = BertModel(self.config)

        self.input_proj = nn.Linear(self.audio_dim, self.config.hidden_size)
        self.bert = base_model
        self.classifier = nn.Linear(self.config.hidden_size, self.num_labels)

        self.to(self.device)

    def train_architecture(self, train_loader: DataLoader, epochs: int, val_loader: Optional[DataLoader] = None, **kwargs: Any) -> None:
        self.train()
        optimizer = optim.AdamW(self.parameters(), lr=self.lr)
        criterion = nn.CrossEntropyLoss()

        for epoch in range(1, epochs + 1):
            total_loss, total_correct, total_samples = 0.0, 0, 0
            for batch in train_loader:
                x = batch.data.to(self.device)
                y = batch.labels.to(self.device)

                if x.dim() == 3:
                    feats = x.permute(0, 2, 1)
                else:
                    B, C, H, W = x.shape
                    feats = x.reshape(B, C * H, W).permute(0, 2, 1)

                if feats.size(-1) != self.audio_dim:
                    self.audio_dim = feats.size(-1)
                    self.input_proj = nn.Linear(self.audio_dim, self.config.hidden_size).to(self.device)

                embeddings = self.input_proj(feats)
                attention_mask = torch.ones(embeddings.size()[:2], dtype=torch.long, device=self.device)

                outputs = self.bert(inputs_embeds=embeddings, attention_mask=attention_mask)
                pooled = outputs.pooler_output

                logits = self.classifier(pooled)
                loss = criterion(logits, y)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                preds = logits.argmax(dim=1)
                total_correct += (preds == y).sum().item()
                total_loss += loss.item() * y.size(0)
                total_samples += y.size(0)

            print(f"Epoch {epoch}/{epochs} - train loss: {total_loss / total_samples:.4f} - acc: {total_correct / total_samples:.4f}")

            if val_loader:
                val_metrics = self.evaluate(val_loader)["summary"]
                print(f"Validation - loss: {val_metrics['loss']:.4f} - acc: {val_metrics['accuracy']:.4f} - f1: {val_metrics['f1']:.4f}")

                # Only report to Ray Tune if it is active
                if session.get_session():
                    session.report({"accuracy": s["accuracy"]})

    def evaluate(self, eval_loader: DataLoader) -> Dict[str, Any]:
        self.eval()
        criterion = nn.CrossEntropyLoss()
        preds_all, trues_all = [], []
        total_loss, total_samples = 0.0, 0

        with torch.no_grad():
            for batch in tqdm(eval_loader, desc="Evaluating BERT"):
                x = batch.data.to(self.device)
                y = batch.labels.to(self.device)

                feats = x.permute(0, 2, 1) if x.dim() == 3 else x.reshape(x.size(0), -1, x.size(-1)).permute(0, 2, 1)
                embeddings = self.input_proj(feats)
                attention_mask = torch.ones(embeddings.size()[:2], dtype=torch.long, device=self.device)

                outputs = self.bert(inputs_embeds=embeddings, attention_mask=attention_mask)
                pooled = outputs.pooler_output
                logits = self.classifier(pooled)

                loss = criterion(logits, y).item()
                preds = logits.argmax(dim=1).cpu().numpy()
                trues = y.cpu().numpy()

                preds_all.extend(preds.tolist())
                trues_all.extend(trues.tolist())
                total_loss += loss * y.size(0)
                total_samples += y.size(0)

        summary = {
            "loss": total_loss / total_samples,
            "accuracy": (torch.tensor(preds_all) == torch.tensor(trues_all)).float().mean().item(),
            "precision": precision_score(trues_all, preds_all, average='macro', zero_division=0),
            "recall": recall_score(trues_all, preds_all, average='macro', zero_division=0),
            "f1": f1_score(trues_all, preds_all, average='macro', zero_division=0)
        }
        return {"summary": summary}

    def predict(self, data_loader: DataLoader) -> Tuple[List[int], List[int]]:
        self.eval()
        preds_all, trues_all = [], []
        with torch.no_grad():
            for batch in data_loader:
                x = batch.data.to(self.device)
                y = batch.labels.to(self.device)
                feats = x.permute(0, 2, 1) if x.dim() == 3 else x.reshape(x.size(0), -1, x.size(-1)).permute(0, 2, 1)
                embeddings = self.input_proj(feats)
                attention_mask = torch.ones(embeddings.size()[:2], dtype=torch.long, device=self.device)
                outputs = self.bert(inputs_embeds=embeddings, attention_mask=attention_mask)
                pooled = outputs.pooler_output
                logits = self.classifier(pooled)
                preds_all.extend(logits.argmax(dim=1).cpu().tolist())
                trues_all.extend(y.cpu().tolist())
        return preds_all, trues_all

    def infer(self, data_loader: DataLoader) -> List[int]:
        self.eval()
        preds_all = []
        with torch.no_grad():
            for batch in data_loader:
                x = batch.data.to(self.device)
                feats = x.permute(0, 2, 1) if x.dim() == 3 else x.reshape(x.size(0), -1, x.size(-1)).permute(0, 2, 1)
                embeddings = self.input_proj(feats)
                attention_mask = torch.ones(embeddings.size()[:2], dtype=torch.long, device=self.device)
                outputs = self.bert(inputs_embeds=embeddings, attention_mask=attention_mask)
                pooled = outputs.pooler_output
                logits = self.classifier(pooled)
                preds_all.extend(logits.argmax(dim=1).cpu().tolist())
        return preds_all

    def save(self, path: str) -> None:
        torch.save(self.state_dict(), path)

    @classmethod
    def load(cls: Type['BERTFineTuner'], path: str, **model_kwargs: Any) -> 'BERTFineTuner':
        model = cls(**model_kwargs)
        state = torch.load(path, map_location=model.device)
        model.load_state_dict(state)
        model.to(model.device)
        return model
