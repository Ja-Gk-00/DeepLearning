import os
import pickle
from typing import Any, Dict, List, Tuple, Optional, Type

import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from ..AbstractModel import BaseModel
from DataObjects.DataLoader import DataLoader


class GMMClassifier(BaseModel):
    """
    Gaussian Mixture Model classifier for FFT-based features.
    Trains one GMM per class on flattened FFT features.
    """
    def __init__(
        self,
        n_components: int = 4,
        covariance_type: str = 'full',
        max_iter: int = 100,
        random_state: Optional[int] = None
    ) -> None:
        """
        Args:
            n_components: number of mixture components per class GMM
            covariance_type: 'full', 'tied', 'diag', or 'spherical'
            max_iter: EM iterations
            random_state: random seed
        """
        self.n_components = n_components
        self.covariance_type = covariance_type
        self.max_iter = max_iter
        self.random_state = random_state
        # Will hold class_label -> GMM model
        self.models: Dict[int, GaussianMixture] = {}
        # Label mapping from DataLoader
        self.class_to_idx: Dict[str, int] = {}

    def train_architecture(
        self,
        train_loader: DataLoader,
        epochs: int = 1,
        val_loader: Optional[DataLoader] = None,
        **kwargs: Any
    ) -> None:
        # store mapping
        self.class_to_idx = train_loader.class_to_idx
        # collect data by class
        features_by_class: Dict[int, List[np.ndarray]] = {}
        for batch in train_loader:
            arrs = batch.data.numpy()
            labels = batch.labels.numpy()
            for x, y in zip(arrs, labels):
                vec = x.flatten()
                features_by_class.setdefault(y, []).append(vec)
        # fit a GMM for each class
        for cls, feats in features_by_class.items():
            X = np.stack(feats, axis=0)
            gmm = GaussianMixture(
                n_components=self.n_components,
                covariance_type=self.covariance_type,
                max_iter=self.max_iter,
                random_state=self.random_state
            )
            gmm.fit(X)
            self.models[cls] = gmm
        print(f"Trained GMMs for classes: {list(self.models.keys())}")

        # validation check
        if val_loader is not None:
            results = self.evaluate(val_loader)
            summary = results['summary']
            print(
                f"Validation - loss: {summary['loss']:.4f} "
                f"accuracy: {summary['accuracy']:.4f} "
                f"precision: {summary['precision']:.4f} "
                f"recall: {summary['recall']:.4f} "
                f"f1: {summary['f1']:.4f}"
            )

    def evaluate(
        self,
        eval_loader: DataLoader
    ) -> Dict[str, Any]:

        preds_all: List[int] = []
        trues_all: List[int] = []
        metrics_history: List[Dict[str, float]] = []

        for batch in eval_loader:
            arrs = batch.data.numpy()
            labels = batch.labels.numpy()
            batch_preds = []
            for x, y in zip(arrs, labels):
                vec = x.flatten()[None, :]
                # compute log-likelihood for each class GMM
                scores = {cls: model.score(vec) for cls, model in self.models.items()}
                # choose class with highest likelihood
                pred = max(scores, key=scores.get)
                batch_preds.append(pred)
            acc = accuracy_score(labels, batch_preds)
            pr = precision_score(labels, batch_preds, average='macro', zero_division=0)
            rc = recall_score(labels, batch_preds, average='macro', zero_division=0)
            f1 = f1_score(labels, batch_preds, average='macro', zero_division=0)
            batch_loss = -np.mean([self.models[p].score(x.flatten()[None, :]) for p, x in zip(batch_preds, arrs)])
            metrics_history.append({
                'loss': batch_loss,
                'accuracy': acc,
                'precision': pr,
                'recall': rc,
                'f1': f1
            })
            preds_all.extend(batch_preds)
            trues_all.extend(labels.tolist())
        # summary
        summary = {
            'loss': np.mean([m['loss'] for m in metrics_history]),
            'accuracy': accuracy_score(trues_all, preds_all),
            'precision': precision_score(trues_all, preds_all, average='macro', zero_division=0),
            'recall': recall_score(trues_all, preds_all, average='macro', zero_division=0),
            'f1': f1_score(trues_all, preds_all, average='macro', zero_division=0)
        }
        return {'metrics_history': metrics_history, 'summary': summary}

    def predict(
        self,
        data_loader: DataLoader
    ) -> Tuple[List[int], List[int]]:
        
        preds_all: List[int] = []
        trues_all: List[int] = []
        for batch in data_loader:
            arrs = batch.data.numpy()
            labels = batch.labels.numpy()
            for x, y in zip(arrs, labels):
                vec = x.flatten()[None, :]
                scores = {cls: model.score(vec) for cls, model in self.models.items()}
                pred = max(scores, key=scores.get)
                preds_all.append(pred)
                trues_all.append(int(y))
        return preds_all, trues_all

    def infer(
        self,
        data_loader: DataLoader
    ) -> List[int]:

        preds_all: List[int] = []
        for batch in data_loader:
            arrs = batch.data.numpy()
            for x in arrs:
                vec = x.flatten()[None, :]
                scores = {cls: model.score(vec) for cls, model in self.models.items()}
                pred = max(scores, key=scores.get)
                preds_all.append(pred)
        return preds_all

    def save(self, path: str) -> None:

        with open(path, 'wb') as f:
            pickle.dump({
                'models': self.models,
                'class_to_idx': self.class_to_idx
            }, f)

    @classmethod
    def load(
        cls: Type['GMMClassifier'],
        path: str,
        **kwargs: Any
    ) -> 'GMMClassifier':

        with open(path, 'rb') as f:
            data = pickle.load(f)
        obj = cls(**kwargs)
        obj.models = data['models']
        obj.class_to_idx = data['class_to_idx']
        return obj
