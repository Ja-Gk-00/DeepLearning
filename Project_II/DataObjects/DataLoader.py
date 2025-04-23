import os
from typing import List, Tuple, Iterator, Optional

import numpy as np
import torch
import librosa


def default_loader(path: str, data_type: str, sr: int):
    """
    Load data based on data_type:
      - 'raw': load waveform via librosa
      - 'spectrogram', 'mfcc': load .npy
      - 'fft', 'moving', 'scattering': load .npz
    Returns a Tensor-ready numpy array.
    """
    ext = os.path.splitext(path)[1].lower()
    if data_type == 'raw':
        y, _ = librosa.load(path, sr=sr)
        return y.astype(np.float32)
    elif data_type in ('spectrogram', 'mfcc'):
        return np.load(path)
    elif data_type in ('fft', 'moving', 'scattering'):
        npz = np.load(path)
        if len(npz.files) == 1:
            return npz[npz.files[0]]
        return np.stack([npz[key] for key in npz.files], axis=0)
    else:
        raise ValueError(f"Unsupported data_type '{data_type}'")


class DataBatch:
    def __init__(self, data: torch.Tensor, labels: torch.Tensor) -> None:
        self.data = data
        self.labels = labels


class DataLoader:
    def __init__(
        self,
        data_dir: str,
        data_type: str,
        batch_size: int = 32,
        shuffle: bool = True,
        sr: int = 16000,
        max_per_class: Optional[int] = None,
        subset: Optional[List[str]] = None
    ) -> None:
        """
        DataLoader for various precomputed audio features, with optional label filtering.

        Args:
            data_dir: root directory with subfolders per class
            data_type: one of ['raw','spectrogram','mfcc','fft','moving','scattering']
            batch_size: samples per batch
            shuffle: whether to shuffle at each epoch
            sr: sample rate for 'raw' type
            max_per_class: limit number of samples per class
            subset: list of class names to include (default None = all)
        """
        self.data_dir = data_dir
        self.data_type = data_type
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.sr = sr
        self.max_per_class = max_per_class
        self.subset = subset
        self.samples: List[Tuple[str, int]] = []
        self.class_to_idx = {}
        self._prepare_dataset()

    def _prepare_dataset(self) -> None:
        """
        Scan subdirectories, mapping class names to indices and
        collecting file paths matching the data_type and optional subset.
        """
        # identify all classes
        all_classes = sorted([
            d for d in os.listdir(self.data_dir)
            if os.path.isdir(os.path.join(self.data_dir, d))
        ])
        # filter by subset if provided
        if self.subset:
            classes = [c for c in all_classes if c in self.subset]
        else:
            classes = all_classes
        self.class_to_idx = {cls: idx for idx, cls in enumerate(classes)}

        # determine valid extensions
        if self.data_type == 'raw':
            exts = ('.wav',)
        elif self.data_type in ('spectrogram', 'mfcc'):
            exts = ('.npy',)
        elif self.data_type in ('fft', 'moving', 'scattering'):
            exts = ('.npz',)
        else:
            raise ValueError(f"Unsupported data_type '{self.data_type}'")

        # collect samples
        for cls in classes:
            cls_idx = self.class_to_idx[cls]
            cls_dir = os.path.join(self.data_dir, cls)
            count = 0
            for fname in sorted(os.listdir(cls_dir)):
                if not fname.lower().endswith(exts):
                    continue
                if self.max_per_class and count >= self.max_per_class:
                    break
                path = os.path.join(cls_dir, fname)
                self.samples.append((path, cls_idx))
                count += 1

    def __iter__(self) -> Iterator[DataBatch]:
        """
        Yield batches of DataBatch(data, labels).
        """
        if self.shuffle:
            np.random.shuffle(self.samples)

        for start in range(0, len(self.samples), self.batch_size):
            batch = self.samples[start:start + self.batch_size]
            data_list = []
            labels_list = []
            for path, label in batch:
                try:
                    arr = default_loader(path, self.data_type, self.sr)
                except Exception as e:
                    print(f"Error loading {path}: {e}")
                    continue
                tensor = torch.tensor(arr, dtype=torch.float32)
                if tensor.dim() == 2:
                    tensor = tensor.unsqueeze(0)
                data_list.append(tensor)
                labels_list.append(label)

            if data_list:
                batch_data = torch.stack(data_list)
                batch_labels = torch.tensor(labels_list, dtype=torch.long)
                yield DataBatch(batch_data, batch_labels)

    def __len__(self) -> int:
        """
        Number of batches per epoch.
        """
        return (len(self.samples) + self.batch_size - 1) // self.batch_size

    def __add__(self, other: 'DataLoader') -> 'DataLoader':
        """
        Concatenate two DataLoaders with same settings.
        """
        if self.data_dir != other.data_dir or self.data_type != other.data_type:
            raise ValueError("Cannot add DataLoaders with different data_dir or data_type")
        new = DataLoader(
            self.data_dir,
            self.data_type,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            sr=self.sr,
            max_per_class=self.max_per_class,
            subset=self.subset
        )
        new.samples = self.samples + other.samples
        new.class_to_idx = self.class_to_idx
        return new
