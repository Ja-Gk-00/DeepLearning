import os
import numpy as np
import torch
from typing import List, Tuple, Iterator, Optional

class DataBatch:
    def __init__(self, data: torch.Tensor, labels: torch.Tensor) -> None:
        self.data: torch.Tensor = data
        self.labels: torch.Tensor = labels

class DataLoader:
    def __init__(
        self,
        data_dir: str,
        batch_size: int = 32,
        shuffle: bool = True,
        convert_mode: str = "grayscale",
        max_per_class: Optional[int] = None
    ) -> None:
        self.data_dir: str = data_dir
        self.batch_size: int = batch_size
        self.shuffle: bool = shuffle
        self.convert_mode: str = convert_mode.lower()
        self.max_per_class: Optional[int] = max_per_class
        self.samples: List[Tuple[str, int]] = []
        self.class_to_idx: dict[str, int] = {}
        self._prepare_dataset()

    def _prepare_dataset(self) -> None:
        classes: List[str] = sorted(
            [d for d in os.listdir(self.data_dir) if os.path.isdir(os.path.join(self.data_dir, d))]
        )
        self.class_to_idx = {cls: idx for idx, cls in enumerate(classes)}
        for cls in classes:
            cls_dir: str = os.path.join(self.data_dir, cls)
            count: int = 0
            files: List[str] = sorted(os.listdir(cls_dir))
            for file in files:
                if file.endswith(".npy") or file.endswith(".pkl"):
                    if self.max_per_class is not None and count >= self.max_per_class:
                        continue
                    file_path: str = os.path.join(cls_dir, file)
                    self.samples.append((file_path, self.class_to_idx[cls]))
                    count += 1

    def __iter__(self) -> Iterator[DataBatch]:
        if self.shuffle:
            np.random.shuffle(self.samples)
        for i in range(0, len(self.samples), self.batch_size):
            batch_samples: List[Tuple[str, int]] = self.samples[i:i + self.batch_size]
            data_list: List[torch.Tensor] = []
            labels_list: List[int] = []
            for file_path, label in batch_samples:
                if file_path.endswith(".npy"):
                    arr: np.ndarray = np.load(file_path)
                elif file_path.endswith(".pkl"):
                    import pickle
                    with open(file_path, "rb") as f:
                        arr = pickle.load(f)
                else:
                    continue
                tensor: torch.Tensor = torch.tensor(arr, dtype=torch.float32)
                if tensor.dim() == 2:
                    tensor = tensor.unsqueeze(0)
                    if self.convert_mode == "color":
                        tensor = tensor.repeat(3, 1, 1)
                elif tensor.dim() == 3:
                    if tensor.shape[0] != 1 and tensor.shape[0] != 3 and tensor.shape[-1] in [1, 3]:
                        tensor = tensor.permute(2, 0, 1)
                    if self.convert_mode == "grayscale" and tensor.shape[0] == 3:
                        gray = 0.2989 * tensor[0] + 0.5870 * tensor[1] + 0.1140 * tensor[2]
                        tensor = gray.unsqueeze(0)
                    elif self.convert_mode == "color" and tensor.shape[0] == 1:
                        tensor = tensor.repeat(3, 1, 1)
                data_list.append(tensor)
                labels_list.append(label)
            if data_list:
                batch_data: torch.Tensor = torch.stack(data_list)
                batch_labels: torch.Tensor = torch.tensor(labels_list, dtype=torch.long)
                yield DataBatch(batch_data, batch_labels)

    def __add__(self, other: "DataLoader") -> "DataLoader":
        new_loader = DataLoader(self.data_dir, self.batch_size, self.shuffle, self.convert_mode, self.max_per_class)
        new_loader.samples = self.samples + other.samples
        new_loader.class_to_idx = self.class_to_idx
        return new_loader

    def __len__(self) -> int:
        return (len(self.samples) + self.batch_size - 1) // self.batch_size
