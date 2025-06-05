#!/usr/bin/env python3
import os
import random
from pathlib import Path
from typing import List, Tuple, Iterator, Union

import torch
import numpy as np

from torch.nn.functional import interpolate

class DataBatch:
    def __init__(self, file_paths: List[Path], labels: List[str]) -> None:
        self.data: List[Tuple[torch.Tensor, str]] = []
        for fp, label in zip(file_paths, labels):
            ext = fp.suffix.lower()
            if ext == ".pt":
                tensor = torch.load(fp)
            elif ext == ".npy":
                arr = np.load(fp)
                tensor = torch.from_numpy(arr)
            else:
                continue
            self.data.append((tensor, label))

    def __iter__(self) -> Iterator[Tuple[torch.Tensor, str]]:
        for tensor, label in self.data:
            yield tensor, label

    def __len__(self) -> int:
        return len(self.data)


class DataLoader:
    def __init__(
        self, data_dir: Union[str, Path], batch_size: int = 32, shuffle: bool = True, fraction: float = 1.0
    ) -> None:
        self.data_dir: Path = Path(data_dir)
        self.batch_size: int = batch_size
        self.shuffle: bool = shuffle
        self.data_list: List[Tuple[Path, str]] = []

        for ext in ("*.pt", "*.npy"):
            for fp in self.data_dir.rglob(ext):
                if 'cat' in fp.parent.name.lower():
                    label = 'cat'
                elif 'dog' in fp.parent.name.lower():
                    label = 'dog'
                else:
                    label = fp.parent.name.lower()
                if label not in ("cat", "dog"):
                    continue
                self.data_list.append((fp, label))

        if not (0 < fraction <= 1):
            raise ValueError("Fraction must be between 0 and 1")
        cutoff = int(len(self.data_list) * fraction)
        self.data_list = self.data_list[:cutoff]

        if self.shuffle:
            random.shuffle(self.data_list)

        self._build_batches()

    @classmethod
    def from_list(
        cls,
        data_list: List[Tuple[Path, str]],
        batch_size: int = 32,
        shuffle: bool = False,
    ) -> "DataLoader":
        loader = cls.__new__(cls)
        loader.data_dir = None  # type: ignore
        loader.batch_size = batch_size
        loader.shuffle = shuffle
        loader.data_list = data_list.copy()
        if shuffle:
            random.shuffle(loader.data_list)
        loader._build_batches()
        return loader

    def _build_batches(self) -> None:
        self.batches: List[DataBatch] = []
        for i in range(0, len(self.data_list), self.batch_size):
            chunk = self.data_list[i : i + self.batch_size]
            file_paths, labels = zip(*chunk)
            self.batches.append(DataBatch(list(file_paths), list(labels)))

    def __len__(self) -> int:
        return len(self.batches)

    def __iter__(self) -> Iterator[DataBatch]:
        for batch in self.batches:
            yield batch

    def __add__(self, other: "DataLoader") -> "DataLoader":
        if not isinstance(other, DataLoader):
            raise TypeError("Can only add DataLoader with DataLoader")
        if self.batch_size != other.batch_size:
            raise ValueError("Batch sizes must match to add DataLoaders")
        combined: List[Tuple[Path, str]] = self.data_list + other.data_list
        return DataLoader.from_list(
            combined, batch_size=self.batch_size, shuffle=self.shuffle
        )

    def __truediv__(self, fraction: float) -> "DataLoader":
        if not (0 < fraction <= 1):
            raise ValueError("Fraction must be between 0 and 1")
        cutoff = int(len(self.data_list) * fraction)
        subset: List[Tuple[Path, str]] = self.data_list[:cutoff]
        return DataLoader.from_list(subset, batch_size=self.batch_size, shuffle=False)

    def split(
        self, train_frac: float, test_frac: float, valid_frac: float
    ) -> Tuple["DataLoader", "DataLoader", "DataLoader"]:
        total = train_frac + test_frac + valid_frac
        if total <= 0:
            raise ValueError("Fractions must sum to a positive number")
        t_frac = train_frac / total
        te_frac = test_frac / total
        data_copy = self.data_list.copy()
        random.shuffle(data_copy)
        n = len(data_copy)
        n_train = int(n * t_frac)
        n_test = int(n * te_frac)
        train_list = data_copy[:n_train]
        test_list = data_copy[n_train : n_train + n_test]
        valid_list = data_copy[n_train + n_test :]
        train_loader = DataLoader.from_list(
            train_list, batch_size=self.batch_size, shuffle=False
        )
        test_loader = DataLoader.from_list(
            test_list, batch_size=self.batch_size, shuffle=False
        )
        valid_loader = DataLoader.from_list(
            valid_list, batch_size=self.batch_size, shuffle=False
        )
        return train_loader, test_loader, valid_loader

    def reshape_all(self, height: int, width: int) -> None:
        for fp, _ in self.data_list:
            ext = fp.suffix.lower()
            if ext == ".pt":
                tensor = torch.load(fp)
            elif ext == ".npy":
                arr = np.load(fp)
                tensor = torch.from_numpy(arr)
            else:
                continue

            tensor = tensor.unsqueeze(0) if tensor.ndim == 3 else tensor
            resized = interpolate(tensor.unsqueeze(0), size=(height, width), mode="bilinear", align_corners=False)
            resized = resized.squeeze(0)

            if ext == ".pt":
                torch.save(resized, fp)  # type: ignore
            else:
                np.save(fp.with_suffix(".npy"), resized.cpu().numpy())
