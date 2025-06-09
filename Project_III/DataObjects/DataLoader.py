#!/usr/bin/env python3
import os
import random
from pathlib import Path
from typing import List, Tuple, Iterator, Union, Callable, Dict, Any
import random

import torch
import numpy as np
from torch.nn.functional import interpolate
from PIL import Image
from torchvision import transforms

RAW_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".gif"}


class DataBatch:
    def __init__(self, file_paths: List[Path], labels: List[str], dim_shape: int) -> None:
        tensors = []
        self.labels: List[str] = []
        for fp, label in zip(file_paths, labels):
            tensor = self._load(fp)
            tensors.append(tensor)
            self.labels.append(label)

        preprocess = transforms.Compose(
    [
        transforms.Resize((dim_shape, dim_shape)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ]
    )
        tensors = [preprocess(image) for image in tensors]
        self.data: torch.Tensor = torch.stack(tensors)


    def _load(self, fp: Path) -> torch.Tensor:
        ext = fp.suffix.lower()
        if ext == ".pt":
            return torch.load(fp)
        if ext == ".npy":
            return torch.from_numpy(np.load(fp))
        if ext in RAW_EXTENSIONS:
            img = Image.open(fp).convert("RGB")
            return img
        raise RuntimeError(f"Unsupported extension {ext}")

    def __iter__(self) -> Iterator[Tuple[torch.Tensor, str]]:
        return iter(self.data)

    def __len__(self) -> int:
        return len(self.data)

class DataLoader:
    def __init__(
        self,
        data_dir: Union[str, Path],
        batch_size: int = 32,
        shuffle: bool = True,
        fraction: float = 1.0,
        raw: bool = False,
        dim_shape: int = 128,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.dim_shape = dim_shape

        exts = RAW_EXTENSIONS if raw else {".pt", ".npy"}
        self.data_list: List[Tuple[Path, str]] = []

        for ext in exts:
            for fp in self.data_dir.rglob(f"*{ext}"):
                if fraction < 1.0 and random.random() > fraction:
                    continue
                name = fp.name.lower()
                if "cat" in name:
                    label = "cat"
                elif "dog" in name:
                    label = "dog"
                else:
                    label = "other"
                self.data_list.append((fp, label))

        self._build_batches()


    @classmethod
    def from_list(
        cls,
        data_list: List[Tuple[Path, str]],
        batch_size: int = 32,
        shuffle: bool = False,
        dim_shape: int = 128
    ) -> "DataLoader":
        loader = cls.__new__(cls)
        loader.data_dir = None  # type: ignore
        loader.batch_size = batch_size
        loader.shuffle = shuffle
        loader.data_list = data_list.copy()
        loader.dim_shape = dim_shape
        if shuffle:
            random.shuffle(loader.data_list)
        loader._build_batches()
        return loader

    def _build_batches(self) -> None:
        if self.batch_size == 1:
            # store raw pairs
            self.batches = self.data_list.copy()
        else:
            self.batches = []
            for i in range(0, len(self.data_list), self.batch_size):
                chunk = self.data_list[i : i + self.batch_size]
                fps, labels = zip(*chunk)
                self.batches.append(DataBatch(list(fps), list(labels), dim_shape=self.dim_shape))

    def __len__(self) -> int:
        return len(self.batches)

    def __iter__(self) -> Iterator[Any]:
        if self.batch_size == 1:
            for fp, label in self.batches:
                ext = fp.suffix.lower()
                if ext == ".pt":
                    tensor = torch.load(fp)
                elif ext == ".npy":
                    tensor = torch.from_numpy(np.load(fp))
                else:
                    img = Image.open(fp).convert("RGB")
                    tensor = transforms.ToTensor()(img)
                yield tensor, label
        else:
            for batch in self.batches:
                yield batch

    def __add__(self, other: "DataLoader") -> "DataLoader":
        if not isinstance(other, DataLoader):
            raise TypeError("Can only add DataLoader with DataLoader")
        if self.batch_size != other.batch_size:
            raise ValueError("Batch sizes must match to add DataLoaders")
        combined = self.data_list + other.data_list
        return DataLoader.from_list(combined, batch_size=self.batch_size, shuffle=self.shuffle)

    def __truediv__(self, fraction: float) -> "DataLoader":
        if not (0 < fraction <= 1):
            raise ValueError("Fraction must be between 0 and 1")
        cutoff = int(len(self.data_list) * fraction)
        subset = self.data_list[:cutoff]
        return DataLoader.from_list(subset, batch_size=self.batch_size, shuffle=False)

    def split(
        self,
        train_frac: float,
        test_frac: float,
        valid_frac: float
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
        return (
            DataLoader.from_list(data_copy[:n_train], batch_size=self.batch_size),
            DataLoader.from_list(data_copy[n_train:n_train + n_test], batch_size=self.batch_size),
            DataLoader.from_list(data_copy[n_train + n_test:], batch_size=self.batch_size),
        )

    def reshape_all(self, height: int, width: int) -> None:
        for fp, _ in self.data_list:
            ext = fp.suffix.lower()
            if ext == ".pt":
                tensor = torch.load(fp)
            elif ext == ".npy":
                tensor = torch.from_numpy(np.load(fp))
            else:
                img = Image.open(fp).convert("RGB")
                tensor = transforms.ToTensor()(img)
            t = tensor.unsqueeze(0) if tensor.ndim == 3 else tensor.unsqueeze(0)
            resized = interpolate(t, size=(height, width), mode="bilinear", align_corners=False).squeeze(0)
            if ext == ".pt":
                torch.save(resized, fp)
            elif ext == ".npy":
                np.save(fp.with_suffix(".npy"), resized.cpu().numpy())
            else:
                transforms.ToPILImage()(resized).save(fp)

    def apply_transform(
        self, transform_fn: Callable[[Dict[str, List[Any]]], Dict[str, Any]]
    ) -> Dict[str, Any]:
        examples: Dict[str, List[Any]] = {"image": []}
        for fp, _ in self.data_list:
            if fp.suffix.lower() in RAW_EXTENSIONS:
                img = Image.open(fp).convert("RGB")
            else:
                tensor = torch.load(fp) if fp.suffix.lower() == ".pt" else torch.from_numpy(np.load(fp))
                img = transforms.ToPILImage()(tensor)
            examples["image"].append(img)
        return transform_fn(examples)
