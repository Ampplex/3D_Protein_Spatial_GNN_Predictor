from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader


@dataclass(frozen=True)
class SplitData:
    train: list[int]
    val: list[int]
    test: list[int]


def load_proteins(root: str | Path = "./data") -> TUDataset:
    return TUDataset(root=str(root), name="PROTEINS")


def make_stratified_split(dataset: TUDataset, seed: int = 42,
                          test_size: float = 0.10, val_size: float = 0.10) -> SplitData:
    if test_size <= 0 or val_size <= 0 or test_size + val_size >= 1:
        raise ValueError("test_size and val_size must be > 0 and sum to < 1")

    indices = np.arange(len(dataset))
    labels = np.asarray([int(dataset[i].y.item()) for i in indices])

    train_idx, holdout_idx = train_test_split(
        indices, test_size=test_size + val_size,
        stratify=labels, random_state=seed,
    )
    holdout_labels = labels[holdout_idx]
    relative_test = test_size / (test_size + val_size)
    val_idx, test_idx = train_test_split(
        holdout_idx, test_size=relative_test,
        stratify=holdout_labels, random_state=seed,
    )

    return SplitData(train=train_idx.tolist(), val=val_idx.tolist(), test=test_idx.tolist())


def make_loaders(dataset: TUDataset, split: SplitData, batch_size: int = 32):
    return (
        DataLoader(dataset[split.train], batch_size=batch_size, shuffle=True),
        DataLoader(dataset[split.val], batch_size=batch_size, shuffle=False),
        DataLoader(dataset[split.test], batch_size=batch_size, shuffle=False),
    )
