from __future__ import annotations

import argparse
import copy
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset
from torch_geometric.loader import DataLoader

from src.data.pdb_dataset import PDBDataset
from src.models.structural_gnns import StructuralGNN
from src.utils.reproducibility import set_seed

SEED = 42
BATCH_SIZE = 32
HIDDEN_CHANNELS = 64
EMBEDDING_DIM = 64
DROPOUT = 0.30
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 5e-4
MAX_EPOCHS = 100
PATIENCE = 15


def stratified_indices(dataset, seed: int = SEED):
    indices = np.arange(len(dataset))
    labels = np.asarray([int(dataset[i].y.item()) for i in indices])
    counts = Counter(labels.tolist())

    if set(counts) != {0, 1}:
        raise ValueError(
            f"Structural training requires both classes. Found class counts: {dict(counts)}"
        )
    if min(counts.values()) < 10:
        raise ValueError(
            f"Too few examples in a class for a 3-way stratified split: {dict(counts)}"
        )

    train_idx, holdout_idx = train_test_split(
        indices, test_size=0.20, stratify=labels, random_state=seed
    )
    val_idx, test_idx = train_test_split(
        holdout_idx,
        test_size=0.50,
        stratify=labels[holdout_idx],
        random_state=seed,
    )
    return train_idx.tolist(), val_idx.tolist(), test_idx.tolist()


def evaluate(model, loader, device, criterion):
    model.eval()
    total_loss, total_items = 0.0, 0
    y_true, y_pred, y_prob = [], [], []

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            logits = model(batch)
            loss = criterion(logits, batch.y)
            probs = torch.softmax(logits, dim=1)[:, 1]
            preds = logits.argmax(dim=1)
            n = batch.y.size(0)
            total_loss += float(loss.item()) * n
            total_items += n
            y_true.extend(batch.y.cpu().numpy().tolist())
            y_pred.extend(preds.cpu().numpy().tolist())
            y_prob.extend(probs.cpu().numpy().tolist())

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    y_prob = np.asarray(y_prob)

    try:
        auc = float(roc_auc_score(y_true, y_prob))
    except ValueError:
        auc = float("nan")

    return {
        "loss": total_loss / max(total_items, 1),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "auc": auc,
    }


def main():
    parser = argparse.ArgumentParser(description="Train a structural protein GNN")
    parser.add_argument("--model", required=True, choices=["GCN", "GraphSAGE", "GAT"])
    parser.add_argument("--root", default="./data")
    parser.add_argument("--feature-mode", default="baseline", choices=["baseline", "A1", "A2", "A3", "A4"])
    args = parser.parse_args()

    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    dataset = PDBDataset(Path(args.root), feature_mode=args.feature_mode)
    train_idx, val_idx, test_idx = stratified_indices(dataset)
    print(f"Split sizes: train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}")

    train_loader = DataLoader(Subset(dataset, train_idx), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(Subset(dataset, val_idx), batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(Subset(dataset, test_idx), batch_size=BATCH_SIZE, shuffle=False)

    model = StructuralGNN(
        model_type=args.model,
        in_channels=dataset[0].x.size(1),
        hidden_channels=HIDDEN_CHANNELS,
        embedding_dim=EMBEDDING_DIM,
        num_classes=2,
        dropout=DROPOUT,
        use_edge_attr=(args.feature_mode == "A4"),
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    criterion = nn.CrossEntropyLoss()

    best_val_loss = float("inf")
    best_state = None
    patience_counter = 0
    history = []

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        running_loss, count = 0.0, 0

        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch)
            loss = criterion(logits, batch.y)
            loss.backward()
            optimizer.step()
            n = batch.y.size(0)
            running_loss += float(loss.item()) * n
            count += n

        train_loss = running_loss / max(count, 1)
        val = evaluate(model, val_loader, device, criterion)
        history.append({"epoch": epoch, "train_loss": train_loss, **{f"val_{k}": v for k, v in val.items()}})

        print(
            f"Epoch {epoch:03d} | train_loss={train_loss:.4f} | "
            f"val_loss={val['loss']:.4f} | val_acc={val['accuracy']:.4f} | val_f1={val['f1']:.4f}"
        )

        if val["loss"] < best_val_loss:
            best_val_loss = val["loss"]
            best_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= PATIENCE:
            print(f"Early stopping at epoch {epoch}")
            break

    if best_state is None:
        raise RuntimeError("Training did not produce a checkpointable model")

    model.load_state_dict(best_state)
    test_metrics = evaluate(model, test_loader, device, criterion)

    checkpoints = Path("./checkpoints")
    results = Path("./results")
    checkpoints.mkdir(exist_ok=True)
    results.mkdir(exist_ok=True)

    torch.save(
        {
            "model_type": args.model,
            "model_state_dict": model.state_dict(),
            "input_dim": dataset[0].x.size(1),
            "hidden_channels": HIDDEN_CHANNELS,
            "embedding_dim": EMBEDDING_DIM,
            "num_classes": 2,
            "seed": SEED,
            "cutoff_angstrom": float(dataset.cutoff_angstrom),
            "split": {"train": train_idx, "val": val_idx, "test": test_idx},
        },
        checkpoints / f"{args.model.lower()}_structural_{args.feature_mode}.pt",
    )

    payload = {
        "model": args.model,
        "dataset_size": len(dataset),
        "input_dim": dataset[0].x.size(1),
        "cutoff_angstrom": float(dataset.cutoff_angstrom),
        "test_metrics": test_metrics,
        "history": history,
    }
    (results / f"{args.model.lower()}_structural_{args.feature_mode}.json").write_text(json.dumps(payload, indent=2))

    print("\nTest metrics:")
    for key, value in test_metrics.items():
        print(f"  {key}: {value:.4f}")


if __name__ == "__main__":
    main()
