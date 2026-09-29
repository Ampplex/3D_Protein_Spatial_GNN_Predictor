from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import numpy as np

from src.data.pdb_dataset import PDBDataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect structural PDB graphs")
    parser.add_argument("--root", default="./data")
    args = parser.parse_args()

    dataset = PDBDataset(Path(args.root))
    labels: list[int] = []
    node_counts: list[int] = []
    edge_counts: list[int] = []
    feature_dims: set[int] = set()

    print("\nStructural dataset inspection")
    print("=" * 60)

    for i in range(len(dataset)):
        data = dataset[i]
        label = int(data.y.item())
        labels.append(label)
        node_counts.append(int(data.num_nodes))
        edge_counts.append(int(data.edge_index.size(1)))
        feature_dims.add(int(data.x.size(1)))

        if i < 10:
            row = dataset.metadata[i]
            print(
                f"{row['pdb_id']:>8} | label={label} | "
                f"nodes={data.num_nodes:>4} | edges={data.edge_index.size(1):>6} | "
                f"features={data.x.size(1)}"
            )

    counts = Counter(labels)
    print("\nSummary")
    print(f"Graphs:          {len(dataset)}")
    print(f"Node features:   {sorted(feature_dims)}")
    print(f"Mean nodes:      {np.mean(node_counts):.2f}")
    print(f"Mean edges:      {np.mean(edge_counts):.2f}")
    print(f"Class 0:         {counts[0]}")
    print(f"Class 1:         {counts[1]}")

    if feature_dims != {23}:
        raise SystemExit(f"Expected exactly 23 node features; found {sorted(feature_dims)}")


if __name__ == "__main__":
    main()
