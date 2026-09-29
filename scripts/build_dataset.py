from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from src.data.pdb_dataset import PDBDataset


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build/cache PyG graphs from locally downloaded labelled PDBs."
    )
    parser.add_argument("--root", default="./data")
    parser.add_argument("--cutoff", type=float, default=8.0)
    parser.add_argument("--chain", default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    dataset = PDBDataset(
        root=Path(args.root),
        cutoff_angstrom=args.cutoff,
        chain_id=args.chain,
        force_reprocess=args.force,
    )

    labels = Counter(int(dataset[i].y.item()) for i in range(len(dataset)))

    print(f"Built/verified {len(dataset)} graph(s)")
    print(f"Class 0: {labels[0]}")
    print(f"Class 1: {labels[1]}")

    for i in range(min(len(dataset), 5)):
        data = dataset[i]
        row = dataset.metadata[i]
        print(
            f"{row['pdb_id']}: label={int(data.y.item())}, "
            f"nodes={data.num_nodes}, edges={data.edge_index.size(1)}, "
            f"x={tuple(data.x.shape)}"
        )


if __name__ == "__main__":
    main()
