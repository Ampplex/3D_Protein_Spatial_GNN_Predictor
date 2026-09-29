from __future__ import annotations

import argparse
from pathlib import Path

import torch

from src.data import build_protein_graph, parse_pdb


def main():
    parser = argparse.ArgumentParser(
        description="Parse one PDB file and convert it to a PyG protein graph."
    )
    parser.add_argument("pdb", type=Path)
    parser.add_argument("--chain", default=None)
    parser.add_argument("--threshold", type=float, default=8.0)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    residues = parse_pdb(args.pdb, chain_id=args.chain)
    graph = build_protein_graph(
        residues,
        threshold=args.threshold,
    )

    print(f"PDB: {args.pdb}")
    print(f"Chain filter: {args.chain if args.chain else 'all'}")
    print(f"Residues / nodes: {graph.num_residues}")
    print(f"Node feature shape: {tuple(graph.x.shape)}")
    print(f"Edge index shape: {tuple(graph.edge_index.shape)}")
    print(f"Edge attributes: {tuple(graph.edge_attr.shape)}")
    print(f"Coordinate shape: {tuple(graph.pos.shape)}")
    print(f"8 Å contacts (directed edges): {graph.edge_index.shape[1]}")

    if graph.edge_attr.numel():
        distances = graph.edge_attr.view(-1)
        print(
            "Distance range (Å): "
            f"{distances.min().item():.3f} - {distances.max().item():.3f}"
        )

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        torch.save(graph, args.output)
        print(f"Saved graph: {args.output}")


if __name__ == "__main__":
    main()
