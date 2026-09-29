from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import torch


def main():
    parser = argparse.ArgumentParser(description="Visualize a saved protein graph.")
    parser.add_argument("graph", type=Path)
    parser.add_argument("--output", type=Path, default=Path("results/protein_graph.png"))
    parser.add_argument("--max-nodes", type=int, default=250)
    args = parser.parse_args()

    data = torch.load(args.graph, weights_only=False)

    n = int(data.x.shape[0])
    if n > args.max_nodes:
        raise ValueError(
            f"Graph has {n} nodes. Increase --max-nodes to visualize it."
        )

    g = nx.Graph()
    g.add_nodes_from(range(n))

    edges = data.edge_index.t().tolist()
    for u, v in edges:
        if u != v:
            g.add_edge(u, v)

    pos = nx.spring_layout(g, seed=42)

    args.output.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(9, 8))
    nx.draw_networkx_edges(g, pos, ax=ax, alpha=0.25, width=0.8)
    nx.draw_networkx_nodes(
        g,
        pos,
        ax=ax,
        node_size=80,
        node_color="white",
        edgecolors="black",
        linewidths=0.7,
    )
    ax.set_title(
        f"Protein residue graph | nodes={n} | unique contacts={g.number_of_edges()}"
    )
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(args.output, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
