from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch_geometric.data import Data

from .pdb_parser import ResidueRecord
from .residue_features import build_feature_matrix


@dataclass(frozen=True)
class GraphBuildResult:
    data: Data
    residues: list[ResidueRecord]


def build_spatial_graph(
    records: list[ResidueRecord],
    cutoff_angstrom: float = 8.0,
    include_self_loops: bool = False,
    feature_mode: str = "baseline",
) -> GraphBuildResult:
    """Build a directed residue-contact graph using C-alpha distance.

    An edge i -> j exists when the Euclidean C-alpha distance is strictly
    smaller than ``cutoff_angstrom``. The reverse edge j -> i is also stored.
    ``edge_attr[:, 0]`` contains the corresponding C-alpha distance in Å.
    """
    if cutoff_angstrom <= 0:
        raise ValueError("cutoff_angstrom must be positive")
    if not records:
        raise ValueError("Cannot build graph from zero residues")

    features = build_feature_matrix(records, mode=feature_mode)
    
    if feature_mode == "A3":
        # Sequence graph (i, i+1)
        n = len(records)
        if n > 1:
            src = np.arange(n - 1)
            dst = np.arange(1, n)
            src_all = np.concatenate([src, dst])
            dst_all = np.concatenate([dst, src])
        else:
            src_all = np.array([], dtype=np.int64)
            dst_all = np.array([], dtype=np.int64)
            
        edge_index = torch.tensor(np.stack([src_all, dst_all], axis=0), dtype=torch.long) if len(src_all) > 0 else torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.ones((len(src_all), 1), dtype=torch.float32)
        
        data = Data(
            x=torch.tensor(features, dtype=torch.float32),
            edge_index=edge_index,
            edge_attr=edge_attr,
            pos=torch.tensor(np.stack([r.coord for r in records]).astype(np.float32), dtype=torch.float32),
            num_nodes=n,
        )
        return GraphBuildResult(data=data, residues=records)

    coords = np.stack([r.coord for r in records]).astype(np.float32)
    n = len(records)

    deltas = coords[:, None, :] - coords[None, :, :]
    distances = np.linalg.norm(deltas, axis=-1)

    mask = distances < float(cutoff_angstrom)
    if not include_self_loops:
        np.fill_diagonal(mask, False)

    src, dst = np.nonzero(mask)
    edge_distances = distances[src, dst].astype(np.float32)

    edge_index = torch.tensor(
        np.stack([src, dst], axis=0),
        dtype=torch.long,
    )
    edge_attr = torch.tensor(edge_distances[:, None], dtype=torch.float32)

    data = Data(
        x=torch.tensor(features, dtype=torch.float32),
        edge_index=edge_index,
        edge_attr=edge_attr,
        pos=torch.tensor(coords, dtype=torch.float32),
        num_nodes=n,
    )

    return GraphBuildResult(data=data, residues=records)
