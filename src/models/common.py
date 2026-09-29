from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool


class GraphClassifierBase(nn.Module):
    """Shared graph-level classifier head and embedding contract."""

    def __init__(self, embedding_dim: int, hidden_channels: int, num_classes: int, dropout: float) -> None:
        super().__init__()
        self.dropout = dropout
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, num_classes),
        )

    def pool(self, node_embeddings: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        return global_mean_pool(node_embeddings, batch)

    def classify(self, graph_embedding: torch.Tensor) -> torch.Tensor:
        return self.classifier(graph_embedding)
