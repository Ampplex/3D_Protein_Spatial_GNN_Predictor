from __future__ import annotations

import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv

from .common import GraphClassifierBase


class ProteinGraphSAGE(GraphClassifierBase):
    """3-layer GraphSAGE with graph-level mean pooling and MLP classification."""

    def __init__(self, in_channels: int, hidden_channels: int = 64, embedding_dim: int = 64,
                 num_classes: int = 2, dropout: float = 0.30) -> None:
        super().__init__(embedding_dim, hidden_channels, num_classes, dropout)
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)
        self.conv3 = SAGEConv(hidden_channels, embedding_dim)

    def encode_nodes(self, data) -> torch.Tensor:
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.conv2(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.conv3(x, edge_index))
        return x

    def encode_graph(self, data) -> torch.Tensor:
        return self.pool(self.encode_nodes(data), data.batch)

    def forward(self, data, return_embedding: bool = False) -> torch.Tensor:
        graph_embedding = self.encode_graph(data)
        if return_embedding:
            return graph_embedding
        return self.classify(graph_embedding)
