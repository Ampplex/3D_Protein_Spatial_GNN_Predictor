from __future__ import annotations

import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv

from .common import GraphClassifierBase


class ProteinGAT(GraphClassifierBase):
    """3-layer GAT with graph-level mean pooling and MLP classification."""

    def __init__(self, in_channels: int, hidden_channels: int = 64, embedding_dim: int = 64,
                 num_classes: int = 2, dropout: float = 0.30) -> None:
        super().__init__(embedding_dim, hidden_channels, num_classes, dropout)
        # Using heads=1 to keep dimensions exactly comparable to GCN and GraphSAGE
        self.conv1 = GATConv(in_channels, hidden_channels, heads=1)
        self.conv2 = GATConv(hidden_channels, hidden_channels, heads=1)
        self.conv3 = GATConv(hidden_channels, embedding_dim, heads=1)

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
