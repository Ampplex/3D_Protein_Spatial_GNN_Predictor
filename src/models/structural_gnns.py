from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GCNConv, SAGEConv, global_mean_pool


class StructuralGNN(nn.Module):
    """Common architecture shell with interchangeable GCN/SAGE/GAT layers."""

    def __init__(
        self,
        model_type: str,
        in_channels: int = 23,
        hidden_channels: int = 64,
        embedding_dim: int = 64,
        num_classes: int = 2,
        dropout: float = 0.30,
        use_edge_attr: bool = False,
    ) -> None:
        super().__init__()
        model_type = model_type.upper()
        if model_type not in {"GCN", "GRAPHSAGE", "GAT"}:
            raise ValueError("model_type must be GCN, GraphSAGE, or GAT")

        self.model_type = model_type
        self.dropout = dropout
        self.use_edge_attr = use_edge_attr

        if model_type == "GCN":
            conv = GCNConv
            self.conv1 = conv(in_channels, hidden_channels)
            self.conv2 = conv(hidden_channels, hidden_channels)
            self.conv3 = conv(hidden_channels, embedding_dim)
        elif model_type == "GRAPHSAGE":
            conv = SAGEConv
            self.conv1 = conv(in_channels, hidden_channels)
            self.conv2 = conv(hidden_channels, hidden_channels)
            self.conv3 = conv(hidden_channels, embedding_dim)
        else:
            self.conv1 = GATConv(in_channels, hidden_channels, heads=1, edge_dim=1 if use_edge_attr else None)
            self.conv2 = GATConv(hidden_channels, hidden_channels, heads=1, edge_dim=1 if use_edge_attr else None)
            self.conv3 = GATConv(hidden_channels, embedding_dim, heads=1, edge_dim=1 if use_edge_attr else None)

        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, num_classes),
        )

    def encode_nodes(self, data) -> torch.Tensor:
        x, edge_index = data.x, data.edge_index
        edge_attr = getattr(data, "edge_attr", None)
        
        if self.use_edge_attr and edge_attr is not None and edge_attr.numel() > 0:
            if self.model_type == "GCN":
                ew = 1.0 / (edge_attr.view(-1) + 1e-5)
                x = F.relu(self.conv1(x, edge_index, edge_weight=ew))
                x = F.dropout(x, p=self.dropout, training=self.training)
                x = F.relu(self.conv2(x, edge_index, edge_weight=ew))
                x = F.dropout(x, p=self.dropout, training=self.training)
                x = F.relu(self.conv3(x, edge_index, edge_weight=ew))
                return x
            elif self.model_type == "GAT":
                x = F.relu(self.conv1(x, edge_index, edge_attr=edge_attr))
                x = F.dropout(x, p=self.dropout, training=self.training)
                x = F.relu(self.conv2(x, edge_index, edge_attr=edge_attr))
                x = F.dropout(x, p=self.dropout, training=self.training)
                x = F.relu(self.conv3(x, edge_index, edge_attr=edge_attr))
                return x
                
        # Fallback to standard
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.conv2(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.conv3(x, edge_index))
        return x

    def encode_graph(self, data) -> torch.Tensor:
        node_embeddings = self.encode_nodes(data)
        return global_mean_pool(node_embeddings, data.batch)

    def forward(self, data, return_embedding: bool = False) -> torch.Tensor:
        graph_embedding = self.encode_graph(data)
        if return_embedding:
            return graph_embedding
        return self.classifier(graph_embedding)
