"""
HeteroGraphSAGE model for fraud detection on heterogeneous graphs.
Uses HeteroConv with SAGEConv layers for message passing.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, SAGEConv, BatchNorm
from torch_geometric.data import HeteroData


class HeteroGraphSAGE(nn.Module):
    """
    Heterogeneous GraphSAGE model for fraud node classification.
    
    Architecture:
    - Multiple HeteroConv layers with SAGEConv
    - BatchNorm after each layer
    - Dropout for regularization
    - Residual connections (when dimensions match)
    - Final linear classifier on 'txn' node type
    
    Args:
        metadata: Tuple of (node_types, edge_types) from HeteroData
        in_channels: Input feature dimension (for 'txn' nodes)
        hidden_channels: Hidden dimension
        num_layers: Number of GNN layers
        num_classes: Number of output classes (2 for fraud detection)
        dropout: Dropout rate
    """
    
    def __init__(
        self,
        metadata,
        in_channels: int,
        hidden_channels: int = 128,
        num_layers: int = 2,
        num_classes: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()
        
        self.num_layers = num_layers
        self.dropout = dropout
        
        node_types, edge_types = metadata
        
        # Input projection for txn nodes
        self.input_proj = nn.Linear(in_channels, hidden_channels)
        
        # Entity embedding layers (for entity nodes without features)
        self.entity_embeds = nn.ModuleDict()
        for ntype in node_types:
            if ntype != "txn":
                self.entity_embeds[ntype] = nn.LazyLinear(hidden_channels)
        
        # HeteroConv layers
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        
        for i in range(num_layers):
            conv_dict = {}
            for et in edge_types:
                conv_dict[et] = SAGEConv((-1, -1), hidden_channels)
            
            self.convs.append(HeteroConv(conv_dict, aggr='mean'))
            self.norms.append(BatchNorm(hidden_channels))
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels // 2, num_classes),
        )
    
    def forward(self, x_dict, edge_index_dict):
        """
        Forward pass.
        
        Args:
            x_dict: Dictionary mapping node types to feature tensors
            edge_index_dict: Dictionary mapping edge types to edge index tensors
            
        Returns:
            Logits for 'txn' nodes, shape (num_txn_nodes, num_classes)
        """
        # Project all node features to hidden_channels
        h_dict = {}
        for ntype, x in x_dict.items():
            if ntype == "txn":
                h_dict[ntype] = self.input_proj(x)
            elif ntype in self.entity_embeds:
                h_dict[ntype] = self.entity_embeds[ntype](x)
            else:
                h_dict[ntype] = torch.zeros(
                    x.shape[0], self.input_proj.out_features,
                    device=x.device
                )
        
        # Message passing
        for i, (conv, norm) in enumerate(zip(self.convs, self.norms)):
            h_dict_new = conv(h_dict, edge_index_dict)
            
            # Apply normalization, activation, and dropout
            out_dict = {}
            for ntype in h_dict_new:
                h = h_dict_new[ntype]
                if h is None:
                    continue
                if h.dim() == 2 and h.shape[0] > 0:
                    h = norm(h)
                    h = F.relu(h)
                    h = F.dropout(h, p=self.dropout, training=self.training)
                    
                    # Residual connection
                    if ntype in h_dict and h_dict[ntype].shape == h.shape:
                        h = h + h_dict[ntype]
                out_dict[ntype] = h
            
            # Keep nodes from h_dict that weren't updated
            for ntype in h_dict:
                if ntype not in out_dict:
                    out_dict[ntype] = h_dict[ntype]
            
            h_dict = out_dict
        
        # Classify txn nodes
        out = self.classifier(h_dict["txn"])
        return out
