"""
HeteroGAT model for fraud detection on heterogeneous graphs.
Uses HeteroConv with GATConv layers for multi-head attention message passing.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, GATConv, BatchNorm
from torch_geometric.data import HeteroData


class HeteroGAT(nn.Module):
    """
    Heterogeneous Graph Attention Network for fraud node classification.
    
    Architecture:
    - Multiple HeteroConv layers with GATConv (multi-head attention)
    - BatchNorm after each layer
    - Dropout for regularization
    - Residual connections
    - Final linear classifier on 'txn' node type
    
    Args:
        metadata: Tuple of (node_types, edge_types) from HeteroData
        in_channels: Input feature dimension
        hidden_channels: Hidden dimension
        num_layers: Number of GNN layers
        num_heads: Number of attention heads
        num_classes: Number of output classes
        dropout: Dropout rate
    """
    
    def __init__(
        self,
        metadata,
        in_channels: int,
        hidden_channels: int = 128,
        num_layers: int = 2,
        num_heads: int = 4,
        num_classes: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()
        
        self.num_layers = num_layers
        self.dropout = dropout
        self.num_heads = num_heads
        
        node_types, edge_types = metadata
        
        # Head dimension
        head_dim = hidden_channels // num_heads
        
        # Input projection
        self.input_proj = nn.Linear(in_channels, hidden_channels)
        
        # Entity embeddings
        self.entity_embeds = nn.ModuleDict()
        for ntype in node_types:
            if ntype != "txn":
                self.entity_embeds[ntype] = nn.LazyLinear(hidden_channels)
        
        # GAT conv layers
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        
        for i in range(num_layers):
            conv_dict = {}
            for et in edge_types:
                if i == 0:
                    conv_dict[et] = GATConv(
                        (-1, -1), head_dim,
                        heads=num_heads, concat=True,
                        dropout=dropout, add_self_loops=False,
                    )
                else:
                    conv_dict[et] = GATConv(
                        (-1, -1), head_dim,
                        heads=num_heads, concat=True,
                        dropout=dropout, add_self_loops=False,
                    )
            
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
        Forward pass with multi-head attention.
        
        Args:
            x_dict: Node feature dictionaries
            edge_index_dict: Edge index dictionaries
            
        Returns:
            Logits for 'txn' nodes
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
        
        for i, (conv, norm) in enumerate(zip(self.convs, self.norms)):
            h_dict_new = conv(h_dict, edge_index_dict)
            
            out_dict = {}
            for ntype in h_dict_new:
                h = h_dict_new[ntype]
                if h is None:
                    continue
                if h.dim() == 2 and h.shape[0] > 0:
                    h = norm(h)
                    h = F.elu(h)
                    h = F.dropout(h, p=self.dropout, training=self.training)
                    
                    if ntype in h_dict and h_dict[ntype].shape == h.shape:
                        h = h + h_dict[ntype]
                out_dict[ntype] = h
            
            # Keep nodes from h_dict that weren't updated
            for ntype in h_dict:
                if ntype not in out_dict:
                    out_dict[ntype] = h_dict[ntype]
            
            h_dict = out_dict
        
        out = self.classifier(h_dict["txn"])
        return out
