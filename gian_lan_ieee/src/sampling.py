"""
Sampling module for handling class imbalance in fraud detection.
Provides Focal Loss, class weights, and weighted sampling strategies.
"""

import torch
import numpy as np
from torch_geometric.loader import NeighborLoader
from torch_geometric.data import HeteroData

from src.config import (
    NUM_NEIGHBORS, BATCH_SIZE,
    FOCAL_LOSS_ALPHA, FOCAL_LOSS_GAMMA,
)
from src.utils import FocalLoss


class ImbalanceSampler:
    """
    Handles class imbalance for fraud detection GNN training.
    
    Strategies:
    - Focal Loss: Down-weights easy examples, focuses on hard ones
    - Class weighting: Inverse frequency weighting for CrossEntropyLoss
    - Weighted NeighborLoader: Stratified mini-batch sampling
    
    Usage:
        sampler = ImbalanceSampler(data)
        loss_fn = sampler.get_focal_loss()
        train_loader = sampler.get_neighbor_loader('train')
    """
    
    def __init__(self, data: HeteroData):
        self.data = data
        self._compute_class_weights()
    
    def _compute_class_weights(self):
        """Compute inverse frequency class weights."""
        y = self.data["txn"].y.cpu().numpy()
        
        n_total = len(y)
        n_fraud = (y == 1).sum()
        n_non_fraud = n_total - n_fraud
        
        # Inverse frequency weighting
        if n_fraud > 0:
            w_non_fraud = n_total / (2 * n_non_fraud)
            w_fraud = n_total / (2 * n_fraud)
        else:
            w_non_fraud = 1.0
            w_fraud = 1.0
        
        self.class_weights = torch.tensor([w_non_fraud, w_fraud], dtype=torch.float32)
        self.fraud_ratio = n_fraud / n_total if n_total > 0 else 0.0
        
        print(f"Class weights: non-fraud={w_non_fraud:.4f}, fraud={w_fraud:.4f}")
        print(f"Fraud ratio: {self.fraud_ratio:.4f} ({n_fraud:,}/{n_total:,})")
    
    def get_focal_loss(
        self,
        alpha: float = FOCAL_LOSS_ALPHA,
        gamma: float = FOCAL_LOSS_GAMMA,
    ) -> FocalLoss:
        """
        Get Focal Loss criterion for imbalanced training.
        
        Args:
            alpha: Weighting factor for fraud class
            gamma: Focusing parameter
            
        Returns:
            FocalLoss module
        """
        return FocalLoss(alpha=alpha, gamma=gamma)
    
    def get_weighted_ce_loss(self, device: torch.device = None) -> torch.nn.CrossEntropyLoss:
        """
        Get class-weighted CrossEntropyLoss.
        
        Returns:
            CrossEntropyLoss with class weights
        """
        weights = self.class_weights
        if device is not None:
            weights = weights.to(device)
        return torch.nn.CrossEntropyLoss(weight=weights)
    
    def get_neighbor_loader(
        self,
        split: str = 'train',
        num_neighbors: list = None,
        batch_size: int = BATCH_SIZE,
        shuffle: bool = True,
    ) -> NeighborLoader:
        """
        Create a NeighborLoader for mini-batch training on the heterogeneous graph.
        
        Args:
            split: 'train', 'val', or 'test'
            num_neighbors: List of neighbor counts per layer
            batch_size: Batch size
            shuffle: Whether to shuffle
            
        Returns:
            NeighborLoader instance
        """
        if num_neighbors is None:
            num_neighbors = NUM_NEIGHBORS
        
        mask_name = f"{split}_mask"
        if not hasattr(self.data["txn"], mask_name):
            raise ValueError(f"Mask '{mask_name}' not found in data['txn']")
        
        mask = getattr(self.data["txn"], mask_name)
        input_nodes = ("txn", mask)
        
        loader = NeighborLoader(
            self.data,
            num_neighbors=num_neighbors,
            batch_size=batch_size,
            input_nodes=input_nodes,
            shuffle=shuffle,
        )
        
        print(f"Created {split} NeighborLoader: "
              f"{mask.sum():,} nodes, batch_size={batch_size}, "
              f"neighbors={num_neighbors}")
        
        return loader
    
    def get_all_loaders(
        self,
        num_neighbors: list = None,
        batch_size: int = BATCH_SIZE,
    ) -> dict:
        """
        Create NeighborLoaders for all splits.
        
        Returns:
            Dictionary with 'train', 'val', 'test' loaders
        """
        loaders = {}
        for split in ['train', 'val', 'test']:
            shuffle = (split == 'train')
            loaders[split] = self.get_neighbor_loader(
                split=split,
                num_neighbors=num_neighbors,
                batch_size=batch_size,
                shuffle=shuffle,
            )
        return loaders
