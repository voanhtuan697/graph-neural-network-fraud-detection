"""
Trainer module for GNN models on the IEEE-CIS Fraud Detection dataset.
Handles the training loop, validation, early stopping, and model checkpointing.
"""

import time
import torch
import torch.nn as nn
import numpy as np
from copy import deepcopy
from pathlib import Path
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

from src.config import (
    LEARNING_RATE, WEIGHT_DECAY, NUM_EPOCHS, PATIENCE,
    MODEL_DIR, DEVICE,
)
from src.utils import EarlyStopping, save_model, print_separator


class GNNTrainer:
    """
    Unified trainer for HeteroGraphSAGE, HeteroGAT, and HeteroRGCN models.
    
    Features:
    - Training loop with early stopping
    - CosineAnnealing LR scheduling
    - Gradient clipping
    - Focal Loss / Weighted CE support
    - Best model checkpointing
    - Training curves logging
    
    Usage:
        trainer = GNNTrainer(model, loss_fn)
        history = trainer.train(train_loader, val_loader)
    """
    
    def __init__(
        self,
        model: nn.Module,
        loss_fn: nn.Module,
        lr: float = LEARNING_RATE,
        weight_decay: float = WEIGHT_DECAY,
        device: torch.device = DEVICE,
        model_name: str = "gnn_model",
    ):
        self.model = model.to(device)
        self.loss_fn = loss_fn
        self.device = device
        self.model_name = model_name
        
        self.optimizer = torch.optim.Adam(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=NUM_EPOCHS, eta_min=1e-6)
        
        self.best_model_state = None
        self.history = {
            'train_loss': [], 'val_loss': [],
            'train_f1': [], 'val_f1': [],
            'val_precision': [], 'val_recall': [],
            'val_auc': [],
        }
    
    def train(
        self,
        train_loader,
        val_loader,
        num_epochs: int = NUM_EPOCHS,
        patience: int = PATIENCE,
        grad_clip: float = 1.0,
    ) -> dict:
        """
        Train the model.
        
        Args:
            train_loader: Training NeighborLoader
            val_loader: Validation NeighborLoader
            num_epochs: Maximum number of epochs
            patience: Early stopping patience
            grad_clip: Max gradient norm for clipping
            
        Returns:
            Training history dictionary
        """
        print_separator(f"TRAINING {self.model_name.upper()}")
        print(f"Device: {self.device}")
        print(f"Epochs: {num_epochs}, Patience: {patience}")
        print(f"Optimizer: Adam (lr={self.optimizer.param_groups[0]['lr']:.1e})")
        
        early_stopper = EarlyStopping(patience=patience, mode='max')
        
        for epoch in range(1, num_epochs + 1):
            start_time = time.time()
            
            # Train
            train_loss, train_f1 = self._train_epoch(train_loader, grad_clip)
            
            # Validate
            val_loss, val_metrics = self._validate(val_loader)
            
            # Step scheduler
            self.scheduler.step()
            
            elapsed = time.time() - start_time
            
            # Log
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_f1'].append(train_f1)
            self.history['val_f1'].append(val_metrics['f1'])
            self.history['val_precision'].append(val_metrics['precision'])
            self.history['val_recall'].append(val_metrics['recall'])
            self.history['val_auc'].append(val_metrics.get('auc', 0))
            
            print(
                f"Epoch {epoch:3d}/{num_epochs} | "
                f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
                f"Val F1: {val_metrics['f1']:.4f} | "
                f"Val AUC: {val_metrics.get('auc', 0):.4f} | "
                f"Time: {elapsed:.1f}s"
            )
            
            # Save best model
            if val_metrics['f1'] > 0 and (
                self.best_model_state is None or
                val_metrics['f1'] > max(self.history['val_f1'][:-1], default=0)
            ):
                self.best_model_state = deepcopy(self.model.state_dict())
                save_path = str(MODEL_DIR / f"{self.model_name}_best.pt")
                save_model(self.model, save_path, epoch=epoch, metrics=val_metrics)
            
            # Early stopping
            if early_stopper(val_metrics['f1']):
                print(f"\nEarly stopping at epoch {epoch}")
                break
        
        # Restore best model
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
            print("Restored best model weights.")
        
        print_separator("TRAINING COMPLETE")
        return self.history
    
    def _train_epoch(self, train_loader, grad_clip: float):
        """Run one training epoch."""
        self.model.train()
        total_loss = 0
        all_preds = []
        all_labels = []
        num_batches = 0
        
        for batch in train_loader:
            batch = batch.to(self.device)
            self.optimizer.zero_grad()
            
            x_dict = {nt: batch[nt].x for nt in batch.node_types if hasattr(batch[nt], 'x') and batch[nt].x is not None}
            edge_index_dict = {et: batch[et].edge_index for et in batch.edge_types if hasattr(batch[et], 'edge_index')}
            
            out = self.model(x_dict, edge_index_dict)
            
            # Get mask for input nodes
            mask = batch["txn"].train_mask if hasattr(batch["txn"], 'train_mask') else torch.ones(out.shape[0], dtype=torch.bool, device=self.device)
            
            if mask.sum() == 0:
                continue
            
            y = batch["txn"].y[mask]
            pred = out[mask]
            
            loss = self.loss_fn(pred, y)
            loss.backward()
            
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)
            
            self.optimizer.step()
            
            total_loss += loss.item()
            all_preds.extend(pred.argmax(dim=-1).cpu().numpy())
            all_labels.extend(y.cpu().numpy())
            num_batches += 1
        
        avg_loss = total_loss / max(num_batches, 1)
        f1 = f1_score(all_labels, all_preds, average='binary', zero_division=0)
        
        return avg_loss, f1
    
    @torch.no_grad()
    def _validate(self, val_loader):
        """Run validation."""
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_probs = []
        all_labels = []
        num_batches = 0
        
        for batch in val_loader:
            batch = batch.to(self.device)
            
            x_dict = {nt: batch[nt].x for nt in batch.node_types if hasattr(batch[nt], 'x') and batch[nt].x is not None}
            edge_index_dict = {et: batch[et].edge_index for et in batch.edge_types if hasattr(batch[et], 'edge_index')}
            
            out = self.model(x_dict, edge_index_dict)
            
            mask = batch["txn"].val_mask if hasattr(batch["txn"], 'val_mask') else torch.ones(out.shape[0], dtype=torch.bool, device=self.device)
            
            if mask.sum() == 0:
                continue
            
            y = batch["txn"].y[mask]
            pred = out[mask]
            
            loss = self.loss_fn(pred, y)
            total_loss += loss.item()
            
            probs = torch.softmax(pred, dim=-1)[:, 1].cpu().numpy()
            all_preds.extend(pred.argmax(dim=-1).cpu().numpy())
            all_probs.extend(probs)
            all_labels.extend(y.cpu().numpy())
            num_batches += 1
        
        avg_loss = total_loss / max(num_batches, 1)
        
        metrics = {
            'f1': f1_score(all_labels, all_preds, average='binary', zero_division=0),
            'precision': precision_score(all_labels, all_preds, average='binary', zero_division=0),
            'recall': recall_score(all_labels, all_preds, average='binary', zero_division=0),
        }
        
        try:
            if len(set(all_labels)) > 1:
                metrics['auc'] = roc_auc_score(all_labels, all_probs)
            else:
                metrics['auc'] = 0.0
        except ValueError:
            metrics['auc'] = 0.0
        
        return avg_loss, metrics
    
    @torch.no_grad()
    def predict(self, test_loader):
        """Run predictions on test set."""
        self.model.eval()
        all_preds = []
        all_probs = []
        all_labels = []
        
        for batch in test_loader:
            batch = batch.to(self.device)
            
            x_dict = {nt: batch[nt].x for nt in batch.node_types if hasattr(batch[nt], 'x') and batch[nt].x is not None}
            edge_index_dict = {et: batch[et].edge_index for et in batch.edge_types if hasattr(batch[et], 'edge_index')}
            
            out = self.model(x_dict, edge_index_dict)
            
            mask = batch["txn"].test_mask if hasattr(batch["txn"], 'test_mask') else torch.ones(out.shape[0], dtype=torch.bool, device=self.device)
            
            if mask.sum() == 0:
                continue
            
            probs = torch.softmax(out[mask], dim=-1)[:, 1].cpu().numpy()
            preds = out[mask].argmax(dim=-1).cpu().numpy()
            labels = batch["txn"].y[mask].cpu().numpy()
            
            all_preds.extend(preds)
            all_probs.extend(probs)
            all_labels.extend(labels)
        
        return np.array(all_labels), np.array(all_preds), np.array(all_probs)
