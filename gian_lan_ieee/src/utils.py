"""
Utility functions for the IEEE-CIS Fraud Detection GNN project.
"""

import time
import functools
import torch
import numpy as np
from pathlib import Path
from src.config import DEVICE


def timer(func):
    """Decorator that prints the execution time of a function."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start
        print(f"[{func.__name__}] completed in {elapsed:.2f}s")
        return result
    return wrapper


def get_device():
    """Return the configured device (cuda or cpu)."""
    return DEVICE


def save_model(model, path: str, epoch: int = None, optimizer=None, metrics=None):
    """
    Save model checkpoint.
    
    Args:
        model: PyTorch model
        path: Save path
        epoch: Current epoch number
        optimizer: Optimizer state
        metrics: Dictionary of metrics
    """
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'epoch': epoch,
    }
    if optimizer is not None:
        checkpoint['optimizer_state_dict'] = optimizer.state_dict()
    if metrics is not None:
        checkpoint['metrics'] = metrics
    
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, path)
    print(f"Model saved to {path}")


def load_model(model, path: str, optimizer=None):
    """
    Load model checkpoint.
    
    Args:
        model: PyTorch model
        path: Checkpoint path
        optimizer: Optional optimizer to restore state
        
    Returns:
        Dictionary with epoch and metrics info
    """
    checkpoint = torch.load(path, map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    info = {
        'epoch': checkpoint.get('epoch', None),
        'metrics': checkpoint.get('metrics', None),
    }
    print(f"Model loaded from {path}")
    return info


class FocalLoss(torch.nn.Module):
    """
    Focal Loss for addressing class imbalance in fraud detection.
    
    Focal Loss = -alpha * (1 - p_t)^gamma * log(p_t)
    
    Args:
        alpha: Weighting factor for the rare class
        gamma: Focusing parameter  
        reduction: 'mean', 'sum', or 'none'
    """
    
    def __init__(self, alpha: float = 0.75, gamma: float = 2.0, reduction: str = 'mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = torch.nn.functional.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        
        # Apply alpha weighting
        alpha_t = torch.where(targets == 1, self.alpha, 1 - self.alpha)
        
        focal_loss = alpha_t * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss


class EarlyStopping:
    """
    Early stopping to prevent overfitting.
    
    Args:
        patience: Number of epochs to wait after last improvement
        min_delta: Minimum change to qualify as an improvement
        mode: 'min' for loss, 'max' for metrics like AUC
    """
    
    def __init__(self, patience: int = 10, min_delta: float = 0.0, mode: str = 'min'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
    
    def __call__(self, score: float) -> bool:
        if self.best_score is None:
            self.best_score = score
            return False
        
        if self.mode == 'min':
            improved = score < self.best_score - self.min_delta
        else:
            improved = score > self.best_score + self.min_delta
        
        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                return True
        
        return False


def print_separator(title: str = "", char: str = "=", length: int = 80):
    """Print a formatted separator line."""
    if title:
        print(f"\n{char * length}")
        print(title)
        print(f"{char * length}")
    else:
        print(char * length)
