"""
Centralized configuration for the IEEE-CIS Fraud Detection GNN project.
All paths, hyperparameters, entity schemas, and random seeds are defined here.
"""

import os
import random
import numpy as np
import torch
from pathlib import Path

# ============================================================================
# Paths
# ============================================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "output"
MODEL_DIR = PROJECT_ROOT / "models"

EDA_OUTPUT_DIR = OUTPUT_DIR / "eda"
GRAPH_OUTPUT_DIR = OUTPUT_DIR / "graphs"
METRICS_OUTPUT_DIR = OUTPUT_DIR / "metrics"

# Data file paths
PATH_TRAIN_TRANSACTION = DATA_DIR / "train_transaction.csv"
PATH_TRAIN_IDENTITY = DATA_DIR / "train_identity.csv"
PATH_TEST_TRANSACTION = DATA_DIR / "test_transaction.csv"
PATH_TEST_IDENTITY = DATA_DIR / "test_identity.csv"

# Ensure output directories exist
for d in [EDA_OUTPUT_DIR, GRAPH_OUTPUT_DIR, METRICS_OUTPUT_DIR, MODEL_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ============================================================================
# Random Seed
# ============================================================================
SEED = 42


def set_seed(seed: int = SEED):
    """Set random seed for reproducibility across all libraries."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ============================================================================
# Device
# ============================================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================================
# Data Loading
# ============================================================================
# Columns to use from transaction table
USECOLS_TRANSACTION = [
    "TransactionID", "isFraud", "TransactionDT", "TransactionAmt",
    "card1", "card2", "card3", "card4", "card5", "card6",
    "addr1", "addr2", "P_emaildomain", "R_emaildomain"
]

# Columns to use from identity table
USECOLS_IDENTITY = [
    "TransactionID", "DeviceInfo", "DeviceType", "id_30", "id_31"
]

# Subsample fraction (set to None for full data, e.g., 0.25 for 25%)
SUBSAMPLE_FRAC = None

# ============================================================================
# Entity Schema for Graph Construction
# ============================================================================
# Format: (column_name_in_data, entity_type_name)
ENTITY_SCHEMA = [
    ("card1", "card1"),
    ("card2", "card2"),
    ("card3", "card3"),
    ("card4", "card4"),
    ("card5", "card5"),
    ("card6", "card6"),
    ("addr1", "addr1"),
    ("addr2", "addr2"),
    ("P_emaildomain", "p_email"),
    ("R_emaildomain", "r_email"),
    ("DeviceInfo", "device"),
    ("DeviceType", "devtype"),
    ("id_30", "os"),
    ("id_31", "browser"),
]

# Entity indexer settings
MIN_FREQ = 3           # Minimum frequency for entity to be included
KEEP_TOP_K = None      # Keep top K entities (None for all)
ADD_REVERSE_EDGES = True  # Add reverse edges to the graph

# ============================================================================
# Train/Val/Test Split
# ============================================================================
TRAIN_RATIO = 0.7
VAL_RATIO = 0.1
TEST_RATIO = 0.2

# ============================================================================
# GNN Hyperparameters
# ============================================================================
HIDDEN_DIM = 128
NUM_LAYERS = 2
NUM_HEADS = 4          # For GAT
DROPOUT = 0.3
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-5
NUM_EPOCHS = 50
PATIENCE = 10          # Early stopping patience

# NeighborLoader settings
NUM_NEIGHBORS = [15, 10]  # Number of neighbors per layer
BATCH_SIZE = 512

# Focal Loss settings
FOCAL_LOSS_ALPHA = 0.75
FOCAL_LOSS_GAMMA = 2.0

# ============================================================================
# Shared key edges (txn-txn)
# ============================================================================
TXN_TXN_SHARED_KEY = "card1"
TXN_TXN_MAX_DEGREE = 20
