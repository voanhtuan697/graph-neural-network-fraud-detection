"""
Feature Engineering module for IEEE-CIS Fraud Detection.
Handles feature extraction and transformation for graph node features.
"""

import numpy as np
import polars as pl
from sklearn.preprocessing import RobustScaler
from src.utils import timer


class FeatureEngineer:
    """
    Extracts and transforms features from the IEEE-CIS dataset
    for use as node features in the heterogeneous graph.
    
    Features include:
    - Transaction amount (z-score normalized, log-transformed)
    - Time features (hour, day_of_week with cyclical encoding)
    
    Usage:
        fe = FeatureEngineer(df)
        X, y = fe.build_features()
    """
    
    def __init__(self, df: pl.DataFrame):
        self.df = df
        self.scaler = RobustScaler()
    
    @timer
    def build_features(self):
        """
        Build feature matrix X and label vector y from the dataframe.
        
        Returns:
            X: numpy array of shape (num_transactions, num_features), float32
            y: numpy array of shape (num_transactions,), int64
        """
        print("Building transaction features...")
        
        # Amount features
        amt_np = self.df["TransactionAmt"].cast(pl.Float64).fill_null(0.0).to_numpy()
        mu = float(np.mean(amt_np))
        sd = float(np.std(amt_np))
        sd = sd if sd != 0 else 1.0
        amt_z = (amt_np - mu) / (sd + 1e-6)
        log_amt = np.log1p(amt_np)
        
        # Time features with cyclical encoding
        sec_np = self.df["TransactionDT"].cast(pl.Float64).fill_null(0.0).to_numpy()
        hour = (sec_np % 86400.0) / 3600.0
        dow = ((sec_np // 86400.0) % 7.0)
        
        # Stack features
        X = np.stack([
            amt_z,
            log_amt,
            np.sin(2 * np.pi * hour / 24.0),
            np.cos(2 * np.pi * hour / 24.0),
            np.sin(2 * np.pi * dow / 7.0),
            np.cos(2 * np.pi * dow / 7.0),
        ], axis=1).astype(np.float32)
        X = np.ascontiguousarray(X)
        
        # Labels
        y = self.df.select(
            pl.col("isFraud").fill_null(0).cast(pl.Int64)
        )["isFraud"].to_numpy()
        
        print(f"  Features shape: {X.shape}")
        print(f"  Labels shape: {y.shape} (fraud: {y.sum():,})")
        
        return X, y
