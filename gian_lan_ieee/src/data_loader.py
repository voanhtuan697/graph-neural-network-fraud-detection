"""
Data Loader module for the IEEE-CIS Fraud Detection dataset.
Handles loading, merging, and basic preprocessing of transaction and identity data.
"""

import polars as pl
import numpy as np
from pathlib import Path
from src.config import (
    PATH_TRAIN_TRANSACTION, PATH_TRAIN_IDENTITY,
    USECOLS_TRANSACTION, USECOLS_IDENTITY,
    SUBSAMPLE_FRAC, SEED
)
from src.utils import timer, print_separator


class IEEECISDataLoader:
    """
    Loads and merges IEEE-CIS Fraud Detection dataset using Polars
    for high-performance data processing.
    
    Usage:
        loader = IEEECISDataLoader()
        df = loader.load()
    """
    
    def __init__(
        self,
        path_transaction: str = None,
        path_identity: str = None,
        usecols_tran: list = None,
        usecols_id: list = None,
        subsample_frac: float = None,
        seed: int = SEED,
    ):
        self.path_transaction = Path(path_transaction or PATH_TRAIN_TRANSACTION)
        self.path_identity = Path(path_identity or PATH_TRAIN_IDENTITY)
        self.usecols_tran = usecols_tran or USECOLS_TRANSACTION
        self.usecols_id = usecols_id or USECOLS_IDENTITY
        self.subsample_frac = subsample_frac if subsample_frac is not None else SUBSAMPLE_FRAC
        self.seed = seed
        self.df = None
    
    @timer
    def load(self) -> pl.DataFrame:
        """
        Load and merge transaction + identity data.
        
        Returns:
            Polars DataFrame with merged data and txn_index column
        """
        print_separator("LOADING DATA")
        
        # Lazy scan for efficiency
        lt = pl.scan_csv(str(self.path_transaction), infer_schema_length=2048).select(self.usecols_tran)
        li = pl.scan_csv(str(self.path_identity), infer_schema_length=2048).select(self.usecols_id)
        
        # Left join on TransactionID
        lf = lt.join(li, on="TransactionID", how="left")
        df = lf.collect(streaming=False)
        
        # Optional subsampling
        if self.subsample_frac and self.subsample_frac < 1.0:
            print(f"Sampling {self.subsample_frac * 100}% of data...")
            df = df.sample(fraction=self.subsample_frac, with_replacement=False, seed=self.seed)
        
        # Add row index
        df = df.with_row_index(name="txn_index")
        
        self.df = df
        self._print_summary()
        
        return df
    
    def _print_summary(self):
        """Print dataset summary statistics."""
        if self.df is None:
            return
        
        df = self.df
        num_txn = df.height
        fraud_count = int(df["isFraud"].sum())
        fraud_rate = fraud_count / num_txn * 100
        
        print(f"Loaded {num_txn:,} transactions")
        print(f"Fraud rate: {fraud_rate:.2f}% ({fraud_count:,} fraud transactions)")
        print(f"txn_index range: 0 to {num_txn - 1}")
        print(f"Columns: {df.columns}")
        print_separator()
    
    def get_fraud_stats(self) -> dict:
        """Return fraud statistics as a dictionary."""
        if self.df is None:
            raise ValueError("Data not loaded yet. Call load() first.")
        
        df = self.df
        num_txn = df.height
        fraud_count = int(df["isFraud"].sum())
        non_fraud_count = num_txn - fraud_count
        
        return {
            "total_transactions": num_txn,
            "fraud_count": fraud_count,
            "non_fraud_count": non_fraud_count,
            "fraud_rate": fraud_count / num_txn,
            "imbalance_ratio": non_fraud_count / max(fraud_count, 1),
        }
    
    def get_missing_stats(self) -> pl.DataFrame:
        """Return missing value statistics for each column."""
        if self.df is None:
            raise ValueError("Data not loaded yet. Call load() first.")
        
        df = self.df
        stats = []
        for col in df.columns:
            null_count = df[col].null_count()
            total = df.height
            stats.append({
                "column": col,
                "null_count": null_count,
                "null_percentage": null_count / total * 100,
            })
        
        return pl.from_dicts(stats).sort("null_percentage", descending=True)
