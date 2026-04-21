"""
Graph Feature Extractor for IEEE-CIS dataset.
Extracts topological features from the transaction graph WITHOUT using GNN.
These features capture graph structure for use with traditional ML models.

Features extracted per transaction:
- Entity degree (how many txns share the same card/addr/email/device)
- Amount statistics per entity (mean, std of amt for same card1/addr1)
- Time gap features (time since last txn with same card1)
- Entity diversity (how many unique entities a txn connects to)
"""

import numpy as np
import polars as pl
from src.utils import timer


class GraphFeatureExtractor:
    """
    Extracts graph-derived features from the IEEE-CIS dataset.
    Uses only training data statistics to avoid data leakage.
    
    Usage:
        gfe = GraphFeatureExtractor()
        graph_features = gfe.fit_transform(df, train_mask)
    """
    
    # Entity columns to compute degree features
    DEGREE_COLS = ["card1", "card2", "card3", "addr1", "addr2",
                   "P_emaildomain", "R_emaildomain", "DeviceType", "DeviceInfo"]
    
    # Entity columns for amount aggregation
    AGG_COLS = ["card1", "addr1", "P_emaildomain"]
    
    # Column for time-gap features
    TIME_COL = "card1"
    
    def __init__(self):
        self.degree_maps = {}
        self.amt_stats = {}
        self._fitted = False
    
    @timer
    def fit_transform(self, df: pl.DataFrame, train_mask: np.ndarray = None) -> np.ndarray:
        """
        Fit on training data and transform entire dataset.
        
        Args:
            df: Full dataset with txn_index
            train_mask: Boolean mask for training rows (for leakage-free stats)
            
        Returns:
            numpy array of graph features, shape (N, num_graph_features)
        """
        if train_mask is None:
            train_mask = np.ones(df.height, dtype=bool)
        
        print("Extracting graph features...")
        
        # Fit statistics on training data only
        train_df = df.filter(pl.Series(train_mask))
        self._fit_degree_maps(train_df)
        self._fit_amt_stats(train_df)
        
        # Transform entire dataset
        features = []
        feat_names = []
        
        # 1. Degree features
        deg_feats, deg_names = self._extract_degree_features(df)
        features.append(deg_feats)
        feat_names.extend(deg_names)
        
        # 2. Amount aggregation features
        amt_feats, amt_names = self._extract_amt_features(df)
        features.append(amt_feats)
        feat_names.extend(amt_names)
        
        # 3. Time gap features
        time_feats, time_names = self._extract_time_features(df)
        features.append(time_feats)
        feat_names.extend(time_names)
        
        # 4. Entity diversity
        div_feats, div_names = self._extract_diversity_features(df)
        features.append(div_feats)
        feat_names.extend(div_names)
        
        X_graph = np.hstack(features).astype(np.float32)
        X_graph = np.nan_to_num(X_graph, nan=0.0, posinf=0.0, neginf=0.0)
        
        self._fitted = True
        self.feature_names = feat_names
        print(f"  Graph features: {X_graph.shape[1]} dimensions")
        print(f"  Feature names: {feat_names}")
        
        return X_graph
    
    def _fit_degree_maps(self, train_df: pl.DataFrame):
        """Compute entity degree from training data."""
        for col in self.DEGREE_COLS:
            if col not in train_df.columns:
                continue
            counts = (
                train_df.select(pl.col(col).cast(pl.Utf8).fill_null("__NULL__"))
                .group_by(col).len().rename({"len": "cnt"})
            )
            self.degree_maps[col] = dict(
                zip(counts[col].to_list(), counts["cnt"].to_list())
            )
    
    def _fit_amt_stats(self, train_df: pl.DataFrame):
        """Compute amount statistics per entity from training data."""
        for col in self.AGG_COLS:
            if col not in train_df.columns:
                continue
            stats = (
                train_df.select([
                    pl.col(col).cast(pl.Utf8).fill_null("__NULL__").alias(col),
                    pl.col("TransactionAmt").cast(pl.Float64).fill_null(0.0),
                ])
                .group_by(col).agg([
                    pl.col("TransactionAmt").mean().alias("amt_mean"),
                    pl.col("TransactionAmt").std().alias("amt_std"),
                    pl.col("TransactionAmt").max().alias("amt_max"),
                ])
            )
            self.amt_stats[col] = {}
            for row in stats.iter_rows(named=True):
                self.amt_stats[col][row[col]] = {
                    "mean": row["amt_mean"] or 0.0,
                    "std": row["amt_std"] or 0.0,
                    "max": row["amt_max"] or 0.0,
                }
    
    def _extract_degree_features(self, df: pl.DataFrame):
        """Map entity degrees to each transaction."""
        features = []
        names = []
        for col in self.DEGREE_COLS:
            if col not in df.columns or col not in self.degree_maps:
                continue
            deg_map = self.degree_maps[col]
            vals = df[col].cast(pl.Utf8).fill_null("__NULL__").to_list()
            deg = np.array([deg_map.get(v, 0) for v in vals], dtype=np.float32)
            features.append(np.log1p(deg).reshape(-1, 1))
            names.append(f"degree_{col}")
        return np.hstack(features) if features else np.zeros((df.height, 0)), names
    
    def _extract_amt_features(self, df: pl.DataFrame):
        """Map entity-level amount stats to each transaction."""
        features = []
        names = []
        for col in self.AGG_COLS:
            if col not in df.columns or col not in self.amt_stats:
                continue
            stats = self.amt_stats[col]
            vals = df[col].cast(pl.Utf8).fill_null("__NULL__").to_list()
            
            means = np.array([stats.get(v, {}).get("mean", 0) for v in vals], dtype=np.float32)
            stds = np.array([stats.get(v, {}).get("std", 0) for v in vals], dtype=np.float32)
            maxs = np.array([stats.get(v, {}).get("max", 0) for v in vals], dtype=np.float32)
            
            # Amount deviation from entity mean
            amt = df["TransactionAmt"].cast(pl.Float64).fill_null(0.0).to_numpy()
            dev = np.where(stds > 0, (amt - means) / (stds + 1e-6), 0.0)
            
            features.extend([
                means.reshape(-1, 1),
                stds.reshape(-1, 1),
                dev.reshape(-1, 1).astype(np.float32),
            ])
            names.extend([f"amt_mean_{col}", f"amt_std_{col}", f"amt_dev_{col}"])
        
        return np.hstack(features) if features else np.zeros((df.height, 0)), names
    
    def _extract_time_features(self, df: pl.DataFrame):
        """Extract time gap features (time since last txn with same entity)."""
        col = self.TIME_COL
        if col not in df.columns:
            return np.zeros((df.height, 0)), []
        
        # Sort by time, compute time gap within same entity group
        pdf = df.select([
            "txn_index",
            pl.col(col).cast(pl.Utf8).fill_null("__NULL__"),
            pl.col("TransactionDT").cast(pl.Float64).fill_null(0.0),
        ]).sort("TransactionDT")
        
        # Group by entity and compute time differences
        time_gaps = np.zeros(df.height, dtype=np.float32)
        txn_counts = np.zeros(df.height, dtype=np.float32)
        
        last_time = {}
        count_map = {}
        
        for row in pdf.iter_rows():
            idx, entity, t = row
            if entity in last_time:
                time_gaps[idx] = t - last_time[entity]
            count_map[entity] = count_map.get(entity, 0) + 1
            txn_counts[idx] = count_map[entity]
            last_time[entity] = t
        
        features = np.stack([
            np.log1p(time_gaps),
            np.log1p(txn_counts),
        ], axis=1)
        
        return features, [f"time_gap_{col}", f"txn_count_{col}"]
    
    def _extract_diversity_features(self, df: pl.DataFrame):
        """Count unique entities per transaction (entity diversity)."""
        entity_cols = [c for c in self.DEGREE_COLS if c in df.columns]
        
        # Count non-null entities per row
        non_null_count = np.zeros(df.height, dtype=np.float32)
        for col in entity_cols:
            mask = df[col].is_not_null().to_numpy().astype(np.float32)
            non_null_count += mask
        
        return non_null_count.reshape(-1, 1), ["entity_diversity"]
