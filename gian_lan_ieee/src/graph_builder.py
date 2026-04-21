"""
Graph Builder module for IEEE-CIS Fraud Detection.
Constructs a heterogeneous graph (HeteroData) from the dataset using PyTorch Geometric.
"""

import json
import numpy as np
import polars as pl
import torch
from pathlib import Path
from torch_geometric.data import HeteroData

from src.config import (
    ENTITY_SCHEMA, MIN_FREQ, KEEP_TOP_K, ADD_REVERSE_EDGES,
    TRAIN_RATIO, VAL_RATIO, DEVICE,
    TXN_TXN_SHARED_KEY, TXN_TXN_MAX_DEGREE,
)
from src.feature_engineering import FeatureEngineer
from src.utils import timer, print_separator


class HeteroGraphBuilder:
    """
    Builds a PyTorch Geometric HeteroData object from the IEEE-CIS dataset.
    
    The graph includes:
    - Transaction nodes with features
    - Entity nodes (card, address, email, device, etc.)
    - Bipartite edges (txn -> entity)
    - Shared-identity edges (txn -> txn via same card1)
    - Reverse edges for message passing
    - Time-based train/val/test masks
    
    Usage:
        builder = HeteroGraphBuilder(df)
        data = builder.build()
    """
    
    def __init__(
        self,
        df: pl.DataFrame,
        entity_schema: list = None,
        min_freq: int = MIN_FREQ,
        keep_top_k: int = KEEP_TOP_K,
        add_reverse_edges: bool = ADD_REVERSE_EDGES,
        device: torch.device = DEVICE,
    ):
        self.df = df
        self.entity_schema = entity_schema or ENTITY_SCHEMA
        self.min_freq = min_freq
        self.keep_top_k = keep_top_k
        self.add_reverse_edges = add_reverse_edges
        self.device = device
        self.idx_map = {}
        self.data = None
    
    @timer
    def build(self) -> HeteroData:
        """
        Build the complete heterogeneous graph.
        
        Returns:
            HeteroData object with nodes, edges, features, labels, and masks
        """
        print_separator("BUILDING HETEROGENEOUS GRAPH")
        
        data = HeteroData()
        num_txn = self.df.height
        data["txn"].num_nodes = num_txn
        print(f"Transaction nodes: {num_txn:,}")
        
        # Build features and labels
        fe = FeatureEngineer(self.df)
        X, y = fe.build_features()
        data["txn"].x = torch.from_numpy(X).contiguous()
        data["txn"].y = torch.tensor(y, dtype=torch.long).contiguous()
        
        # Build entity indexers
        self._build_entity_indexers()
        
        # Add entity nodes
        self._add_entity_nodes(data)
        
        # Build edges
        self._build_edges(data, num_txn)
        
        # Build txn-txn edges
        self._build_shared_key_edges(data, num_txn)
        
        # Create time-based masks
        self._create_time_masks(data, num_txn)
        
        # Move to device
        data = data.to(self.device)
        
        # Verify
        self._verify_data(data)
        
        self.data = data
        return data
    
    def _build_entity_indexers(self):
        """Build entity indexers mapping entity values to unique IDs."""
        print("\nBUILDING ENTITY INDEXERS")
        
        for col, etype in self.entity_schema:
            vc = (
                self.df.select(pl.col(col).cast(pl.Utf8).alias(col))
                .with_columns(pl.col(col).fill_null(""))
                .group_by(col).len().rename({"len": "cnt"})
                .filter(pl.col(col) != "")
                .sort("cnt", descending=True)
            )
            
            if self.keep_top_k:
                vc = vc.head(self.keep_top_k)
            if self.min_freq and self.min_freq > 1:
                vc = vc.filter(pl.col("cnt") >= self.min_freq)
            
            vc = vc.with_row_index(name="eid").select([col, "eid"])
            self.idx_map[etype] = dict(zip(vc[col].to_list(), vc["eid"].to_list()))
            print(f"  {etype}: {len(self.idx_map[etype]):,} unique values")
    
    def _add_entity_nodes(self, data: HeteroData):
        """Add entity nodes with dummy features to the HeteroData."""
        print("\nAdding entity nodes...")
        for _, etype in self.entity_schema:
            num_nodes = len(self.idx_map.get(etype, {}))
            data[etype].num_nodes = num_nodes
            # Assign dummy features so NeighborLoader always produces valid x tensors
            data[etype].x = torch.zeros(num_nodes, 1, dtype=torch.float32)
            print(f"  {etype}: {num_nodes:,} nodes, x={data[etype].x.shape}")
        print(f"  Added {len(self.entity_schema)} entity types")
    
    def _build_edges_for(self, col: str, etype: str, num_txn: int):
        """Build edges for a specific entity type."""
        mapping = self.idx_map.get(etype, {})
        if not mapping:
            return None
        
        map_df = pl.DataFrame({col: list(mapping.keys()), "eid": list(mapping.values())})
        edges_df = (
            self.df.select(["txn_index", pl.col(col).cast(pl.Utf8).fill_null("").alias(col)])
            .join(map_df, on=col, how="inner")
            .select(["txn_index", "eid"])
        )
        
        if edges_df.height == 0:
            return None
        
        src = torch.tensor(edges_df["txn_index"].to_numpy(), dtype=torch.long)
        dst = torch.tensor(edges_df["eid"].to_numpy(), dtype=torch.long)
        
        # Safety check
        assert src.max() < num_txn, f"Source index {src.max()} >= num_txn {num_txn}"
        assert dst.max() < len(mapping), f"Dest index {dst.max()} >= num_entities {len(mapping)}"
        
        return torch.stack([src, dst], dim=0).contiguous()
    
    def _build_edges(self, data: HeteroData, num_txn: int):
        """Build all txn->entity edges."""
        print("\nBUILDING EDGES")
        
        for col, etype in self.entity_schema:
            eidx = self._build_edges_for(col, etype, num_txn)
            if eidx is None:
                continue
            
            key = ("txn", f"has_{etype}", etype)
            data[key].edge_index = eidx
            
            if self.add_reverse_edges:
                rkey = (etype, f"rev_has_{etype}", "txn")
                data[rkey].edge_index = torch.stack([eidx[1], eidx[0]], dim=0).contiguous()
            
            print(f"  {key}: {eidx.shape[1]:,} edges")
    
    def _build_shared_key_edges(
        self, data: HeteroData, num_txn: int,
        key: str = TXN_TXN_SHARED_KEY,
        max_degree: int = TXN_TXN_MAX_DEGREE,
    ):
        """Build txn-txn edges for transactions sharing the same entity."""
        print(f"\nBuilding txn-txn edges on shared {key}...")
        
        groups = (
            self.df.select(["txn_index", pl.col(key).cast(pl.Utf8).fill_null("").alias(key)])
            .to_pandas()
        )
        
        by = {}
        for idx, val in zip(groups["txn_index"].values, groups[key].values):
            if not val:
                continue
            by.setdefault(val, []).append(int(idx))
        
        rows, cols = [], []
        for val, idxs in by.items():
            if len(idxs) < 2:
                continue
            L = len(idxs)
            for i in range(L):
                for j in range(max(0, i - max_degree), min(L, i + max_degree + 1)):
                    if j == i:
                        continue
                    rows.append(idxs[i])
                    cols.append(idxs[j])
        
        if rows:
            edge_index = torch.tensor([rows, cols], dtype=torch.long)
            
            # Safety check
            max_idx = edge_index.max().item()
            assert max_idx < num_txn, f"txn-txn edge index {max_idx} >= num_txn {num_txn}"
            
            edge_index = torch.unique(edge_index, dim=1).contiguous()
            rel = ("txn", f"same_{key}", "txn")
            data[rel].edge_index = edge_index
            print(f"  {rel}: {edge_index.shape[1]:,} edges")
    
    def _create_time_masks(self, data: HeteroData, num_txn: int):
        """Create time-based train/val/test masks."""
        print("\nCREATING TRAIN/VAL/TEST SPLITS")
        
        time_col = self.df["TransactionDT"].cast(pl.Float64)
        t1 = float(time_col.quantile(TRAIN_RATIO))
        t2 = float(time_col.quantile(TRAIN_RATIO + VAL_RATIO))
        
        times = time_col.fill_null(t1).to_numpy()
        n = len(times)
        
        train_mask = torch.from_numpy((times <= t1)).bool()
        val_mask = torch.from_numpy((times > t1) & (times <= t2)).bool()
        test_mask = torch.from_numpy((times > t2)).bool()
        
        # Safety fallback for random split
        if train_mask.sum() == 0 or val_mask.sum() == 0 or test_mask.sum() == 0:
            idx = torch.randperm(n)
            n_train = int(n * TRAIN_RATIO)
            n_val = int(n * VAL_RATIO)
            
            train_mask = torch.zeros(n, dtype=torch.bool)
            train_mask[idx[:n_train]] = True
            val_mask = torch.zeros(n, dtype=torch.bool)
            val_mask[idx[n_train:n_train + n_val]] = True
            test_mask = torch.zeros(n, dtype=torch.bool)
            test_mask[idx[n_train + n_val:]] = True
        
        data["txn"].train_mask = train_mask
        data["txn"].val_mask = val_mask
        data["txn"].test_mask = test_mask
        
        print(f"  Train: {train_mask.sum():,} ({train_mask.sum() / n * 100:.1f}%)")
        print(f"  Val:   {val_mask.sum():,} ({val_mask.sum() / n * 100:.1f}%)")
        print(f"  Test:  {test_mask.sum():,} ({test_mask.sum() / n * 100:.1f}%)")
    
    def _verify_data(self, data: HeteroData):
        """Verify the integrity of the HeteroData object."""
        print_separator("VERIFYING DATA INTEGRITY")
        
        print(f"✓ Node types: {len(data.node_types)}")
        print(f"✓ Edge types: {len(data.edge_types)}")
        
        for et in data.edge_types:
            ei = data[et].edge_index
            src_type, rel_name, dst_type = et
            
            if ei.numel() > 0:
                src_max = ei[0].max().item()
                dst_max = ei[1].max().item()
                src_nodes = data[src_type].num_nodes
                dst_nodes = data[dst_type].num_nodes
                
                status = "✓"
                if src_max >= src_nodes or dst_max >= dst_nodes:
                    status = "✗ ERROR"
                
                print(f"  {status} {et}: {ei.shape[1]:,} edges")
        
        print("\n✓ Data verification complete")
        print_separator()
    
    def save_indexers(self, path: str):
        """Save entity indexers to JSON."""
        Path(path).write_text(json.dumps(self.idx_map))
        print(f"Saved indexers to {path}")
    
    def load_indexers(self, path: str):
        """Load entity indexers from JSON."""
        self.idx_map = json.loads(Path(path).read_text())
        print(f"Loaded indexers from {path}")
