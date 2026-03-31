"""
Heterogeneous Graph Construction Module
=========================================
Build a bipartite heterogeneous graph from Sparkov transaction data:
  - Node types: Customer (cc_num), Merchant, Category
  - Edge types: Customer <-> Merchant (TRANSACTS_AT), Merchant <-> Category (BELONGS_TO)
  - Transaction attributes aggregated as edge properties

IMPORTANT: Graph is built from TRAINING data only to prevent data leakage.
"""

import pandas as pd
import numpy as np
import networkx as nx
from typing import Tuple, Dict
from tqdm import tqdm


def build_heterogeneous_graph(
    df_train: pd.DataFrame,
) -> nx.Graph:
    """
    Build a heterogeneous bipartite graph from training transaction data.
    
    Node types and their prefixes:
      - Customer: 'C_<cc_num>'
      - Merchant: 'M_<merchant>'
      - Category: 'CAT_<category>'
    
    Edge types:
      - Customer <-> Merchant: aggregated transaction statistics
      - Merchant <-> Category: membership relation
    
    Parameters
    ----------
    df_train : pd.DataFrame
        Training data with columns: cc_num, merchant, category, amt, unix_time, is_fraud
    
    Returns
    -------
    nx.Graph
        Heterogeneous graph.
    """
    print("=" * 60)
    print("[STEP] Building Heterogeneous Graph from Training Data")
    print("=" * 60)

    G = nx.Graph()

    # --- Add Customer nodes ---
    customers = df_train['cc_num'].unique()
    for c in customers:
        G.add_node(f"C_{c}", node_type='customer', cc_num=c)
    print(f"[INFO] Added {len(customers)} Customer nodes")

    # --- Add Merchant nodes ---
    merchants = df_train['merchant'].unique()
    for m in merchants:
        G.add_node(f"M_{m}", node_type='merchant', merchant=m)
    print(f"[INFO] Added {len(merchants)} Merchant nodes")

    # --- Add Category nodes ---
    categories = df_train['category'].unique()
    for cat in categories:
        G.add_node(f"CAT_{cat}", node_type='category', category=cat)
    print(f"[INFO] Added {len(categories)} Category nodes")

    # --- Add Customer <-> Merchant edges (aggregated) ---
    print("[INFO] Aggregating Customer-Merchant transaction statistics...")
    cm_agg = df_train.groupby(['cc_num', 'merchant']).agg(
        txn_count=('amt', 'count'),
        total_amt=('amt', 'sum'),
        avg_amt=('amt', 'mean'),
        std_amt=('amt', 'std'),
        fraud_count=('is_fraud', 'sum'),
        min_time=('unix_time', 'min'),
        max_time=('unix_time', 'max'),
    ).reset_index()
    cm_agg['std_amt'] = cm_agg['std_amt'].fillna(0)

    for _, row in tqdm(cm_agg.iterrows(), total=len(cm_agg),
                       desc="Adding C-M edges"):
        c_node = f"C_{row['cc_num']}"
        m_node = f"M_{row['merchant']}"
        G.add_edge(
            c_node, m_node,
            edge_type='transacts_at',
            txn_count=row['txn_count'],
            total_amt=row['total_amt'],
            avg_amt=row['avg_amt'],
            std_amt=row['std_amt'],
            fraud_count=row['fraud_count'],
            min_time=row['min_time'],
            max_time=row['max_time'],
        )

    print(f"[INFO] Added {len(cm_agg)} Customer-Merchant edges")

    # --- Add Merchant <-> Category edges ---
    mc_pairs = df_train[['merchant', 'category']].drop_duplicates()
    for _, row in mc_pairs.iterrows():
        m_node = f"M_{row['merchant']}"
        cat_node = f"CAT_{row['category']}"
        G.add_edge(m_node, cat_node, edge_type='belongs_to')

    print(f"[INFO] Added {len(mc_pairs)} Merchant-Category edges")

    # --- Summary ---
    print(f"\n[INFO] Graph summary:")
    print(f"  Total nodes: {G.number_of_nodes()}")
    print(f"  Total edges: {G.number_of_edges()}")
    print(f"  Customer nodes: {len(customers)}")
    print(f"  Merchant nodes: {len(merchants)}")
    print(f"  Category nodes: {len(categories)}")
    print(f"  Density: {nx.density(G):.6f}")

    return G


def get_graph_stats(G: nx.Graph) -> Dict:
    """Get detailed statistics of the heterogeneous graph."""
    # Count by node type
    node_types = {}
    for node, data in G.nodes(data=True):
        nt = data.get('node_type', 'unknown')
        node_types[nt] = node_types.get(nt, 0) + 1

    # Count by edge type
    edge_types = {}
    for u, v, data in G.edges(data=True):
        et = data.get('edge_type', 'unknown')
        edge_types[et] = edge_types.get(et, 0) + 1

    # Degree stats
    degrees = [d for _, d in G.degree()]

    stats = {
        'num_nodes': G.number_of_nodes(),
        'num_edges': G.number_of_edges(),
        'node_types': node_types,
        'edge_types': edge_types,
        'avg_degree': np.mean(degrees),
        'max_degree': np.max(degrees),
        'min_degree': np.min(degrees),
        'density': nx.density(G),
        'num_connected_components': nx.number_connected_components(G),
    }
    return stats


def get_nodes_by_type(G: nx.Graph, node_type: str) -> list:
    """Get all nodes of a specific type."""
    return [n for n, d in G.nodes(data=True) if d.get('node_type') == node_type]
