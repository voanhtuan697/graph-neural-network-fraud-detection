"""
Graph Feature Extraction Module
=================================
Extract 12 groups of graph-based features from the heterogeneous graph.
Features are computed at Customer/Merchant level, then mapped to each transaction.

Groups:
  1. User unique merchants         (Customer)
  2. Transaction frequency         (Customer)
  3. Avg/Std amount                (Customer)
  4. Merchant fraud rate           (Merchant) - Bayesian smoothed, training labels only
  5. Merchant degree               (Merchant)
  6. Degree centrality             (Customer)
  7. PageRank                      (Customer)
  8. Community detection (Louvain) (Both) - encoded as community size
  9. Neighbor fraud ratio          (Customer) - training labels only
 10. Shared fraud entity count     (Merchant) - training labels only
 11. Time-based burst              (Transaction)
 12. Sequence pattern              (Transaction)

IMPORTANT: Features using is_fraud labels (4, 9, 10) are computed from TRAINING
data only. Test transactions reuse training-derived node-level features.
Merchant fraud rate uses strong Bayesian smoothing to prevent overfitting.

For the COMBINED scenario (tabular + graph), label-dependent features use
leave-one-out encoding for training data to prevent target leakage.
"""

import pandas as pd
import numpy as np
import networkx as nx
import community as community_louvain
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm
from typing import Dict, Tuple, Optional


# =============================================================================
# Node-level feature computation (computed once on training graph)
# =============================================================================

def compute_customer_features(
    G: nx.Graph,
    df_train: pd.DataFrame
) -> pd.DataFrame:
    """
    Compute customer-level features from graph + training data.
    Returns DataFrame indexed by cc_num.
    """
    print("[INFO] Computing customer-level features...")

    customers = df_train['cc_num'].unique()

    # --- Feature 1: Unique merchants ---
    user_merchants = df_train.groupby('cc_num')['merchant'].nunique()

    # --- Feature 2: Transaction frequency (txn / day) ---
    df_tmp = df_train.copy()
    df_tmp['date'] = pd.to_datetime(df_tmp['trans_date_trans_time']).dt.date
    user_days = df_tmp.groupby('cc_num')['date'].nunique()
    user_txn_count = df_train.groupby('cc_num').size()
    user_freq = user_txn_count / user_days.clip(lower=1)

    # --- Feature 3: Avg/Std amount ---
    user_amt_stats = df_train.groupby('cc_num')['amt'].agg(['mean', 'std'])
    user_amt_stats.columns = ['user_avg_amt', 'user_std_amt']
    user_amt_stats['user_std_amt'] = user_amt_stats['user_std_amt'].fillna(0)

    # --- Feature 6: Degree centrality ---
    print("  [6/12] Computing degree centrality...")
    degree_centrality = nx.degree_centrality(G)

    # --- Feature 7: PageRank ---
    print("  [7/12] Computing PageRank...")
    pagerank = nx.pagerank(G, max_iter=100, tol=1e-06)

    # --- Feature 8: Community detection (Louvain) ---
    print("  [8/12] Computing community detection (Louvain)...")
    communities = community_louvain.best_partition(G, random_state=42)
    community_sizes = {}
    for node, comm in communities.items():
        community_sizes[comm] = community_sizes.get(comm, 0) + 1

    # --- Feature 9: Neighbor fraud ratio (Bayesian smoothed) ---
    print("  [9/12] Computing neighbor fraud ratio...")
    global_fraud_rate = df_train['is_fraud'].mean()
    C = 100
    merchant_fraud = df_train.groupby('merchant').agg(
        fraud_txn=('is_fraud', 'sum'),
        total_txn=('is_fraud', 'count')
    )
    merchant_fraud['fraud_rate_smoothed'] = (
        merchant_fraud['fraud_txn'] + C * global_fraud_rate
    ) / (merchant_fraud['total_txn'] + C)

    neighbor_fraud_ratio = {}
    for c in customers:
        c_node = f"C_{c}"
        if c_node not in G:
            neighbor_fraud_ratio[c] = global_fraud_rate
            continue
        neighbors = list(G.neighbors(c_node))
        merchant_neighbors = [
            n for n in neighbors
            if G.nodes[n].get('node_type') == 'merchant'
        ]
        if len(merchant_neighbors) == 0:
            neighbor_fraud_ratio[c] = global_fraud_rate
            continue
        fraud_rates = []
        for mn in merchant_neighbors:
            m_name = G.nodes[mn].get('merchant', '')
            if m_name in merchant_fraud.index:
                fraud_rates.append(merchant_fraud.loc[m_name, 'fraud_rate_smoothed'])
            else:
                fraud_rates.append(global_fraud_rate)
        neighbor_fraud_ratio[c] = np.mean(fraud_rates)

    # --- Assemble customer features ---
    cust_feats = pd.DataFrame(index=customers)
    cust_feats.index.name = 'cc_num'
    cust_feats['user_unique_merchants'] = user_merchants
    cust_feats['user_txn_frequency'] = user_freq
    cust_feats['user_avg_amt'] = user_amt_stats['user_avg_amt']
    cust_feats['user_std_amt'] = user_amt_stats['user_std_amt']
    cust_feats['customer_degree_centrality'] = cust_feats.index.map(
        lambda c: degree_centrality.get(f"C_{c}", 0.0)
    )
    cust_feats['customer_pagerank'] = cust_feats.index.map(
        lambda c: pagerank.get(f"C_{c}", 0.0)
    )
    cust_feats['customer_community_size'] = cust_feats.index.map(
        lambda c: community_sizes.get(communities.get(f"C_{c}", -1), 0)
    )
    cust_feats['customer_neighbor_fraud_ratio'] = cust_feats.index.map(
        lambda c: neighbor_fraud_ratio.get(c, global_fraud_rate)
    )

    cust_feats = cust_feats.fillna(0)
    print(f"[INFO] Customer features shape: {cust_feats.shape}")
    return cust_feats


def compute_merchant_features(
    G: nx.Graph,
    df_train: pd.DataFrame
) -> pd.DataFrame:
    """
    Compute merchant-level features from graph + training data.
    Uses strong Bayesian smoothing for fraud rate to prevent overfitting.
    Returns DataFrame indexed by merchant name.
    """
    print("[INFO] Computing merchant-level features...")

    # --- Feature 4: Merchant fraud rate (Bayesian smoothed) ---
    print("  [4/12] Computing merchant fraud rate (Bayesian smoothed)...")
    global_fraud_rate = df_train['is_fraud'].mean()
    C = 100  # strong smoothing

    m_raw = df_train.groupby('merchant').agg(
        fraud_count=('is_fraud', 'sum'),
        total_count=('is_fraud', 'count'),
        merchant_total_amt=('amt', 'sum'),
    )

    m_raw['merchant_fraud_rate'] = (
        m_raw['fraud_count'] + C * global_fraud_rate
    ) / (m_raw['total_count'] + C)

    m_stats = pd.DataFrame(index=m_raw.index)
    m_stats['merchant_fraud_rate'] = m_raw['merchant_fraud_rate']
    m_stats['merchant_txn_count'] = m_raw['total_count']
    m_stats['merchant_total_amt'] = m_raw['merchant_total_amt']

    # --- Feature 5: Merchant degree (unique customers) ---
    print("  [5/12] Computing merchant degree...")
    m_degree = df_train.groupby('merchant')['cc_num'].nunique()
    m_stats['merchant_degree'] = m_degree

    # --- Graph metrics ---
    degree_centrality = nx.degree_centrality(G)
    pagerank = nx.pagerank(G, max_iter=100, tol=1e-06)
    communities = community_louvain.best_partition(G, random_state=42)
    community_sizes = {}
    for node, comm in communities.items():
        community_sizes[comm] = community_sizes.get(comm, 0) + 1

    m_stats['merchant_degree_centrality'] = m_stats.index.map(
        lambda m: degree_centrality.get(f"M_{m}", 0.0)
    )
    m_stats['merchant_pagerank'] = m_stats.index.map(
        lambda m: pagerank.get(f"M_{m}", 0.0)
    )
    m_stats['merchant_community_size'] = m_stats.index.map(
        lambda m: community_sizes.get(communities.get(f"M_{m}", -1), 0)
    )

    # --- Feature 10: Shared fraud entity count ---
    print("  [10/12] Computing shared fraud entity count...")
    fraud_txns = df_train[df_train['is_fraud'] == 1]
    fraud_users_per_merchant = fraud_txns.groupby('merchant')['cc_num'].nunique()
    m_stats['merchant_shared_fraud_users'] = fraud_users_per_merchant.reindex(
        m_stats.index
    ).fillna(0).astype(int)

    m_stats = m_stats.fillna(0)
    m_stats.index.name = 'merchant'
    print(f"[INFO] Merchant features shape: {m_stats.shape}")
    return m_stats


def compute_temporal_features(
    df: pd.DataFrame,
    window_hours: int = 1
) -> pd.DataFrame:
    """
    Compute transaction-level temporal features.
    Feature 11: Time-based burst
    Feature 12: Sequence pattern
    """
    print(f"  [11-12/12] Computing temporal features (window={window_hours}h)...")

    df = df.copy()
    if not pd.api.types.is_datetime64_any_dtype(df['trans_date_trans_time']):
        df['trans_date_trans_time'] = pd.to_datetime(df['trans_date_trans_time'])

    df = df.sort_values(['cc_num', 'trans_date_trans_time'])

    window_sec = window_hours * 3600
    burst_counts = []
    seq_lengths = []

    for cc_num, group in tqdm(df.groupby('cc_num'),
                              desc="Temporal features",
                              total=df['cc_num'].nunique()):
        times = group['unix_time'].values
        merchants = group['merchant'].values

        for i in range(len(group)):
            current_time = times[i]
            mask = (times >= current_time - window_sec) & (times <= current_time)
            burst_counts.append(mask.sum())
            seq_lengths.append(len(set(merchants[mask])))

    result = pd.DataFrame({
        'user_txn_burst': burst_counts,
        'user_merchant_sequence_len': seq_lengths
    }, index=df.index)

    return result


# =============================================================================
# Leave-One-Out Target Encoding for label-dependent features
# =============================================================================

def compute_loo_merchant_fraud_rate(df: pd.DataFrame) -> pd.Series:
    """
    Compute leave-one-out merchant fraud rate for training data.
    For each transaction, the fraud rate is computed excluding that transaction.
    This prevents the model from memorizing the exact label.
    """
    print("[INFO] Computing LOO merchant fraud rate for training data...")
    global_fraud_rate = df['is_fraud'].mean()
    C = 50  # smoothing

    merchant_stats = df.groupby('merchant')['is_fraud'].agg(['sum', 'count'])
    merchant_stats.columns = ['total_fraud', 'total_count']

    loo_rates = []
    for idx, row in df.iterrows():
        m = row['merchant']
        total_fraud = merchant_stats.loc[m, 'total_fraud']
        total_count = merchant_stats.loc[m, 'total_count']
        # Leave-one-out: subtract current observation
        loo_fraud = total_fraud - row['is_fraud']
        loo_count = total_count - 1
        if loo_count <= 0:
            loo_rates.append(global_fraud_rate)
        else:
            rate = (loo_fraud + C * global_fraud_rate) / (loo_count + C)
            loo_rates.append(rate)

    return pd.Series(loo_rates, index=df.index, name='merchant_fraud_rate_loo')


# =============================================================================
# Main extraction function
# =============================================================================

def extract_graph_features(
    G: nx.Graph,
    df_train: pd.DataFrame,
    df_target: pd.DataFrame,
    customer_features: pd.DataFrame = None,
    merchant_features: pd.DataFrame = None,
    scaler: StandardScaler = None,
    fit_scaler: bool = False,
) -> Tuple[pd.DataFrame, Optional[StandardScaler]]:
    """
    Extract all graph features for a target DataFrame (train or test).
    Features are scaled with StandardScaler for proper combination with tabular features.
    """
    print("=" * 60)
    print("[STEP] Extracting Graph Features")
    print("=" * 60)

    if customer_features is None:
        customer_features = compute_customer_features(G, df_train)
    if merchant_features is None:
        merchant_features = compute_merchant_features(G, df_train)

    # --- Map customer features to each transaction ---
    print("[INFO] Mapping customer features to transactions...")
    cust_cols = customer_features.columns.tolist()
    cust_mapped = df_target[['cc_num']].merge(
        customer_features, left_on='cc_num', right_index=True, how='left'
    )[cust_cols]
    cust_mapped.index = df_target.index

    cust_medians = customer_features.median()
    cust_mapped = cust_mapped.fillna(cust_medians)

    # --- Map merchant features to each transaction ---
    print("[INFO] Mapping merchant features to transactions...")
    merch_cols = merchant_features.columns.tolist()
    merch_mapped = df_target[['merchant']].merge(
        merchant_features, left_on='merchant', right_index=True, how='left'
    )[merch_cols]
    merch_mapped.index = df_target.index

    merch_medians = merchant_features.median()
    merch_mapped = merch_mapped.fillna(merch_medians)

    # --- Temporal features ---
    temporal_feats = compute_temporal_features(df_target)
    temporal_feats = temporal_feats.reindex(df_target.index)

    # --- Combine all graph features ---
    graph_features = pd.concat([cust_mapped, merch_mapped, temporal_feats], axis=1)

    # --- Scale graph features ---
    if fit_scaler:
        scaler = StandardScaler()
        graph_features_scaled = pd.DataFrame(
            scaler.fit_transform(graph_features),
            columns=graph_features.columns,
            index=graph_features.index
        )
        print("[INFO] Fitted and applied StandardScaler to graph features")
    elif scaler is not None:
        graph_features_scaled = pd.DataFrame(
            scaler.transform(graph_features),
            columns=graph_features.columns,
            index=graph_features.index
        )
        print("[INFO] Applied pre-fitted StandardScaler to graph features")
    else:
        graph_features_scaled = graph_features
        print("[INFO] No scaling applied to graph features")

    print(f"\n[INFO] Total graph features: {graph_features_scaled.shape[1]}")
    print(f"[INFO] Feature names: {list(graph_features_scaled.columns)}")
    print(f"[INFO] Shape: {graph_features_scaled.shape}")

    return graph_features_scaled, scaler


def combine_features(
    X_tabular: pd.DataFrame,
    graph_features: pd.DataFrame
) -> pd.DataFrame:
    """Combine tabular and graph features by positional alignment."""
    tab_reset = X_tabular.reset_index(drop=True)
    graph_reset = graph_features.reset_index(drop=True)
    X_combined = pd.concat([tab_reset, graph_reset], axis=1)

    nan_count = X_combined.isna().sum().sum()
    if nan_count > 0:
        print(f"[WARN] {nan_count} NaN values detected after combining features! Filling with 0.")
        X_combined = X_combined.fillna(0)

    print(f"[INFO] Combined features: {X_combined.shape[1]} "
          f"(tabular: {X_tabular.shape[1]}, graph: {graph_features.shape[1]})")
    return X_combined


def analyze_graph_features_by_class(
    graph_features: pd.DataFrame,
    y: pd.Series
) -> pd.DataFrame:
    """Compare graph feature distributions between fraud and legit classes."""
    df = graph_features.copy()
    df['is_fraud'] = y.values

    print("\n" + "=" * 90)
    print("  Graph Feature Analysis: Fraud vs Legit")
    print("=" * 90)
    print(f"  {'Feature':<35} | {'Legit Mean':>12} | {'Fraud Mean':>12} | {'Ratio':>8}")
    print("-" * 90)

    rows = []
    for col in graph_features.columns:
        fraud_mean = df[df['is_fraud'] == 1][col].mean()
        legit_mean = df[df['is_fraud'] == 0][col].mean()
        ratio = fraud_mean / legit_mean if legit_mean != 0 else float('inf')
        print(f"  {col:<35} | {legit_mean:>12.6f} | {fraud_mean:>12.6f} | {ratio:>8.3f}")
        rows.append({
            'feature': col,
            'legit_mean': legit_mean,
            'fraud_mean': fraud_mean,
            'ratio': ratio
        })

    print("=" * 90)
    return pd.DataFrame(rows)
