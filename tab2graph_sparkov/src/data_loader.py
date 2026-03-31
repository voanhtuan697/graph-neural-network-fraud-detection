"""
Data Loading & Preprocessing Module
=====================================
- Load Sparkov Credit Card Fraud Detection dataset (Train/Test CSVs)
- Downsample majority class to reduce computation time for graph construction
- Feature engineering: temporal, geographic, demographic features
- StandardScaler on numeric features
- Preserve original indices for graph feature mapping
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from typing import Tuple


def load_data(
    train_path: str = "data/SparkovTrain.csv",
    test_path: str = "data/SparkovTest.csv"
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load Sparkov train and test datasets."""
    print("=" * 60)
    print("[STEP] Loading Sparkov Dataset")
    print("=" * 60)

    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)

    print(f"[INFO] Train shape: {df_train.shape}")
    print(f"[INFO] Test shape:  {df_test.shape}")
    print(f"[INFO] Train fraud: {df_train['is_fraud'].sum()} / {len(df_train)} "
          f"({df_train['is_fraud'].mean()*100:.3f}%)")
    print(f"[INFO] Test fraud:  {df_test['is_fraud'].sum()} / {len(df_test)} "
          f"({df_test['is_fraud'].mean()*100:.3f}%)")

    return df_train, df_test


def downsample(
    df: pd.DataFrame,
    target_total: int = 50000,
    random_state: int = 42,
    label_col: str = "is_fraud"
) -> pd.DataFrame:
    """
    Downsample the majority class (legit), keeping ALL fraud transactions.

    Parameters
    ----------
    df : pd.DataFrame
        Full dataset.
    target_total : int
        Approximate total number of samples to keep.
    random_state : int
        Random seed for reproducibility.
    label_col : str
        Name of the label column.

    Returns
    -------
    pd.DataFrame
        Downsampled dataset with reset index.
    """
    fraud = df[df[label_col] == 1]
    legit = df[df[label_col] == 0]

    n_fraud = len(fraud)
    n_legit_keep = target_total - n_fraud

    if n_legit_keep >= len(legit):
        print(f"[INFO] No downsampling needed. Keeping all {len(df)} samples.")
        return df.reset_index(drop=True)

    legit_sampled = legit.sample(n=n_legit_keep, random_state=random_state)
    df_down = pd.concat([fraud, legit_sampled], axis=0).sample(
        frac=1, random_state=random_state
    ).reset_index(drop=True)

    fraud_rate = df_down[label_col].mean() * 100
    print(f"[INFO] Downsampled: {n_fraud} fraud + {n_legit_keep} legit = {len(df_down)} total")
    print(f"[INFO] New fraud rate: {fraud_rate:.2f}%")

    return df_down


def haversine_distance(lat1, lon1, lat2, lon2):
    """Compute haversine distance in km between two lat/lon points (vectorized)."""
    R = 6371  # Earth radius in km
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    return R * c


def engineer_tabular_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineer tabular features from raw Sparkov columns.

    Creates:
    - Temporal: hour, day_of_week, month
    - Geographic: distance (customer-to-merchant haversine)
    - Demographic: age, gender_encoded
    - Financial: amt_log
    - Category: one-hot encoded

    Returns
    -------
    pd.DataFrame
        Feature-engineered DataFrame (drops PII and raw text columns).
    """
    df = df.copy()

    # --- Temporal features ---
    df['trans_date_trans_time'] = pd.to_datetime(df['trans_date_trans_time'])
    df['hour'] = df['trans_date_trans_time'].dt.hour
    df['day_of_week'] = df['trans_date_trans_time'].dt.dayofweek
    df['month'] = df['trans_date_trans_time'].dt.month

    # --- Geographic features ---
    df['distance'] = haversine_distance(
        df['lat'], df['long'], df['merch_lat'], df['merch_long']
    )

    # --- Demographic features ---
    df['dob'] = pd.to_datetime(df['dob'])
    ref_date = df['trans_date_trans_time'].max()
    df['age'] = (ref_date - df['dob']).dt.days / 365.25

    # Gender encoding
    df['gender_encoded'] = (df['gender'] == 'M').astype(int)

    # --- Financial features ---
    df['amt_log'] = np.log1p(df['amt'])

    # --- Category one-hot ---
    cat_dummies = pd.get_dummies(df['category'], prefix='cat', dtype=int)
    df = pd.concat([df, cat_dummies], axis=1)

    return df


def prepare_features(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    scale: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series,
           list, StandardScaler]:
    """
    Prepare final tabular feature matrices.

    Parameters
    ----------
    df_train, df_test : pd.DataFrame
        Feature-engineered DataFrames.
    scale : bool
        Whether to apply StandardScaler to numeric features.

    Returns
    -------
    X_train, X_test : pd.DataFrame
        Feature matrices.
    y_train, y_test : pd.Series
        Labels.
    feature_names : list
        List of feature column names.
    scaler : StandardScaler
        Fitted scaler (or None if scale=False).
    """
    # Define feature columns
    numeric_cols = [
        'amt', 'amt_log', 'lat', 'long', 'city_pop',
        'merch_lat', 'merch_long', 'hour', 'day_of_week',
        'month', 'age', 'distance', 'gender_encoded'
    ]

    # Category dummies
    cat_cols = [c for c in df_train.columns if c.startswith('cat_')]

    feature_cols = numeric_cols + cat_cols
    label_col = 'is_fraud'

    X_train = df_train[feature_cols].copy()
    X_test = df_test[feature_cols].copy()
    y_train = df_train[label_col].copy()
    y_test = df_test[label_col].copy()

    # Scale numeric features
    scaler = None
    if scale:
        scaler = StandardScaler()
        X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
        X_test[numeric_cols] = scaler.transform(X_test[numeric_cols])

    print(f"[INFO] Tabular features: {len(feature_cols)} columns")
    print(f"[INFO] X_train shape: {X_train.shape}, X_test shape: {X_test.shape}")
    print(f"[INFO] y_train fraud: {y_train.sum()}, y_test fraud: {y_test.sum()}")

    return X_train, X_test, y_train, y_test, feature_cols, scaler


def get_raw_columns_for_graph(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract columns needed for graph construction (before scaling).
    Returns cc_num, merchant, category, amt, unix_time, is_fraud, trans_date_trans_time.
    """
    cols = ['cc_num', 'merchant', 'category', 'amt', 'unix_time', 'is_fraud']
    if 'trans_date_trans_time' in df.columns:
        cols.append('trans_date_trans_time')
    return df[cols].copy()
