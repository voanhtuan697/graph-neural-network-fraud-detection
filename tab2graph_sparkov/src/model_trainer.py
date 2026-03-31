"""
Model Training & Evaluation Module
====================================
Train and evaluate ML models for fraud detection across 3 scenarios:
  1. Tabular features only
  2. Graph features only
  3. Tabular + Graph features (combined)

Models: Logistic Regression, Random Forest, XGBoost, LightGBM
Metrics: Precision, Recall, F1-Score (for fraud class = 1)
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)
from typing import Dict, Tuple


def get_models(scale_pos_weight: float = 1.0) -> Dict[str, object]:
    """
    Get dictionary of ML models configured for class imbalance.

    Parameters
    ----------
    scale_pos_weight : float
        Ratio of negative / positive samples for XGBoost/LightGBM.
        Set to n_legit / n_fraud for proper balancing.

    Returns
    -------
    dict
        Model name -> model instance.
    """
    models = {
        'Logistic Regression': LogisticRegression(
            max_iter=1000,
            random_state=42,
            class_weight='balanced',
            solver='lbfgs',
            n_jobs=-1
        ),
        'Random Forest': RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            min_samples_leaf=5,
            random_state=42,
            class_weight='balanced',
            n_jobs=-1
        ),
        'XGBoost': XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.1,
            scale_pos_weight=scale_pos_weight,
            random_state=42,
            eval_metric='logloss',
            early_stopping_rounds=30,
            n_jobs=-1,
            verbosity=0
        ),
        'LightGBM': LGBMClassifier(
            n_estimators=300,
            max_depth=8,
            learning_rate=0.1,
            is_unbalance=True,
            random_state=42,
            n_jobs=-1,
            verbose=-1,
            early_stopping_rounds=30,
        ),
    }
    return models


def train_and_evaluate(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    scenario_name: str = "Tabular Only",
    scale_pos_weight: float = None
) -> Tuple[Dict[str, dict], Dict[str, object]]:
    """
    Train all 4 models and evaluate on test set.
    Uses a validation split from training data for early stopping (no test leakage).
    """
    from sklearn.model_selection import train_test_split

    if scale_pos_weight is None:
        n_legit = (y_train == 0).sum()
        n_fraud = (y_train == 1).sum()
        scale_pos_weight = n_legit / max(n_fraud, 1)

    print(f"\n{'='*70}")
    print(f"  Scenario: {scenario_name}")
    print(f"  Features: {X_train.shape[1]} | Train: {X_train.shape[0]} | Test: {X_test.shape[0]}")
    print(f"  Train fraud: {y_train.sum()} ({y_train.mean()*100:.2f}%)")
    print(f"  Scale pos weight: {scale_pos_weight:.2f}")
    print(f"{'='*70}")

    # Create validation split for early stopping (15% of training data)
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.15, random_state=42, stratify=y_train
    )

    models = get_models(scale_pos_weight=scale_pos_weight)
    results = {}
    trained_models = {}

    for name, model in models.items():
        print(f"\n  [Training] {name}...")

        # XGBoost/LightGBM: use validation split for early stopping
        if name == 'XGBoost':
            model.fit(
                X_tr, y_tr,
                eval_set=[(X_val, y_val)],
                verbose=False
            )
        elif name == 'LightGBM':
            model.fit(
                X_tr, y_tr,
                eval_set=[(X_val, y_val)],
            )
        else:
            model.fit(X_train, y_train)

        trained_models[name] = model

        # Predict
        y_pred = model.predict(X_test)

        # Metrics (for fraud class = 1)
        precision = precision_score(y_test, y_pred, pos_label=1, zero_division=0)
        recall = recall_score(y_test, y_pred, pos_label=1, zero_division=0)
        f1 = f1_score(y_test, y_pred, pos_label=1, zero_division=0)
        cm = confusion_matrix(y_test, y_pred)
        report = classification_report(
            y_test, y_pred, target_names=['Legit', 'Fraud'], zero_division=0
        )

        results[name] = {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'confusion_matrix': cm,
            'classification_report': report,
            'y_pred': y_pred
        }

        print(f"    Precision: {precision:.4f}")
        print(f"    Recall:    {recall:.4f}")
        print(f"    F1-Score:  {f1:.4f}")
        print(f"    Confusion Matrix:\n{cm}")

    return results, trained_models


def create_comparison_table(
    results_dict: Dict[str, Dict[str, dict]]
) -> pd.DataFrame:
    """
    Create a comprehensive comparison table across all scenarios.

    Parameters
    ----------
    results_dict : dict
        {scenario_name: {model_name: {precision, recall, f1, ...}}}

    Returns
    -------
    pd.DataFrame
        Comparison table.
    """
    rows = []
    scenarios = list(results_dict.keys())

    # Collect all model names
    all_models = list(next(iter(results_dict.values())).keys())

    for model_name in all_models:
        for scenario in scenarios:
            r = results_dict[scenario][model_name]
            rows.append({
                'Model': model_name,
                'Scenario': scenario,
                'Precision': r['precision'],
                'Recall': r['recall'],
                'F1-Score': r['f1']
            })

        # Add improvement row (last scenario vs first)
        if len(scenarios) >= 2:
            r_first = results_dict[scenarios[0]][model_name]
            r_last = results_dict[scenarios[-1]][model_name]
            rows.append({
                'Model': model_name,
                'Scenario': '[Delta] Improvement (Combined vs Tabular)',
                'Precision': r_last['precision'] - r_first['precision'],
                'Recall': r_last['recall'] - r_first['recall'],
                'F1-Score': r_last['f1'] - r_first['f1']
            })

    return pd.DataFrame(rows)


def get_feature_importance(
    trained_models: Dict[str, object],
    feature_names: list,
    top_n: int = 20
) -> Dict[str, pd.DataFrame]:
    """
    Extract feature importance from trained models.

    Returns
    -------
    dict
        model_name -> DataFrame with top features and their importance.
    """
    importance_dict = {}

    for name, model in trained_models.items():
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            df_imp = pd.DataFrame({
                'feature': feature_names,
                'importance': importances
            }).sort_values('importance', ascending=False).head(top_n)
            importance_dict[name] = df_imp

        elif hasattr(model, 'coef_'):
            coefs = np.abs(model.coef_[0])
            df_imp = pd.DataFrame({
                'feature': feature_names,
                'importance': coefs
            }).sort_values('importance', ascending=False).head(top_n)
            importance_dict[name] = df_imp

    return importance_dict
