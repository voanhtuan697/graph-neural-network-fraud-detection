"""
ML Trainer module for comparing traditional ML models on IEEE-CIS dataset.
Supports 3 scenarios: tabular-only, graph-only, and combined features.
Optimized for CPU training with time-efficient settings.
"""

import time
import numpy as np
import warnings
warnings.filterwarnings("ignore")

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    f1_score, precision_score, recall_score,
    roc_auc_score, average_precision_score,
    classification_report,
)

from src.utils import print_separator


def _get_models(n_samples: int):
    """
    Return dict of ML models with CPU-optimized settings.
    Settings are tuned based on dataset size for speed.
    """
    fast = n_samples > 200_000
    
    models = {
        "Decision Tree": DecisionTreeClassifier(
            max_depth=12,
            min_samples_leaf=50,
            class_weight="balanced",
            random_state=42,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200 if not fast else 100,
            max_depth=14,
            min_samples_leaf=30,
            class_weight="balanced",
            n_jobs=-1,
            random_state=42,
        ),
        "KNN": KNeighborsClassifier(
            n_neighbors=7,
            metric="manhattan",
            n_jobs=-1,
        ),
        "Naive Bayes": GaussianNB(),
        "Bagging": BaggingClassifier(
            n_estimators=50 if not fast else 30,
            max_samples=0.8,
            n_jobs=-1,
            random_state=42,
        ),
    }
    
    # Add LightGBM
    try:
        from lightgbm import LGBMClassifier
        models["LightGBM"] = LGBMClassifier(
            n_estimators=500,
            max_depth=8,
            learning_rate=0.05,
            num_leaves=63,
            min_child_samples=50,
            scale_pos_weight=25,
            subsample=0.8,
            colsample_bytree=0.8,
            n_jobs=-1,
            random_state=42,
            verbose=-1,
        )
    except ImportError:
        print("  LightGBM not installed, skipping.")
    
    # Add XGBoost
    try:
        from xgboost import XGBClassifier
        models["XGBoost"] = XGBClassifier(
            n_estimators=500,
            max_depth=8,
            learning_rate=0.05,
            scale_pos_weight=25,
            subsample=0.8,
            colsample_bytree=0.8,
            tree_method="hist",
            n_jobs=-1,
            random_state=42,
            verbosity=0,
            eval_metric="logloss",
        )
    except ImportError:
        print("  XGBoost not installed, skipping.")
    
    return models


class MLTrainer:
    """
    Trains and evaluates 7 ML models across 3 feature scenarios.
    
    Scenarios:
    1. Tabular only (original features)
    2. Graph only (entity degree, amt stats, time gaps, diversity)  
    3. Combined (tabular + graph)
    
    Usage:
        trainer = MLTrainer()
        all_results = trainer.run_all_scenarios(
            X_tab_train, X_tab_test,
            X_graph_train, X_graph_test,
            y_train, y_test
        )
    """
    
    def __init__(self, subsample_knn: int = 80_000):
        """
        Args:
            subsample_knn: Max samples for KNN training (KNN is O(n²))
        """
        self.subsample_knn = subsample_knn
        self.all_results = {}
    
    def train_evaluate(
        self,
        X_train: np.ndarray,
        X_test: np.ndarray, 
        y_train: np.ndarray,
        y_test: np.ndarray,
        scenario_name: str = "Tabular",
    ) -> dict:
        """
        Train all models on one scenario.
        
        Returns:
            Dict of {model_name: {precision, recall, f1, auc_roc, auc_pr, time}}
        """
        print_separator(f"SCENARIO: {scenario_name}")
        print(f"  Train: {X_train.shape}, Test: {X_test.shape}")
        print(f"  Fraud train: {y_train.sum():,}/{len(y_train):,} ({y_train.mean()*100:.2f}%)")
        print(f"  Fraud test:  {y_test.sum():,}/{len(y_test):,} ({y_test.mean()*100:.2f}%)")
        
        models = _get_models(len(X_train))
        results = {}
        
        for name, model in models.items():
            print(f"\n  Training {name}...", end=" ", flush=True)
            t0 = time.time()
            
            try:
                # Subsample for KNN (too slow on large data)
                if name == "KNN" and len(X_train) > self.subsample_knn:
                    rng = np.random.RandomState(42)
                    idx = rng.choice(len(X_train), self.subsample_knn, replace=False)
                    model.fit(X_train[idx], y_train[idx])
                else:
                    model.fit(X_train, y_train)
                
                y_pred = model.predict(X_test)
                
                # Get probabilities
                if hasattr(model, "predict_proba"):
                    y_prob = model.predict_proba(X_test)[:, 1]
                else:
                    y_prob = y_pred.astype(float)
                
                elapsed = time.time() - t0
                
                p = precision_score(y_test, y_pred, zero_division=0)
                r = recall_score(y_test, y_pred, zero_division=0)
                f = f1_score(y_test, y_pred, zero_division=0)
                
                try:
                    auc = roc_auc_score(y_test, y_prob)
                except:
                    auc = 0.0
                try:
                    ap = average_precision_score(y_test, y_prob)
                except:
                    ap = 0.0
                
                results[name] = {
                    "precision": p, "recall": r, "f1": f,
                    "auc_roc": auc, "auc_pr": ap,
                    "time": elapsed,
                    "y_pred": y_pred, "y_prob": y_prob,
                }
                
                print(f"F1={f:.4f} AUC={auc:.4f} ({elapsed:.1f}s)")
                
            except Exception as e:
                print(f"ERROR: {e}")
                results[name] = {
                    "precision": 0, "recall": 0, "f1": 0,
                    "auc_roc": 0, "auc_pr": 0, "time": 0,
                    "y_pred": np.zeros_like(y_test),
                    "y_prob": np.zeros_like(y_test, dtype=float),
                }
        
        self.all_results[scenario_name] = results
        return results
    
    def run_all_scenarios(
        self,
        X_tab_train, X_tab_test,
        X_graph_train, X_graph_test,
        y_train, y_test,
    ) -> dict:
        """Run all 3 scenarios."""
        
        # Scenario 1: Tabular only
        self.train_evaluate(X_tab_train, X_tab_test, y_train, y_test, "Tabular Only")
        
        # Scenario 2: Graph only
        self.train_evaluate(X_graph_train, X_graph_test, y_train, y_test, "Graph Only")
        
        # Scenario 3: Combined
        X_comb_train = np.hstack([X_tab_train, X_graph_train])
        X_comb_test = np.hstack([X_tab_test, X_graph_test])
        self.train_evaluate(X_comb_train, X_comb_test, y_train, y_test, "Combined")
        
        return self.all_results
    
    def print_comparison(self):
        """Print comparison table across all scenarios."""
        print_separator("ML MODEL COMPARISON ACROSS SCENARIOS")
        
        # Header
        scenarios = list(self.all_results.keys())
        models = list(next(iter(self.all_results.values())).keys())
        
        for scenario in scenarios:
            print(f"\n{'─'*75}")
            print(f"  {scenario}")
            print(f"{'─'*75}")
            print(f"  {'Model':<18} {'Precision':>10} {'Recall':>10} {'F1':>10} {'AUC-ROC':>10} {'AUC-PR':>10}")
            print(f"  {'─'*68}")
            
            results = self.all_results[scenario]
            for name, m in results.items():
                print(f"  {name:<18} {m['precision']:>10.4f} {m['recall']:>10.4f} "
                      f"{m['f1']:>10.4f} {m['auc_roc']:>10.4f} {m['auc_pr']:>10.4f}")
        
        # Best model summary
        print(f"\n{'='*75}")
        print("  BEST F1 PER SCENARIO")
        print(f"{'='*75}")
        for scenario in scenarios:
            results = self.all_results[scenario]
            best_name = max(results, key=lambda k: results[k]["f1"])
            best = results[best_name]
            print(f"  {scenario:<20} → {best_name:<18} F1={best['f1']:.4f}")
