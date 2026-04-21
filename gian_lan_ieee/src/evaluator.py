"""
Evaluator module for model performance assessment.
Generates metrics tables, confusion matrices, ROC curves, and PR curves.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_curve, auc,
    precision_recall_curve, average_precision_score,
    f1_score, precision_score, recall_score, roc_auc_score,
)
from src.config import METRICS_OUTPUT_DIR


class ModelEvaluator:
    """
    Evaluates model performance and generates visualizations.
    
    Provides:
    - Metrics table (Precision, Recall, F1, AUC-ROC, AUC-PR)
    - Confusion matrix visualization
    - ROC curve plot
    - Precision-Recall curve plot
    - Model comparison table
    
    Usage:
        evaluator = ModelEvaluator()
        evaluator.evaluate(y_true, y_pred, y_prob, model_name="HeteroSAGE")
        evaluator.plot_comparison()
    """
    
    def __init__(self, output_dir: str = None):
        self.output_dir = Path(output_dir or METRICS_OUTPUT_DIR)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results = {}
    
    def _save_figure(self, name: str, dpi: int = 150):
        path = self.output_dir / f"{name}.png"
        plt.savefig(str(path), dpi=dpi, bbox_inches='tight', pad_inches=0.1)
        plt.close()
        print(f"  Saved: {path}")
    
    def evaluate(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_prob: np.ndarray,
        model_name: str = "Model",
    ) -> dict:
        """
        Evaluate model and generate all reports.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_prob: Prediction probabilities for positive class
            model_name: Name of the model for labeling
            
        Returns:
            Dictionary of metrics
        """
        print(f"\n{'=' * 60}")
        print(f"Evaluating: {model_name}")
        print(f"{'=' * 60}")
        
        metrics = self._compute_metrics(y_true, y_pred, y_prob)
        self.results[model_name] = metrics
        
        self._print_classification_report(y_true, y_pred, model_name)
        self.plot_confusion_matrix(y_true, y_pred, model_name)
        self.plot_roc_curve(y_true, y_prob, model_name)
        self.plot_pr_curve(y_true, y_prob, model_name)
        
        return metrics
    
    def _compute_metrics(self, y_true, y_pred, y_prob) -> dict:
        """Compute all evaluation metrics."""
        metrics = {
            'precision': precision_score(y_true, y_pred, average='binary', zero_division=0),
            'recall': recall_score(y_true, y_pred, average='binary', zero_division=0),
            'f1': f1_score(y_true, y_pred, average='binary', zero_division=0),
        }
        
        try:
            if len(set(y_true)) > 1:
                metrics['auc_roc'] = roc_auc_score(y_true, y_prob)
                metrics['auc_pr'] = average_precision_score(y_true, y_prob)
            else:
                metrics['auc_roc'] = 0.0
                metrics['auc_pr'] = 0.0
        except ValueError:
            metrics['auc_roc'] = 0.0
            metrics['auc_pr'] = 0.0
        
        print(f"\n  Precision: {metrics['precision']:.4f}")
        print(f"  Recall:    {metrics['recall']:.4f}")
        print(f"  F1-Score:  {metrics['f1']:.4f}")
        print(f"  AUC-ROC:   {metrics['auc_roc']:.4f}")
        print(f"  AUC-PR:    {metrics['auc_pr']:.4f}")
        
        return metrics
    
    def _print_classification_report(self, y_true, y_pred, model_name):
        """Print detailed classification report."""
        print(f"\n  Classification Report ({model_name}):")
        print(classification_report(
            y_true, y_pred,
            target_names=['Non-Fraud', 'Fraud'],
            zero_division=0
        ))
    
    def plot_confusion_matrix(self, y_true, y_pred, model_name: str):
        """Plot and save confusion matrix."""
        print(f"  Plotting confusion matrix for {model_name}...")
        
        cm = confusion_matrix(y_true, y_pred)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Non-Fraud', 'Fraud'],
                    yticklabels=['Non-Fraud', 'Fraud'], ax=ax)
        ax.set_xlabel('Predicted', fontsize=12)
        ax.set_ylabel('Actual', fontsize=12)
        ax.set_title(f'Confusion Matrix - {model_name}', fontsize=14, fontweight='bold')
        
        self._save_figure(f"confusion_matrix_{model_name.lower().replace(' ', '_')}")
    
    def plot_roc_curve(self, y_true, y_prob, model_name: str):
        """Plot and save ROC curve."""
        print(f"  Plotting ROC curve for {model_name}...")
        
        if len(set(y_true)) <= 1:
            print("  Skipping ROC curve - only one class present.")
            return
        
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        roc_auc = auc(fpr, tpr)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(fpr, tpr, color='#e74c3c', lw=2,
                label=f'{model_name} (AUC = {roc_auc:.4f})')
        ax.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--', label='Random')
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1.02])
        ax.set_xlabel('False Positive Rate', fontsize=12)
        ax.set_ylabel('True Positive Rate', fontsize=12)
        ax.set_title(f'ROC Curve - {model_name}', fontsize=14, fontweight='bold')
        ax.legend(loc='lower right', fontsize=11)
        ax.grid(alpha=0.3)
        
        self._save_figure(f"roc_curve_{model_name.lower().replace(' ', '_')}")
    
    def plot_pr_curve(self, y_true, y_prob, model_name: str):
        """Plot and save Precision-Recall curve."""
        print(f"  Plotting PR curve for {model_name}...")
        
        if len(set(y_true)) <= 1:
            print("  Skipping PR curve - only one class present.")
            return
        
        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        ap = average_precision_score(y_true, y_prob)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(recall, precision, color='#3498db', lw=2,
                label=f'{model_name} (AP = {ap:.4f})')
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1.02])
        ax.set_xlabel('Recall', fontsize=12)
        ax.set_ylabel('Precision', fontsize=12)
        ax.set_title(f'Precision-Recall Curve - {model_name}', fontsize=14, fontweight='bold')
        ax.legend(loc='lower left', fontsize=11)
        ax.grid(alpha=0.3)
        
        self._save_figure(f"pr_curve_{model_name.lower().replace(' ', '_')}")
    
    def plot_comparison(self):
        """Plot comparison table of all evaluated models."""
        if not self.results:
            print("No results to compare.")
            return
        
        print("\n" + "=" * 60)
        print("MODEL COMPARISON")
        print("=" * 60)
        
        # Print table
        header = f"{'Model':<20} {'Precision':>10} {'Recall':>10} {'F1':>10} {'AUC-ROC':>10} {'AUC-PR':>10}"
        print(header)
        print("-" * len(header))
        
        for model_name, metrics in self.results.items():
            print(
                f"{model_name:<20} "
                f"{metrics['precision']:>10.4f} "
                f"{metrics['recall']:>10.4f} "
                f"{metrics['f1']:>10.4f} "
                f"{metrics['auc_roc']:>10.4f} "
                f"{metrics['auc_pr']:>10.4f}"
            )
        
        # Plot comparison bar chart
        if len(self.results) > 1:
            self._plot_comparison_chart()
    
    def _plot_comparison_chart(self):
        """Plot bar chart comparing all models."""
        print("\nPlotting model comparison chart...")
        
        models = list(self.results.keys())
        metric_names = ['precision', 'recall', 'f1', 'auc_roc', 'auc_pr']
        metric_labels = ['Precision', 'Recall', 'F1-Score', 'AUC-ROC', 'AUC-PR']
        
        x = np.arange(len(metric_labels))
        width = 0.8 / len(models)
        
        fig, ax = plt.subplots(figsize=(14, 6))
        
        colors = plt.cm.Set2(np.linspace(0, 1, len(models)))
        
        for i, (model_name, color) in enumerate(zip(models, colors)):
            values = [self.results[model_name][m] for m in metric_names]
            offset = (i - len(models) / 2 + 0.5) * width
            bars = ax.bar(x + offset, values, width, label=model_name,
                         color=color, edgecolor='black', linewidth=0.5)
            
            for bar, val in zip(bars, values):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                       f'{val:.3f}', ha='center', fontsize=7, fontweight='bold')
        
        ax.set_ylabel('Score', fontsize=12)
        ax.set_title('Model Performance Comparison', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(metric_labels, fontsize=11)
        ax.set_ylim(0, 1.15)
        ax.legend(fontsize=10)
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        self._save_figure("model_comparison")
    
    def plot_training_curves(self, history: dict, model_name: str):
        """Plot training and validation curves."""
        print(f"Plotting training curves for {model_name}...")
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        epochs = range(1, len(history['train_loss']) + 1)
        
        # Loss curves
        axes[0].plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
        axes[0].plot(epochs, history['val_loss'], 'r-', label='Val Loss', linewidth=2)
        axes[0].set_xlabel('Epoch', fontsize=12)
        axes[0].set_ylabel('Loss', fontsize=12)
        axes[0].set_title(f'Loss Curves - {model_name}', fontsize=13, fontweight='bold')
        axes[0].legend(fontsize=10)
        axes[0].grid(alpha=0.3)
        
        # F1 curves
        axes[1].plot(epochs, history['train_f1'], 'b-', label='Train F1', linewidth=2)
        axes[1].plot(epochs, history['val_f1'], 'r-', label='Val F1', linewidth=2)
        axes[1].set_xlabel('Epoch', fontsize=12)
        axes[1].set_ylabel('F1 Score', fontsize=12)
        axes[1].set_title(f'F1 Score Curves - {model_name}', fontsize=13, fontweight='bold')
        axes[1].legend(fontsize=10)
        axes[1].grid(alpha=0.3)
        
        plt.tight_layout()
        self._save_figure(f"training_curves_{model_name.lower().replace(' ', '_')}")
