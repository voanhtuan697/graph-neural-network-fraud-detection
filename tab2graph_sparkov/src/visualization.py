"""
Visualization Module
=====================
Generate charts and plots for comparing model performance across scenarios:
  1. Grouped bar chart: Precision/Recall/F1 per model per scenario
  2. Feature importance bar charts (top-20)
  3. Confusion matrix heatmaps
  4. Graph statistics summary
  5. Class distribution plot
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List
import os


# Style configuration
plt.rcParams.update({
    'figure.figsize': (12, 6),
    'font.size': 11,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.dpi': 150,
})

SCENARIO_COLORS = {
    'Tabular Only': '#3498db',
    'Graph Only': '#e74c3c',
    'Tabular + Graph': '#2ecc71',
}


def plot_comparison_bars(
    comparison_df: pd.DataFrame,
    save_path: str = "results/comparison_bars.png"
) -> None:
    """
    Plot grouped bar chart comparing Precision, Recall, F1 across scenarios.
    One subplot per metric; bars grouped by model, colored by scenario.
    """
    # Filter out improvement rows
    df = comparison_df[~comparison_df['Scenario'].str.contains('Delta')].copy()
    scenarios = df['Scenario'].unique()
    models = df['Model'].unique()
    metrics = ['Precision', 'Recall', 'F1-Score']

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    fig.suptitle('Model Performance Comparison: Tabular vs Graph vs Combined',
                 fontsize=16, fontweight='bold', y=1.02)

    x = np.arange(len(models))
    width = 0.25

    for ax_idx, metric in enumerate(metrics):
        ax = axes[ax_idx]
        for i, scenario in enumerate(scenarios):
            vals = df[df['Scenario'] == scenario].set_index('Model')[metric]
            vals = vals.reindex(models)
            color = SCENARIO_COLORS.get(scenario, f'C{i}')
            bars = ax.bar(x + i * width, vals.values, width,
                         label=scenario, color=color, edgecolor='white',
                         linewidth=0.5)
            # Add value labels on bars
            for bar, val in zip(bars, vals.values):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                       f'{val:.3f}', ha='center', va='bottom', fontsize=8,
                       fontweight='bold')

        ax.set_title(metric, fontsize=14, fontweight='bold')
        ax.set_xticks(x + width)
        ax.set_xticklabels(models, rotation=15, ha='right')
        ax.set_ylim(0, 1.05)
        ax.set_ylabel(metric)
        ax.legend(loc='lower right', fontsize=9)
        ax.grid(axis='y', alpha=0.3)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.show()
    print(f"[INFO] Saved comparison chart to {save_path}")


def plot_improvement_bars(
    comparison_df: pd.DataFrame,
    save_path: str = "results/improvement_bars.png"
) -> None:
    """
    Plot improvement (delta) from Tabular Only to Tabular + Graph.
    """
    df = comparison_df[comparison_df['Scenario'].str.contains('Delta')].copy()
    if df.empty:
        print("[WARN] No improvement data to plot.")
        return

    models = df['Model'].unique()
    metrics = ['Precision', 'Recall', 'F1-Score']

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(models))
    width = 0.25

    colors = ['#3498db', '#e74c3c', '#2ecc71']
    for i, metric in enumerate(metrics):
        vals = df.set_index('Model')[metric].reindex(models)
        bars = ax.bar(x + i * width, vals.values, width,
                     label=metric, color=colors[i], edgecolor='white')
        for bar, val in zip(bars, vals.values):
            color = 'green' if val > 0 else 'red'
            ax.text(bar.get_x() + bar.get_width() / 2,
                   bar.get_height() + 0.002 if val >= 0 else bar.get_height() - 0.015,
                   f'{val:+.4f}', ha='center', va='bottom', fontsize=9,
                   fontweight='bold', color=color)

    ax.set_title('Performance Improvement: Tabular+Graph vs Tabular Only',
                fontsize=14, fontweight='bold')
    ax.set_xticks(x + width)
    ax.set_xticklabels(models, rotation=15, ha='right')
    ax.set_ylabel('Delta Score')
    ax.axhline(y=0, color='black', linewidth=0.8, linestyle='-')
    ax.legend(loc='upper left')
    ax.grid(axis='y', alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.show()
    print(f"[INFO] Saved improvement chart to {save_path}")


def plot_feature_importance(
    importance_dict: Dict[str, pd.DataFrame],
    scenario_name: str = "",
    save_path: str = "results/feature_importance.png"
) -> None:
    """
    Plot feature importance for each model (horizontal bar chart).
    """
    n_models = len(importance_dict)
    if n_models == 0:
        print("[WARN] No feature importance data to plot.")
        return

    fig, axes = plt.subplots(1, n_models, figsize=(8 * n_models, 8))
    if n_models == 1:
        axes = [axes]

    colors = sns.color_palette("viridis", 20)

    for ax, (model_name, df_imp) in zip(axes, importance_dict.items()):
        df_plot = df_imp.sort_values('importance', ascending=True).tail(20)

        # Color graph features differently
        bar_colors = []
        graph_feature_keywords = [
            'user_', 'merchant_', 'customer_', 'community',
            'pagerank', 'degree', 'burst', 'sequence', 'neighbor', 'shared'
        ]
        for feat in df_plot['feature']:
            is_graph = any(kw in feat.lower() for kw in graph_feature_keywords)
            bar_colors.append('#e74c3c' if is_graph else '#3498db')

        ax.barh(range(len(df_plot)), df_plot['importance'].values,
               color=bar_colors, edgecolor='white', linewidth=0.5)
        ax.set_yticks(range(len(df_plot)))
        ax.set_yticklabels(df_plot['feature'].values, fontsize=9)
        ax.set_title(f'{model_name}\n{scenario_name}', fontsize=12, fontweight='bold')
        ax.set_xlabel('Importance')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        # Legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='#3498db', label='Tabular Feature'),
            Patch(facecolor='#e74c3c', label='Graph Feature')
        ]
        ax.legend(handles=legend_elements, loc='lower right', fontsize=9)

    plt.suptitle('Feature Importance (Top 20)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.show()
    print(f"[INFO] Saved feature importance to {save_path}")


def plot_confusion_matrices(
    results_dict: Dict[str, Dict[str, dict]],
    save_path: str = "results/confusion_matrices.png"
) -> None:
    """
    Plot confusion matrices for all models × scenarios.
    """
    scenarios = list(results_dict.keys())
    models = list(next(iter(results_dict.values())).keys())

    n_rows = len(scenarios)
    n_cols = len(models)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    if n_rows == 1:
        axes = [axes]

    for i, scenario in enumerate(scenarios):
        for j, model in enumerate(models):
            ax = axes[i][j] if n_cols > 1 else axes[i]
            cm = results_dict[scenario][model]['confusion_matrix']
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                       xticklabels=['Legit', 'Fraud'],
                       yticklabels=['Legit', 'Fraud'])
            ax.set_title(f'{model}\n({scenario})', fontsize=10, fontweight='bold')
            ax.set_ylabel('Actual')
            ax.set_xlabel('Predicted')

    plt.suptitle('Confusion Matrices', fontsize=14, fontweight='bold')
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.show()
    print(f"[INFO] Saved confusion matrices to {save_path}")


def plot_class_distribution(
    y_train_original: pd.Series,
    y_train_downsampled: pd.Series,
    y_test: pd.Series,
    save_path: str = "results/class_distribution.png"
) -> None:
    """Plot class distribution before/after downsampling."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    datasets = [
        ("Train (Original)", y_train_original),
        ("Train (Downsampled)", y_train_downsampled),
        ("Test (Downsampled)", y_test),
    ]
    colors = ['#3498db', '#e74c3c']

    for ax, (title, y) in zip(axes, datasets):
        counts = y.value_counts().sort_index()
        bars = ax.bar(['Legit (0)', 'Fraud (1)'], counts.values, color=colors)
        ax.set_title(title, fontweight='bold')
        ax.set_ylabel('Count')
        for bar, val in zip(bars, counts.values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                   f'{val:,}', ha='center', va='bottom', fontweight='bold')
        fraud_rate = y.mean() * 100
        ax.text(0.95, 0.95, f'Fraud: {fraud_rate:.1f}%',
               transform=ax.transAxes, ha='right', va='top',
               fontsize=11, fontweight='bold',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    plt.suptitle('Class Distribution', fontsize=14, fontweight='bold')
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.show()
    print(f"[INFO] Saved class distribution to {save_path}")


def plot_graph_feature_analysis(
    analysis_df: pd.DataFrame,
    save_path: str = "results/graph_feature_analysis.png"
) -> None:
    """Plot fraud/legit ratio for each graph feature."""
    fig, ax = plt.subplots(figsize=(14, 6))

    df = analysis_df.sort_values('ratio', ascending=True)
    colors = ['#e74c3c' if r > 1.5 or r < 0.67 else '#95a5a6'
              for r in df['ratio'].values]

    bars = ax.barh(range(len(df)), df['ratio'].values,
                  color=colors, edgecolor='white')
    ax.set_yticks(range(len(df)))
    ax.set_yticklabels(df['feature'].values, fontsize=9)
    ax.set_xlabel('Fraud Mean / Legit Mean Ratio')
    ax.set_title('Graph Feature Discriminative Power\n(Red = Strong signal, ratio far from 1.0)',
                fontsize=13, fontweight='bold')
    ax.axvline(x=1.0, color='black', linewidth=1, linestyle='--', alpha=0.7)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    for bar, val in zip(bars, df['ratio'].values):
        ax.text(bar.get_width() + 0.05, bar.get_y() + bar.get_height() / 2,
               f'{val:.2f}', ha='left', va='center', fontsize=9)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.show()
    print(f"[INFO] Saved graph feature analysis to {save_path}")


def print_results_table(comparison_df: pd.DataFrame) -> None:
    """Pretty-print the comparison results table."""
    print("\n" + "=" * 85)
    print("  FINAL COMPARISON RESULTS")
    print("=" * 85)
    print(f"  {'Model':<22} | {'Scenario':<38} | {'Prec':>7} | {'Recall':>7} | {'F1':>7}")
    print("-" * 85)

    for _, row in comparison_df.iterrows():
        if 'Delta' in row['Scenario']:
            # Improvement row
            print(f"  {row['Model']:<22} | {row['Scenario']:<38} | "
                  f"{row['Precision']:>+7.4f} | {row['Recall']:>+7.4f} | {row['F1-Score']:>+7.4f}")
            print("-" * 85)
        else:
            print(f"  {row['Model']:<22} | {row['Scenario']:<38} | "
                  f"{row['Precision']:>7.4f} | {row['Recall']:>7.4f} | {row['F1-Score']:>7.4f}")

    print("=" * 85)
