"""
Exploratory Data Analysis module for IEEE-CIS Fraud Detection.
Generates visualization plots and saves them to the output/eda/ directory.
"""

import polars as pl
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import seaborn as sns
from pathlib import Path
from src.config import EDA_OUTPUT_DIR


class EDAAnalyzer:
    """
    Performs exploratory data analysis on the IEEE-CIS dataset.
    All plots are automatically saved to the output/eda/ directory.
    
    Usage:
        analyzer = EDAAnalyzer(df)
        analyzer.run_all()
    """
    
    def __init__(self, df: pl.DataFrame, output_dir: str = None):
        self.df = df
        self.output_dir = Path(output_dir or EDA_OUTPUT_DIR)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def _save_figure(self, name: str, dpi: int = 150):
        """Save the current figure to the output directory."""
        path = self.output_dir / f"{name}.png"
        plt.savefig(str(path), dpi=dpi, bbox_inches='tight', pad_inches=0.1)
        plt.close()
        print(f"  Saved: {path}")
    
    def plot_class_distribution(self):
        """Plot fraud vs non-fraud distribution as pie chart and bar chart."""
        print("Plotting class distribution...")
        
        fraud_counts = self.df.group_by("isFraud").len().sort("isFraud")
        labels = ["Non-Fraud", "Fraud"]
        counts = fraud_counts["len"].to_list()
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Pie chart
        colors = ['#2ecc71', '#e74c3c']
        axes[0].pie(counts, labels=labels, autopct='%1.2f%%', colors=colors,
                    startangle=90, explode=(0, 0.1))
        axes[0].set_title("Fraud vs Non-Fraud Distribution", fontsize=14, fontweight='bold')
        
        # Bar chart
        axes[1].bar(labels, counts, color=colors, edgecolor='black', linewidth=0.5)
        for i, v in enumerate(counts):
            axes[1].text(i, v + v * 0.02, f"{v:,}", ha='center', fontsize=11, fontweight='bold')
        axes[1].set_title("Transaction Counts", fontsize=14, fontweight='bold')
        axes[1].set_ylabel("Count")
        
        plt.tight_layout()
        self._save_figure("class_distribution")
    
    def plot_transaction_amount_distribution(self):
        """Plot distribution of TransactionAmt."""
        print("Plotting transaction amount distribution...")
        
        amt = self.df["TransactionAmt"].to_numpy()
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Regular distribution
        axes[0].hist(amt, bins=100, color='#3498db', edgecolor='black', linewidth=0.5, alpha=0.7)
        axes[0].set_title("Transaction Amount Distribution", fontsize=13, fontweight='bold')
        axes[0].set_xlabel("Amount")
        axes[0].set_ylabel("Frequency")
        axes[0].set_xlim(0, np.percentile(amt, 99))
        
        # Log distribution
        log_amt = np.log1p(amt)
        axes[1].hist(log_amt, bins=100, color='#9b59b6', edgecolor='black', linewidth=0.5, alpha=0.7)
        axes[1].set_title("Log(1 + Amount) Distribution", fontsize=13, fontweight='bold')
        axes[1].set_xlabel("Log(1 + Amount)")
        axes[1].set_ylabel("Frequency")
        
        plt.tight_layout()
        self._save_figure("transaction_amount_distribution")
    
    def plot_fraud_by_category(self, column: str, top_n: int = 10):
        """Plot fraud rate by categorical column."""
        print(f"Plotting fraud rate by {column}...")
        
        stats = (
            self.df
            .with_columns(pl.col(column).cast(pl.Utf8).fill_null("Unknown"))
            .group_by(column)
            .agg([
                pl.len().alias("total"),
                pl.col("isFraud").sum().alias("fraud_count"),
            ])
            .with_columns(
                (pl.col("fraud_count") / pl.col("total") * 100).alias("fraud_rate")
            )
            .sort("total", descending=True)
            .head(top_n)
        )
        
        categories = stats[column].to_list()
        fraud_rates = stats["fraud_rate"].to_list()
        totals = stats["total"].to_list()
        
        fig, ax1 = plt.subplots(figsize=(12, 5))
        
        x = np.arange(len(categories))
        bars = ax1.bar(x, totals, color='#3498db', alpha=0.7, label='Total Count')
        ax1.set_xlabel(column, fontsize=12)
        ax1.set_ylabel("Total Count", color='#3498db', fontsize=12)
        ax1.set_xticks(x)
        ax1.set_xticklabels(categories, rotation=45, ha='right', fontsize=9)
        
        ax2 = ax1.twinx()
        ax2.plot(x, fraud_rates, 'ro-', linewidth=2, markersize=8, label='Fraud Rate (%)')
        ax2.set_ylabel("Fraud Rate (%)", color='red', fontsize=12)
        
        plt.title(f"Fraud Analysis by {column} (Top {top_n})", fontsize=14, fontweight='bold')
        
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
        
        plt.tight_layout()
        self._save_figure(f"fraud_by_{column.lower()}")
    
    def plot_missing_values(self):
        """Plot missing value heatmap."""
        print("Plotting missing values...")
        
        cols = self.df.columns
        null_pcts = []
        for col in cols:
            pct = self.df[col].null_count() / self.df.height * 100
            null_pcts.append(pct)
        
        # Filter columns with >0% missing
        filtered = [(c, p) for c, p in zip(cols, null_pcts) if p > 0]
        if not filtered:
            print("  No missing values found.")
            return
        
        filtered.sort(key=lambda x: x[1], reverse=True)
        col_names = [f[0] for f in filtered]
        pcts = [f[1] for f in filtered]
        
        fig, ax = plt.subplots(figsize=(10, max(4, len(col_names) * 0.4)))
        
        colors = ['#e74c3c' if p > 50 else '#f39c12' if p > 20 else '#3498db' for p in pcts]
        bars = ax.barh(col_names, pcts, color=colors, edgecolor='black', linewidth=0.5)
        
        for bar, pct in zip(bars, pcts):
            ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                    f'{pct:.1f}%', va='center', fontsize=9)
        
        ax.set_xlabel("Missing %", fontsize=12)
        ax.set_title("Missing Values by Column", fontsize=14, fontweight='bold')
        ax.invert_yaxis()
        
        plt.tight_layout()
        self._save_figure("missing_values")
    
    def plot_transaction_time(self):
        """Plot transaction distribution over time."""
        print("Plotting transaction time distribution...")
        
        sec_np = self.df["TransactionDT"].cast(pl.Float64).fill_null(0.0).to_numpy()
        hour = (sec_np % 86400) / 3600
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # By hour of day
        axes[0].hist(hour, bins=24, color='#2ecc71', edgecolor='black', linewidth=0.5, alpha=0.7)
        axes[0].set_title("Transactions by Hour of Day", fontsize=13, fontweight='bold')
        axes[0].set_xlabel("Hour")
        axes[0].set_ylabel("Count")
        
        # Fraud vs non-fraud by hour
        is_fraud = self.df["isFraud"].to_numpy()
        axes[1].hist(hour[is_fraud == 0], bins=24, alpha=0.6, color='#2ecc71', label='Non-Fraud')
        axes[1].hist(hour[is_fraud == 1], bins=24, alpha=0.6, color='#e74c3c', label='Fraud')
        axes[1].set_title("Fraud vs Non-Fraud by Hour", fontsize=13, fontweight='bold')
        axes[1].set_xlabel("Hour")
        axes[1].set_ylabel("Count")
        axes[1].legend()
        
        plt.tight_layout()
        self._save_figure("transaction_time")
    
    def plot_correlation_matrix(self):
        """Plot correlation matrix for numeric features."""
        print("Plotting correlation matrix...")
        
        numeric_cols = ["TransactionAmt", "TransactionDT"]
        card_cols = [c for c in ["card1", "card2", "card3", "card5"] if c in self.df.columns]
        addr_cols = [c for c in ["addr1", "addr2"] if c in self.df.columns]
        
        cols_to_use = numeric_cols + card_cols + addr_cols + ["isFraud"]
        
        df_pd = self.df.select([
            pl.col(c).cast(pl.Float64).fill_null(0.0) for c in cols_to_use
        ]).to_pandas()
        
        corr = df_pd.corr()
        
        fig, ax = plt.subplots(figsize=(10, 8))
        mask = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r',
                    center=0, ax=ax, square=True, linewidths=0.5)
        ax.set_title("Feature Correlation Matrix", fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        self._save_figure("correlation_matrix")
    
    def run_all(self):
        """Run all EDA analyses and save plots."""
        print("=" * 60)
        print("Running Full EDA Analysis")
        print("=" * 60)
        
        self.plot_class_distribution()
        self.plot_transaction_amount_distribution()
        self.plot_missing_values()
        self.plot_transaction_time()
        self.plot_correlation_matrix()
        
        # Category-specific fraud analysis  
        for col in ["card4", "card6", "DeviceType", "P_emaildomain"]:
            if col in self.df.columns:
                self.plot_fraud_by_category(col)
        
        print("\n" + "=" * 60)
        print(f"All EDA plots saved to: {self.output_dir}")
        print("=" * 60)
