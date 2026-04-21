"""
Graph Visualization module for IEEE-CIS Fraud Detection.
Provides tools to visualize graph structure, statistics, and embeddings.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import networkx as nx
import torch
from torch_geometric.data import HeteroData
from pathlib import Path
from src.config import GRAPH_OUTPUT_DIR, SEED


class GraphVisualizer:
    """
    Visualizes heterogeneous graph structure and statistics.
    All plots are saved to output/graphs/ directory.
    
    Usage:
        viz = GraphVisualizer(data)
        viz.plot_all()
    """
    
    def __init__(self, data: HeteroData, output_dir: str = None):
        self.data = data
        self.output_dir = Path(output_dir or GRAPH_OUTPUT_DIR)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def _save_figure(self, name: str, dpi: int = 150):
        path = self.output_dir / f"{name}.png"
        plt.savefig(str(path), dpi=dpi, bbox_inches='tight', pad_inches=0.1)
        plt.close()
        print(f"  Saved: {path}")
    
    def plot_schema(self):
        """Visualize the graph schema (node types and edge types)."""
        print("Plotting graph schema...")
        
        G = nx.DiGraph()
        
        # Add node types
        for ntype in self.data.node_types:
            num = self.data[ntype].num_nodes
            G.add_node(ntype, label=f"{ntype}\n({num:,})", ntype=ntype)
        
        # Add edge types
        for et in self.data.edge_types:
            src_type, rel_name, dst_type = et
            num_edges = self.data[et].edge_index.shape[1] if self.data[et].edge_index.numel() > 0 else 0
            G.add_edge(src_type, dst_type, label=f"{rel_name}\n({num_edges:,})")
        
        fig, ax = plt.subplots(figsize=(16, 12))
        pos = nx.spring_layout(G, seed=SEED, k=2)
        
        # Color nodes by type
        colors = []
        for node in G.nodes():
            if node == 'txn':
                colors.append('#e74c3c')
            else:
                colors.append('#3498db')
        
        nx.draw_networkx_nodes(G, pos, node_size=2000, node_color=colors, alpha=0.8, ax=ax)
        nx.draw_networkx_labels(G, pos, labels=nx.get_node_attributes(G, 'label'),
                                font_size=8, font_weight='bold', ax=ax)
        nx.draw_networkx_edges(G, pos, edge_color='gray', arrows=True,
                               arrowsize=15, connectionstyle='arc3,rad=0.1', ax=ax)
        edge_labels = nx.get_edge_attributes(G, 'label')
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=6, ax=ax)
        
        ax.set_title("Heterogeneous Graph Schema", fontsize=16, fontweight='bold')
        ax.axis('off')
        
        self._save_figure("graph_schema")
    
    def plot_edge_statistics(self):
        """Plot bar chart of edge counts by type."""
        print("Plotting edge statistics...")
        
        edge_types = []
        edge_counts = []
        
        for et in self.data.edge_types:
            src_type, rel_name, dst_type = et
            num = self.data[et].edge_index.shape[1] if self.data[et].edge_index.numel() > 0 else 0
            # Skip reverse edges for cleaner visualization
            if not rel_name.startswith("rev_"):
                edge_types.append(f"{src_type}→{dst_type}\n({rel_name})")
                edge_counts.append(num)
        
        fig, ax = plt.subplots(figsize=(14, 6))
        
        colors = plt.cm.Set3(np.linspace(0, 1, len(edge_types)))
        bars = ax.barh(edge_types, edge_counts, color=colors, edgecolor='black', linewidth=0.5)
        
        for bar, count in zip(bars, edge_counts):
            ax.text(bar.get_width() + max(edge_counts) * 0.01,
                    bar.get_y() + bar.get_height() / 2,
                    f'{count:,}', va='center', fontsize=9)
        
        ax.set_xlabel("Number of Edges", fontsize=12)
        ax.set_title("Edge Type Statistics", fontsize=14, fontweight='bold')
        ax.invert_yaxis()
        
        plt.tight_layout()
        self._save_figure("edge_statistics")
    
    def plot_node_statistics(self):
        """Plot bar chart of node counts by type."""
        print("Plotting node statistics...")
        
        node_types = []
        node_counts = []
        
        for ntype in self.data.node_types:
            node_types.append(ntype)
            node_counts.append(self.data[ntype].num_nodes)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        colors = ['#e74c3c' if nt == 'txn' else '#3498db' for nt in node_types]
        bars = ax.bar(node_types, node_counts, color=colors, edgecolor='black', linewidth=0.5)
        
        for bar, count in zip(bars, node_counts):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(node_counts) * 0.01,
                    f'{count:,}', ha='center', fontsize=9, fontweight='bold')
        
        ax.set_ylabel("Number of Nodes", fontsize=12)
        ax.set_title("Node Type Statistics", fontsize=14, fontweight='bold')
        plt.xticks(rotation=45, ha='right')
        
        plt.tight_layout()
        self._save_figure("node_statistics")
    
    def plot_subgraph(self, num_nodes: int = 50):
        """
        Sample and visualize a small subgraph using NetworkX.
        
        Args:
            num_nodes: Maximum number of transaction nodes to sample
        """
        print(f"Plotting subgraph (max {num_nodes} txn nodes)...")
        
        G = nx.DiGraph()
        
        # Sample transaction nodes
        total_txn = self.data["txn"].num_nodes
        sample_idx = np.random.choice(total_txn, min(num_nodes, total_txn), replace=False)
        sample_set = set(sample_idx.tolist())
        
        # Add sampled txn nodes
        for idx in sample_idx:
            label = "F" if hasattr(self.data["txn"], 'y') and self.data["txn"].y[idx].item() == 1 else "N"
            G.add_node(f"txn_{idx}", ntype="txn", label=label)
        
        # Add edges that connect sampled nodes
        for et in self.data.edge_types:
            src_type, rel_name, dst_type = et
            if rel_name.startswith("rev_"):
                continue
            
            ei = self.data[et].edge_index.cpu().numpy()
            
            if src_type == "txn" and dst_type == "txn":
                for i in range(ei.shape[1]):
                    s, d = int(ei[0, i]), int(ei[1, i])
                    if s in sample_set and d in sample_set:
                        G.add_edge(f"txn_{s}", f"txn_{d}", rel=rel_name)
            elif src_type == "txn":
                for i in range(ei.shape[1]):
                    s, d = int(ei[0, i]), int(ei[1, i])
                    if s in sample_set:
                        entity_node = f"{dst_type}_{d}"
                        if not G.has_node(entity_node):
                            G.add_node(entity_node, ntype=dst_type, label=dst_type)
                        G.add_edge(f"txn_{s}", entity_node, rel=rel_name)
        
        if len(G.nodes()) == 0:
            print("  No edges found in subgraph sample.")
            return
        
        fig, ax = plt.subplots(figsize=(16, 12))
        pos = nx.spring_layout(G, seed=SEED, k=1.5)
        
        # Color by node type
        node_colors = []
        for node in G.nodes():
            ntype = G.nodes[node].get('ntype', 'unknown')
            if ntype == 'txn':
                label = G.nodes[node].get('label', 'N')
                node_colors.append('#e74c3c' if label == 'F' else '#2ecc71')
            else:
                node_colors.append('#3498db')
        
        nx.draw(G, pos, node_color=node_colors, node_size=300,
                font_size=6, with_labels=True,
                labels={n: G.nodes[n].get('label', n.split('_')[0]) for n in G.nodes()},
                edge_color='gray', arrows=True, arrowsize=8, ax=ax)
        
        ax.set_title(f"Subgraph Sample ({len(G.nodes())} nodes, {len(G.edges())} edges)",
                     fontsize=14, fontweight='bold')
        ax.axis('off')
        
        self._save_figure("subgraph_sample")
    
    def plot_all(self):
        """Run all visualizations."""
        print("=" * 60)
        print("Generating Graph Visualizations")
        print("=" * 60)
        
        self.plot_schema()
        self.plot_node_statistics()
        self.plot_edge_statistics()
        self.plot_subgraph()
        
        print(f"\nAll graph visualizations saved to: {self.output_dir}")
