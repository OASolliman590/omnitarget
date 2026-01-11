"""
Network Pharmacology Visualization Module

This module provides functions to visualize gene interaction networks
and drug-target relationships using interactive and static plots.

Visualization types:
- Gene Interaction Network (static PNG and interactive HTML)
- Drug-Gene Targeting Network
- Pathway Enrichment Visualizations
"""

import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import json

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
import numpy as np

# Optional: Plotly for interactive visualizations
try:
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

logger = logging.getLogger(__name__)

# Color schemes
COLORS = {
    'target': '#E41A1C',      # Red - primary targets
    'direct': '#377EB8',      # Blue - direct interactions
    'downstream': '#4DAF4A',  # Green - downstream effects
    'feedback': '#FF7F00',    # Orange - feedback loops
    'drug': '#984EA3',        # Purple - drugs
    'pathway': '#A65628'      # Brown - pathways
}


class NetworkVisualizer:
    """
    Visualizer for gene interaction and drug-target networks.
    
    Supports both static (matplotlib/PNG) and interactive (Plotly/HTML) outputs.
    """
    
    def __init__(self, style: str = 'publication', dpi: int = 300):
        """
        Initialize the visualizer.
        
        Args:
            style: Visual style ('publication', 'presentation', 'dark')
            dpi: Resolution for static images
        """
        self.style = style
        self.dpi = dpi
        self._setup_style()
    
    def _setup_style(self):
        """Configure matplotlib style."""
        plt.rcParams['figure.dpi'] = 150
        plt.rcParams['savefig.dpi'] = self.dpi
        plt.rcParams['font.family'] = 'sans-serif'
        plt.rcParams['font.size'] = 10
        
        if self.style == 'dark':
            plt.style.use('dark_background')
    
    def visualize_gene_network(
        self,
        network_data: Dict[str, Any],
        output_path: Path,
        show_all_labels: bool = True,
        interactive: bool = False,
        title: str = "Gene Interaction Network"
    ) -> Path:
        """
        Visualize gene interaction network.
        
        Args:
            network_data: Dictionary with 'nodes' and 'edges' keys
            output_path: Output directory for generated files
            show_all_labels: Whether to show all gene names
            interactive: Generate interactive HTML (requires Plotly)
            title: Plot title
            
        Returns:
            Path to generated file
        """
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Build NetworkX graph
        G = self._build_graph(network_data)
        
        if interactive and PLOTLY_AVAILABLE:
            return self._plot_interactive_network(G, output_path, title)
        else:
            return self._plot_static_network(G, output_path, show_all_labels, title)
    
    def visualize_drug_gene_network(
        self,
        genes: List[str],
        drugs: List[Dict[str, Any]],
        output_path: Path,
        show_all_labels: bool = True,
        interactive: bool = False,
        title: str = "Drug-Gene Targeting Network"
    ) -> Path:
        """
        Visualize drug-gene targeting relationships.
        
        Args:
            genes: List of target gene symbols
            drugs: List of drug dictionaries with drug_name, target_protein, score
            output_path: Output directory
            show_all_labels: Whether to show all labels
            interactive: Generate interactive HTML
            title: Plot title
            
        Returns:
            Path to generated file
        """
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Build bipartite graph
        G = nx.Graph()
        
        # Add gene nodes
        for gene in genes:
            G.add_node(gene, node_type='gene')
        
        # Add drug nodes and edges
        for drug in drugs:
            drug_name = str(drug.get('drug_name', 'Unknown'))[:15]
            target = drug.get('target_protein')
            score = drug.get('repurposing_score', 0.5)
            
            if target:
                if target not in G.nodes():
                    G.add_node(target, node_type='gene')
                
                G.add_node(drug_name, node_type='drug', score=float(score))
                G.add_edge(drug_name, target)
        
        if interactive and PLOTLY_AVAILABLE:
            return self._plot_interactive_drug_network(G, output_path, title)
        else:
            return self._plot_static_drug_network(G, output_path, show_all_labels, title)
    
    def _build_graph(self, network_data: Dict[str, Any]) -> nx.DiGraph:
        """Build NetworkX graph from network data."""
        G = nx.DiGraph()
        
        nodes = network_data.get('nodes', {})
        edges = network_data.get('edges', [])
        
        for node_id, attrs in nodes.items():
            G.add_node(node_id, **attrs)
        
        for edge in edges:
            source = edge.get('source')
            target = edge.get('target')
            if source and target:
                G.add_edge(source, target, 
                          edge_type=edge.get('type', 'default'),
                          weight=edge.get('score', 1.0))
        
        return G
    
    def _plot_static_network(
        self,
        G: nx.DiGraph,
        output_path: Path,
        show_all_labels: bool,
        title: str
    ) -> Path:
        """Generate static PNG network visualization."""
        fig, ax = plt.subplots(figsize=(16, 14))
        
        # Layout
        pos = nx.kamada_kawai_layout(G, scale=2)
        
        # Draw edges by type
        for etype, color, style, width in [
            ('downstream', COLORS['downstream'], '-', 0.8),
            ('direct', COLORS['direct'], '-', 1.5),
            ('feedback', COLORS['feedback'], '--', 1.0)
        ]:
            edges = [(u, v) for u, v in G.edges() 
                    if G.edges[u, v].get('edge_type') == etype]
            if edges:
                nx.draw_networkx_edges(
                    G, pos, edgelist=edges, edge_color=color,
                    style=style, width=width, alpha=0.6, ax=ax,
                    arrows=True, arrowsize=10, connectionstyle='arc3,rad=0.1'
                )
        
        # Draw nodes by type
        for ntype, color in COLORS.items():
            if ntype in ['drug', 'pathway']:
                continue
            nodes = [n for n in G.nodes() if G.nodes[n].get('type') == ntype]
            if nodes:
                sizes = [300 + G.nodes[n].get('degree', 5) * 15 for n in nodes]
                if ntype == 'target':
                    sizes = [s * 1.5 for s in sizes]
                nx.draw_networkx_nodes(
                    G, pos, nodelist=nodes, node_color=color,
                    node_size=sizes, alpha=0.9, ax=ax,
                    edgecolors='white', linewidths=2
                )
        
        # Draw labels
        if show_all_labels:
            label_pos = {k: (v[0], v[1] + 0.08) for k, v in pos.items()}
            nx.draw_networkx_labels(G, label_pos, font_size=7, 
                                   font_weight='bold', ax=ax)
        
        # Legend
        legend_elements = [
            mpatches.Patch(color=COLORS['target'], label='Primary Targets'),
            mpatches.Patch(color=COLORS['direct'], label='Direct Targets'),
            mpatches.Patch(color=COLORS['downstream'], label='Downstream Effects'),
            mpatches.Patch(color=COLORS['feedback'], label='Feedback Loops'),
        ]
        ax.legend(handles=legend_elements, loc='upper left', fontsize=10, framealpha=0.9)
        
        ax.set_title(f'{title}\n{G.number_of_nodes()} genes | {G.number_of_edges()} interactions',
                    fontsize=14, fontweight='bold', pad=20)
        ax.axis('off')
        
        plt.tight_layout()
        output_file = output_path / 'gene_network.png'
        plt.savefig(output_file, dpi=self.dpi, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        plt.close()
        
        logger.info(f"Saved static network: {output_file}")
        return output_file
    
    def _plot_interactive_network(
        self,
        G: nx.DiGraph,
        output_path: Path,
        title: str
    ) -> Path:
        """Generate interactive Plotly network visualization."""
        pos = nx.kamada_kawai_layout(G, scale=2)
        
        fig = go.Figure()
        
        # Draw edges
        edge_x, edge_y = [], []
        for u, v in G.edges():
            x0, y0 = pos[u]
            x1, y1 = pos[v]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
        
        fig.add_trace(go.Scatter(
            x=edge_x, y=edge_y, mode='lines',
            line=dict(color='rgba(150,150,150,0.5)', width=1),
            hoverinfo='none', name='Interactions'
        ))
        
        # Draw nodes by type
        for ntype, color in COLORS.items():
            if ntype in ['drug', 'pathway']:
                continue
            nodes = [n for n in G.nodes() if G.nodes[n].get('type') == ntype]
            if not nodes:
                continue
            
            x_vals = [pos[n][0] for n in nodes]
            y_vals = [pos[n][1] for n in nodes]
            sizes = [15 + G.nodes[n].get('degree', 5) for n in nodes]
            
            hovertexts = [
                f"<b>{n}</b><br>Type: {ntype}<br>"
                f"Effect: {G.nodes[n].get('effect', 0):.3f}"
                for n in nodes
            ]
            
            fig.add_trace(go.Scatter(
                x=x_vals, y=y_vals, mode='markers+text',
                marker=dict(size=sizes, color=color, line=dict(color='white', width=1)),
                text=nodes, textposition='top center', textfont=dict(size=8),
                hovertext=hovertexts, hoverinfo='text',
                name=ntype.title()
            ))
        
        fig.update_layout(
            title=dict(text=f'<b>{title}</b>', font=dict(size=16), x=0.5),
            showlegend=True,
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            template='plotly_white',
            width=1100, height=900
        )
        
        output_file = output_path / 'gene_network_interactive.html'
        fig.write_html(str(output_file), include_plotlyjs='cdn')
        
        logger.info(f"Saved interactive network: {output_file}")
        return output_file
    
    def _plot_static_drug_network(
        self,
        G: nx.Graph,
        output_path: Path,
        show_all_labels: bool,
        title: str
    ) -> Path:
        """Generate static PNG drug-gene network."""
        fig, ax = plt.subplots(figsize=(16, 14))
        
        # Bipartite layout
        genes = [n for n in G.nodes() if G.nodes[n].get('node_type') == 'gene']
        drugs = [n for n in G.nodes() if G.nodes[n].get('node_type') == 'drug']
        
        pos = {}
        for i, gene in enumerate(sorted(genes)):
            pos[gene] = (2, -(i - len(genes)/2) * 0.5)
        for i, drug in enumerate(drugs):
            pos[drug] = (-1, -(i - len(drugs)/2) * 0.3)
        
        # Draw edges
        nx.draw_networkx_edges(G, pos, alpha=0.4, width=1, 
                              edge_color='#888888', ax=ax)
        
        # Draw gene nodes
        gene_colors = [COLORS['target'] if g in ['AXL', 'AKT1', 'MAPK1', 'RELA'] 
                      else COLORS['direct'] for g in genes]
        nx.draw_networkx_nodes(G, pos, nodelist=genes, node_color=gene_colors,
                              node_size=500, alpha=0.9, ax=ax,
                              edgecolors='white', linewidths=2)
        
        # Draw drug nodes
        drug_scores = [G.nodes[d].get('score', 0.5) for d in drugs]
        cmap = plt.cm.RdYlGn
        norm = plt.Normalize(vmin=0.5, vmax=0.8)
        drug_colors = [cmap(norm(s)) for s in drug_scores]
        
        nx.draw_networkx_nodes(G, pos, nodelist=drugs, node_color=drug_colors,
                              node_size=250, node_shape='s', alpha=0.9, ax=ax,
                              edgecolors='black', linewidths=1)
        
        # Draw labels
        if show_all_labels:
            gene_label_pos = {g: (pos[g][0] + 0.15, pos[g][1]) for g in genes}
            nx.draw_networkx_labels(G, gene_label_pos, 
                                   labels={g: g for g in genes},
                                   font_size=9, font_weight='bold', ax=ax,
                                   horizontalalignment='left')
            
            drug_label_pos = {d: (pos[d][0] - 0.1, pos[d][1]) for d in drugs}
            nx.draw_networkx_labels(G, drug_label_pos,
                                   labels={d: d for d in drugs},
                                   font_size=7, ax=ax,
                                   horizontalalignment='right')
        
        # Legend
        legend_elements = [
            mpatches.Patch(color=COLORS['target'], label='Primary Targets'),
            mpatches.Patch(color=COLORS['direct'], label='Gene Targets'),
            mpatches.Patch(color='#4DAF4A', label='High Score Drugs'),
            mpatches.Patch(color='#FFFF00', label='Medium Score Drugs'),
        ]
        ax.legend(handles=legend_elements, loc='upper left', fontsize=9, framealpha=0.9)
        
        # Colorbar
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, shrink=0.5, pad=0.02)
        cbar.set_label('Repurposing Score', fontsize=10)
        
        # Column labels
        ax.text(-1, max(p[1] for p in pos.values()) + 0.8, 'DRUGS', fontsize=14,
               fontweight='bold', ha='center', color=COLORS['drug'])
        ax.text(2, max(p[1] for p in pos.values()) + 0.8, 'GENE TARGETS', fontsize=14,
               fontweight='bold', ha='center', color=COLORS['direct'])
        
        ax.set_title(f'{title}\n{len(drugs)} drugs → {len(genes)} gene targets',
                    fontsize=14, fontweight='bold', pad=20)
        ax.axis('off')
        
        plt.tight_layout()
        output_file = output_path / 'drug_gene_network.png'
        plt.savefig(output_file, dpi=self.dpi, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
        plt.close()
        
        logger.info(f"Saved drug-gene network: {output_file}")
        return output_file
    
    def _plot_interactive_drug_network(
        self,
        G: nx.Graph,
        output_path: Path,
        title: str
    ) -> Path:
        """Generate interactive Plotly drug-gene network."""
        genes = [n for n in G.nodes() if G.nodes[n].get('node_type') == 'gene']
        drugs = [n for n in G.nodes() if G.nodes[n].get('node_type') == 'drug']
        
        pos = {}
        for i, gene in enumerate(sorted(genes)):
            pos[gene] = (3, -(i - len(genes)/2) * 0.5)
        for i, drug in enumerate(drugs):
            pos[drug] = (-1.5, -(i - len(drugs)/2) * 0.35)
        
        fig = go.Figure()
        
        # Draw edges
        edge_x, edge_y = [], []
        for u, v in G.edges():
            x0, y0 = pos[u]
            x1, y1 = pos[v]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
        
        fig.add_trace(go.Scatter(
            x=edge_x, y=edge_y, mode='lines',
            line=dict(color='rgba(100,100,100,0.4)', width=1),
            hoverinfo='none', name='Drug-Target'
        ))
        
        # Gene nodes
        fig.add_trace(go.Scatter(
            x=[pos[g][0] for g in genes],
            y=[pos[g][1] for g in genes],
            mode='markers+text',
            marker=dict(size=25, color=COLORS['direct'], 
                       line=dict(color='white', width=2)),
            text=genes, textposition='middle right', textfont=dict(size=10),
            hovertext=[f'<b>{g}</b><br>Drugs: {G.degree(g)}' for g in genes],
            hoverinfo='text', name='Genes'
        ))
        
        # Drug nodes
        drug_scores = [G.nodes[d].get('score', 0.5) for d in drugs]
        fig.add_trace(go.Scatter(
            x=[pos[d][0] for d in drugs],
            y=[pos[d][1] for d in drugs],
            mode='markers+text',
            marker=dict(size=15, color=drug_scores, colorscale='RdYlGn',
                       cmin=0.5, cmax=0.8, symbol='square',
                       colorbar=dict(title='Score', x=-0.15)),
            text=[d[:10] for d in drugs], textposition='middle left',
            textfont=dict(size=7),
            hovertext=[f'<b>{d}</b><br>Score: {G.nodes[d].get("score", 0):.3f}' 
                      for d in drugs],
            hoverinfo='text', name='Drugs'
        ))
        
        fig.update_layout(
            title=dict(text=f'<b>{title}</b>', font=dict(size=16), x=0.5),
            showlegend=True,
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            template='plotly_white',
            width=1000, height=900
        )
        
        output_file = output_path / 'drug_gene_network_interactive.html'
        fig.write_html(str(output_file), include_plotlyjs='cdn')
        
        logger.info(f"Saved interactive drug-gene network: {output_file}")
        return output_file
