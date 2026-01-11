"""
Scenario 1: Disease Network Visualizer

Generates visualizations for disease network construction including:
- Disease-pathway network graph
- Pathway centrality analysis
- Gene interaction network
- Multi-panel summary report
"""

import logging
from collections import defaultdict
from pathlib import Path
from typing import Dict, Any, List, Optional

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import Rectangle

from src.visualization.base import BaseVisualizer
from src.visualization.styles import VisualizationStyles

logger = logging.getLogger(__name__)


class DiseaseNetworkVisualizer(BaseVisualizer):
    """Visualizer for Scenario 1: Disease Network Construction."""
    
    def __init__(self, style: str = 'publication'):
        super().__init__(style)
        VisualizationStyles.apply_style(style)
    
    def visualize(
        self,
        data: Dict[str, Any],
        output_dir: str,
        interactive: bool = False,
        formats: List[str] = ['png']
    ) -> List[Path]:
        """
        Generate all visualizations for Scenario 1.
        
        Args:
            data: Scenario 1 results data
            output_dir: Output directory for figures
            interactive: Generate interactive HTML versions
            formats: Output formats for static figures
            
        Returns:
            List of generated file paths
        """
        output_path = self.create_output_dir(output_dir, 1)
        generated_files = []
        
        logger.info("Generating Scenario 1 visualizations")
        
        # 1. Disease-pathway network
        if self._validate_data(data, ['network_nodes', 'network_edges']):
            try:
                fig = self.plot_disease_network(data)
                files = self.save_figure(fig, output_path, 'S1_disease_network', formats)
                generated_files.extend(files)
                
                if interactive:
                    html_file = self.plot_disease_network_interactive(data, output_path)
                    if html_file:
                        generated_files.append(html_file)
            except Exception as e:
                logger.error(f"Failed to generate disease network plot: {e}")
        
        # 2. Pathway centrality
        if self._validate_data(data, ['pathway_enrichment']):
            try:
                fig = self.plot_pathway_centrality(data)
                files = self.save_figure(fig, output_path, 'S1_pathway_centrality', formats)
                generated_files.extend(files)
            except Exception as e:
                logger.error(f"Failed to generate pathway centrality plot: {e}")
        
        # 3. Gene network
        if self._validate_data(data, ['network_nodes', 'network_edges']):
            try:
                fig = self.plot_gene_network(data)
                files = self.save_figure(fig, output_path, 'S1_gene_network', formats)
                generated_files.extend(files)
                
                if interactive:
                    html_file = self.plot_gene_network_interactive(data, output_path)
                    if html_file:
                        generated_files.append(html_file)
            except Exception as e:
                logger.error(f"Failed to generate gene network plot: {e}")
        
        # 3b. Expression heatmap
        if self._validate_data(data, ['expression_profiles']):
            try:
                fig = self.plot_expression_heatmap(data)
                files = self.save_figure(fig, output_path, 'S1_expression_heatmap', formats)
                generated_files.extend(files)
            except Exception as e:
                logger.error(f"Failed to generate expression heatmap: {e}")
        
        # 4. Summary report (multi-panel)
        try:
            fig = self.generate_summary_report(data)
            files = self.save_figure(fig, output_path, 'S1_summary_report', formats)
            generated_files.extend(files)
        except Exception as e:
            logger.error(f"Failed to generate summary report: {e}")

        # 5. Pathway sunburst diagram (interactive)
        if self._validate_data(data, ['pathways']):
            try:
                html_file = self.plot_pathway_sunburst(data, output_path)
                if html_file:
                    generated_files.append(html_file)
            except Exception as e:
                logger.error(f"Failed to generate pathway sunburst: {e}")
        
        logger.info(f"Generated {len(generated_files)} files for Scenario 1")
        return generated_files
    
    def plot_disease_network(self, data: Dict[str, Any]) -> plt.Figure:
        """
        Generate disease-pathway network visualization.
        
        Args:
            data: Scenario 1 results data
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=VisualizationStyles.get_figure_size('network'))
        
        # Build network
        G = self._build_network_graph(data)
        
        if len(G.nodes()) == 0:
            ax.text(0.5, 0.5, 'No network data available', 
                   ha='center', va='center', fontsize=14)
            ax.axis('off')
            return fig
        
        # Layout
        pos = nx.spring_layout(G, k=2, iterations=50, seed=42)
        
        # Get node types
        node_types = nx.get_node_attributes(G, 'type')
        
        # Color nodes by type
        colors = []
        sizes = []
        for node in G.nodes():
            node_type = node_types.get(node, 'gene')
            if node_type == 'disease':
                colors.append(VisualizationStyles.get_color('biological', 'danger'))
                sizes.append(800)
            elif node_type == 'pathway':
                colors.append(VisualizationStyles.get_color('biological', 'pathway'))
                sizes.append(600)
            else:
                colors.append(VisualizationStyles.get_color('network', 'node_default'))
                sizes.append(400)
        
        # Draw network
        nx.draw_networkx_nodes(
            G, pos, node_color=colors, node_size=sizes,
            alpha=0.8, linewidths=2, edgecolors='black', ax=ax
        )
        
        # Draw edges with varying thickness
        edge_weights = [G[u][v].get('weight', 1.0) for u, v in G.edges()]
        if edge_weights:
            max_weight = max(edge_weights)
            edge_widths = [2 * w / max_weight if max_weight > 0 else 1 for w in edge_weights]
        else:
            edge_widths = [1.0] * len(G.edges())
        
        nx.draw_networkx_edges(
            G, pos, width=edge_widths, alpha=0.5,
            edge_color='#999999', ax=ax
        )
        
        # Draw labels
        nx.draw_networkx_labels(G, pos, font_size=8, font_weight='bold', ax=ax)
        
        ax.set_title('Disease-Pathway Network', fontsize=16, fontweight='bold', pad=20)
        ax.axis('off')
        
        # Add legend
        legend_elements = [
            plt.Line2D([0], [0], marker='o', color='w', 
                      markerfacecolor=VisualizationStyles.get_color('biological', 'danger'),
                      markersize=10, label='Disease'),
            plt.Line2D([0], [0], marker='o', color='w',
                      markerfacecolor=VisualizationStyles.get_color('biological', 'pathway'),
                      markersize=10, label='Pathway'),
            plt.Line2D([0], [0], marker='o', color='w',
                      markerfacecolor=VisualizationStyles.get_color('network', 'node_default'),
                      markersize=10, label='Gene/Protein'),
        ]
        ax.legend(handles=legend_elements, loc='upper right', framealpha=0.9)
        
        plt.tight_layout()
        return fig
    
    def plot_pathway_centrality(self, data: Dict[str, Any]) -> plt.Figure:
        """
        Generate pathway centrality bar plot.
        
        Args:
            data: Scenario 1 results data
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=VisualizationStyles.get_figure_size('barplot'))
        
        pathway_enrichment = data.get('pathway_enrichment', [])
        
        if not pathway_enrichment:
            ax.text(0.5, 0.5, 'No pathway enrichment data available',
                   ha='center', va='center', fontsize=14)
            ax.axis('off')
            return fig
        
        # Extract pathway names and scores
        pathways = []
        scores = []
        
        for pathway in pathway_enrichment[:20]:  # Top 20 pathways
            if isinstance(pathway, dict):
                name = pathway.get('term', pathway.get('description', 'Unknown'))
                # Use -log10(p-value) as score
                # p_value may be a string (e.g., '2.1e-27'), convert to float
                p_value_raw = pathway.get('p_value', pathway.get('fdr', 1.0))
                try:
                    if isinstance(p_value_raw, str):
                        p_value = float(p_value_raw)
                    else:
                        p_value = float(p_value_raw)
                except (ValueError, TypeError):
                    p_value = 1.0  # Default if conversion fails
                
                if p_value > 0:
                    score = -np.log10(p_value)
                else:
                    score = 10  # Very significant
                pathways.append(name)
                scores.append(score)
        
        if not pathways:
            ax.text(0.5, 0.5, 'No valid pathway data',
                   ha='center', va='center', fontsize=14)
            ax.axis('off')
            return fig
        
        # Sort by score
        sorted_data = sorted(zip(pathways, scores), key=lambda x: x[1], reverse=True)
        pathways, scores = zip(*sorted_data)
        
        # Create horizontal bar plot
        colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(pathways)))
        y_pos = np.arange(len(pathways))
        
        bars = ax.barh(y_pos, scores, color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels([self._truncate_label(p, 50) for p in pathways])
        ax.set_xlabel('-log10(p-value)', fontsize=12, fontweight='bold')
        ax.set_title('Pathway Enrichment Significance', fontsize=16, fontweight='bold', pad=20)
        ax.invert_yaxis()
        ax.grid(axis='x', alpha=0.3, linestyle='--')
        
        # Add significance threshold line
        if max(scores) > 1.3:  # -log10(0.05) = 1.3
            ax.axvline(x=1.3, color='red', linestyle='--', linewidth=2, alpha=0.7, label='p=0.05')
            ax.legend()
        
        plt.tight_layout()
        return fig
    
    def plot_gene_network(self, data: Dict[str, Any]) -> plt.Figure:
        """
        Generate gene interaction network visualization.
        
        Args:
            data: Scenario 1 results data
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=VisualizationStyles.get_figure_size('network'))
        
        # Build network (genes only)
        G = self._build_network_graph(data, genes_only=True)
        
        if len(G.nodes()) == 0:
            ax.text(0.5, 0.5, 'No gene network data available',
                   ha='center', va='center', fontsize=14)
            ax.axis('off')
            return fig
        
        # Layout
        pos = nx.spring_layout(G, k=1.5, iterations=50, seed=42)
        
        # Calculate node degrees for sizing
        degrees = dict(G.degree())
        max_degree = max(degrees.values()) if degrees else 1
        node_sizes = [300 + (degrees.get(node, 1) / max_degree) * 700 for node in G.nodes()]
        
        # Calculate centrality for coloring
        try:
            betweenness = nx.betweenness_centrality(G)
            node_colors = [betweenness.get(node, 0) for node in G.nodes()]
        except:
            node_colors = ['#4E79A7'] * len(G.nodes())
        
        # Draw network
        nx.draw_networkx_nodes(
            G, pos, node_size=node_sizes, node_color=node_colors,
            cmap='YlOrRd', alpha=0.8, linewidths=2, edgecolors='black', ax=ax
        )
        
        nx.draw_networkx_edges(
            G, pos, alpha=0.3, edge_color='#666666', width=1.0, ax=ax
        )
        
        # Draw labels for high-degree nodes only
        high_degree_nodes = {node: node for node in G.nodes() if degrees.get(node, 0) > np.percentile(list(degrees.values()), 75)}
        nx.draw_networkx_labels(G, pos, labels=high_degree_nodes, font_size=8, font_weight='bold', ax=ax)
        
        ax.set_title('Gene Interaction Network', fontsize=16, fontweight='bold', pad=20)
        ax.axis('off')
        
        # Add colorbar for centrality
        if isinstance(node_colors, list) and all(isinstance(c, (int, float)) for c in node_colors):
            sm = plt.cm.ScalarMappable(cmap='YlOrRd', norm=plt.Normalize(vmin=0, vmax=max(node_colors)))
            sm.set_array([])
            cbar = plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label('Betweenness Centrality', rotation=270, labelpad=20)
        
        plt.tight_layout()
        return fig
    
    def generate_summary_report(self, data: Dict[str, Any]) -> plt.Figure:
        """
        Generate multi-panel summary report.
        
        Args:
            data: Scenario 1 results data
            
        Returns:
            Matplotlib figure with multiple subplots
        """
        fig = plt.figure(figsize=(16, 12))
        gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
        
        # Panel 1: Network statistics
        ax1 = fig.add_subplot(gs[0, 0])
        self._plot_network_stats(ax1, data)
        
        # Panel 2: Pathway enrichment (top 10)
        ax2 = fig.add_subplot(gs[0, 1])
        self._plot_top_pathways(ax2, data)
        
        # Panel 3: Network visualization
        ax3 = fig.add_subplot(gs[1, :])
        self._plot_mini_network(ax3, data)
        
        # Panel 4: Validation metrics
        ax4 = fig.add_subplot(gs[2, 0])
        self._plot_validation_metrics(ax4, data)
        
        # Panel 5: Cancer markers (if available)
        ax5 = fig.add_subplot(gs[2, 1])
        self._plot_cancer_markers(ax5, data)
        
        fig.suptitle('Scenario 1: Disease Network Analysis Summary', 
                    fontsize=18, fontweight='bold', y=0.995)
        
        return fig
    
    def plot_disease_network_interactive(self, data: Dict[str, Any], output_path: Path) -> Optional[Path]:
        """Generate interactive disease network using PyVis."""
        try:
            from pyvis.network import Network
            
            G = self._build_network_graph(data)
            
            if len(G.nodes()) == 0:
                logger.warning("No network data for interactive visualization")
                return None
            
            net = Network(height='750px', width='100%', bgcolor='#ffffff', font_color='black')
            net.from_nx(G)
            
            # Customize appearance
            net.set_options("""
            {
              "physics": {
                "enabled": true,
                "barnesHut": {
                  "gravitationalConstant": -8000,
                  "springLength": 200,
                  "springConstant": 0.001
                }
              }
            }
            """)
            
            output_file = output_path / 'S1_disease_network_interactive.html'
            net.save_graph(str(output_file))
            logger.info(f"Saved interactive network: {output_file}")
            return output_file
            
        except ImportError:
            logger.warning("PyVis not available for interactive visualization")
            return None
        except Exception as e:
            logger.error(f"Failed to generate interactive network: {e}")
            return None
    
    def plot_gene_network_interactive(self, data: Dict[str, Any], output_path: Path) -> Optional[Path]:
        """Generate interactive gene network using PyVis."""
        try:
            from pyvis.network import Network
            
            G = self._build_network_graph(data, genes_only=True)
            
            if len(G.nodes()) == 0:
                logger.warning("No gene network data for interactive visualization")
                return None
            
            net = Network(height='750px', width='100%', bgcolor='#ffffff', font_color='black')
            net.from_nx(G)
            
            output_file = output_path / 'S1_gene_network_interactive.html'
            net.save_graph(str(output_file))
            logger.info(f"Saved interactive gene network: {output_file}")
            return output_file
            
        except ImportError:
            logger.warning("PyVis not available for interactive visualization")
            return None
        except Exception as e:
            logger.error(f"Failed to generate interactive gene network: {e}")
            return None
    
    # Helper methods
    
    def _build_network_graph(self, data: Dict[str, Any], genes_only: bool = False) -> nx.Graph:
        """Build NetworkX graph from data."""
        G = nx.Graph()
        
        nodes = data.get('network_nodes', [])
        edges = data.get('network_edges', [])
        
        # Add nodes
        for node in nodes:
            if isinstance(node, dict):
                node_id = node.get('id', node.get('gene', node.get('name', '')))
                node_type = node.get('type', 'gene')
                
                if genes_only and node_type != 'gene':
                    continue
                
                G.add_node(node_id, type=node_type, **node)
        
        # Add edges
        for edge in edges:
            if isinstance(edge, dict):
                source = edge.get('source', edge.get('protein_a', ''))
                target = edge.get('target', edge.get('protein_b', ''))
                weight = edge.get('weight', edge.get('confidence_score', edge.get('score', 1.0)))
                
                if source in G.nodes() and target in G.nodes():
                    # Don't pass weight separately if it's already in edge dict
                    edge_data = {k: v for k, v in edge.items() if k not in ['source', 'target']}
                    if 'weight' not in edge_data:
                        edge_data['weight'] = weight
                    G.add_edge(source, target, **edge_data)
        
        return G
    
    def _plot_network_stats(self, ax, data: Dict[str, Any]):
        """Plot network statistics as table."""
        stats = {
            'Nodes': data.get('network_nodes_count', len(data.get('network_nodes', []))),
            'Edges': data.get('network_edges_count', len(data.get('network_edges', []))),
            'Pathways': len(data.get('pathway_enrichment', [])),
            'Diseases': len(data.get('disease_associations', [])),
            'Validation Score': f"{data.get('validation_score', 0.0):.2f}",
        }
        
        ax.axis('tight')
        ax.axis('off')
        
        table_data = [[key, str(value)] for key, value in stats.items()]
        table = ax.table(cellText=table_data, colLabels=['Metric', 'Value'],
                        cellLoc='left', loc='center', colWidths=[0.6, 0.4])
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        
        # Style header
        for i in range(2):
            table[(0, i)].set_facecolor('#4E79A7')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        ax.set_title('Network Statistics', fontsize=12, fontweight='bold', pad=10)
    
    def _plot_top_pathways(self, ax, data: Dict[str, Any]):
        """Plot top pathways as horizontal bar chart."""
        pathway_enrichment = data.get('pathway_enrichment', [])[:10]
        
        if not pathway_enrichment:
            ax.text(0.5, 0.5, 'No pathway data', ha='center', va='center')
            ax.axis('off')
            return
        
        names = []
        scores = []
        for p in pathway_enrichment:
            if isinstance(p, dict):
                names.append(self._truncate_label(p.get('term', 'Unknown'), 30))
                p_val_raw = p.get('p_value', p.get('fdr', 1.0))
                try:
                    # Convert string p-values to float
                    if isinstance(p_val_raw, str):
                        p_val = float(p_val_raw)
                    else:
                        p_val = float(p_val_raw)
                except (ValueError, TypeError):
                    p_val = 1.0
                scores.append(-np.log10(p_val) if p_val > 0 else 10)
        
        y_pos = np.arange(len(names))
        ax.barh(y_pos, scores, color='#377eb8', alpha=0.7)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(names, fontsize=8)
        ax.set_xlabel('-log10(p-value)', fontsize=9)
        ax.set_title('Top 10 Enriched Pathways', fontsize=12, fontweight='bold', pad=10)
        ax.invert_yaxis()
        ax.grid(axis='x', alpha=0.3)
    
    def _plot_mini_network(self, ax, data: Dict[str, Any]):
        """Plot simplified network visualization."""
        G = self._build_network_graph(data, genes_only=True)
        
        if len(G.nodes()) == 0:
            ax.text(0.5, 0.5, 'No network data', ha='center', va='center')
            ax.axis('off')
            return
        
        # Limit to top nodes by degree
        if len(G.nodes()) > 50:
            degrees = dict(G.degree())
            top_nodes = sorted(degrees.keys(), key=lambda x: degrees[x], reverse=True)[:50]
            G = G.subgraph(top_nodes)
        
        pos = nx.spring_layout(G, k=2, iterations=30, seed=42)
        nx.draw_networkx_nodes(G, pos, node_size=200, node_color='#4E79A7', 
                              alpha=0.7, ax=ax)
        nx.draw_networkx_edges(G, pos, alpha=0.2, width=0.5, ax=ax)
        
        ax.set_title('Gene Interaction Network (Top Nodes)', fontsize=12, fontweight='bold', pad=10)
        ax.axis('off')
    
    def _plot_validation_metrics(self, ax, data: Dict[str, Any]):
        """Plot validation metrics."""
        validation = data.get('validation_metrics', {})
        
        metrics = {
            'Pathway Coverage': validation.get('pathway_coverage', 0.0),
            'Network Density': validation.get('network_density', 0.0),
            'Data Completeness': validation.get('data_completeness', 0.0),
            'Confidence Score': validation.get('confidence_score', 0.0),
        }
        
        names = list(metrics.keys())
        values = list(metrics.values())
        
        colors = ['#2ca02c' if v >= 0.7 else '#ff7f0e' if v >= 0.4 else '#d62728' for v in values]
        
        y_pos = np.arange(len(names))
        ax.barh(y_pos, values, color=colors, alpha=0.7, edgecolor='black', linewidth=0.5)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(names, fontsize=9)
        ax.set_xlabel('Score', fontsize=9)
        ax.set_xlim(0, 1.0)
        ax.set_title('Validation Metrics', fontsize=12, fontweight='bold', pad=10)
        ax.invert_yaxis()
        ax.grid(axis='x', alpha=0.3)
        
        # Add value labels
        for i, v in enumerate(values):
            ax.text(v + 0.02, i, f'{v:.2f}', va='center', fontsize=9)
    
    def _plot_cancer_markers(self, ax, data: Dict[str, Any]):
        """Plot cancer marker information."""
        cancer_markers = data.get('cancer_markers', [])
        
        if not cancer_markers:
            ax.text(0.5, 0.5, 'No cancer marker data', ha='center', va='center')
            ax.axis('off')
            return
        
        marker_count = len(cancer_markers)
        ax.text(0.5, 0.7, f'{marker_count}', ha='center', va='center', 
               fontsize=48, fontweight='bold', color='#d62728')
        ax.text(0.5, 0.4, 'Cancer Markers\nIdentified', ha='center', va='center',
               fontsize=14)
        
        # List top markers
        if marker_count > 0:
            top_markers = cancer_markers[:5]
            marker_names = [m.get('gene', m.get('name', 'Unknown')) if isinstance(m, dict) else str(m) 
                           for m in top_markers]
            ax.text(0.5, 0.15, '\n'.join(marker_names), ha='center', va='top',
                   fontsize=10, style='italic')
        
        ax.set_title('Cancer Markers', fontsize=12, fontweight='bold', pad=10)
        ax.axis('off')
    
    def _truncate_label(self, label: str, max_length: int = 40) -> str:
        """Truncate long labels."""
        if len(label) <= max_length:
            return label
        return label[:max_length-3] + '...'
    
    def plot_expression_heatmap(self, data: Dict[str, Any]) -> plt.Figure:
        """Generate tissue expression heatmap for top disease genes."""
        fig, ax = plt.subplots(figsize=(12, 9))
        profiles = data.get('expression_profiles') or []
        if not profiles:
            ax.text(0.5, 0.5, 'No expression profiles available',
                    ha='center', va='center', fontsize=14)
            ax.axis('off')
            return fig
        
        rows = []
        level_map = {
            'not detected': 0,
            'low': 1,
            'medium': 2,
            'high': 3,
            'very high': 4
        }
        for prof in profiles:
            if not isinstance(prof, dict):
                continue
            gene = prof.get('gene')
            tissue = prof.get('tissue')
            level = prof.get('expression_level')
            if not gene or not tissue or level is None:
                continue
            if isinstance(level, str):
                value = level_map.get(level.lower(), 2)
            else:
                value = float(level)
            rows.append({'gene': gene, 'tissue': tissue, 'expression': value})
        
        if not rows:
            ax.text(0.5, 0.5, 'Expression profile format not recognized',
                    ha='center', va='center', fontsize=14)
            ax.axis('off')
            return fig
        
        df = pd.DataFrame(rows)
        top_genes = (
            df.groupby('gene')['expression']
            .var()
            .sort_values(ascending=False)
            .head(40)
            .index
        )
        top_tissues = (
            df.groupby('tissue')['expression']
            .mean()
            .sort_values(ascending=False)
            .head(12)
            .index
        )
        matrix = (
            df[df['gene'].isin(top_genes) & df['tissue'].isin(top_tissues)]
            .pivot_table(index='gene', columns='tissue', values='expression', aggfunc='mean')
            .reindex(index=top_genes, columns=top_tissues)
        )
        if matrix.empty:
            ax.text(0.5, 0.5, 'Not enough overlapping gene/tissue data',
                    ha='center', va='center', fontsize=14)
            ax.axis('off')
            return fig
        
        sns.heatmap(
            matrix,
            cmap='viridis',
            linewidths=0.5,
            linecolor='white',
            annot=True,
            fmt='.1f',
            cbar_kws={'label': 'Expression Level (0-4)'},
            ax=ax
        )
        ax.set_title('Top Disease Genes – Tissue Expression Heatmap',
                     fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel('Tissue', fontsize=12, fontweight='bold')
        ax.set_ylabel('Gene', fontsize=12, fontweight='bold')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        return fig

    def plot_pathway_sunburst(
        self,
        data: Dict[str, Any],
        output_path: Path
    ) -> Optional[Path]:
        """Generate interactive sunburst showing pathway hierarchy and enrichment."""
        try:
            import plotly.graph_objects as go
        except ImportError:
            logger.warning("Plotly not available; skipping S1 sunburst visualization")
            return None
        
        pathways = data.get('pathways') or []
        if not pathways:
            logger.warning("Scenario 1 pathways missing; sunburst skipped")
            return None
        
        enrichment = (data.get('enrichment_results') or {}).get('enrichment', {})
        pvalue_lookup: Dict[str, float] = {}
        for entries in enrichment.values():
            for entry in entries or []:
                term = (entry.get('term') or entry.get('name') or '').lower()
                try:
                    pval = float(entry.get('p_value'))
                except (TypeError, ValueError):
                    continue
                if term:
                    pvalue_lookup[term] = pval
        
        def get_pathway_pvalue(pathway: Dict[str, Any]) -> float:
            for key in (pathway.get('id'), pathway.get('name')):
                if key and key.lower() in pvalue_lookup:
                    return max(pvalue_lookup[key.lower()], 1e-6)
            return 0.05
        
        categories = defaultdict(list)
        for pathway in pathways:
            category = (pathway.get('source_db') or 'other').upper()
            categories[category].append(pathway)
        
        ids = ['root']
        labels = ['Disease Pathways']
        parents = ['']
        values = [sum(len(p.get('genes', [])) or 1 for p in pathways)]
        colors = [0.0]
        
        for category, items in categories.items():
            cat_id = f"category_{category}"
            ids.append(cat_id)
            labels.append(category)
            parents.append('root')
            values.append(sum(len(p.get('genes', [])) or 1 for p in items))
            category_p = min((get_pathway_pvalue(p) for p in items), default=0.05)
            colors.append(-np.log10(category_p))
            
            for pathway in items:
                pathway_id = pathway.get('id') or pathway.get('name')
                if not pathway_id:
                    continue
                ids.append(pathway_id)
                labels.append(pathway.get('name', pathway_id))
                parents.append(cat_id)
                gene_count = len(pathway.get('genes', [])) or 1
                values.append(gene_count)
                colors.append(-np.log10(get_pathway_pvalue(pathway)))
        
        fig = go.Figure(
            go.Sunburst(
                ids=ids,
                labels=labels,
                parents=parents,
                values=values,
                branchvalues='total',
                marker=dict(
                    colors=colors,
                    colorscale='Turbo',
                    colorbar=dict(title='-log10(p-value)')
                ),
                hovertemplate='<b>%{label}</b><br>Genes: %{value}<br>-log10(p): %{color:.2f}<extra></extra>'
            )
        )
        fig.update_layout(
            title_text='Scenario 1 Pathway Hierarchy',
            margin=dict(l=0, r=0, t=50, b=0)
        )
        
        output_file = output_path / 'S1_pathway_sunburst.html'
        fig.write_html(str(output_file), include_plotlyjs='cdn')
        logger.info(f"Saved S1 pathway sunburst: {output_file}")
        return output_file
