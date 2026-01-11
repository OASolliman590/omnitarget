"""
Scenario 2: Target Analysis Visualizer

Generates visualizations for target-centric analysis including:
- Target-centric interaction network
- Pathway enrichment dot plot
- Druggability score comparison
- Tissue expression profile heatmap
"""

import logging
from pathlib import Path
from typing import Dict, Any, List, Optional

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import seaborn as sns
import pandas as pd

from src.visualization.base import BaseVisualizer
from src.visualization.styles import VisualizationStyles

logger = logging.getLogger(__name__)


class TargetAnalysisVisualizer(BaseVisualizer):
    """Visualizer for Scenario 2: Target Analysis."""
    
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
        Generate all visualizations for Scenario 2.
        
        Args:
            data: Scenario 2 results data
            output_dir: Output directory for figures
            interactive: Generate interactive HTML versions
            formats: Output formats for static figures
            
        Returns:
            List of generated file paths
        """
        output_path = self.create_output_dir(output_dir, 2)
        generated_files = []
        
        logger.info("Generating Scenario 2 visualizations")
        
        # 1. Target interaction network
        if self._validate_data(data, ['interactions']):
            try:
                fig = self.plot_target_interaction_network(data)
                files = self.save_figure(fig, output_path, 'S2_target_interaction_network', formats)
                generated_files.extend(files)
                
                if interactive:
                    html_file = self.plot_target_interaction_network_interactive(data, output_path)
                    if html_file:
                        generated_files.append(html_file)
            except Exception as e:
                logger.error(f"Failed to generate target interaction network: {e}")
        
        # 2. Pathway enrichment
        if self._validate_data(data, ['pathway_memberships']):
            try:
                fig = self.plot_pathway_enrichment(data)
                files = self.save_figure(fig, output_path, 'S2_pathway_enrichment', formats)
                generated_files.extend(files)
            except Exception as e:
                logger.error(f"Failed to generate pathway enrichment plot: {e}")
        
        # 3. Druggability scores
        try:
            fig = self.plot_druggability_scores(data)
            files = self.save_figure(fig, output_path, 'S2_druggability_scores', formats)
            generated_files.extend(files)
        except Exception as e:
            logger.error(f"Failed to generate druggability scores plot: {e}")
        
        # 4. Expression profile
        if self._validate_data(data, ['expression_profile']):
            try:
                fig = self.plot_expression_profile(data)
                files = self.save_figure(fig, output_path, 'S2_expression_profile', formats)
                generated_files.extend(files)
            except Exception as e:
                logger.error(f"Failed to generate expression profile plot: {e}")
        
        logger.info(f"Generated {len(generated_files)} files for Scenario 2")
        return generated_files
    
    def plot_target_interaction_network(self, data: Dict[str, Any]) -> plt.Figure:
        """
        Generate target-centric interaction network.
        
        Args:
            data: Scenario 2 results data
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=VisualizationStyles.get_figure_size('network'))
        
        # Build network
        G = self._build_interaction_network(data)
        target_gene = data.get('gene', data.get('target_gene', 'TARGET'))
        
        if len(G.nodes()) == 0:
            ax.text(0.5, 0.5, 'No interaction data available',
                   ha='center', va='center', fontsize=14)
            ax.axis('off')
            return fig
        
        # Layout - place target at center
        pos = nx.spring_layout(G, k=2, iterations=50, seed=42)
        
        # Adjust target position to center if it exists
        if target_gene in G.nodes():
            pos[target_gene] = np.array([0.5, 0.5])
            # Reposition other nodes around target
            for node in G.nodes():
                if node != target_gene:
                    pos[node] = pos[node] * 0.6 + np.array([0.5, 0.5]) * 0.4
        
        # Node colors and sizes
        node_colors = []
        node_sizes = []
        for node in G.nodes():
            if node == target_gene:
                node_colors.append(VisualizationStyles.get_color('network', 'node_target'))
                node_sizes.append(1200)
            else:
                node_colors.append(VisualizationStyles.get_color('network', 'node_default'))
                node_sizes.append(600)
        
        # Edge weights for thickness
        edge_weights = [G[u][v].get('combined_score', G[u][v].get('confidence_score', 0.5)) for u, v in G.edges()]
        if edge_weights:
            max_weight = max(edge_weights)
            edge_widths = [1 + 4 * w / max_weight if max_weight > 0 else 1.5 for w in edge_weights]
        else:
            edge_widths = [1.5] * len(G.edges())
        
        # Draw network
        nx.draw_networkx_nodes(
            G, pos, node_color=node_colors, node_size=node_sizes,
            alpha=0.8, linewidths=2, edgecolors='black', ax=ax
        )
        
        nx.draw_networkx_edges(
            G, pos, width=edge_widths, alpha=0.6,
            edge_color='#666666', ax=ax
        )
        
        nx.draw_networkx_labels(G, pos, font_size=9, font_weight='bold', ax=ax)
        
        ax.set_title(f'{target_gene} Interaction Network', fontsize=16, fontweight='bold', pad=20)
        ax.axis('off')
        
        # Add legend
        legend_elements = [
            plt.Line2D([0], [0], marker='o', color='w',
                      markerfacecolor=VisualizationStyles.get_color('network', 'node_target'),
                      markersize=12, label='Target Gene'),
            plt.Line2D([0], [0], marker='o', color='w',
                      markerfacecolor=VisualizationStyles.get_color('network', 'node_default'),
                      markersize=10, label='Interacting Protein'),
        ]
        ax.legend(handles=legend_elements, loc='upper right', framealpha=0.9)
        
        # Add statistics
        stats_text = f"Nodes: {len(G.nodes())} | Edges: {len(G.edges())}"
        ax.text(0.02, 0.02, stats_text, transform=ax.transAxes,
               fontsize=10, verticalalignment='bottom',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        return fig
    
    def plot_pathway_enrichment(self, data: Dict[str, Any]) -> plt.Figure:
        """
        Generate pathway enrichment dot plot.
        
        Args:
            data: Scenario 2 results data
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=VisualizationStyles.get_figure_size('barplot'))
        
        pathway_memberships = data.get('pathway_memberships', [])
        
        if not pathway_memberships:
            ax.text(0.5, 0.5, 'No pathway membership data available',
                   ha='center', va='center', fontsize=14)
            ax.axis('off')
            return fig
        
        # Extract pathway information
        pathways = []
        confidences = []
        gene_counts = []
        
        for pathway in pathway_memberships[:20]:  # Top 20
            if isinstance(pathway, dict):
                name = pathway.get('name', pathway.get('pathway_name', 'Unknown'))
                conf = pathway.get('confidence', pathway.get('score', 0.5))
                count = pathway.get('gene_count', pathway.get('size', 10))
                
                pathways.append(self._truncate_label(name, 50))
                confidences.append(conf)
                gene_counts.append(count)
        
        if not pathways:
            ax.text(0.5, 0.5, 'No valid pathway data',
                   ha='center', va='center', fontsize=14)
            ax.axis('off')
            return fig
        
        # Sort by confidence
        sorted_data = sorted(zip(pathways, confidences, gene_counts), 
                           key=lambda x: x[1], reverse=True)
        pathways, confidences, gene_counts = zip(*sorted_data)
        
        # Create dot plot
        y_pos = np.arange(len(pathways))
        
        # Normalize gene counts for point sizes
        if max(gene_counts) > min(gene_counts):
            sizes = 100 + 400 * (np.array(gene_counts) - min(gene_counts)) / (max(gene_counts) - min(gene_counts))
        else:
            sizes = [250] * len(gene_counts)
        
        scatter = ax.scatter(confidences, y_pos, s=sizes, c=confidences,
                           cmap='viridis', alpha=0.7, edgecolors='black', linewidth=1)
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels(pathways)
        ax.set_xlabel('Confidence Score', fontsize=12, fontweight='bold')
        ax.set_title('Pathway Membership', fontsize=16, fontweight='bold', pad=20)
        ax.invert_yaxis()
        ax.grid(axis='x', alpha=0.3, linestyle='--')
        ax.set_xlim(0, 1.0)
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Confidence', rotation=270, labelpad=20)
        
        # Add legend for sizes
        for size, label in [(150, 'Small'), (300, 'Medium'), (450, 'Large')]:
            ax.scatter([], [], s=size, c='gray', alpha=0.5, edgecolors='black',
                      label=f'{label} pathway')
        ax.legend(scatterpoints=1, frameon=True, labelspacing=1, title='Pathway Size',
                 loc='lower right', fontsize=9)
        
        plt.tight_layout()
        return fig
    
    def plot_druggability_scores(self, data: Dict[str, Any]) -> plt.Figure:
        """
        Generate druggability score visualization.
        
        Args:
            data: Scenario 2 results data
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        target_gene = data.get('gene', data.get('target_gene', 'TARGET'))
        druggability_score = data.get('druggability_score', 0.0)
        
        # Component scores if available
        components = {
            'Network\nCentrality': data.get('network_centrality', 0.3),
            'Protein\nFamily': data.get('protein_family_score', 0.2),
            'Structural\nFeatures': data.get('structural_score', 0.15),
            'Expression\nLevel': data.get('expression_score', 0.25),
            'Known\nDrugs': data.get('known_drugs_score', 0.1),
        }
        
        # Create bar plot
        categories = list(components.keys())
        scores = list(components.values())
        
        colors = plt.cm.RdYlGn(np.array(scores))
        bars = ax.barh(categories, scores, color=colors, alpha=0.8,
                      edgecolor='black', linewidth=1.5)
        
        # Add value labels
        for i, (bar, score) in enumerate(zip(bars, scores)):
            width = bar.get_width()
            ax.text(width + 0.02, bar.get_y() + bar.get_height()/2,
                   f'{score:.2f}', ha='left', va='center', fontweight='bold')
        
        ax.set_xlabel('Score', fontsize=12, fontweight='bold')
        ax.set_title(f'{target_gene} Druggability Assessment', fontsize=16, fontweight='bold', pad=20)
        ax.set_xlim(0, 1.0)
        ax.grid(axis='x', alpha=0.3, linestyle='--')
        
        # Add overall score
        ax.text(0.98, 0.98, f'Overall Druggability\nScore: {druggability_score:.2f}',
               transform=ax.transAxes, fontsize=14, fontweight='bold',
               verticalalignment='top', horizontalalignment='right',
               bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        
        plt.tight_layout()
        return fig
    
    def plot_expression_profile(self, data: Dict[str, Any]) -> plt.Figure:
        """
        Generate tissue expression profile heatmap.
        
        Args:
            data: Scenario 2 results data
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(12, 8))
        
        expression_profile = data.get('expression_profile', {})
        target_gene = data.get('gene', data.get('target_gene', 'TARGET'))
        
        if not expression_profile or not isinstance(expression_profile, dict):
            ax.text(0.5, 0.5, 'No expression profile data available',
                   ha='center', va='center', fontsize=14)
            ax.axis('off')
            return fig
        
        # Convert categorical to numeric
        level_map = {
            'not detected': 0,
            'low': 1,
            'medium': 2,
            'high': 3,
            'very high': 4
        }
        
        tissues = []
        values = []
        
        for tissue, level in expression_profile.items():
            if isinstance(level, str):
                numeric_level = level_map.get(level.lower(), 0)
            elif isinstance(level, (int, float)):
                numeric_level = level
            else:
                continue
            
            tissues.append(tissue)
            values.append(numeric_level)
        
        if not tissues:
            ax.text(0.5, 0.5, 'No valid expression data',
                   ha='center', va='center', fontsize=14)
            ax.axis('off')
            return fig
        
        # Sort by expression level
        sorted_data = sorted(zip(tissues, values), key=lambda x: x[1], reverse=True)
        tissues, values = zip(*sorted_data)
        
        # Take top 30 tissues
        tissues = list(tissues)[:30]
        values = list(values)[:30]
        
        # Create heatmap data
        data_matrix = np.array(values).reshape(-1, 1)
        
        # Create heatmap
        sns.heatmap(data_matrix, annot=False, cmap='YlOrRd',
                   cbar_kws={'label': 'Expression Level'},
                   yticklabels=tissues, xticklabels=[target_gene],
                   linewidths=0.5, linecolor='white', ax=ax)
        
        ax.set_title(f'{target_gene} Tissue Expression Profile', 
                    fontsize=16, fontweight='bold', pad=20)
        ax.set_ylabel('Tissue', fontsize=12, fontweight='bold')
        
        # Add expression level reference
        ref_text = "Levels: 0=Not Detected, 1=Low, 2=Medium, 3=High, 4=Very High"
        ax.text(0.5, -0.05, ref_text, transform=ax.transAxes,
               fontsize=9, ha='center', style='italic')
        
        plt.tight_layout()
        return fig
    
    def plot_target_interaction_network_interactive(
        self, 
        data: Dict[str, Any], 
        output_path: Path
    ) -> Optional[Path]:
        """Generate interactive target interaction network using PyVis."""
        try:
            from pyvis.network import Network
            
            G = self._build_interaction_network(data)
            target_gene = data.get('gene', data.get('target_gene', 'TARGET'))
            
            if len(G.nodes()) == 0:
                logger.warning("No interaction data for interactive visualization")
                return None
            
            net = Network(height='750px', width='100%', bgcolor='#ffffff', font_color='black')
            
            # Add nodes with custom colors
            for node in G.nodes():
                color = '#E15759' if node == target_gene else '#4E79A7'
                size = 40 if node == target_gene else 25
                net.add_node(node, label=node, color=color, size=size, title=node)
            
            # Add edges
            for u, v, data in G.edges(data=True):
                weight = data.get('combined_score', data.get('confidence_score', 0.5))
                net.add_edge(u, v, value=weight, title=f"Confidence: {weight:.2f}")
            
            # Customize physics
            net.set_options("""
            {
              "physics": {
                "enabled": true,
                "barnesHut": {
                  "gravitationalConstant": -10000,
                  "springLength": 150,
                  "springConstant": 0.002
                }
              }
            }
            """)
            
            output_file = output_path / 'S2_target_interaction_network_interactive.html'
            net.save_graph(str(output_file))
            logger.info(f"Saved interactive target network: {output_file}")
            return output_file
            
        except ImportError:
            logger.warning("PyVis not available for interactive visualization")
            return None
        except Exception as e:
            logger.error(f"Failed to generate interactive network: {e}")
            return None
    
    # Helper methods
    
    def _build_interaction_network(self, data: Dict[str, Any]) -> nx.Graph:
        """Build NetworkX graph from interaction data."""
        G = nx.Graph()
        
        interactions = data.get('interactions', [])
        target_gene = data.get('gene', data.get('target_gene', 'TARGET'))
        
        # Add target node
        G.add_node(target_gene, node_type='target')
        
        # Add interactions
        for interaction in interactions:
            if isinstance(interaction, dict):
                protein_a = interaction.get('protein_a', interaction.get('source', ''))
                protein_b = interaction.get('protein_b', interaction.get('target', ''))
                score = interaction.get('combined_score', interaction.get('confidence_score', 0.5))
                
                if protein_a and protein_b:
                    G.add_node(protein_a, node_type='interactor')
                    G.add_node(protein_b, node_type='interactor')
                    G.add_edge(protein_a, protein_b, combined_score=score, confidence_score=score)
        
        return G
    
    def _truncate_label(self, label: str, max_length: int = 40) -> str:
        """Truncate long labels."""
        if len(label) <= max_length:
            return label
        return label[:max_length-3] + '...'

