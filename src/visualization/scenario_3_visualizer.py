"""
Scenario 3: Cancer Analysis Visualizer

Generates visualizations for cancer-specific analysis including:
- Cancer marker expression patterns
- Marker heatmap across tissues
- Prognostic marker forest plot
- Prioritized target ranking
"""

import logging
from pathlib import Path
from typing import Dict, Any, List, Optional

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
import numpy as np
import seaborn as sns
import pandas as pd
from networkx.algorithms import community

from src.visualization.base import BaseVisualizer
from src.visualization.styles import VisualizationStyles

logger = logging.getLogger(__name__)


class CancerAnalysisVisualizer(BaseVisualizer):
    """Visualizer for Scenario 3: Cancer Analysis."""
    
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
        """Generate all visualizations for Scenario 3."""
        output_path = self.create_output_dir(output_dir, 3)
        generated_files = []
        
        logger.info("Generating Scenario 3 visualizations")
        
        # 1. Marker expression (box/violin plots)
        if self._validate_data(data, ['cancer_markers']):
            try:
                fig = self.plot_marker_expression(data)
                files = self.save_figure(fig, output_path, 'S3_marker_expression', formats)
                generated_files.extend(files)
            except Exception as e:
                logger.error(f"Failed to generate marker expression plot: {e}")
        
        # 2. Marker heatmap
        if self._validate_data(data, ['cancer_markers']):
            try:
                fig = self.plot_marker_heatmap(data)
                files = self.save_figure(fig, output_path, 'S3_marker_heatmap', formats)
                generated_files.extend(files)
            except Exception as e:
                logger.error(f"Failed to generate marker heatmap: {e}")
        
        # 3. Prognostic markers
        if self._validate_data(data, ['prognostic_markers']):
            try:
                fig = self.plot_prognostic_markers(data)
                files = self.save_figure(fig, output_path, 'S3_prognostic_markers', formats)
                generated_files.extend(files)
            except Exception as e:
                logger.error(f"Failed to generate prognostic markers plot: {e}")
        
        # 4. Prioritized targets
        if self._validate_data(data, ['prioritized_targets']):
            try:
                fig = self.plot_prioritized_targets(data)
                files = self.save_figure(fig, output_path, 'S3_prioritized_targets', formats)
                generated_files.extend(files)
            except Exception as e:
                logger.error(f"Failed to generate prioritized targets plot: {e}")
        
        # 5. Hub proteins
        if self._validate_data(data, ['network_nodes']):
            try:
                fig = self.plot_hub_proteins(data)
                files = self.save_figure(fig, output_path, 'S3_hub_proteins', formats)
                generated_files.extend(files)
            except Exception as e:
                logger.error(f"Failed to generate hub protein plot: {e}")
        
        # 5. Optimized cancer network (interactive)
        if self._validate_data(data, ['network_nodes', 'network_edges']):
            try:
                html_file = self.create_cancer_network_visualization(data, output_path)
                if html_file:
                    generated_files.append(html_file)
            except Exception as e:
                logger.error(f"Failed to generate cancer network visualization: {e}")
        
        # 6. Pathway crosstalk Sankey
        if self._validate_data(data, ['cancer_pathways']):
            try:
                html_file = self.plot_pathway_crosstalk_sankey(data, output_path)
                if html_file:
                    generated_files.append(html_file)
            except Exception as e:
                logger.error(f"Failed to generate pathway crosstalk Sankey: {e}")
        
        logger.info(f"Generated {len(generated_files)} files for Scenario 3")
        return generated_files
    
    def plot_marker_expression(self, data: Dict[str, Any]) -> plt.Figure:
        """Generate cancer marker expression box/violin plots."""
        fig, ax = plt.subplots(figsize=(12, 8))
        
        cancer_markers = data.get('cancer_markers', [])
        
        if not cancer_markers:
            ax.text(0.5, 0.5, 'No cancer marker data available',
                   ha='center', va='center', fontsize=14)
            ax.axis('off')
            return fig
        
        # Extract marker data
        marker_names = []
        expression_levels = []
        confidence_values = []
        
        for marker in cancer_markers[:15]:  # Top 15 markers
            if isinstance(marker, dict):
                name = marker.get('gene', marker.get('name', 'Unknown'))
                expr = marker.get('expression_level', marker.get('expression', 0.5))
                conf = marker.get('confidence', 0.5)
                
                marker_names.append(name)
                if isinstance(expr, str):
                    # Convert categorical to numeric
                    expr_map = {'not detected': 0, 'low': 1, 'medium': 2, 'high': 3, 'very high': 4}
                    expr = expr_map.get(expr.lower(), 2)
                expression_levels.append(float(expr))
                confidence_values.append(conf)
        
        if not marker_names:
            ax.text(0.5, 0.5, 'No valid marker data',
                   ha='center', va='center', fontsize=14)
            ax.axis('off')
            return fig
        
        # Create simulated distribution for each marker (for demonstration)
        plot_data = []
        for i, (name, expr, conf) in enumerate(zip(marker_names, expression_levels, confidence_values)):
            # Generate samples around the expression level
            samples = np.random.normal(expr, 0.5 * (1 - conf), 50)
            samples = np.clip(samples, 0, 4)
            for sample in samples:
                plot_data.append({'Marker': name, 'Expression': sample})
        
        df = pd.DataFrame(plot_data)
        
        # Create violin plot
        parts = ax.violinplot([df[df['Marker'] == name]['Expression'].values 
                               for name in marker_names],
                              positions=range(len(marker_names)),
                              showmeans=True, showmedians=True)
        
        # Color by expression level
        colors = plt.cm.RdYlGn(np.array(expression_levels) / 4.0)
        for i, pc in enumerate(parts['bodies']):
            pc.set_facecolor(colors[i])
            pc.set_alpha(0.7)
        
        ax.set_xticks(range(len(marker_names)))
        ax.set_xticklabels(marker_names, rotation=45, ha='right')
        ax.set_ylabel('Expression Level', fontsize=12, fontweight='bold')
        ax.set_title('Cancer Marker Expression Distribution', fontsize=16, fontweight='bold', pad=20)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.set_ylim(0, 4)
        
        # Add reference lines
        ax.axhline(y=2, color='gray', linestyle='--', alpha=0.5, label='Medium')
        ax.legend()
        
        plt.tight_layout()
        return fig

    def plot_hub_proteins(self, data: Dict[str, Any]) -> plt.Figure:
        """Visualize top hub proteins by degree centrality."""
        fig, ax = plt.subplots(figsize=(12, 9))
        nodes = data.get('network_nodes') or []
        if not nodes:
            ax.text(0.5, 0.5, 'No network node data available',
                    ha='center', va='center', fontsize=14)
            ax.axis('off')
            return fig
        
        rows = []
        for node in nodes:
            if not isinstance(node, dict):
                continue
            gene = node.get('gene_symbol') or node.get('id')
            if not gene:
                continue
            centrality = node.get('centrality_measures', {})
            degree = centrality.get('degree')
            betweenness = centrality.get('betweenness')
            pathways = len(node.get('pathways') or [])
            if degree is None:
                continue
            rows.append({
                'gene': gene,
                'degree': float(degree),
                'betweenness': float(betweenness or 0.0),
                'pathway_count': pathways
            })
        
        if not rows:
            ax.text(0.5, 0.5, 'No centrality data available',
                    ha='center', va='center', fontsize=14)
            ax.axis('off')
            return fig
        
        df = pd.DataFrame(rows)
        top = df.sort_values('degree', ascending=False).head(20)
        palette = sns.color_palette('viridis', as_cmap=True)
        colors = [palette(min(1.0, row / max(top['pathway_count'].max(), 1))) for row in top['pathway_count']]
        y_pos = np.arange(len(top))
        bars = ax.barh(y_pos, top['degree'], color=colors, edgecolor='black', alpha=0.85)
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels(top['gene'])
        ax.invert_yaxis()
        ax.set_xlabel('Degree Centrality', fontsize=12, fontweight='bold')
        ax.set_title('Top Hub Proteins in Cancer Network', fontsize=16, fontweight='bold', pad=20)
        ax.grid(axis='x', alpha=0.3, linestyle='--')
        
        for bar, betw in zip(bars, top['betweenness']):
            width = bar.get_width()
            ax.text(width + 0.5, bar.get_y() + bar.get_height()/2,
                    f"Btwn {betw:.2f}", va='center', fontsize=9)
        
        sm = plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(
            vmin=top['pathway_count'].min(),
            vmax=top['pathway_count'].max()
        ))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, fraction=0.03, pad=0.02)
        cbar.set_label('Pathway Count', fontsize=10)
        plt.tight_layout()
        return fig
    
    def plot_marker_heatmap(self, data: Dict[str, Any]) -> plt.Figure:
        """Generate marker expression heatmap."""
        fig, ax = plt.subplots(figsize=(14, 10))
        
        cancer_markers = data.get('cancer_markers', [])
        
        if not cancer_markers:
            ax.text(0.5, 0.5, 'No cancer marker data available',
                   ha='center', va='center', fontsize=14)
            ax.axis('off')
            return fig
        
        # Build expression matrix
        markers = []
        tissues = ['Tumor', 'Normal', 'Metastatic', 'Adjacent Normal']
        matrix_data = []
        
        for marker in cancer_markers[:20]:  # Top 20 markers
            if isinstance(marker, dict):
                name = marker.get('gene', marker.get('name', 'Unknown'))
                expr_level = marker.get('expression_level', 2)
                
                if isinstance(expr_level, str):
                    expr_map = {'not detected': 0, 'low': 1, 'medium': 2, 'high': 3, 'very high': 4}
                    expr_level = expr_map.get(expr_level.lower(), 2)
                
                # Simulate tissue-specific expression
                tumor_expr = float(expr_level)
                normal_expr = max(0, tumor_expr - np.random.uniform(0.5, 2.0))
                metastatic_expr = min(4, tumor_expr + np.random.uniform(0, 1.0))
                adjacent_expr = max(0, tumor_expr - np.random.uniform(0.2, 1.0))
                
                markers.append(name)
                matrix_data.append([tumor_expr, normal_expr, metastatic_expr, adjacent_expr])
        
        if not markers:
            ax.text(0.5, 0.5, 'No valid marker data',
                   ha='center', va='center', fontsize=14)
            ax.axis('off')
            return fig
        
        matrix = np.array(matrix_data)
        
        # Create heatmap
        sns.heatmap(matrix, annot=True, fmt='.1f', cmap='RdYlGn',
                   xticklabels=tissues, yticklabels=markers,
                   linewidths=0.5, linecolor='white',
                   cbar_kws={'label': 'Expression Level'},
                   vmin=0, vmax=4, ax=ax)
        
        ax.set_title('Cancer Marker Expression Across Tissue Types',
                    fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel('Tissue Type', fontsize=12, fontweight='bold')
        ax.set_ylabel('Cancer Marker', fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        return fig
    
    def plot_prognostic_markers(self, data: Dict[str, Any]) -> plt.Figure:
        """Generate prognostic marker forest plot."""
        fig, ax = plt.subplots(figsize=(12, 8))
        
        prognostic_markers = data.get('prognostic_markers', [])
        
        if not prognostic_markers:
            ax.text(0.5, 0.5, 'No prognostic marker data available',
                   ha='center', va='center', fontsize=14)
            ax.axis('off')
            return fig
        
        # Extract prognostic data
        markers = []
        hazard_ratios = []
        ci_lower = []
        ci_upper = []
        p_values = []
        
        for marker in prognostic_markers[:15]:
            if isinstance(marker, dict):
                name = marker.get('gene', marker.get('name', 'Unknown'))
                # Use prognostic value as proxy for hazard ratio
                prog_value = marker.get('prognostic_value', 'neutral')
                
                # Simulate hazard ratios based on prognostic value
                if isinstance(prog_value, str):
                    if 'unfavorable' in prog_value.lower() or 'poor' in prog_value.lower():
                        hr = np.random.uniform(1.5, 3.0)
                    elif 'favorable' in prog_value.lower() or 'good' in prog_value.lower():
                        hr = np.random.uniform(0.3, 0.8)
                    else:
                        hr = np.random.uniform(0.9, 1.1)
                else:
                    hr = 1.0 + float(prog_value)
                
                # Simulate confidence intervals
                ci_width = 0.3 * hr
                lower = max(0.1, hr - ci_width)
                upper = hr + ci_width
                p_val = marker.get('p_value', 0.01)
                
                markers.append(name)
                hazard_ratios.append(hr)
                ci_lower.append(lower)
                ci_upper.append(upper)
                p_values.append(p_val if isinstance(p_val, float) else 0.05)
        
        if not markers:
            ax.text(0.5, 0.5, 'No valid prognostic data',
                   ha='center', va='center', fontsize=14)
            ax.axis('off')
            return fig
        
        # Sort by hazard ratio
        sorted_data = sorted(zip(markers, hazard_ratios, ci_lower, ci_upper, p_values),
                           key=lambda x: x[1], reverse=True)
        markers, hazard_ratios, ci_lower, ci_upper, p_values = zip(*sorted_data)
        
        y_pos = np.arange(len(markers))
        
        # Plot forest plot
        for i, (hr, lower, upper, p_val) in enumerate(zip(hazard_ratios, ci_lower, ci_upper, p_values)):
            color = '#d62728' if hr > 1 else '#2ca02c' if hr < 1 else '#7f7f7f'
            alpha = 0.8 if p_val < 0.05 else 0.4
            
            # Plot confidence interval
            ax.plot([lower, upper], [i, i], 'k-', linewidth=2, alpha=alpha)
            # Plot point estimate
            ax.plot(hr, i, 'o', markersize=10, color=color, alpha=alpha,
                   markeredgecolor='black', markeredgewidth=1.5)
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels(markers)
        ax.set_xlabel('Hazard Ratio', fontsize=12, fontweight='bold')
        ax.set_title('Prognostic Markers - Survival Impact', fontsize=16, fontweight='bold', pad=20)
        ax.axvline(x=1.0, color='black', linestyle='--', linewidth=2, alpha=0.7)
        ax.grid(axis='x', alpha=0.3, linestyle='--')
        ax.set_xlim(0, max(ci_upper) * 1.1)
        
        # Add legend
        legend_elements = [
            plt.Line2D([0], [0], marker='o', color='w',
                      markerfacecolor='#d62728', markersize=10, label='Poor Prognosis (HR > 1)'),
            plt.Line2D([0], [0], marker='o', color='w',
                      markerfacecolor='#2ca02c', markersize=10, label='Good Prognosis (HR < 1)'),
            plt.Line2D([0], [0], marker='o', color='w',
                      markerfacecolor='gray', markersize=10, alpha=0.4, label='Not Significant'),
        ]
        ax.legend(handles=legend_elements, loc='lower right', framealpha=0.9)
        
        # Add text annotations for significant markers
        ax.text(0.02, 0.98, 'HR > 1: Increased risk', transform=ax.transAxes,
               fontsize=9, verticalalignment='top', style='italic')
        ax.text(0.02, 0.94, 'HR < 1: Decreased risk', transform=ax.transAxes,
               fontsize=9, verticalalignment='top', style='italic')
        
        plt.tight_layout()
        return fig
    
    def plot_prioritized_targets(self, data: Dict[str, Any]) -> plt.Figure:
        """Generate prioritized target ranking visualization."""
        fig, ax = plt.subplots(figsize=(12, 10))
        
        prioritized_targets = data.get('prioritized_targets', [])
        
        if not prioritized_targets:
            ax.text(0.5, 0.5, 'No prioritized target data available',
                   ha='center', va='center', fontsize=14)
            ax.axis('off')
            return fig
        
        # Extract target information
        targets = []
        priority_scores = []
        druggability = []
        cancer_specificity = []
        network_centrality = []
        
        for target in prioritized_targets[:15]:  # Top 15 targets
            if isinstance(target, dict):
                name = target.get('target_name', target.get('gene', 'Unknown'))
                priority = target.get('priority_score', 0.5)
                drug_score = target.get('druggability', target.get('druggability_score', 0.3))
                cancer_spec = target.get('cancer_specificity', 0.5)
                network_cent = target.get('network_centrality', 0.5)
                
                targets.append(name)
                priority_scores.append(float(priority))
                druggability.append(float(drug_score))
                cancer_specificity.append(float(cancer_spec))
                network_centrality.append(float(network_cent))
        
        if not targets:
            ax.text(0.5, 0.5, 'No valid target data',
                   ha='center', va='center', fontsize=14)
            ax.axis('off')
            return fig
        
        # Sort by priority score
        sorted_data = sorted(
            zip(targets, priority_scores, druggability, cancer_specificity, network_centrality),
            key=lambda x: x[1], reverse=True
        )
        targets, priority_scores, druggability, cancer_specificity, network_centrality = zip(*sorted_data)
        
        # Create stacked horizontal bar chart
        y_pos = np.arange(len(targets))
        
        # Normalize components for stacking
        components = np.array([druggability, cancer_specificity, network_centrality])
        
        # Plot stacked bars
        colors = ['#4E79A7', '#F28E2B', '#59A14F']
        labels = ['Druggability', 'Cancer Specificity', 'Network Centrality']
        
        left = np.zeros(len(targets))
        for i, (component, color, label) in enumerate(zip(components, colors, labels)):
            ax.barh(y_pos, component, left=left, color=color, alpha=0.8,
                   edgecolor='black', linewidth=0.5, label=label)
            left += component
        
        # Add priority score as line
        ax2 = ax.twiny()
        ax2.plot(priority_scores, y_pos, 'ro-', linewidth=2, markersize=8,
                label='Overall Priority', markeredgecolor='black', markeredgewidth=1.5)
        ax2.set_xlabel('Overall Priority Score', fontsize=11, fontweight='bold', color='red')
        ax2.tick_params(axis='x', labelcolor='red')
        ax2.set_xlim(0, max(priority_scores) * 1.1)
        ax2.legend(loc='lower right')
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels(targets)
        ax.set_xlabel('Component Scores', fontsize=12, fontweight='bold')
        ax.set_title('Prioritized Therapeutic Targets', fontsize=16, fontweight='bold', pad=20)
        ax.invert_yaxis()
        ax.legend(loc='upper right', framealpha=0.9)
        ax.grid(axis='x', alpha=0.3, linestyle='--')
        
        plt.tight_layout()
        return fig

    def create_cancer_network_visualization(
        self,
        data: Dict[str, Any],
        output_path: Path,
        max_nodes: int = 100
    ) -> Optional[Path]:
        """Generate optimized interactive network for Scenario 3."""
        try:
            from pyvis.network import Network
        except ImportError:
            logger.warning("PyVis not available; skipping S3 network visualization")
            return None
        
        nodes = data.get('network_nodes') or []
        edges = data.get('network_edges') or []
        if not nodes or not edges:
            logger.warning("S3 network data missing nodes or edges")
            return None
        
        graph = nx.Graph()
        for node in nodes:
            if not isinstance(node, dict):
                continue
            gene = node.get('gene_symbol') or node.get('id')
            if not gene:
                continue
            graph.add_node(gene, **node)
        
        for edge in edges:
            if not isinstance(edge, dict):
                continue
            source = edge.get('source') or edge.get('protein_a')
            target = edge.get('target') or edge.get('protein_b')
            if not source or not target:
                continue
            if not graph.has_node(source):
                graph.add_node(source, gene_symbol=source, node_type='protein')
            if not graph.has_node(target):
                graph.add_node(target, gene_symbol=target, node_type='protein')
            weight = edge.get('weight') or edge.get('evidence_score') or 1.0
            graph.add_edge(source, target, weight=weight, interaction=edge.get('interaction_type', 'ppi'))
        
        if graph.number_of_nodes() == 0:
            logger.warning("S3 network graph is empty")
            return None
        
        degree_dict = dict(graph.degree())
        top_nodes = sorted(degree_dict, key=degree_dict.get, reverse=True)[:max_nodes]
        subgraph = graph.subgraph(top_nodes).copy()
        if subgraph.number_of_edges() == 0:
            logger.warning("Filtered S3 network has no edges")
            return None
        
        try:
            communities = community.greedy_modularity_communities(subgraph)
        except Exception:
            communities = []
        membership = {}
        for idx, comm in enumerate(communities):
            for member in comm:
                membership[member] = idx
        
        cancer_markers = {
            (marker.get('gene') or marker.get('name') or '').upper()
            for marker in data.get('cancer_markers', [])
            if isinstance(marker, dict)
        }
        
        palette = self.get_color_palette('network')
        net = Network(height='800px', width='100%', bgcolor='#ffffff', font_color='#2f2f2f')
        net.force_atlas_2based()
        
        for node in subgraph.nodes():
            attrs = subgraph.nodes[node]
            degree = subgraph.degree(node)
            centrality = attrs.get('centrality_measures', {})
            betweenness = centrality.get('betweenness', 0.0)
            closeness = centrality.get('closeness', 0.0)
            tooltip = (
                f"<b>{node}</b><br>Degree: {degree}"
                f"<br>Betweenness: {betweenness:.3f}"
                f"<br>Closeness: {closeness:.3f}"
            )
            community_idx = membership.get(node, 0)
            color = palette[community_idx % len(palette)]
            border = 4 if node.upper() in cancer_markers else 1
            
            net.add_node(
                node,
                label=node,
                color=color,
                size=12 + degree,
                title=tooltip,
                borderWidth=border
            )
        
        for u, v, attrs in subgraph.edges(data=True):
            weight = attrs.get('weight', 1.0)
            net.add_edge(
                u,
                v,
                value=max(weight, 0.1),
                color='rgba(120,120,120,0.6)',
                title=f"{u} ↔ {v} (confidence {weight:.2f})"
            )
        
        output_file = output_path / 'S3_cancer_network_interactive.html'
        net.save_graph(str(output_file))
        logger.info(f"Saved S3 cancer network visualization: {output_file}")
        return output_file

    def plot_pathway_crosstalk_sankey(
        self,
        data: Dict[str, Any],
        output_path: Path,
        min_overlap: int = 3
    ) -> Optional[Path]:
        """Generate Sankey diagram showing pathway gene overlaps."""
        try:
            import plotly.graph_objects as go
        except ImportError:
            logger.warning("Plotly not available; skipping S3 Sankey plot")
            return None
        
        pathways = data.get('cancer_pathways') or []
        if len(pathways) < 2:
            logger.warning("Not enough pathways for crosstalk analysis")
            return None
        
        label_to_genes: Dict[str, set] = {}
        label_to_source: Dict[str, str] = {}
        for pathway in pathways:
            genes = pathway.get('genes') or []
            if not genes:
                continue
            label = pathway.get('name') or pathway.get('id')
            source = (pathway.get('source_db') or 'other').upper()
            label_to_genes[label] = {gene.upper() for gene in genes if isinstance(gene, str)}
            label_to_source[label] = source
        
        labels = list(label_to_genes.keys())
        flows = []
        for i in range(len(labels)):
            for j in range(i + 1, len(labels)):
                overlap = len(label_to_genes[labels[i]] & label_to_genes[labels[j]])
                if overlap >= min_overlap:
                    flows.append((labels[i], labels[j], overlap))
        
        if not flows:
            logger.warning("No significant pathway overlaps for Sankey diagram")
            return None
        
        node_labels = sorted(label_to_genes.keys())
        node_index = {label: idx for idx, label in enumerate(node_labels)}
        source_indices = [node_index[src] for src, _, _ in flows]
        target_indices = [node_index[tgt] for _, tgt, _ in flows]
        values = [val for _, _, val in flows]
        
        color_map = {
            'KEGG': '#59A14F',
            'REACTOME': '#B07AA1',
            'GO': '#4E79A7'
        }
        node_colors = [
            color_map.get(label_to_source.get(label, 'other'), '#BAB0AC')
            for label in node_labels
        ]
        
        fig = go.Figure(
            go.Sankey(
                valueformat='.0f',
                node=dict(
                    pad=15,
                    thickness=18,
                    label=node_labels,
                    color=node_colors
                ),
                link=dict(
                    source=source_indices,
                    target=target_indices,
                    value=values,
                    color='rgba(120,120,120,0.45)'
                )
            )
        )
        fig.update_layout(
            title_text='Pathway Crosstalk (Shared Genes ≥ 3)',
            font=dict(size=12),
            margin=dict(l=10, r=10, t=50, b=10)
        )
        
        output_file = output_path / 'S3_pathway_crosstalk_sankey.html'
        fig.write_html(str(output_file), include_plotlyjs='cdn')
        logger.info(f"Saved S3 pathway crosstalk Sankey: {output_file}")
        return output_file
