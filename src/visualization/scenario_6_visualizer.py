"""
Scenario 6: Drug Repurposing Visualizer

Generates visualizations for drug repurposing analysis including:
- Drug volcano plot (efficacy vs safety)
- Drug-target bipartite network (static + interactive)
- Repurposing candidate ranking
- Safety profile radar charts
"""

import logging
from collections import defaultdict
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import cm

from src.visualization.base import BaseVisualizer
from src.visualization.styles import VisualizationStyles

logger = logging.getLogger(__name__)


class DrugRepurposingVisualizer(BaseVisualizer):
    """Visualizer for Scenario 6: Drug Repurposing."""
    
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
        """Generate all visualizations for Scenario 6."""
        output_path = self.create_output_dir(output_dir, 6)
        generated_files = []
        
        logger.info("Generating Scenario 6 visualizations")
        
        # 1. Drug volcano plot
        try:
            fig = self.plot_drug_volcano(data)
            files = self.save_figure(fig, output_path, 'S6_drug_volcano', formats)
            generated_files.extend(files)
        except Exception as e:
            logger.error(f"Failed to generate drug volcano plot: {e}")
        
        # 1b. Repurposing score distribution
        try:
            fig = self.plot_score_distribution(data)
            files = self.save_figure(fig, output_path, 'S6_score_distribution', formats)
            generated_files.extend(files)
        except Exception as e:
            logger.error(f"Failed to generate score distribution plot: {e}")
        
        # 2. Drug-target network
        try:
            fig = self.plot_drug_target_network(data)
            files = self.save_figure(fig, output_path, 'S6_drug_target_network', formats)
            generated_files.extend(files)
            
            if interactive:
                html_file = self.plot_drug_target_network_interactive(data, output_path)
                if html_file:
                    generated_files.append(html_file)
        except Exception as e:
            logger.error(f"Failed to generate drug-target network: {e}")
        
        # 3. Repurposing candidates
        try:
            fig = self.plot_repurposing_candidates(data)
            files = self.save_figure(fig, output_path, 'S6_repurposing_candidates', formats)
            generated_files.extend(files)
        except Exception as e:
            logger.error(f"Failed to generate repurposing candidates plot: {e}")
        
        # 3b. Target-drug heatmap
        try:
            fig = self.plot_target_drug_heatmap(data)
            files = self.save_figure(fig, output_path, 'S6_target_drug_heatmap', formats)
            generated_files.extend(files)
        except Exception as e:
            logger.error(f"Failed to generate target-drug heatmap: {e}")
        
        # 4. Safety profiles
        try:
            fig = self.plot_safety_profiles(data)
            files = self.save_figure(fig, output_path, 'S6_safety_profiles', formats)
            generated_files.extend(files)
        except Exception as e:
            logger.error(f"Failed to generate safety profiles plot: {e}")
        
        # 5. Interactive Plotly visualizations (Figures 11-13)
        if interactive:
            # Figure 11: Bioactivity Scatter Plot
            try:
                html_file = self.plot_bioactivity_scatter_plotly(data, output_path)
                if html_file:
                    generated_files.append(html_file)
            except Exception as e:
                logger.error(f"Failed to generate bioactivity scatter: {e}")
            
            # Figure 12: Target Coverage Sunburst
            try:
                html_file = self.plot_target_coverage_sunburst_plotly(data, output_path)
                if html_file:
                    generated_files.append(html_file)
            except Exception as e:
                logger.error(f"Failed to generate target coverage sunburst: {e}")
            
            # Figure 13: Drug Comparison Dashboard
            try:
                html_file = self.plot_drug_comparison_dashboard_plotly(data, output_path)
                if html_file:
                    generated_files.append(html_file)
            except Exception as e:
                logger.error(f"Failed to generate drug comparison dashboard: {e}")
            
            # Interactive Volcano Plot
            try:
                html_file = self.plot_volcano_plotly(data, output_path)
                if html_file:
                    generated_files.append(html_file)
            except Exception as e:
                logger.error(f"Failed to generate interactive volcano: {e}")
        
        logger.info(f"Generated {len(generated_files)} files for Scenario 6")
        return generated_files

    # ------------------------------------------------------------------
    # Internal helpers for network-focused visualizations
    # ------------------------------------------------------------------
    def _build_gene_to_pathways_map(self, pathways: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """Create mapping from gene symbol to pathway metadata."""
        gene_map: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for pathway in pathways or []:
            genes = pathway.get('genes') or []
            if not genes:
                continue
            pathway_payload = {
                'id': pathway.get('id') or pathway.get('name'),
                'name': pathway.get('name', pathway.get('id', 'Unknown pathway')),
                'source_db': (pathway.get('source_db') or 'unknown').lower(),
                'gene_count': len(genes)
            }
            for gene in genes:
                if isinstance(gene, str):
                    gene_map[gene.upper()].append(pathway_payload)
        return gene_map

    def _create_drug_target_network(
        self,
        data: Dict[str, Any],
        max_drugs: int = 25,
        max_pathways_per_gene: int = 2
    ) -> nx.Graph:
        """
        Build a multi-layer graph connecting drugs, targets, and pathways.
        
        Returns:
            networkx.Graph with node/edge metadata ready for plotting.
        """
        candidate_drugs = data.get('candidate_drugs', [])
        if not candidate_drugs:
            return nx.Graph()
        
        # Prioritize highest scoring candidates
        sorted_candidates = sorted(
            candidate_drugs,
            key=lambda c: c.get('repurposing_score', 0.0) or 0.0,
            reverse=True
        )[:max_drugs]
        
        network_validation = data.get('network_validation', {})
        node_lookup = {}
        for node in network_validation.get('nodes', []):
            gene_symbol = (node.get('gene_symbol') or node.get('id') or '').upper()
            if gene_symbol:
                node_lookup[gene_symbol] = node
        
        gene_to_pathways = self._build_gene_to_pathways_map(
            data.get('disease_pathways', [])
        )
        
        drug_scores = [
            c.get('repurposing_score', 0.0) or 0.0
            for c in sorted_candidates
        ]
        score_min = min(drug_scores) if drug_scores else 0.0
        score_max = max(drug_scores) if drug_scores else 1.0
        score_range = score_max - score_min if score_max != score_min else 1.0
        cmap = cm.get_cmap('RdYlGn')
        
        def score_to_color(score: float) -> str:
            normalized = (score - score_min) / score_range if score_range else 0.5
            return mcolors.to_hex(cmap(float(np.clip(normalized, 0, 1))))
        
        target_color = VisualizationStyles.get_color('network', 'edge_default')
        pathway_color_map = {
            'kegg': '#59A14F',
            'reactome': '#B07AA1',
            'go': '#4E79A7'
        }
        
        graph = nx.Graph()
        
        for candidate in sorted_candidates:
            if not isinstance(candidate, dict):
                continue
            drug_id = candidate.get('drug_id') or candidate.get('drug_name')
            if not drug_id:
                continue
            drug_name = candidate.get('drug_name', drug_id)
            score = float(candidate.get('repurposing_score', 0.0) or 0.0)
            bioactivity = candidate.get('bioactivity_nm')
            approval = (
                candidate.get('approval_status')
                or candidate.get('safety_profile', {}).get('approval_status')
                or 'unknown'
            )
            drug_color = score_to_color(score)
            drug_size = 500 + 300 * ((score - score_min) / score_range if score_range else 0.5)
            
            graph.add_node(
                drug_id,
                node_type='drug',
                label=drug_name,
                color=drug_color,
                size=drug_size,
                score=score,
                approval_status=approval,
                bioactivity=bioactivity
            )
            
            target_gene = candidate.get('target_protein') or candidate.get('target_gene')
            if not target_gene:
                continue
            target_gene = str(target_gene).upper()
            
            node_meta = node_lookup.get(target_gene, {})
            centrality = node_meta.get('centrality_measures', {})
            degree = centrality.get('degree') or 0.0
            betweenness = centrality.get('betweenness') or 0.0
            target_size = 350 + 15 * degree
            target_label = node_meta.get('gene_symbol') or target_gene
            
            graph.add_node(
                target_gene,
                node_type='target',
                label=target_label,
                color=target_color,
                size=target_size,
                degree=degree,
                betweenness=betweenness
            )
            
            graph.add_edge(
                drug_id,
                target_gene,
                edge_type='drug-target',
                weight=max(score, 0.1),
                tooltip=f"{drug_name} → {target_label} (score {score:.2f})"
            )
            
            associated_pathways = gene_to_pathways.get(target_gene, [])
            for pathway in associated_pathways[:max_pathways_per_gene]:
                pathway_id = pathway.get('id') or pathway.get('name')
                if not pathway_id:
                    continue
                pathway_label = pathway.get('name', pathway_id)
                source_db = pathway.get('source_db', 'unknown').lower()
                path_color = pathway_color_map.get(source_db, '#BAB0AC')
                path_size = 300 + 8 * min(pathway.get('gene_count', 10), 50)
                
                graph.add_node(
                    pathway_id,
                    node_type='pathway',
                    label=pathway_label,
                    color=path_color,
                    size=path_size,
                    source_db=source_db,
                    gene_count=pathway.get('gene_count', 0)
                )
                
                graph.add_edge(
                    target_gene,
                    pathway_id,
                    edge_type='target-pathway',
                    weight=0.3,
                    tooltip=f"{target_label} → {pathway_label}"
                )
        
        graph.graph['score_range'] = (score_min, score_max)
        return graph

    def _layered_layout(self, graph: nx.Graph) -> Dict[Any, Tuple[float, float]]:
        """Generate deterministic layered layout (drugs → targets → pathways)."""
        layout: Dict[Any, Tuple[float, float]] = {}
        layers = {'drug': [], 'target': [], 'pathway': []}
        for node, attrs in graph.nodes(data=True):
            node_type = attrs.get('node_type', 'target')
            layers.setdefault(node_type, []).append(node)
        
        x_positions = {'drug': 0.0, 'target': 0.5, 'pathway': 1.0}
        for layer_name, nodes in layers.items():
            if not nodes:
                continue
            nodes_sorted = sorted(nodes)
            count = len(nodes_sorted)
            for idx, node in enumerate(nodes_sorted):
                if count == 1:
                    y = 0.5
                else:
                    y = idx / (count - 1)
                layout[node] = (x_positions.get(layer_name, 0.5), y)
        return layout

    def _format_node_tooltip(self, attrs: Dict[str, Any]) -> str:
        """Create HTML tooltip text for interactive visualization."""
        node_type = attrs.get('node_type', '')
        label = attrs.get('label', '')
        if node_type == 'drug':
            score = attrs.get('score')
            approval = attrs.get('approval_status', 'unknown')
            bioactivity = attrs.get('bioactivity')
            tooltip = f"<b>{label}</b><br>Repurposing score: {score:.2f}" if score is not None else f"<b>{label}</b>"
            tooltip += f"<br>Approval: {approval.title() if isinstance(approval, str) else approval}"
            if bioactivity:
                tooltip += f"<br>Bioactivity: {bioactivity} nM"
            return tooltip
        if node_type == 'target':
            degree = attrs.get('degree')
            betweenness = attrs.get('betweenness')
            tooltip = f"<b>{label}</b><br>Node type: Target"
            if degree is not None:
                tooltip += f"<br>Degree: {degree:.0f}"
            if betweenness is not None:
                tooltip += f"<br>Betweenness: {betweenness:.3f}"
            return tooltip
        if node_type == 'pathway':
            source = attrs.get('source_db', 'unknown')
            gene_count = attrs.get('gene_count')
            tooltip = f"<b>{label}</b><br>Source: {source.upper()}"
            if gene_count is not None:
                tooltip += f"<br>Genes: {gene_count}"
            return tooltip
        return label or ''
    
    def plot_drug_volcano(self, data: Dict[str, Any]) -> plt.Figure:
        """Generate volcano plot for drug efficacy vs safety."""
        fig, ax = plt.subplots(figsize=(12, 10))
        
        repurposing_scores = data.get('repurposing_scores', {})
        candidate_drugs = data.get('top_candidates', data.get('candidate_drugs', []))
        
        if not candidate_drugs and not repurposing_scores:
            ax.text(0.5, 0.5, 'No drug repurposing data available',
                   ha='center', va='center', fontsize=14)
            ax.axis('off')
            return fig
        
        # Extract drug data
        drugs = []
        efficacy_scores = []
        safety_scores = []
        
        for drug in candidate_drugs[:50]:  # Top 50 drugs
            if isinstance(drug, dict):
                name = drug.get('drug_name', drug.get('name', 'Unknown'))
                efficacy = drug.get('efficacy_score', drug.get('repurposing_score', 0.5))
                safety = drug.get('safety_score', drug.get('safety_profile', {}).get('overall_score', 0.7))
                
                drugs.append(name)
                efficacy_scores.append(float(efficacy))
                safety_scores.append(float(safety))
        
        if not drugs:
            ax.text(0.5, 0.5, 'No valid drug data',
                   ha='center', va='center', fontsize=14)
            ax.axis('off')
            return fig
        
        # Calculate significance (combine efficacy and safety)
        significance = [-np.log10(max(0.001, 1 - (e * s))) for e, s in zip(efficacy_scores, safety_scores)]
        
        # Color by combined score
        colors = []
        for e, s in zip(efficacy_scores, safety_scores):
            if e > 0.7 and s > 0.7:
                colors.append('#2ca02c')  # High efficacy, high safety
            elif e > 0.7:
                colors.append('#ff7f0e')  # High efficacy, lower safety
            elif s > 0.7:
                colors.append('#1f77b4')  # Lower efficacy, high safety
            else:
                colors.append('#7f7f7f')  # Low both
        
        # Create scatter plot
        scatter = ax.scatter(efficacy_scores, significance, c=colors, s=100,
                           alpha=0.6, edgecolors='black', linewidth=1)
        
        # Label top candidates
        top_indices = sorted(range(len(significance)), key=lambda i: significance[i], reverse=True)[:10]
        for i in top_indices:
            ax.annotate(drugs[i], (efficacy_scores[i], significance[i]),
                       xytext=(5, 5), textcoords='offset points',
                       fontsize=8, alpha=0.8,
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.3))
        
        # Add threshold lines
        ax.axvline(x=0.7, color='red', linestyle='--', linewidth=1.5, alpha=0.5,
                  label='High efficacy threshold')
        ax.axhline(y=-np.log10(0.05), color='blue', linestyle='--', linewidth=1.5, alpha=0.5,
                  label='Significance threshold')
        
        ax.set_xlabel('Repurposing Efficacy Score', fontsize=12, fontweight='bold')
        ax.set_ylabel('-log10(p-value)', fontsize=12, fontweight='bold')
        ax.set_title('Drug Repurposing Volcano Plot', fontsize=16, fontweight='bold', pad=20)
        ax.grid(alpha=0.3, linestyle='--')
        ax.legend(loc='upper left')
        
        # Add color legend
        legend_elements = [
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#2ca02c',
                      markersize=10, label='High Efficacy & Safety'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#ff7f0e',
                      markersize=10, label='High Efficacy'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#1f77b4',
                      markersize=10, label='High Safety'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#7f7f7f',
                      markersize=10, label='Low Both'),
        ]
        ax.legend(handles=legend_elements, loc='upper right', framealpha=0.9)
        
        plt.tight_layout()
        return fig
    
    def plot_score_distribution(self, data: Dict[str, Any]) -> plt.Figure:
        """Visualize distribution of repurposing scores with approval stratification."""
        fig, ax = plt.subplots(figsize=(11, 7))
        candidates = data.get('candidate_drugs', [])
        if not candidates:
            ax.text(0.5, 0.5, 'No repurposing candidates available',
                    ha='center', va='center', fontsize=14)
            ax.axis('off')
            return fig
        
        rows = []
        for cand in candidates:
            if not isinstance(cand, dict):
                continue
            score = cand.get('repurposing_score')
            if score is None:
                continue
            approval = (
                cand.get('approval_status')
                or cand.get('safety_profile', {}).get('approval_status')
                or 'unknown'
            )
            approval = approval.lower()
            if 'approved' in approval or 'phase 4' in approval:
                bucket = 'Approved'
            elif 'phase 3' in approval:
                bucket = 'Phase 3'
            elif 'phase 2' in approval:
                bucket = 'Phase 2'
            else:
                bucket = 'Other'
            rows.append({'score': float(score), 'bucket': bucket})
        
        if not rows:
            ax.text(0.5, 0.5, 'No valid score data',
                    ha='center', va='center', fontsize=14)
            ax.axis('off')
            return fig
        
        df = pd.DataFrame(rows)

        # Handle zero variance case
        score_range = df['score'].max() - df['score'].min()
        if score_range < 0.001:  # Effectively zero variance
            # Use bins instead of binwidth for zero variance
            sns.histplot(
                df,
                x='score',
                hue='bucket',
                multiple='stack',
                bins=1,  # Single bin for zero variance
                palette=['#2ca02c', '#59A14F', '#F28E2B', '#7f7f7f'],
                edgecolor='black',
                alpha=0.85,
                ax=ax
            )
            # Skip KDE for zero variance (would fail)
            ax.axvline(df['score'].mean(), color='black', linewidth=2, linestyle='-', label='Mean (zero variance)')
        else:
            # Normal case with variance
            sns.histplot(
                df,
                x='score',
                hue='bucket',
                multiple='stack',
                binwidth=0.02,
                palette=['#2ca02c', '#59A14F', '#F28E2B', '#7f7f7f'],
                edgecolor='black',
                alpha=0.85,
                ax=ax
            )
            sns.kdeplot(df['score'], color='black', linewidth=2, ax=ax, label='Density')
        
        ax.axvline(0.50, color='red', linestyle='--', linewidth=1.5, alpha=0.7)
        ax.axvline(0.55, color='red', linestyle='--', linewidth=1.5, alpha=0.7)
        ax.text(0.50, ax.get_ylim()[1]*0.95, '0.50 threshold', rotation=90,
                color='red', ha='right', va='top', fontsize=9)
        ax.text(0.55, ax.get_ylim()[1]*0.95, '0.55 threshold', rotation=90,
                color='red', ha='left', va='top', fontsize=9)
        
        summary = df['score'].describe()
        annotation = (
            f"n={len(df)} | mean={summary['mean']:.3f} "
            f"| std={summary['std']:.3f}\n"
            f"min={summary['min']:.3f} | max={summary['max']:.3f}"
        )
        ax.text(
            0.02,
            0.98,
            annotation,
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.7)
        )
        
        ax.set_xlabel('Repurposing Score', fontsize=12, fontweight='bold')
        ax.set_ylabel('Candidate Count', fontsize=12, fontweight='bold')
        ax.set_title('Scenario 6 Repurposing Score Distribution', fontsize=16, fontweight='bold', pad=20)
        ax.set_xlim(0, 1.0)
        ax.grid(alpha=0.3, linestyle='--')
        ax.legend(title='Approval Bucket')
        plt.tight_layout()
        return fig
    
    def plot_drug_target_network(self, data: Dict[str, Any]) -> plt.Figure:
        """Generate layered drug-target-pathway network visualization."""
        fig, ax = plt.subplots(figsize=(14, 10))
        graph = self._create_drug_target_network(data)
        
        if graph.number_of_nodes() == 0:
            ax.text(
                0.5,
                0.5,
                'No drug-target network data available',
                ha='center',
                va='center',
                fontsize=14
            )
            ax.axis('off')
            return fig
        
        pos = self._layered_layout(graph)
        node_groups = defaultdict(list)
        for node, attrs in graph.nodes(data=True):
            node_groups[attrs.get('node_type', 'target')].append(node)
        
        # Draw nodes by type with shapes
        shape_map = {'drug': 's', 'target': 'o', 'pathway': '^'}
        for node_type, nodes in node_groups.items():
            if not nodes:
                continue
            node_colors = [graph.nodes[n].get('color', '#4E79A7') for n in nodes]
            node_sizes = [graph.nodes[n].get('size', 400) for n in nodes]
            nx.draw_networkx_nodes(
                graph,
                pos,
                nodelist=nodes,
                node_color=node_colors,
                node_size=node_sizes,
                alpha=0.9,
                node_shape=shape_map.get(node_type, 'o'),
                edgecolors='black',
                linewidths=0.8,
                ax=ax
            )
        
        # Draw edges with styles
        drug_target_edges = [
            (u, v) for u, v, attrs in graph.edges(data=True)
            if attrs.get('edge_type') == 'drug-target'
        ]
        target_pathway_edges = [
            (u, v) for u, v, attrs in graph.edges(data=True)
            if attrs.get('edge_type') == 'target-pathway'
        ]
        
        if drug_target_edges:
            widths = [
                1.5 + graph.edges[e].get('weight', 0.3) * 2
                for e in drug_target_edges
            ]
            nx.draw_networkx_edges(
                graph,
                pos,
                edgelist=drug_target_edges,
                width=widths,
                alpha=0.6,
                edge_color='#6b6b6b',
                ax=ax
            )
        
        if target_pathway_edges:
            nx.draw_networkx_edges(
                graph,
                pos,
                edgelist=target_pathway_edges,
                width=1.2,
                alpha=0.4,
                style='dashed',
                edge_color='#9c9c9c',
                ax=ax
            )
        
        # Labels
        labels = {node: graph.nodes[node].get('label', node)[:20] for node in graph.nodes()}
        nx.draw_networkx_labels(
            graph,
            pos,
            labels=labels,
            font_size=8,
            font_weight='bold',
            ax=ax
        )
        
        ax.set_title('Drug → Target → Pathway Network', fontsize=16, fontweight='bold', pad=20)
        ax.axis('off')
        
        # Add legend with shapes
        legend_elements = [
            plt.Line2D([0], [0], marker='s', color='w',
                       markerfacecolor='#4E79A7', markeredgecolor='black',
                       markersize=10, label='Drugs'),
            plt.Line2D([0], [0], marker='o', color='w',
                       markerfacecolor=VisualizationStyles.get_color('network', 'edge_default'),
                       markeredgecolor='black', markersize=10, label='Targets'),
            plt.Line2D([0], [0], marker='^', color='w',
                       markerfacecolor='#59A14F', markeredgecolor='black',
                       markersize=10, label='Pathways (KEGG/Reactome)')
        ]
        ax.legend(handles=legend_elements, loc='upper left', framealpha=0.9)
        
        # Add statistics and colorbar
        score_min, score_max = graph.graph.get('score_range', (0.0, 1.0))
        sm = plt.cm.ScalarMappable(
            cmap=cm.get_cmap('RdYlGn'),
            norm=mcolors.Normalize(vmin=score_min, vmax=score_max if score_max != score_min else score_min + 1)
        )
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, fraction=0.03, pad=0.02)
        cbar.set_label('Repurposing Score', fontsize=10)
        
        stats_text = (
            f"Nodes: {graph.number_of_nodes()} | Edges: {graph.number_of_edges()}"
        )
        ax.text(
            0.5,
            0.02,
            stats_text,
            transform=ax.transAxes,
            fontsize=10,
            ha='center',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.6)
        )
        
        plt.tight_layout()
        return fig
    
    def plot_repurposing_candidates(self, data: Dict[str, Any]) -> plt.Figure:
        """Generate ranked visualization of repurposing candidates."""
        fig, ax = plt.subplots(figsize=(12, 10))
        
        candidate_drugs = data.get('top_candidates', data.get('candidate_drugs', []))
        
        if not candidate_drugs:
            ax.text(0.5, 0.5, 'No repurposing candidate data available',
                   ha='center', va='center', fontsize=14)
            ax.axis('off')
            return fig
        
        # Extract and sort candidates
        drugs = []
        scores = []
        phases = []
        
        for drug in candidate_drugs[:20]:  # Top 20
            if isinstance(drug, dict):
                name = drug.get('drug_name', drug.get('name', 'Unknown'))
                score = drug.get('repurposing_score', drug.get('efficacy_score', 0.5))
                phase = (
                    drug.get('approval_status')
                    or drug.get('clinical_phase')
                    or drug.get('development_phase', 'Preclinical')
                )
                
                drugs.append(name)
                scores.append(float(score))
                phases.append(str(phase))
        
        if not drugs:
            ax.text(0.5, 0.5, 'No valid candidate data',
                   ha='center', va='center', fontsize=14)
            ax.axis('off')
            return fig
        
        # Sort by score
        sorted_data = sorted(zip(drugs, scores, phases), key=lambda x: x[1], reverse=True)
        drugs, scores, phases = zip(*sorted_data)
        
        # Color by clinical phase
        phase_colors = {
            'approved': '#2ca02c',
            'phase 4': '#2ca02c',
            'phase 3': '#59A14F',
            'phase 2': '#F28E2B',
            'phase 1': '#ff7f0e',
            'preclinical': '#7f7f7f',
            'investigational': '#9467bd',
        }
        
        colors = [phase_colors.get(phase.lower(), '#7f7f7f') for phase in phases]
        
        y_pos = np.arange(len(drugs))
        bars = ax.barh(y_pos, scores, color=colors, alpha=0.7,
                      edgecolor='black', linewidth=1)
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels(drugs)
        ax.set_xlabel('Repurposing Score', fontsize=12, fontweight='bold')
        ax.set_title('Top Repurposing Candidates', fontsize=16, fontweight='bold', pad=20)
        ax.invert_yaxis()
        ax.grid(axis='x', alpha=0.3, linestyle='--')
        ax.set_xlim(0, max(scores) * 1.1)
        
        # Add value labels
        for i, (bar, score) in enumerate(zip(bars, scores)):
            width = bar.get_width()
            ax.text(width + 0.01, bar.get_y() + bar.get_height()/2,
                   f'{score:.2f}', ha='left', va='center', fontweight='bold')
        
        # Add legend
        legend_elements = [
            plt.Line2D([0], [0], color=color, lw=4, label=phase.title())
            for phase, color in phase_colors.items()
        ]
        ax.legend(handles=legend_elements, loc='lower right', framealpha=0.9,
                 title='Clinical Phase')
        
        plt.tight_layout()
        return fig

    def plot_target_drug_heatmap(self, data: Dict[str, Any]) -> plt.Figure:
        """Heatmap of top targets vs top drugs using repurposing scores."""
        fig, ax = plt.subplots(figsize=(12, 10))
        candidates = data.get('candidate_drugs', [])
        rows = []
        for cand in candidates:
            if not isinstance(cand, dict):
                continue
            drug = cand.get('drug_name') or cand.get('drug_id')
            target = cand.get('target_protein') or cand.get('target_gene')
            score = cand.get('repurposing_score')
            if drug and target and score is not None:
                rows.append({'drug': str(drug), 'target': str(target), 'score': float(score)})
        
        if not rows:
            ax.text(0.5, 0.5, 'Insufficient drug-target score data',
                    ha='center', va='center', fontsize=14)
            ax.axis('off')
            return fig
        
        df = pd.DataFrame(rows)
        top_targets = (
            df.groupby('target')['score']
            .mean()
            .sort_values(ascending=False)
            .head(20)
            .index
        )
        top_drugs = (
            df.groupby('drug')['score']
            .mean()
            .sort_values(ascending=False)
            .head(20)
            .index
        )
        matrix = (
            df[df['target'].isin(top_targets) & df['drug'].isin(top_drugs)]
            .pivot_table(index='target', columns='drug', values='score', aggfunc='max')
            .reindex(index=top_targets, columns=top_drugs)
        )
        
        if matrix.empty:
            ax.text(0.5, 0.5, 'Not enough overlap for heatmap',
                    ha='center', va='center', fontsize=14)
            ax.axis('off')
            return fig
        
        sns.heatmap(
            matrix,
            annot=True,
            fmt='.2f',
            cmap='YlGnBu',
            vmin=0,
            vmax=1,
            cbar_kws={'label': 'Repurposing Score'},
            linewidths=0.5,
            linecolor='white',
            ax=ax
        )
        ax.set_xlabel('Top Drugs', fontsize=12, fontweight='bold')
        ax.set_ylabel('Top Targets', fontsize=12, fontweight='bold')
        ax.set_title('Scenario 6 Target-Drug Heatmap', fontsize=16, fontweight='bold', pad=20)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        return fig
    
    def plot_safety_profiles(self, data: Dict[str, Any]) -> plt.Figure:
        """Generate safety profile comparison (radar charts)."""
        fig, axes = plt.subplots(2, 3, figsize=(16, 12), subplot_kw=dict(projection='polar'))
        axes = axes.flatten()
        
        candidate_drugs = data.get('top_candidates', data.get('candidate_drugs', []))
        safety_profiles = data.get('safety_profiles', {})
        
        if not candidate_drugs:
            axes[0].text(0.5, 0.5, 'No safety profile data available',
                        ha='center', va='center', fontsize=14)
            for ax in axes:
                ax.axis('off')
            return fig
        
        # Safety categories
        categories = ['Hepatotoxicity', 'Cardiotoxicity', 'Nephrotoxicity',
                     'CNS Effects', 'GI Effects', 'Overall Safety']
        num_vars = len(categories)
        
        # Compute angle for each axis
        angles = [n / float(num_vars) * 2 * np.pi for n in range(num_vars)]
        angles += angles[:1]
        
        # Plot top 6 drugs
        for idx, drug in enumerate(candidate_drugs[:6]):
            if isinstance(drug, dict):
                drug_name = drug.get('drug_name', drug.get('name', f'Drug {idx+1}'))
                
                # Get safety scores (inverted so higher is safer)
                if isinstance(safety_profiles, dict):
                    profile = safety_profiles.get(drug_name, {})
                else:
                    profile = {}
                
                values = [
                    1.0 - profile.get('hepatotoxicity', 0.2),
                    1.0 - profile.get('cardiotoxicity', 0.2),
                    1.0 - profile.get('nephrotoxicity', 0.2),
                    1.0 - profile.get('cns_effects', 0.2),
                    1.0 - profile.get('gi_effects', 0.2),
                    profile.get('overall_safety', 0.7),
                ]
                values += values[:1]
                
                ax = axes[idx]
                ax.plot(angles, values, 'o-', linewidth=2, label=drug_name)
                ax.fill(angles, values, alpha=0.25)
                ax.set_xticks(angles[:-1])
                ax.set_xticklabels(categories, size=8)
                ax.set_ylim(0, 1)
                ax.set_title(drug_name[:20], size=11, fontweight='bold', pad=20)
                ax.grid(True)
        
        # Hide unused subplots
        for idx in range(len(candidate_drugs), 6):
            axes[idx].axis('off')
        
        fig.suptitle('Drug Safety Profiles Comparison', fontsize=18, fontweight='bold', y=0.98)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        return fig
    
    def plot_drug_target_network_interactive(
        self,
        data: Dict[str, Any],
        output_path: Path
    ) -> Optional[Path]:
        """Generate interactive drug-target network using PyVis."""
        try:
            from pyvis.network import Network
            
            graph = self._create_drug_target_network(data)
            if graph.number_of_nodes() == 0:
                logger.warning("No data available for interactive S6 network")
                return None
            
            net = Network(height='800px', width='100%', bgcolor='#ffffff', font_color='#2f2f2f')
            net.force_atlas_2based()
            
            shape_map = {'drug': 'square', 'target': 'dot', 'pathway': 'triangle'}
            for node, attrs in graph.nodes(data=True):
                tooltip = self._format_node_tooltip(attrs)
                net.add_node(
                    str(node),
                    label=attrs.get('label', str(node)),
                    color=attrs.get('color', '#4E79A7'),
                    shape=shape_map.get(attrs.get('node_type', 'target'), 'dot'),
                    size=max(int(attrs.get('size', 25) / 20), 12),
                    title=tooltip
                )
            
            for u, v, attrs in graph.edges(data=True):
                is_dashed = attrs.get('edge_type') == 'target-pathway'
                net.add_edge(
                    str(u),
                    str(v),
                    value=max(attrs.get('weight', 0.3), 0.1),
                    color='#7f7f7f',
                    title=attrs.get('tooltip', ''),
                    smooth=False,
                    dashes=is_dashed
                )
            
            output_file = output_path / 'S6_drug_target_network_interactive.html'
            net.save_graph(str(output_file))
            logger.info(f"Saved interactive drug-target network: {output_file}")
            return output_file
        
        except ImportError:
            logger.warning("PyVis not available for interactive visualization")
            return None
        except Exception as e:
            logger.error(f"Failed to generate interactive network: {e}")
            return None

    def plot_bioactivity_scatter_plotly(self, data: Dict[str, Any], output_path: Path) -> Optional[Path]:
        """
        Figure 11: Bioactivity Scatter Plot using Plotly.
        Shows drug bioactivity (nM) vs repurposing score with target grouping.
        """
        try:
            import plotly.graph_objects as go
            import plotly.express as px
            
            candidates = data.get('candidate_drugs', [])
            if not candidates:
                logger.warning("No candidate drugs for bioactivity scatter")
                return None
            
            # Extract drug data
            drugs = []
            bioactivities = []
            scores = []
            targets = []
            approvals = []
            
            for cand in candidates[:100]:  # Top 100 for visualization
                if not isinstance(cand, dict):
                    continue
                drug = cand.get('drug_name') or cand.get('drug_id', 'Unknown')
                bio = cand.get('bioactivity_nm')
                score = cand.get('repurposing_score', 0.5)
                target = cand.get('target_protein', 'Unknown')
                approval = (cand.get('approval_status') or 
                           cand.get('safety_profile', {}).get('approval_status', 'unknown'))
                
                if bio is not None:
                    drugs.append(str(drug))
                    bioactivities.append(float(bio))
                    scores.append(float(score))
                    targets.append(str(target))
                    approvals.append(str(approval))
            
            if not drugs:
                logger.warning("No bioactivity data available")
                return None
            
            # Create dataframe for plotly
            df = pd.DataFrame({
                'Drug': drugs,
                'Bioactivity (nM)': bioactivities,
                'Repurposing Score': scores,
                'Target': targets,
                'Approval': approvals
            })
            
            # Create scatter plot
            fig = px.scatter(
                df,
                x='Bioactivity (nM)',
                y='Repurposing Score',
                color='Target',
                size='Repurposing Score',
                hover_name='Drug',
                hover_data=['Approval'],
                log_x=True,
                title='<b>Figure 11: Drug Bioactivity vs Repurposing Score</b>',
                labels={'Bioactivity (nM)': '<b>Bioactivity (nM, log scale)</b>',
                       'Repurposing Score': '<b>Repurposing Score</b>'}
            )
            
            # Add threshold lines
            fig.add_hline(y=0.7, line_dash='dash', line_color='red', 
                         annotation_text='High Score Threshold')
            fig.add_vline(x=10, line_dash='dash', line_color='green',
                         annotation_text='High Potency (<10 nM)')
            
            # Highlight optimal zone
            fig.add_shape(
                type='rect',
                x0=1, x1=10, y0=0.7, y1=1.0,
                fillcolor='rgba(0, 255, 0, 0.1)',
                line=dict(color='rgba(0, 255, 0, 0.3)'),
                layer='below'
            )
            
            fig.add_annotation(
                x=3, y=0.85,
                text='Optimal Zone',
                showarrow=False,
                font=dict(size=12, color='green')
            )
            
            fig.update_layout(
                template='plotly_white',
                width=1000, height=700,
                legend=dict(orientation='h', yanchor='bottom', y=-0.2)
            )
            
            output_file = output_path / 'S6_fig11_bioactivity_scatter.html'
            fig.write_html(str(output_file), include_plotlyjs='cdn')
            logger.info(f"Generated bioactivity scatter: {output_file}")
            return output_file
            
        except ImportError:
            logger.warning("Plotly not available for bioactivity scatter")
            return None
        except Exception as e:
            logger.error(f"Failed to generate bioactivity scatter: {e}")
            return None

    def plot_target_coverage_sunburst_plotly(self, data: Dict[str, Any], output_path: Path) -> Optional[Path]:
        """
        Figure 12: Target Coverage Sunburst using Plotly.
        Hierarchical view of drug coverage across pathways and targets.
        """
        try:
            import plotly.express as px
            
            candidates = data.get('candidate_drugs', [])
            pathways = data.get('disease_pathways', [])
            
            if not candidates:
                logger.warning("No candidate drugs for sunburst")
                return None
            
            # Build hierarchical data
            rows = []
            
            # Get pathway-gene mapping
            pathway_genes = {}
            for pathway in pathways[:10]:  # Top 10 pathways
                if isinstance(pathway, dict):
                    pathway_name = pathway.get('name', pathway.get('id', 'Unknown'))[:30]
                    genes = pathway.get('genes', [])[:20]
                    pathway_genes[pathway_name] = genes
            
            # Map drugs to pathways through targets
            for cand in candidates[:50]:
                if not isinstance(cand, dict):
                    continue
                drug = cand.get('drug_name') or cand.get('drug_id', 'Unknown')
                target = cand.get('target_protein', 'Unknown')
                score = float(cand.get('repurposing_score', 0.5))
                
                # Find pathway for this target
                assigned_pathway = 'Other'
                for pathway_name, genes in pathway_genes.items():
                    if target in genes:
                        assigned_pathway = pathway_name
                        break
                
                rows.append({
                    'root': 'AXL Drug Network',
                    'pathway': assigned_pathway,
                    'target': str(target),
                    'drug': str(drug)[:20],
                    'score': score
                })
            
            if not rows:
                logger.warning("No data for sunburst")
                return None
            
            df = pd.DataFrame(rows)
            
            # Create sunburst
            fig = px.sunburst(
                df,
                path=['root', 'pathway', 'target', 'drug'],
                values='score',
                color='score',
                color_continuous_scale='RdYlGn',
                title='<b>Figure 12: Drug-Target-Pathway Coverage</b>'
            )
            
            fig.update_layout(
                template='plotly_white',
                width=900, height=800
            )
            
            output_file = output_path / 'S6_fig12_target_coverage.html'
            fig.write_html(str(output_file), include_plotlyjs='cdn')
            logger.info(f"Generated target coverage sunburst: {output_file}")
            return output_file
            
        except ImportError:
            logger.warning("Plotly not available for sunburst")
            return None
        except Exception as e:
            logger.error(f"Failed to generate sunburst: {e}")
            return None

    def plot_drug_comparison_dashboard_plotly(self, data: Dict[str, Any], output_path: Path) -> Optional[Path]:
        """
        Figure 13: Drug Comparison Dashboard using Plotly.
        Multi-panel comparison of top drug candidates.
        """
        try:
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots
            
            candidates = data.get('top_candidates', data.get('candidate_drugs', []))[:20]
            if not candidates:
                logger.warning("No candidates for comparison dashboard")
                return None
            
            # Extract data
            drugs = []
            scores = []
            bioactivities = []
            safety_scores = []
            targets = []
            
            for cand in candidates:
                if not isinstance(cand, dict):
                    continue
                drugs.append(str(cand.get('drug_name', cand.get('drug_id', 'Unknown')))[:15])
                scores.append(float(cand.get('repurposing_score', 0.5)))
                bio = cand.get('bioactivity_nm', 50)
                bioactivities.append(float(bio) if bio else 50)
                safety = cand.get('safety_score', 
                                 cand.get('safety_profile', {}).get('overall_score', 0.7))
                safety_scores.append(float(safety))
                targets.append(str(cand.get('target_protein', 'Unknown')))
            
            if not drugs:
                logger.warning("No valid drugs for dashboard")
                return None
            
            # Create subplots
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=[
                    'Repurposing Scores',
                    'Bioactivity (nM, lower = better)',
                    'Safety Scores',
                    'Score vs Safety Trade-off'
                ],
                specs=[[{'type': 'bar'}, {'type': 'bar'}],
                       [{'type': 'bar'}, {'type': 'scatter'}]]
            )
            
            # 1. Repurposing scores bar
            fig.add_trace(go.Bar(
                x=drugs, y=scores,
                marker_color='#2ca02c',
                name='Repurposing Score',
                text=[f'{s:.2f}' for s in scores],
                textposition='outside'
            ), row=1, col=1)
            
            # 2. Bioactivity bar (inverted color - lower is better)
            bio_colors = ['#2ca02c' if b < 10 else '#ff7f0e' if b < 50 else '#d62728' 
                         for b in bioactivities]
            fig.add_trace(go.Bar(
                x=drugs, y=bioactivities,
                marker_color=bio_colors,
                name='Bioactivity',
                text=[f'{b:.1f}' for b in bioactivities],
                textposition='outside'
            ), row=1, col=2)
            
            # 3. Safety scores bar
            fig.add_trace(go.Bar(
                x=drugs, y=safety_scores,
                marker_color='#1f77b4',
                name='Safety Score',
                text=[f'{s:.2f}' for s in safety_scores],
                textposition='outside'
            ), row=2, col=1)
            
            # 4. Trade-off scatter
            fig.add_trace(go.Scatter(
                x=scores, y=safety_scores,
                mode='markers+text',
                marker=dict(size=12, color=scores, colorscale='RdYlGn', 
                           line=dict(color='black', width=1)),
                text=drugs,
                textposition='top center',
                name='Drugs',
                hovertemplate='%{text}<br>Score: %{x:.2f}<br>Safety: %{y:.2f}<extra></extra>'
            ), row=2, col=2)
            
            # Add quadrant lines
            fig.add_hline(y=0.7, line_dash='dash', line_color='gray', row=2, col=2)
            fig.add_vline(x=0.7, line_dash='dash', line_color='gray', row=2, col=2)
            
            fig.update_layout(
                title=dict(
                    text='<b>Figure 13: Top Drug Candidates Comparison Dashboard</b>',
                    font=dict(size=18)
                ),
                showlegend=False,
                template='plotly_white',
                width=1200, height=800
            )
            
            # Update axes
            fig.update_xaxes(tickangle=45, row=1, col=1)
            fig.update_xaxes(tickangle=45, row=1, col=2)
            fig.update_xaxes(tickangle=45, row=2, col=1)
            fig.update_xaxes(title='Repurposing Score', row=2, col=2)
            fig.update_yaxes(title='Safety Score', row=2, col=2)
            
            output_file = output_path / 'S6_fig13_drug_dashboard.html'
            fig.write_html(str(output_file), include_plotlyjs='cdn')
            logger.info(f"Generated drug comparison dashboard: {output_file}")
            return output_file
            
        except ImportError:
            logger.warning("Plotly not available for dashboard")
            return None
        except Exception as e:
            logger.error(f"Failed to generate dashboard: {e}")
            return None

    def plot_volcano_plotly(self, data: Dict[str, Any], output_path: Path) -> Optional[Path]:
        """
        Interactive volcano plot using Plotly.
        Shows efficacy vs significance with hover details.
        """
        try:
            import plotly.express as px
            
            candidates = data.get('top_candidates', data.get('candidate_drugs', []))
            if not candidates:
                logger.warning("No candidates for volcano plot")
                return None
            
            # Prepare data
            rows = []
            for cand in candidates[:100]:
                if not isinstance(cand, dict):
                    continue
                drug = cand.get('drug_name', cand.get('drug_id', 'Unknown'))
                efficacy = float(cand.get('repurposing_score', 0.5))
                safety = float(cand.get('safety_score', 
                              cand.get('safety_profile', {}).get('overall_score', 0.7)))
                target = cand.get('target_protein', 'Unknown')
                
                # Calculate significance
                significance = -np.log10(max(0.001, 1 - (efficacy * safety)))
                
                # Classify
                if efficacy > 0.7 and safety > 0.7:
                    category = 'High Priority'
                elif efficacy > 0.7:
                    category = 'High Efficacy'
                elif safety > 0.7:
                    category = 'High Safety'
                else:
                    category = 'Low Priority'
                
                rows.append({
                    'Drug': str(drug)[:20],
                    'Efficacy': efficacy,
                    'Significance': significance,
                    'Safety': safety,
                    'Target': str(target),
                    'Category': category
                })
            
            if not rows:
                return None
            
            df = pd.DataFrame(rows)
            
            color_map = {
                'High Priority': '#2ca02c',
                'High Efficacy': '#ff7f0e',
                'High Safety': '#1f77b4',
                'Low Priority': '#7f7f7f'
            }
            
            fig = px.scatter(
                df,
                x='Efficacy',
                y='Significance',
                color='Category',
                size='Safety',
                hover_name='Drug',
                hover_data=['Target', 'Safety'],
                color_discrete_map=color_map,
                title='<b>Interactive Volcano Plot: Drug Efficacy vs Significance</b>'
            )
            
            # Add threshold lines
            fig.add_hline(y=-np.log10(0.05), line_dash='dash', line_color='blue',
                         annotation_text='p=0.05')
            fig.add_vline(x=0.7, line_dash='dash', line_color='red',
                         annotation_text='High Efficacy')
            
            fig.update_layout(
                template='plotly_white',
                width=1000, height=700,
                xaxis_title='<b>Repurposing Efficacy Score</b>',
                yaxis_title='<b>-log10(p-value)</b>'
            )
            
            output_file = output_path / 'S6_volcano_interactive.html'
            fig.write_html(str(output_file), include_plotlyjs='cdn')
            logger.info(f"Generated interactive volcano plot: {output_file}")
            return output_file
            
        except ImportError:
            logger.warning("Plotly not available for volcano plot")
            return None
        except Exception as e:
            logger.error(f"Failed to generate volcano plot: {e}")
            return None

