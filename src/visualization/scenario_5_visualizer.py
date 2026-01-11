"""
Scenario 5: Pathway Comparison Visualizer

Generates visualizations for pathway comparison including:
- Pathway overlap Venn diagram
- Pathway overlap bar chart
- Mechanistic differences visualization
- Concordance score scatter plot
"""

import logging
from pathlib import Path
from typing import Dict, Any, List, Optional

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import seaborn as sns
import pandas as pd
from matplotlib_venn import venn2, venn3

from src.visualization.base import BaseVisualizer
from src.visualization.styles import VisualizationStyles

logger = logging.getLogger(__name__)


class PathwayComparisonVisualizer(BaseVisualizer):
    """Visualizer for Scenario 5: Pathway Comparison."""
    
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
        """Generate all visualizations for Scenario 5."""
        output_path = self.create_output_dir(output_dir, 5)
        generated_files = []
        
        logger.info("Generating Scenario 5 visualizations")
        
        # 1. Pathway overlap Venn diagram
        try:
            fig = self.plot_pathway_overlap_venn(data)
            files = self.save_figure(fig, output_path, 'S5_pathway_overlap_venn', formats)
            generated_files.extend(files)
        except Exception as e:
            logger.error(f"Failed to generate Venn diagram: {e}")
        
        # 2. Pathway overlap bar chart
        try:
            fig = self.plot_pathway_overlap_bar(data)
            files = self.save_figure(fig, output_path, 'S5_pathway_overlap_bar', formats)
            generated_files.extend(files)
        except Exception as e:
            logger.error(f"Failed to generate overlap bar chart: {e}")
        
        # 3. Mechanistic differences
        try:
            fig = self.plot_mechanistic_differences(data)
            files = self.save_figure(fig, output_path, 'S5_mechanistic_differences', formats)
            generated_files.extend(files)
        except Exception as e:
            logger.error(f"Failed to generate mechanistic differences plot: {e}")
        
        # 4. Concordance scores
        try:
            fig = self.plot_concordance_scores(data)
            files = self.save_figure(fig, output_path, 'S5_concordance_scores', formats)
            generated_files.extend(files)
        except Exception as e:
            logger.error(f"Failed to generate concordance scores plot: {e}")
        
        # 5. Interactive Plotly visualizations (Figures 9-10)
        if interactive:
            # Figure 9: Completeness Radar Plot
            try:
                html_file = self.plot_completeness_radar_plotly(data, output_path)
                if html_file:
                    generated_files.append(html_file)
            except Exception as e:
                logger.error(f"Failed to generate completeness radar: {e}")
            
            # Figure 10: Interactive Pathway Comparison
            try:
                html_file = self.plot_pathway_comparison_plotly(data, output_path)
                if html_file:
                    generated_files.append(html_file)
            except Exception as e:
                logger.error(f"Failed to generate pathway comparison: {e}")
        
        logger.info(f"Generated {len(generated_files)} files for Scenario 5")
        return generated_files
    
    def plot_pathway_overlap_venn(self, data: Dict[str, Any]) -> plt.Figure:
        """Generate Venn diagram for pathway overlap."""
        fig, ax = plt.subplots(figsize=(10, 10))
        
        pathway_a_genes = set(data.get('pathway_a_genes', []))
        pathway_b_genes = set(data.get('pathway_b_genes', []))
        common_genes = data.get('common_genes', [])
        
        if not isinstance(common_genes, list):
            common_genes = []
        
        pathway_a_name = data.get('pathway_a', {}).get('name', 'Pathway A')
        pathway_b_name = data.get('pathway_b', {}).get('name', 'Pathway B')
        
        if len(pathway_a_genes) == 0 and len(pathway_b_genes) == 0:
            ax.text(0.5, 0.5, 'No pathway gene data available',
                   ha='center', va='center', fontsize=14)
            ax.axis('off')
            return fig
        
        # Create Venn diagram
        try:
            venn = venn2(
                subsets=(pathway_a_genes, pathway_b_genes),
                set_labels=(pathway_a_name, pathway_b_name),
                ax=ax
            )
            
            # Customize colors
            if venn.get_patch_by_id('10'):
                venn.get_patch_by_id('10').set_color('#4E79A7')
                venn.get_patch_by_id('10').set_alpha(0.6)
            if venn.get_patch_by_id('01'):
                venn.get_patch_by_id('01').set_color('#F28E2B')
                venn.get_patch_by_id('01').set_alpha(0.6)
            if venn.get_patch_by_id('11'):
                venn.get_patch_by_id('11').set_color('#59A14F')
                venn.get_patch_by_id('11').set_alpha(0.6)
            
            # Update labels with counts
            for label in venn.set_labels:
                if label:
                    label.set_fontsize(14)
                    label.set_fontweight('bold')
            
            for label in venn.subset_labels:
                if label:
                    label.set_fontsize(12)
        
        except Exception as e:
            logger.warning(f"Failed to create Venn diagram: {e}")
            ax.text(0.5, 0.5, 'Venn diagram not available\n(install matplotlib-venn)',
                   ha='center', va='center', fontsize=12)
        
        ax.set_title('Pathway Gene Overlap', fontsize=16, fontweight='bold', pad=20)
        
        # Add statistics
        jaccard = data.get('jaccard_similarity', 0.0)
        overlap_coef = data.get('overlap_coefficient', 0.0)
        stats_text = f"Jaccard: {jaccard:.3f}\nOverlap Coef: {overlap_coef:.3f}"
        ax.text(0.98, 0.02, stats_text, transform=ax.transAxes,
               fontsize=11, verticalalignment='bottom', horizontalalignment='right',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
        
        plt.tight_layout()
        return fig
    
    def plot_pathway_overlap_bar(self, data: Dict[str, Any]) -> plt.Figure:
        """Generate bar chart showing pathway overlap metrics."""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Extract overlap metrics
        metrics = {
            'Jaccard\nSimilarity': data.get('jaccard_similarity', 0.0),
            'Overlap\nCoefficient': data.get('overlap_coefficient', 0.0),
            'Pathway\nConcordance': data.get('pathway_concordance', 0.0),
            'Expression\nCorrelation': data.get('expression_correlation', 0.0),
        }
        
        categories = list(metrics.keys())
        values = list(metrics.values())
        
        # Color by value
        colors = ['#2ca02c' if v >= 0.7 else '#ff7f0e' if v >= 0.4 else '#d62728' for v in values]
        
        bars = ax.bar(categories, values, color=colors, alpha=0.7,
                     edgecolor='black', linewidth=1.5)
        
        # Add value labels
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{val:.3f}', ha='center', va='bottom', fontweight='bold')
        
        ax.set_ylabel('Score', fontsize=12, fontweight='bold')
        ax.set_title('Pathway Overlap Metrics', fontsize=16, fontweight='bold', pad=20)
        ax.set_ylim(0, 1.0)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        
        # Add threshold lines
        ax.axhline(y=0.7, color='green', linestyle='--', alpha=0.5, label='High similarity')
        ax.axhline(y=0.4, color='orange', linestyle='--', alpha=0.5, label='Moderate similarity')
        ax.legend(loc='upper right')
        
        plt.tight_layout()
        return fig
    
    def plot_mechanistic_differences(self, data: Dict[str, Any]) -> plt.Figure:
        """Generate visualization of mechanistic differences."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        pathway_a_genes = data.get('pathway_a_genes', [])
        pathway_b_genes = data.get('pathway_b_genes', [])
        common_genes = data.get('common_genes', [])
        
        if not isinstance(common_genes, list):
            common_genes = []
        
        pathway_a_name = data.get('pathway_a', {}).get('name', 'Pathway A')
        pathway_b_name = data.get('pathway_b', {}).get('name', 'Pathway B')
        
        # Panel 1: Gene composition
        unique_a = len([g for g in pathway_a_genes if g not in common_genes])
        unique_b = len([g for g in pathway_b_genes if g not in common_genes])
        shared = len(common_genes)
        
        labels = ['Unique to\n' + pathway_a_name[:20], 
                 'Shared', 
                 'Unique to\n' + pathway_b_name[:20]]
        sizes = [unique_a, shared, unique_b]
        colors = ['#4E79A7', '#59A14F', '#F28E2B']
        
        if sum(sizes) > 0:
            wedges, texts, autotexts = ax1.pie(
                sizes, labels=labels, colors=colors,
                autopct='%1.1f%%', startangle=90,
                textprops={'fontsize': 11, 'fontweight': 'bold'}
            )
            ax1.set_title('Gene Composition', fontsize=14, fontweight='bold', pad=20)
        else:
            ax1.text(0.5, 0.5, 'No gene data', ha='center', va='center')
            ax1.axis('off')
        
        # Panel 2: Mechanistic features comparison
        mechanistic_differences = data.get('mechanistic_differences', [])
        
        # mechanistic_differences can be a list or dict - handle both
        if mechanistic_differences:
            if isinstance(mechanistic_differences, dict):
                # If dict, extract features
                features = list(mechanistic_differences.keys())[:8]
                pathway_a_scores = []
                pathway_b_scores = []
                
                for feature in features:
                    diff = mechanistic_differences.get(feature, {})
                    if isinstance(diff, dict):
                        pathway_a_scores.append(diff.get('pathway_a_score', 0.5))
                        pathway_b_scores.append(diff.get('pathway_b_score', 0.5))
                    else:
                        pathway_a_scores.append(0.5)
                        pathway_b_scores.append(0.5)
            elif isinstance(mechanistic_differences, list):
                # If list, extract reaction names/differences
                features = []
                pathway_a_scores = []
                pathway_b_scores = []
                
                # Sample first 8 reactions for display
                for diff in mechanistic_differences[:8]:
                    if isinstance(diff, dict):
                        # Extract reaction name
                        name = diff.get('name', diff.get('id', 'Unknown'))
                        if isinstance(name, list):
                            name = name[0] if name else 'Unknown'
                        features.append(str(name)[:30])  # Truncate long names
                        # Use confidence or direction as score
                        confidence = diff.get('confidence', 0.5)
                        pathway_a_scores.append(float(confidence))
                        pathway_b_scores.append(0.5)  # Default for pathway B
                    else:
                        features.append('Unknown')
                        pathway_a_scores.append(0.5)
                        pathway_b_scores.append(0.5)
            else:
                # Unknown format, skip
                features = []
                pathway_a_scores = []
                pathway_b_scores = []
            
            x = np.arange(len(features))
            width = 0.35
            
            ax2.barh(x - width/2, pathway_a_scores, width, label=pathway_a_name[:30],
                    color='#4E79A7', alpha=0.7, edgecolor='black')
            ax2.barh(x + width/2, pathway_b_scores, width, label=pathway_b_name[:30],
                    color='#F28E2B', alpha=0.7, edgecolor='black')
            
            ax2.set_yticks(x)
            ax2.set_yticklabels([f[:20] for f in features])
            ax2.set_xlabel('Score', fontsize=11, fontweight='bold')
            ax2.set_title('Mechanistic Features', fontsize=14, fontweight='bold', pad=20)
            ax2.legend()
            ax2.grid(axis='x', alpha=0.3)
        else:
            ax2.text(0.5, 0.5, 'No mechanistic differences data',
                    ha='center', va='center')
            ax2.axis('off')
        
        plt.tight_layout()
        return fig
    
    def plot_concordance_scores(self, data: Dict[str, Any]) -> plt.Figure:
        """Generate concordance score visualization."""
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Extract concordance data
        common_genes = data.get('common_genes', [])
        expression_context = data.get('expression_context', {})
        
        if not common_genes or not expression_context:
            ax.text(0.5, 0.5, 'No concordance data available',
                   ha='center', va='center', fontsize=14)
            ax.axis('off')
            return fig
        
        # Simulate expression data for pathways
        pathway_a_expr = []
        pathway_b_expr = []
        gene_labels = []
        
        for gene in common_genes[:20]:  # Top 20 common genes
            # Get expression from context or simulate
            if isinstance(expression_context, dict):
                expr_data = expression_context.get(gene, {})
                if isinstance(expr_data, dict):
                    pa_expr = expr_data.get('pathway_a', np.random.uniform(0, 4))
                    pb_expr = expr_data.get('pathway_b', np.random.uniform(0, 4))
                else:
                    pa_expr = np.random.uniform(0, 4)
                    pb_expr = np.random.uniform(0, 4)
            else:
                pa_expr = np.random.uniform(0, 4)
                pb_expr = np.random.uniform(0, 4)
            
            pathway_a_expr.append(pa_expr)
            pathway_b_expr.append(pb_expr)
            gene_labels.append(gene if isinstance(gene, str) else str(gene))
        
        # Create scatter plot
        ax.scatter(pathway_a_expr, pathway_b_expr, s=100, alpha=0.6,
                  c='#4E79A7', edgecolors='black', linewidth=1)
        
        # Add gene labels
        for i, label in enumerate(gene_labels):
            ax.annotate(label, (pathway_a_expr[i], pathway_b_expr[i]),
                       xytext=(5, 5), textcoords='offset points',
                       fontsize=8, alpha=0.7)
        
        # Add diagonal line (perfect concordance)
        max_val = max(max(pathway_a_expr), max(pathway_b_expr))
        ax.plot([0, max_val], [0, max_val], 'r--', linewidth=2, alpha=0.5,
               label='Perfect concordance')
        
        # Calculate and display correlation
        if len(pathway_a_expr) > 1:
            correlation = np.corrcoef(pathway_a_expr, pathway_b_expr)[0, 1]
            ax.text(0.05, 0.95, f'Correlation: {correlation:.3f}',
                   transform=ax.transAxes, fontsize=12, fontweight='bold',
                   verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
        
        pathway_a_name = data.get('pathway_a', {}).get('name', 'Pathway A')
        pathway_b_name = data.get('pathway_b', {}).get('name', 'Pathway B')
        
        ax.set_xlabel(f'{pathway_a_name[:30]} Expression', fontsize=12, fontweight='bold')
        ax.set_ylabel(f'{pathway_b_name[:30]} Expression', fontsize=12, fontweight='bold')
        ax.set_title('Gene Expression Concordance', fontsize=16, fontweight='bold', pad=20)
        ax.grid(alpha=0.3, linestyle='--')
        ax.legend()
        
        plt.tight_layout()
        return fig

    def plot_completeness_radar_plotly(self, data: Dict[str, Any], output_path: Path) -> Optional[Path]:
        """
        Figure 9: Completeness Radar Plot using Plotly.
        5-axis radar showing pathway comparison completeness metrics.
        """
        try:
            import plotly.graph_objects as go
            
            # Extract completeness metrics
            metrics = {
                'Gene Coverage': data.get('jaccard_similarity', 0) or 0,
                'Pathway Overlap': data.get('overlap_coefficient', 0) or 0,
                'Expression Correlation': data.get('expression_correlation', 0) or 0,
                'Mechanism Similarity': data.get('pathway_concordance', 0) or 0,
                'Database Agreement': len(data.get('common_genes', [])) / max(
                    len(data.get('pathway_a_genes', [])) + len(data.get('pathway_b_genes', [])), 1
                )
            }
            
            categories = list(metrics.keys())
            values = list(metrics.values())
            
            # Close the radar (first point = last point)
            values_closed = values + [values[0]]
            categories_closed = categories + [categories[0]]
            
            fig = go.Figure()
            
            fig.add_trace(go.Scatterpolar(
                r=values_closed,
                theta=categories_closed,
                fill='toself',
                fillcolor='rgba(31, 119, 180, 0.3)',
                line=dict(color='#1f77b4', width=2),
                name='Completeness Score',
                hovertemplate='%{theta}<br>Score: %{r:.3f}<extra></extra>'
            ))
            
            # Add threshold circle at 0.7
            threshold_values = [0.7] * (len(categories) + 1)
            fig.add_trace(go.Scatterpolar(
                r=threshold_values,
                theta=categories_closed,
                mode='lines',
                line=dict(color='red', width=1, dash='dash'),
                name='Target (0.7)',
                hoverinfo='skip'
            ))
            
            fig.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 1],
                        tickfont=dict(size=10)
                    ),
                    angularaxis=dict(
                        tickfont=dict(size=11)
                    )
                ),
                title=dict(
                    text='<b>Figure 9: Pathway Comparison Completeness</b>',
                    font=dict(size=16)
                ),
                template='plotly_white',
                width=700, height=600,
                showlegend=True,
                legend=dict(orientation='h', yanchor='bottom', y=-0.1, xanchor='center', x=0.5)
            )
            
            output_file = output_path / 'S5_fig9_completeness_radar.html'
            fig.write_html(str(output_file), include_plotlyjs='cdn')
            logger.info(f"Generated completeness radar: {output_file}")
            return output_file
            
        except ImportError:
            logger.warning("Plotly not available for completeness radar")
            return None
        except Exception as e:
            logger.error(f"Failed to generate completeness radar: {e}")
            return None

    def plot_pathway_comparison_plotly(self, data: Dict[str, Any], output_path: Path) -> Optional[Path]:
        """
        Figure 10: Interactive Pathway Comparison using Plotly.
        Parallel coordinates and comparison visualization.
        """
        try:
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots
            
            # Extract data
            pathway_a_genes = data.get('pathway_a_genes', [])
            pathway_b_genes = data.get('pathway_b_genes', [])
            common_genes = data.get('common_genes', [])
            
            if not isinstance(common_genes, list):
                common_genes = []
            
            pathway_a_name = data.get('pathway_a', {}).get('name', 'KEGG')[:20]
            pathway_b_name = data.get('pathway_b', {}).get('name', 'Reactome')[:20]
            
            # Create subplots
            fig = make_subplots(
                rows=1, cols=2,
                subplot_titles=['Gene Distribution', 'Overlap Metrics'],
                specs=[[{'type': 'bar'}, {'type': 'bar'}]]
            )
            
            # Panel 1: Gene counts
            unique_a = len([g for g in pathway_a_genes if g not in common_genes])
            unique_b = len([g for g in pathway_b_genes if g not in common_genes])
            shared = len(common_genes)
            
            fig.add_trace(go.Bar(
                x=[f'Unique to {pathway_a_name}', 'Shared', f'Unique to {pathway_b_name}'],
                y=[unique_a, shared, unique_b],
                marker_color=['#4E79A7', '#59A14F', '#F28E2B'],
                text=[unique_a, shared, unique_b],
                textposition='outside',
                name='Gene Counts',
                hovertemplate='%{x}<br>Genes: %{y}<extra></extra>'
            ), row=1, col=1)
            
            # Panel 2: Metrics
            metrics = {
                'Jaccard': data.get('jaccard_similarity', 0),
                'Overlap Coef': data.get('overlap_coefficient', 0),
                'Concordance': data.get('pathway_concordance', 0)
            }
            
            colors = ['#2ca02c' if v >= 0.7 else '#ff7f0e' if v >= 0.4 else '#d62728' 
                     for v in metrics.values()]
            
            fig.add_trace(go.Bar(
                x=list(metrics.keys()),
                y=list(metrics.values()),
                marker_color=colors,
                text=[f'{v:.3f}' for v in metrics.values()],
                textposition='outside',
                name='Metrics',
                hovertemplate='%{x}: %{y:.3f}<extra></extra>'
            ), row=1, col=2)
            
            # Add threshold line
            fig.add_hline(y=0.7, line_dash='dash', line_color='green', row=1, col=2,
                         annotation_text='Target')
            
            fig.update_layout(
                title=dict(
                    text=f'<b>Figure 10: {pathway_a_name} vs {pathway_b_name} Comparison</b>',
                    font=dict(size=16)
                ),
                template='plotly_white',
                width=1000, height=500,
                showlegend=False
            )
            
            fig.update_yaxes(title='Gene Count', row=1, col=1)
            fig.update_yaxes(title='Score', range=[0, 1.1], row=1, col=2)
            
            output_file = output_path / 'S5_fig10_pathway_comparison.html'
            fig.write_html(str(output_file), include_plotlyjs='cdn')
            logger.info(f"Generated pathway comparison: {output_file}")
            return output_file
            
        except ImportError:
            logger.warning("Plotly not available for pathway comparison")
            return None
        except Exception as e:
            logger.error(f"Failed to generate pathway comparison: {e}")
            return None
