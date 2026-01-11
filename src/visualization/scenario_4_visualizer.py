"""
Scenario 4: MRA Simulation Visualizer

Generates visualizations for multi-target simulation including:
- Network propagation with effect overlays
- Synergy matrix heatmap
- Perturbation effect bar charts
- Feedback loop network diagram
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


class MRASimulationVisualizer(BaseVisualizer):
    """Visualizer for Scenario 4: MRA Simulation."""
    
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
        """Generate all visualizations for Scenario 4."""
        output_path = self.create_output_dir(output_dir, 4)
        generated_files = []
        
        logger.info("Generating Scenario 4 visualizations")
        
        # 1. Network propagation
        try:
            fig = self.plot_network_propagation(data)
            files = self.save_figure(fig, output_path, 'S4_network_propagation', formats)
            generated_files.extend(files)
            
            if interactive:
                html_file = self.plot_network_propagation_interactive(data, output_path)
                if html_file:
                    generated_files.append(html_file)
        except Exception as e:
            logger.error(f"Failed to generate network propagation plot: {e}")
        
        # 2. Synergy matrix
        if self._validate_data(data, ['synergy_analysis']):
            try:
                fig = self.plot_synergy_matrix(data)
                files = self.save_figure(fig, output_path, 'S4_synergy_matrix', formats)
                generated_files.extend(files)
            except Exception as e:
                logger.error(f"Failed to generate synergy matrix: {e}")
        
        # 3. Perturbation effects
        try:
            fig = self.plot_perturbation_effects(data)
            files = self.save_figure(fig, output_path, 'S4_perturbation_effects', formats)
            generated_files.extend(files)
        except Exception as e:
            logger.error(f"Failed to generate perturbation effects plot: {e}")
        
        # 4. Feedback loops
        try:
            fig = self.plot_feedback_loops(data)
            files = self.save_figure(fig, output_path, 'S4_feedback_loops', formats)
            generated_files.extend(files)
        except Exception as e:
            logger.error(f"Failed to generate feedback loops plot: {e}")
        
        # 5. Interactive Plotly visualizations (Figures 6-8)
        if interactive:
            # Figure 6: Cascade Tree
            try:
                html_file = self.plot_cascade_tree_plotly(data, output_path)
                if html_file:
                    generated_files.append(html_file)
            except Exception as e:
                logger.error(f"Failed to generate cascade tree: {e}")
            
            # Figure 7: Effect Gradient Map
            try:
                html_file = self.plot_effect_gradient_plotly(data, output_path)
                if html_file:
                    generated_files.append(html_file)
            except Exception as e:
                logger.error(f"Failed to generate effect gradient: {e}")
            
            # Figure 8: Feedback Loops Plotly
            try:
                html_file = self.plot_feedback_loops_plotly(data, output_path)
                if html_file:
                    generated_files.append(html_file)
            except Exception as e:
                logger.error(f"Failed to generate feedback loops Plotly: {e}")
            
            # Centrality Comparison
            try:
                html_file = self.plot_centrality_comparison_plotly(data, output_path)
                if html_file:
                    generated_files.append(html_file)
            except Exception as e:
                logger.error(f"Failed to generate centrality comparison: {e}")
        
        logger.info(f"Generated {len(generated_files)} files for Scenario 4")
        return generated_files
    
    def plot_network_propagation(self, data: Dict[str, Any]) -> plt.Figure:
        """Generate network propagation visualization with effect overlays."""
        fig, ax = plt.subplots(figsize=(14, 12))
        
        # Build network from combined effects
        combined_effects = data.get('combined_effects', {})
        individual_results = data.get('individual_results', [])
        
        G = nx.Graph()
        
        # Add nodes from affected proteins
        affected_nodes = combined_effects.get('affected_nodes', {})
        if isinstance(affected_nodes, dict):
            for node, effect in affected_nodes.items():
                if isinstance(effect, dict):
                    G.add_node(node, effect=effect.get('effect', 0.0))
                else:
                    G.add_node(node, effect=float(effect))
        
        # Add edges from network perturbation
        network_pert = data.get('network_perturbation', {})
        edges = network_pert.get('affected_edges', [])
        for edge in edges:
            if isinstance(edge, dict):
                source = edge.get('source', '')
                target = edge.get('target', '')
                if source and target and source in G.nodes() and target in G.nodes():
                    G.add_edge(source, target, weight=edge.get('weight', 1.0))
        
        if len(G.nodes()) == 0:
            ax.text(0.5, 0.5, 'No network propagation data available',
                   ha='center', va='center', fontsize=14)
            ax.axis('off')
            return fig
        
        # Layout
        pos = nx.spring_layout(G, k=2, iterations=50, seed=42)
        
        # Node colors based on effect magnitude
        node_effects = [G.nodes[node].get('effect', 0) for node in G.nodes()]
        node_colors = node_effects
        
        # Node sizes
        node_sizes = [500 + abs(effect) * 500 for effect in node_effects]
        
        # Draw network
        nodes = nx.draw_networkx_nodes(
            G, pos, node_color=node_colors, node_size=node_sizes,
            cmap='RdBu_r', vmin=-1, vmax=1, alpha=0.8,
            linewidths=2, edgecolors='black', ax=ax
        )
        
        nx.draw_networkx_edges(G, pos, alpha=0.4, width=1.5, ax=ax)
        nx.draw_networkx_labels(G, pos, font_size=8, font_weight='bold', ax=ax)
        
        ax.set_title('Network Propagation: Perturbation Effects',
                    fontsize=16, fontweight='bold', pad=20)
        ax.axis('off')
        
        # Add colorbar
        sm = plt.cm.ScalarMappable(cmap='RdBu_r', norm=plt.Normalize(vmin=-1, vmax=1))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Perturbation Effect', rotation=270, labelpad=20)
        
        # Add legend
        legend_text = f"Nodes: {len(G.nodes())} | Edges: {len(G.edges())}"
        ax.text(0.02, 0.02, legend_text, transform=ax.transAxes,
               fontsize=10, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        return fig
    
    def plot_synergy_matrix(self, data: Dict[str, Any]) -> plt.Figure:
        """Generate synergy matrix heatmap."""
        fig, ax = plt.subplots(figsize=(12, 10))
        
        synergy_analysis = data.get('synergy_analysis', {})
        
        if not synergy_analysis:
            ax.text(0.5, 0.5, 'No synergy analysis data available',
                   ha='center', va='center', fontsize=14)
            ax.axis('off')
            return fig
        
        # Extract pairwise synergies
        synergy_pairs = synergy_analysis.get('pairwise_synergies', [])
        
        if not synergy_pairs:
            ax.text(0.5, 0.5, 'No pairwise synergy data',
                   ha='center', va='center', fontsize=14)
            ax.axis('off')
            return fig
        
        # Build synergy matrix
        targets = set()
        for pair in synergy_pairs:
            if isinstance(pair, dict):
                targets.add(pair.get('target_a', ''))
                targets.add(pair.get('target_b', ''))
        
        targets = sorted(list(targets))
        n = len(targets)
        
        if n == 0:
            ax.text(0.5, 0.5, 'No valid targets in synergy data',
                   ha='center', va='center', fontsize=14)
            ax.axis('off')
            return fig
        
        # Create matrix
        matrix = np.zeros((n, n))
        for pair in synergy_pairs:
            if isinstance(pair, dict):
                target_a = pair.get('target_a', '')
                target_b = pair.get('target_b', '')
                synergy_score = pair.get('synergy_score', 0.0)
                
                if target_a in targets and target_b in targets:
                    i = targets.index(target_a)
                    j = targets.index(target_b)
                    matrix[i, j] = synergy_score
                    matrix[j, i] = synergy_score
        
        # Create heatmap
        sns.heatmap(matrix, annot=True, fmt='.2f', cmap='RdYlGn',
                   xticklabels=targets, yticklabels=targets,
                   linewidths=1, linecolor='white',
                   cbar_kws={'label': 'Synergy Score'},
                   vmin=-1, vmax=1, center=0, ax=ax)
        
        ax.set_title('Target Combination Synergy Matrix',
                    fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel('Target', fontsize=12, fontweight='bold')
        ax.set_ylabel('Target', fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        return fig
    
    def plot_perturbation_effects(self, data: Dict[str, Any]) -> plt.Figure:
        """Generate multi-panel perturbation effect summary."""
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        axes = axes.flatten()
        
        individual_results = [
            res for res in (data.get('individual_results') or [])
            if isinstance(res, dict)
        ]
        combined_effects = data.get('combined_effects', {})
        
        top_targets = individual_results[:5]
        effect_colormap = {'activate': '#2ca02c', 'inhibit': '#d62728'}
        for ax, result in zip(axes[:5], top_targets):
            target = result.get('target_node') or result.get('target') or 'Unknown'
            affected = result.get('affected_nodes') or {}
            if not affected:
                ax.text(0.5, 0.5, f'No data for {target}', ha='center', va='center')
                ax.axis('off')
                continue
            top_nodes = sorted(
                affected.items(),
                key=lambda kv: abs(kv[1]) if isinstance(kv[1], (int, float)) else 0,
                reverse=True
            )[:6]
            labels = [node for node, _ in top_nodes]
            values = [float(val) if isinstance(val, (int, float)) else 0.0 for _, val in top_nodes]
            colors = effect_colormap.get(result.get('mode', '').lower(), '#59A14F')
            bars = ax.bar(labels, values, color=colors, alpha=0.8, edgecolor='black')
            ax.set_title(f"{target} perturbation", fontsize=12, fontweight='bold')
            ax.axhline(0, color='black', linewidth=1)
            ax.set_ylabel('Effect magnitude')
            ax.tick_params(axis='x', rotation=30, labelsize=9)
            for bar, val in zip(bars, values):
                ax.text(bar.get_x() + bar.get_width()/2, val,
                        f'{val:.2f}', ha='center',
                        va='bottom' if val >= 0 else 'top', fontsize=8)
        
        # Hide unused target panels
        for ax in axes[len(top_targets):5]:
            ax.axis('off')
        
        # Combined effects panel (last axis)
        combined_ax = axes[-1]
        if combined_effects:
            total_individual = sum(
                float(res.get('impact_score', 0.0))
                for res in individual_results
            )
            combined_score = combined_effects.get('overall_effect', total_individual)
            synergy = combined_effects.get('synergy_score', combined_score - total_individual)
            values = [total_individual, combined_score, synergy]
            labels = ['Sum of Individuals', 'Combined Effect', 'Synergy']
            colors = ['#4E79A7', '#F28E2B', '#59A14F']
            bars = combined_ax.bar(labels, values, color=colors, alpha=0.85, edgecolor='black')
            combined_ax.set_ylabel('Effect Score', fontsize=12, fontweight='bold')
            combined_ax.set_title('Combined vs Individual Effects', fontsize=13, fontweight='bold')
            combined_ax.grid(axis='y', alpha=0.3)
            for bar, val in zip(bars, values):
                combined_ax.text(bar.get_x() + bar.get_width()/2, val,
                                 f'{val:.2f}', ha='center', va='bottom', fontweight='bold')
        else:
            combined_ax.text(0.5, 0.5, 'No combined effect data',
                             ha='center', va='center')
            combined_ax.axis('off')
        
        plt.tight_layout()
        return fig
    
    def plot_feedback_loops(self, data: Dict[str, Any]) -> plt.Figure:
        """Generate feedback loop network diagram."""
        fig, ax = plt.subplots(figsize=(12, 10))
        
        network_pert = data.get('network_perturbation', {})
        feedback_loops = network_pert.get('feedback_loops', [])
        
        if not feedback_loops:
            ax.text(0.5, 0.5, 'No feedback loop data available',
                   ha='center', va='center', fontsize=14)
            ax.axis('off')
            return fig
        
        # Build feedback loop network
        G = nx.DiGraph()
        
        for loop in feedback_loops:
            if isinstance(loop, dict):
                nodes = loop.get('nodes', loop.get('participants', []))
                loop_type = loop.get('type', 'unknown')
                
                # Add cycle edges
                for i in range(len(nodes)):
                    source = nodes[i]
                    target = nodes[(i + 1) % len(nodes)]
                    G.add_edge(source, target, loop_type=loop_type)
        
        if len(G.nodes()) == 0:
            ax.text(0.5, 0.5, 'No valid feedback loop data',
                   ha='center', va='center', fontsize=14)
            ax.axis('off')
            return fig
        
        # Layout
        pos = nx.spring_layout(G, k=3, iterations=50, seed=42)
        
        # Draw network
        nx.draw_networkx_nodes(
            G, pos, node_size=800, node_color='#4E79A7',
            alpha=0.8, linewidths=2, edgecolors='black', ax=ax
        )
        
        # Draw edges with arrows
        nx.draw_networkx_edges(
            G, pos, edge_color='#666666', alpha=0.6,
            width=2, arrows=True, arrowsize=20,
            arrowstyle='->', connectionstyle='arc3,rad=0.1', ax=ax
        )
        
        nx.draw_networkx_labels(G, pos, font_size=10, font_weight='bold', ax=ax)
        
        ax.set_title('Feedback Loops in Perturbed Network',
                    fontsize=16, fontweight='bold', pad=20)
        ax.axis('off')
        
        # Add statistics
        stats_text = f"Feedback Loops: {len(feedback_loops)} | Nodes: {len(G.nodes())}"
        ax.text(0.02, 0.02, stats_text, transform=ax.transAxes,
               fontsize=10, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        return fig
    
    def plot_network_propagation_interactive(
        self,
        data: Dict[str, Any],
        output_path: Path
    ) -> Optional[Path]:
        """Generate interactive network propagation using PyVis."""
        try:
            from pyvis.network import Network
            
            combined_effects = data.get('combined_effects', {})
            affected_nodes = combined_effects.get('affected_nodes', {})
            
            if not affected_nodes:
                logger.warning("No network data for interactive visualization")
                return None
            
            net = Network(height='750px', width='100%', bgcolor='#ffffff', font_color='black')
            
            # Add nodes
            for node, effect in affected_nodes.items():
                if isinstance(effect, dict):
                    effect_val = effect.get('effect', 0.0)
                else:
                    effect_val = float(effect)
                
                # Color based on effect
                if effect_val > 0:
                    color = '#2ca02c'
                elif effect_val < 0:
                    color = '#d62728'
                else:
                    color = '#7f7f7f'
                
                size = 20 + abs(effect_val) * 30
                net.add_node(node, label=node, color=color, size=size,
                           title=f"Effect: {effect_val:.2f}")
            
            # Add edges if available
            network_pert = data.get('network_perturbation', {})
            edges = network_pert.get('affected_edges', [])
            for edge in edges:
                if isinstance(edge, dict):
                    source = edge.get('source', '')
                    target = edge.get('target', '')
                    if source and target:
                        net.add_edge(source, target)
            
            output_file = output_path / 'S4_network_propagation_interactive.html'
            net.save_graph(str(output_file))
            logger.info(f"Saved interactive network propagation: {output_file}")
            return output_file
            
        except ImportError:
            logger.warning("PyVis not available for interactive visualization")
            return None
        except Exception as e:
            logger.error(f"Failed to generate interactive network: {e}")
            return None

    def plot_cascade_tree_plotly(self, data: Dict[str, Any], output_path: Path) -> Optional[Path]:
        """
        Figure 6: Propagation Cascade Tree using Plotly.
        Shows AXL at root with perturbation effects cascading through levels.
        """
        try:
            import plotly.graph_objects as go
            
            individual_results = data.get('individual_results', [])
            if not individual_results:
                logger.warning("No individual results for cascade tree")
                return None
            
            # Get first target (usually AXL) for cascade
            primary_result = individual_results[0] if individual_results else {}
            target = primary_result.get('target_node', 'AXL')
            direct_targets = primary_result.get('direct_targets', [])
            downstream = primary_result.get('downstream', [])
            affected_nodes = primary_result.get('affected_nodes', {})
            
            # Build tree structure
            nodes = [target]
            levels = {target: 0}
            
            # Level 1: direct targets
            for dt in direct_targets[:10]:  # Limit to 10 for visibility
                if dt not in levels:
                    nodes.append(dt)
                    levels[dt] = 1
            
            # Level 2-3: downstream (excluding direct)
            downstream_set = set(downstream) - set(direct_targets) - {target}
            for i, dn in enumerate(list(downstream_set)[:15]):
                if dn not in levels:
                    nodes.append(dn)
                    levels[dn] = 2 if i < 8 else 3
            
            # Position nodes in tree layout
            level_counts = {}
            x_positions = {}
            y_positions = {}
            
            for node in nodes:
                level = levels.get(node, 3)
                if level not in level_counts:
                    level_counts[level] = 0
                level_counts[level] += 1
                
            level_current = {0: 0, 1: 0, 2: 0, 3: 0}
            for node in nodes:
                level = levels.get(node, 3)
                count = level_counts.get(level, 1)
                x_positions[node] = (level_current[level] - count / 2 + 0.5)
                y_positions[node] = -level
                level_current[level] += 1
            
            # Node colors based on effect magnitude
            node_colors = []
            node_sizes = []
            node_texts = []
            
            for node in nodes:
                effect = affected_nodes.get(node, 0)
                if isinstance(effect, dict):
                    effect = effect.get('effect', 0)
                effect = float(effect) if effect else 0
                
                # Color scale: red (inhibited) to green (activated)
                if effect > 0.5:
                    color = '#d62728'  # Strong effect
                elif effect > 0.1:
                    color = '#ff7f0e'  # Medium effect
                elif effect > 0:
                    color = '#ffd700'  # Low effect
                else:
                    color = '#1f77b4'  # Root/no effect
                
                node_colors.append(color)
                node_sizes.append(20 + abs(effect) * 40)
                node_texts.append(f"{node}<br>Effect: {effect:.3f}")
            
            # Create edges
            edge_x = []
            edge_y = []
            
            # Connect root to level 1
            for dt in direct_targets[:10]:
                if dt in nodes:
                    edge_x.extend([x_positions[target], x_positions[dt], None])
                    edge_y.extend([y_positions[target], y_positions[dt], None])
            
            # Create Plotly figure
            fig = go.Figure()
            
            # Add edges
            fig.add_trace(go.Scatter(
                x=edge_x, y=edge_y,
                mode='lines',
                line=dict(color='rgba(100,100,100,0.5)', width=1.5),
                hoverinfo='none',
                name='Connections'
            ))
            
            # Add nodes
            fig.add_trace(go.Scatter(
                x=[x_positions[n] for n in nodes],
                y=[y_positions[n] for n in nodes],
                mode='markers+text',
                marker=dict(
                    size=node_sizes,
                    color=node_colors,
                    line=dict(color='black', width=2)
                ),
                text=nodes,
                textposition='bottom center',
                hovertext=node_texts,
                hoverinfo='text',
                name='Genes'
            ))
            
            fig.update_layout(
                title=dict(
                    text=f'<b>Figure 6: Propagation Cascade from {target} Inhibition</b>',
                    font=dict(size=18)
                ),
                showlegend=False,
                hovermode='closest',
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False,
                          scaleanchor='x', scaleratio=0.5),
                template='plotly_white',
                width=1000, height=700,
                annotations=[
                    dict(x=0.02, y=0.98, xref='paper', yref='paper',
                         text=f"<b>Level 0:</b> Target | <b>Level 1:</b> Direct ({len(direct_targets)}) | "
                              f"<b>Level 2-3:</b> Downstream ({len(downstream)})",
                         showarrow=False, font=dict(size=11), bgcolor='rgba(255,255,255,0.8)')
                ]
            )
            
            output_file = output_path / 'S4_fig6_cascade_tree.html'
            fig.write_html(str(output_file), include_plotlyjs='cdn')
            logger.info(f"Generated cascade tree: {output_file}")
            return output_file
            
        except ImportError:
            logger.warning("Plotly not available for cascade tree")
            return None
        except Exception as e:
            logger.error(f"Failed to generate cascade tree: {e}")
            return None

    def plot_effect_gradient_plotly(self, data: Dict[str, Any], output_path: Path) -> Optional[Path]:
        """
        Figure 7: Effect Size Gradient Map using Plotly.
        Heatmap showing effect magnitude across all targets and affected nodes.
        """
        try:
            import plotly.graph_objects as go
            import plotly.express as px
            
            individual_results = data.get('individual_results', [])
            if not individual_results:
                logger.warning("No individual results for effect gradient")
                return None
            
            # Build matrix: rows=affected nodes, columns=targets
            targets = [r.get('target_node', f'T{i}') for i, r in enumerate(individual_results)]
            
            # Get top affected nodes across all targets
            all_affected = set()
            for result in individual_results:
                affected = result.get('affected_nodes', {})
                if isinstance(affected, dict):
                    # Get top by effect magnitude
                    sorted_nodes = sorted(affected.items(), 
                                         key=lambda x: abs(x[1]) if isinstance(x[1], (int, float)) else 0,
                                         reverse=True)
                    all_affected.update([n for n, _ in sorted_nodes[:20]])
            
            nodes = sorted(list(all_affected))[:25]  # Limit for readability
            
            # Build effect matrix
            matrix = []
            for node in nodes:
                row = []
                for result in individual_results:
                    affected = result.get('affected_nodes', {})
                    effect = affected.get(node, 0)
                    if isinstance(effect, dict):
                        effect = effect.get('effect', 0)
                    row.append(float(effect) if effect else 0)
                matrix.append(row)
            
            # Create heatmap
            fig = go.Figure(data=go.Heatmap(
                z=matrix,
                x=targets,
                y=nodes,
                colorscale='RdYlBu_r',
                zmin=0, zmax=1,
                colorbar=dict(title='Effect Magnitude'),
                hovertemplate='Target: %{x}<br>Gene: %{y}<br>Effect: %{z:.3f}<extra></extra>'
            ))
            
            fig.update_layout(
                title=dict(
                    text='<b>Figure 7: Effect Size Gradient Map</b><br>'
                         '<sub>Perturbation effect magnitudes across targets and downstream genes</sub>',
                    font=dict(size=16)
                ),
                xaxis=dict(title='<b>Target (Inhibited)</b>', tickangle=45),
                yaxis=dict(title='<b>Affected Gene</b>'),
                template='plotly_white',
                width=900, height=800
            )
            
            output_file = output_path / 'S4_fig7_effect_gradient.html'
            fig.write_html(str(output_file), include_plotlyjs='cdn')
            logger.info(f"Generated effect gradient map: {output_file}")
            return output_file
            
        except ImportError:
            logger.warning("Plotly not available for effect gradient")
            return None
        except Exception as e:
            logger.error(f"Failed to generate effect gradient: {e}")
            return None

    def plot_feedback_loops_plotly(self, data: Dict[str, Any], output_path: Path) -> Optional[Path]:
        """
        Figure 8: Feedback Loop Diagram using Plotly.
        Circular diagrams showing regulatory feedback loops.
        """
        try:
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots
            import math
            
            network_pert = data.get('network_perturbation', {})
            feedback_loops = network_pert.get('feedback_loops', [])
            
            if not feedback_loops:
                logger.warning("No feedback loops for diagram")
                return None
            
            # Group loops by source target
            loops_by_target = {}
            for loop in feedback_loops:
                if isinstance(loop, dict):
                    source = loop.get('source_target', 'Unknown')
                    if source not in loops_by_target:
                        loops_by_target[source] = []
                    loops_by_target[source].append(loop)
            
            # Create subplots - one per target
            n_targets = min(len(loops_by_target), 5)
            fig = make_subplots(
                rows=1, cols=n_targets,
                subplot_titles=[f'{t} Loops' for t in list(loops_by_target.keys())[:n_targets]],
                horizontal_spacing=0.08
            )
            
            colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
            
            for idx, (target, loops) in enumerate(list(loops_by_target.items())[:n_targets]):
                col = idx + 1
                
                # For each target, show up to 5 loops
                for loop_idx, loop in enumerate(loops[:5]):
                    nodes = loop.get('nodes', [])
                    if len(nodes) < 2:
                        continue
                    
                    n = len(nodes)
                    # Position nodes in a circle
                    radius = 0.3 + loop_idx * 0.15
                    angles = [2 * math.pi * i / n for i in range(n)]
                    
                    x = [radius * math.cos(a) for a in angles]
                    y = [radius * math.sin(a) for a in angles]
                    
                    # Close the loop
                    x.append(x[0])
                    y.append(y[0])
                    
                    color = colors[loop_idx % len(colors)]
                    
                    # Add loop path
                    fig.add_trace(go.Scatter(
                        x=x, y=y,
                        mode='lines+markers',
                        line=dict(color=color, width=2),
                        marker=dict(size=12, color=color, line=dict(color='white', width=2)),
                        name=f'Loop {loop_idx + 1}',
                        text=nodes + [nodes[0]],
                        hovertext=[f'{nodes[i]} → {nodes[(i+1) % n]}' for i in range(n)] + [''],
                        hoverinfo='text+name',
                        showlegend=(idx == 0)
                    ), row=1, col=col)
                    
                    # Add node labels
                    for i, (xi, yi, label) in enumerate(zip(x[:-1], y[:-1], nodes)):
                        fig.add_annotation(
                            x=xi, y=yi,
                            text=label,
                            showarrow=False,
                            font=dict(size=8, color='white'),
                            bgcolor=color,
                            bordercolor=color,
                            borderwidth=1,
                            borderpad=2,
                            xref=f'x{col}' if col > 1 else 'x',
                            yref=f'y{col}' if col > 1 else 'y'
                        )
            
            fig.update_layout(
                title=dict(
                    text='<b>Figure 8: Feedback Loop Circuits</b><br>'
                         f'<sub>{len(feedback_loops)} regulatory loops across {len(loops_by_target)} targets</sub>',
                    font=dict(size=16)
                ),
                template='plotly_white',
                width=300 * n_targets + 100,
                height=500,
                showlegend=True
            )
            
            # Remove axes
            for i in range(1, n_targets + 1):
                fig.update_xaxes(showgrid=False, zeroline=False, showticklabels=False, row=1, col=i)
                fig.update_yaxes(showgrid=False, zeroline=False, showticklabels=False, 
                                scaleanchor=f'x{i}' if i > 1 else 'x', row=1, col=i)
            
            output_file = output_path / 'S4_fig8_feedback_loops.html'
            fig.write_html(str(output_file), include_plotlyjs='cdn')
            logger.info(f"Generated feedback loops diagram: {output_file}")
            return output_file
            
        except ImportError:
            logger.warning("Plotly not available for feedback loops")
            return None
        except Exception as e:
            logger.error(f"Failed to generate feedback loops: {e}")
            return None

    def plot_centrality_comparison_plotly(self, data: Dict[str, Any], output_path: Path) -> Optional[Path]:
        """
        Figure 4 (adapted): Target Centrality Comparison using Plotly.
        Grouped bar chart showing network vs betweenness centrality.
        """
        try:
            import plotly.graph_objects as go
            
            centrality_data = data.get('centrality_data', [])
            if not centrality_data:
                # Try to extract from individual_results
                individual_results = data.get('individual_results', [])
                centrality_data = []
                for result in individual_results:
                    if isinstance(result, dict):
                        impact = result.get('network_impact', {})
                        centrality_data.append({
                            'target': result.get('target_node', 'Unknown'),
                            'network_centrality': impact.get('network_centrality', 0),
                            'betweenness_centrality': impact.get('betweenness_centrality', 0),
                            'network_coverage': impact.get('network_coverage', 0)
                        })
            
            if not centrality_data:
                logger.warning("No centrality data for comparison")
                return None
            
            targets = [d['target'] for d in centrality_data]
            network_cent = [d.get('network_centrality', 0) for d in centrality_data]
            betweenness = [d.get('betweenness_centrality', 0) for d in centrality_data]
            coverage = [d.get('network_coverage', 0) for d in centrality_data]
            
            fig = go.Figure()
            
            fig.add_trace(go.Bar(
                name='Network Centrality',
                x=targets, y=network_cent,
                marker_color='#1f77b4',
                text=[f'{v:.2f}' for v in network_cent],
                textposition='outside'
            ))
            
            fig.add_trace(go.Bar(
                name='Betweenness Centrality',
                x=targets, y=betweenness,
                marker_color='#ff7f0e',
                text=[f'{v:.2f}' for v in betweenness],
                textposition='outside'
            ))
            
            fig.add_trace(go.Bar(
                name='Network Coverage',
                x=targets, y=coverage,
                marker_color='#2ca02c',
                text=[f'{v:.2f}' for v in coverage],
                textposition='outside'
            ))
            
            fig.update_layout(
                title=dict(
                    text='<b>Target Centrality Comparison</b><br>'
                         '<sub>Network metrics for MRA simulation targets</sub>',
                    font=dict(size=16)
                ),
                xaxis=dict(title='<b>Target</b>'),
                yaxis=dict(title='<b>Centrality Score</b>', range=[0, 1.1]),
                barmode='group',
                template='plotly_white',
                width=800, height=500,
                legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='center', x=0.5)
            )
            
            output_file = output_path / 'S4_centrality_comparison.html'
            fig.write_html(str(output_file), include_plotlyjs='cdn')
            logger.info(f"Generated centrality comparison: {output_file}")
            return output_file
            
        except ImportError:
            logger.warning("Plotly not available for centrality comparison")
            return None
        except Exception as e:
            logger.error(f"Failed to generate centrality comparison: {e}")
            return None

