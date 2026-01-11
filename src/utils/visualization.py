"""
Network visualization utilities.
"""

from typing import Dict, List, Any, Optional
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np


class NetworkVisualizer:
    """Utility class for network visualization."""
    
    @staticmethod
    def create_network_plot(
        network: nx.Graph,
        title: str = "Network",
        node_size: int = 300,
        node_color: str = 'lightblue',
        edge_color: str = 'gray',
        layout: str = 'spring'
    ) -> plt.Figure:
        """
        Create a network visualization plot.
        
        Args:
            network: NetworkX graph
            title: Plot title
            node_size: Size of nodes
            node_color: Color of nodes
            edge_color: Color of edges
            layout: Layout algorithm ('spring', 'circular', 'random')
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Choose layout
        if layout == 'spring':
            pos = nx.spring_layout(network, k=1, iterations=50)
        elif layout == 'circular':
            pos = nx.circular_layout(network)
        elif layout == 'random':
            pos = nx.random_layout(network)
        else:
            pos = nx.spring_layout(network)
        
        # Draw network
        nx.draw_networkx_nodes(
            network, pos, 
            node_size=node_size, 
            node_color=node_color,
            ax=ax
        )
        
        nx.draw_networkx_edges(
            network, pos,
            edge_color=edge_color,
            alpha=0.5,
            ax=ax
        )
        
        # Add labels
        nx.draw_networkx_labels(network, pos, ax=ax)
        
        ax.set_title(title)
        ax.axis('off')
        
        return fig
    
    @staticmethod
    def create_centrality_plot(
        network: nx.Graph,
        centrality_type: str = 'betweenness'
    ) -> plt.Figure:
        """
        Create a centrality visualization plot.
        
        Args:
            network: NetworkX graph
            centrality_type: Type of centrality ('betweenness', 'closeness', 'degree')
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Calculate centrality
        if centrality_type == 'betweenness':
            centrality = nx.betweenness_centrality(network)
        elif centrality_type == 'closeness':
            centrality = nx.closeness_centrality(network)
        elif centrality_type == 'degree':
            centrality = nx.degree_centrality(network)
        else:
            centrality = nx.degree_centrality(network)
        
        # Sort by centrality
        sorted_centrality = sorted(centrality.items(), key=lambda x: x[1], reverse=True)
        nodes, values = zip(*sorted_centrality[:20])  # Top 20 nodes
        
        # Create bar plot
        ax.bar(range(len(nodes)), values)
        ax.set_xlabel('Nodes')
        ax.set_ylabel(f'{centrality_type.title()} Centrality')
        ax.set_title(f'{centrality_type.title()} Centrality Distribution')
        ax.set_xticks(range(len(nodes)))
        ax.set_xticklabels(nodes, rotation=45, ha='right')
        
        plt.tight_layout()
        return fig
    
    @staticmethod
    def create_community_plot(
        network: nx.Graph,
        communities: Optional[Dict[str, int]] = None
    ) -> plt.Figure:
        """
        Create a community visualization plot.
        
        Args:
            network: NetworkX graph
            communities: Community assignments (optional)
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Detect communities if not provided
        if communities is None:
            try:
                import networkx.algorithms.community as nx_comm
                communities = nx_comm.louvain_communities(network)
                # Convert to node -> community mapping
                node_communities = {}
                for i, community in enumerate(communities):
                    for node in community:
                        node_communities[node] = i
                communities = node_communities
            except ImportError:
                # Fallback to simple community detection
                communities = {node: 0 for node in network.nodes()}
        
        # Create color map
        unique_communities = set(communities.values())
        colors = plt.cm.Set3(np.linspace(0, 1, len(unique_communities)))
        node_colors = [colors[communities[node]] for node in network.nodes()]
        
        # Draw network
        pos = nx.spring_layout(network, k=1, iterations=50)
        nx.draw_networkx_nodes(
            network, pos,
            node_color=node_colors,
            node_size=300,
            ax=ax
        )
        nx.draw_networkx_edges(network, pos, alpha=0.5, ax=ax)
        nx.draw_networkx_labels(network, pos, ax=ax)
        
        ax.set_title('Network Communities')
        ax.axis('off')
        
        return fig
    
    @staticmethod
    def create_simulation_plot(
        simulation_results: Dict[str, float],
        title: str = "Simulation Results"
    ) -> plt.Figure:
        """
        Create a simulation results visualization plot.
        
        Args:
            simulation_results: Dictionary of node -> effect strength
            title: Plot title
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Sort results by effect strength
        sorted_results = sorted(simulation_results.items(), key=lambda x: abs(x[1]), reverse=True)
        nodes, effects = zip(*sorted_results[:20])  # Top 20 nodes
        
        # Create bar plot
        colors = ['red' if effect < 0 else 'blue' for effect in effects]
        bars = ax.bar(range(len(nodes)), effects, color=colors, alpha=0.7)
        
        ax.set_xlabel('Nodes')
        ax.set_ylabel('Effect Strength')
        ax.set_title(title)
        ax.set_xticks(range(len(nodes)))
        ax.set_xticklabels(nodes, rotation=45, ha='right')
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        
        # Add legend
        ax.legend(['Inhibition', 'Activation'], loc='upper right')
        
        plt.tight_layout()
        return fig
    
    @staticmethod
    def save_network_to_file(
        network: nx.Graph,
        filename: str,
        format: str = 'png'
    ) -> None:
        """
        Save network visualization to file.
        
        Args:
            network: NetworkX graph
            filename: Output filename
            format: Output format ('png', 'pdf', 'svg')
        """
        fig = NetworkVisualizer.create_network_plot(network)
        fig.savefig(filename, format=format, dpi=300, bbox_inches='tight')
        plt.close(fig)
    
    @staticmethod
    def create_interactive_plot(
        network: nx.Graph,
        output_file: str = 'network.html'
    ) -> None:
        """
        Create an interactive network visualization.
        
        Args:
            network: NetworkX graph
            output_file: Output HTML file
        """
        try:
            import plotly.graph_objects as go
            import plotly.express as px
            
            # Get node positions
            pos = nx.spring_layout(network, k=1, iterations=50)
            
            # Extract node and edge data
            node_x = [pos[node][0] for node in network.nodes()]
            node_y = [pos[node][1] for node in network.nodes()]
            node_text = [f"Node: {node}" for node in network.nodes()]
            
            edge_x = []
            edge_y = []
            for edge in network.edges():
                x0, y0 = pos[edge[0]]
                x1, y1 = pos[edge[1]]
                edge_x.extend([x0, x1, None])
                edge_y.extend([y0, y1, None])
            
            # Create plot
            fig = go.Figure()
            
            # Add edges
            fig.add_trace(go.Scatter(
                x=edge_x, y=edge_y,
                line=dict(width=0.5, color='#888'),
                hoverinfo='none',
                mode='lines'
            ))
            
            # Add nodes
            fig.add_trace(go.Scatter(
                x=node_x, y=node_y,
                mode='markers',
                hoverinfo='text',
                text=node_text,
                marker=dict(
                    size=10,
                    color='lightblue',
                    line=dict(width=2, color='black')
                )
            ))
            
            # Update layout
            fig.update_layout(
                title="Interactive Network Visualization",
                showlegend=False,
                hovermode='closest',
                margin=dict(b=20,l=5,r=5,t=40),
                annotations=[ dict(
                    text="Interactive network visualization",
                    showarrow=False,
                    xref="paper", yref="paper",
                    x=0.005, y=-0.002,
                    xanchor='left', yanchor='bottom',
                    font=dict(color='black', size=12)
                )],
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
            )
            
            # Save to file
            fig.write_html(output_file)
            
        except ImportError:
            print("Plotly not available. Install with: pip install plotly")
            # Fallback to static plot
            NetworkVisualizer.save_network_to_file(network, output_file.replace('.html', '.png'))
