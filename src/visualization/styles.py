"""
Visualization Style Configuration

Provides publication-quality style presets, color palettes,
and configuration options for consistent visualization output.
"""

from typing import Dict, Any, List
import matplotlib.pyplot as plt
import matplotlib as mpl


class VisualizationStyles:
    """Centralized style management for visualizations."""
    
    # Color palettes for different visualization types
    COLOR_PALETTES = {
        'default': {
            'primary': '#1f77b4',
            'secondary': '#ff7f0e',
            'success': '#2ca02c',
            'danger': '#d62728',
            'warning': '#ff7f00',
            'info': '#17becf',
            'palette': ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
                       '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
        },
        'biological': {
            'upregulated': '#d62728',
            'downregulated': '#2ca02c',
            'neutral': '#7f7f7f',
            'pathway': '#377eb8',
            'protein': '#ff7f00',
            'palette': ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00',
                       '#ffff33', '#a65628', '#f781bf', '#999999']
        },
        'heatmap_blue_red': {
            'low': '#053061',
            'mid': '#f7f7f7',
            'high': '#67001f',
            'palette': ['#053061', '#2166ac', '#4393c3', '#92c5de', '#d1e5f0',
                       '#fddbc7', '#f4a582', '#d6604d', '#b2182b', '#67001f']
        },
        'network': {
            'node_default': '#4E79A7',
            'node_target': '#E15759',
            'node_driver': '#59A14F',
            'edge_default': '#999999',
            'edge_strong': '#333333',
            'palette': ['#4E79A7', '#F28E2B', '#E15759', '#76B7B2', '#59A14F',
                       '#EDC948', '#B07AA1', '#FF9DA7', '#9C755F', '#BAB0AC']
        },
        'categorical': {
            'palette': ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6',
                       '#1abc9c', '#34495e', '#e67e22', '#95a5a6', '#d35400']
        }
    }
    
    # Style presets for different use cases
    STYLE_PRESETS = {
        'publication': {
            'figure.dpi': 300,
            'savefig.dpi': 300,
            'font.size': 10,
            'font.family': 'sans-serif',
            'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
            'axes.labelsize': 12,
            'axes.titlesize': 14,
            'axes.linewidth': 1.0,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
            'xtick.major.width': 1.0,
            'ytick.major.width': 1.0,
            'legend.fontsize': 10,
            'legend.frameon': True,
            'legend.framealpha': 0.8,
            'figure.titlesize': 16,
            'figure.figsize': (10, 8),
            'grid.alpha': 0.3,
            'grid.linestyle': '--',
        },
        'presentation': {
            'figure.dpi': 150,
            'savefig.dpi': 150,
            'font.size': 14,
            'font.family': 'sans-serif',
            'axes.labelsize': 16,
            'axes.titlesize': 18,
            'axes.linewidth': 1.5,
            'xtick.labelsize': 14,
            'ytick.labelsize': 14,
            'xtick.major.width': 1.5,
            'ytick.major.width': 1.5,
            'legend.fontsize': 14,
            'figure.titlesize': 20,
            'figure.figsize': (12, 9),
            'grid.alpha': 0.4,
            'grid.linestyle': '--',
        },
        'notebook': {
            'figure.dpi': 100,
            'savefig.dpi': 100,
            'font.size': 11,
            'font.family': 'sans-serif',
            'axes.labelsize': 11,
            'axes.titlesize': 13,
            'axes.linewidth': 1.0,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
            'legend.fontsize': 10,
            'figure.titlesize': 14,
            'figure.figsize': (10, 7),
            'grid.alpha': 0.3,
        }
    }
    
    @classmethod
    def apply_style(cls, style_name: str = 'publication'):
        """
        Apply a style preset to matplotlib.
        
        Args:
            style_name: Name of style preset ('publication', 'presentation', 'notebook')
        """
        if style_name not in cls.STYLE_PRESETS:
            raise ValueError(f"Unknown style: {style_name}. Available: {list(cls.STYLE_PRESETS.keys())}")
        
        # Apply base seaborn style
        plt.style.use('seaborn-v0_8-whitegrid')
        
        # Apply custom parameters
        for key, value in cls.STYLE_PRESETS[style_name].items():
            mpl.rcParams[key] = value
    
    @classmethod
    def get_palette(cls, palette_name: str = 'default') -> List[str]:
        """
        Get color palette by name.
        
        Args:
            palette_name: Name of color palette
            
        Returns:
            List of color hex codes
        """
        if palette_name not in cls.COLOR_PALETTES:
            return cls.COLOR_PALETTES['default']['palette']
        return cls.COLOR_PALETTES[palette_name]['palette']
    
    @classmethod
    def get_color(cls, palette_name: str, color_name: str) -> str:
        """
        Get specific color from palette.
        
        Args:
            palette_name: Name of color palette
            color_name: Name of specific color
            
        Returns:
            Color hex code
        """
        palette = cls.COLOR_PALETTES.get(palette_name, cls.COLOR_PALETTES['default'])
        return palette.get(color_name, palette.get('primary', '#1f77b4'))
    
    @classmethod
    def get_network_style(cls) -> Dict[str, Any]:
        """
        Get default style configuration for network visualizations.
        
        Returns:
            Dictionary of NetworkX visualization parameters
        """
        return {
            'node_size': 500,
            'node_color': cls.get_color('network', 'node_default'),
            'edge_color': cls.get_color('network', 'edge_default'),
            'alpha': 0.8,
            'linewidths': 2,
            'width': 1.5,
            'font_size': 10,
            'font_weight': 'bold',
            'with_labels': True,
        }
    
    @classmethod
    def get_heatmap_style(cls) -> Dict[str, Any]:
        """
        Get default style configuration for heatmap visualizations.
        
        Returns:
            Dictionary of heatmap visualization parameters
        """
        return {
            'cmap': 'RdBu_r',
            'center': 0,
            'linewidths': 0.5,
            'linecolor': 'white',
            'cbar_kws': {'label': 'Value'},
            'square': True,
            'robust': True,
        }
    
    @classmethod
    def get_figure_size(cls, figure_type: str, style: str = 'publication') -> tuple:
        """
        Get appropriate figure size for different visualization types.
        
        Args:
            figure_type: Type of figure ('network', 'heatmap', 'barplot', 'scatter', 'multi')
            style: Style preset name
            
        Returns:
            Tuple of (width, height) in inches
        """
        base_sizes = cls.STYLE_PRESETS.get(style, cls.STYLE_PRESETS['publication'])
        base_fig = base_sizes['figure.figsize']
        
        sizes = {
            'network': (12, 10),
            'heatmap': (10, 8),
            'barplot': (10, 6),
            'scatter': (8, 8),
            'violin': (10, 6),
            'volcano': (10, 8),
            'multi': (16, 12),
            'wide': (14, 6),
            'tall': (8, 12),
        }
        
        return sizes.get(figure_type, base_fig)

