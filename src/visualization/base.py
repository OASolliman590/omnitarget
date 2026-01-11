"""
Base Visualizer Module

Provides common functionality for all scenario visualizers including:
- JSON parsing and data extraction
- Style configuration and color schemes
- Output path management
- Format conversion utilities
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from abc import ABC, abstractmethod

import matplotlib.pyplot as plt
import matplotlib as mpl

logger = logging.getLogger(__name__)


class BaseVisualizer(ABC):
    """
    Base class for all scenario visualizers.
    
    Provides common functionality for loading data, managing output,
    and applying consistent styling across all visualizations.
    """
    
    def __init__(self, style: str = 'publication'):
        """
        Initialize base visualizer.
        
        Args:
            style: Style preset ('publication', 'presentation', 'notebook')
        """
        self.style = style
        self._setup_style()
        
    def _setup_style(self):
        """Set up matplotlib style based on preset."""
        if self.style == 'publication':
            plt.style.use('seaborn-v0_8-paper')
            mpl.rcParams['figure.dpi'] = 300
            mpl.rcParams['savefig.dpi'] = 300
            mpl.rcParams['font.size'] = 10
            mpl.rcParams['axes.labelsize'] = 12
            mpl.rcParams['axes.titlesize'] = 14
            mpl.rcParams['xtick.labelsize'] = 10
            mpl.rcParams['ytick.labelsize'] = 10
            mpl.rcParams['legend.fontsize'] = 10
            mpl.rcParams['figure.titlesize'] = 16
        elif self.style == 'presentation':
            plt.style.use('seaborn-v0_8-talk')
            mpl.rcParams['figure.dpi'] = 150
            mpl.rcParams['savefig.dpi'] = 150
            mpl.rcParams['font.size'] = 14
        elif self.style == 'notebook':
            plt.style.use('seaborn-v0_8-notebook')
            mpl.rcParams['figure.dpi'] = 100
            mpl.rcParams['savefig.dpi'] = 100
    
    def load_results(self, json_path: str) -> Dict[str, Any]:
        """
        Load and parse results from JSON file.
        
        Args:
            json_path: Path to JSON results file
            
        Returns:
            Dictionary containing parsed results
            
        Raises:
            FileNotFoundError: If JSON file doesn't exist
            json.JSONDecodeError: If JSON is malformed
        """
        json_path = Path(json_path)
        if not json_path.exists():
            raise FileNotFoundError(f"Results file not found: {json_path}")
        
        logger.info(f"Loading results from {json_path}")
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        return data
    
    def extract_scenario_data(self, results: Dict[str, Any], scenario_id: int) -> Optional[Dict[str, Any]]:
        """
        Extract data for a specific scenario from results.
        
        Args:
            results: Full results dictionary
            scenario_id: Scenario ID to extract
            
        Returns:
            Scenario-specific data or None if not found
        """
        if 'results' not in results:
            logger.warning("No 'results' key found in JSON")
            return None
        
        for scenario in results['results']:
            if scenario.get('scenario_id') == scenario_id:
                logger.info(f"Found scenario {scenario_id}: {scenario.get('scenario_name')}")
                return scenario.get('data', {})
        
        logger.warning(f"Scenario {scenario_id} not found in results")
        return None
    
    def create_output_dir(self, output_dir: str, scenario_id: int) -> Path:
        """
        Create output directory for scenario visualizations.
        
        Args:
            output_dir: Base output directory
            scenario_id: Scenario ID
            
        Returns:
            Path to scenario-specific output directory
        """
        output_path = Path(output_dir) / f"scenario_{scenario_id}"
        output_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Created output directory: {output_path}")
        return output_path
    
    def save_figure(
        self,
        fig: plt.Figure,
        output_path: Path,
        filename: str,
        formats: List[str] = ['png']
    ) -> List[Path]:
        """
        Save figure in multiple formats.
        
        Args:
            fig: Matplotlib figure to save
            output_path: Output directory path
            filename: Base filename (without extension)
            formats: List of formats ('png', 'pdf', 'svg')
            
        Returns:
            List of saved file paths
        """
        saved_files = []
        
        for fmt in formats:
            file_path = output_path / f"{filename}.{fmt}"
            try:
                fig.savefig(
                    file_path,
                    format=fmt,
                    dpi=300 if fmt == 'png' else None,
                    bbox_inches='tight',
                    facecolor='white',
                    edgecolor='none'
                )
                saved_files.append(file_path)
                logger.info(f"Saved figure: {file_path}")
            except Exception as e:
                logger.error(f"Failed to save figure as {fmt}: {e}")
        
        plt.close(fig)
        return saved_files
    
    def get_color_palette(self, palette_name: str = 'default') -> List[str]:
        """
        Get color palette for visualizations.
        
        Args:
            palette_name: Name of color palette
            
        Returns:
            List of color hex codes
        """
        palettes = {
            'default': ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
                       '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'],
            'biological': ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00',
                          '#ffff33', '#a65628', '#f781bf', '#999999'],
            'heatmap': ['#053061', '#2166ac', '#4393c3', '#92c5de', '#d1e5f0',
                       '#fddbc7', '#f4a582', '#d6604d', '#b2182b', '#67001f'],
            'network': ['#4E79A7', '#F28E2B', '#E15759', '#76B7B2', '#59A14F',
                       '#EDC948', '#B07AA1', '#FF9DA7', '#9C755F', '#BAB0AC'],
        }
        return palettes.get(palette_name, palettes['default'])
    
    @abstractmethod
    def visualize(
        self,
        data: Dict[str, Any],
        output_dir: str,
        interactive: bool = False,
        formats: List[str] = ['png']
    ) -> List[Path]:
        """
        Generate all visualizations for the scenario.
        
        Args:
            data: Scenario-specific data
            output_dir: Output directory for figures
            interactive: Generate interactive HTML versions
            formats: Output formats for static figures
            
        Returns:
            List of generated file paths
        """
        pass
    
    def _validate_data(self, data: Dict[str, Any], required_keys: List[str]) -> bool:
        """
        Validate that data contains required keys.
        
        Args:
            data: Data dictionary to validate
            required_keys: List of required keys
            
        Returns:
            True if all keys present, False otherwise
        """
        missing_keys = [key for key in required_keys if key not in data]
        if missing_keys:
            logger.warning(f"Missing required keys: {missing_keys}")
            return False
        return True
    
    def _safe_get(self, data: Dict[str, Any], key: str, default: Any = None) -> Any:
        """
        Safely get value from dictionary with default.
        
        Args:
            data: Dictionary to access
            key: Key to retrieve
            default: Default value if key not found
            
        Returns:
            Value or default
        """
        return data.get(key, default)

