"""
Basic tests for visualization module.

Run with: pytest tests/test_visualization.py
"""

import pytest
from pathlib import Path
import json

from src.visualization.base import BaseVisualizer
from src.visualization.orchestrator import VisualizationOrchestrator
from src.visualization.styles import VisualizationStyles


class DummyVisualizer(BaseVisualizer):
    """Concrete subclass for BaseVisualizer tests."""

    def visualize(self, data, output_dir, interactive=False, formats=None):
        return []


class TestBaseVisualizer:
    """Test BaseVisualizer functionality."""
    
    def test_style_setup(self):
        """Test style configuration."""
        viz = DummyVisualizer(style='publication')
        viz._setup_style()
        # Should not raise an error
        assert viz.style == 'publication'
    
    def test_get_color_palette(self):
        """Test color palette retrieval."""
        viz = DummyVisualizer()
        colors = viz.get_color_palette('default')
        assert len(colors) == 10
        assert all(c.startswith('#') for c in colors)
    
    def test_output_dir_creation(self):
        """Test output directory creation."""
        viz = DummyVisualizer()
        output_path = viz.create_output_dir('test_output', 1)
        assert output_path.exists()
        assert output_path.name == 'scenario_1'
        # Clean up
        output_path.rmdir()
        output_path.parent.rmdir()


class TestVisualizationStyles:
    """Test VisualizationStyles functionality."""
    
    def test_get_palette(self):
        """Test palette retrieval."""
        palette = VisualizationStyles.get_palette('default')
        assert len(palette) == 10
    
    def test_get_color(self):
        """Test specific color retrieval."""
        color = VisualizationStyles.get_color('biological', 'upregulated')
        assert color.startswith('#')
    
    def test_get_network_style(self):
        """Test network style configuration."""
        style = VisualizationStyles.get_network_style()
        assert 'node_size' in style
        assert 'node_color' in style
        assert 'with_labels' in style
    
    def test_get_figure_size(self):
        """Test figure size calculation."""
        size = VisualizationStyles.get_figure_size('network', 'publication')
        assert len(size) == 2
        assert all(isinstance(s, (int, float)) for s in size)


class TestVisualizationOrchestrator:
    """Test VisualizationOrchestrator functionality."""
    
    @pytest.fixture
    def mock_results_file(self, tmp_path):
        """Create a mock results JSON file."""
        results = {
            "timestamp": "2025-01-01T00:00:00",
            "results": [
                {
                    "scenario_id": 1,
                    "scenario_name": "Disease Network",
                    "data": {
                        "network_nodes": [{"id": "test1"}, {"id": "test2"}],
                        "network_edges": [{"source": "test1", "target": "test2"}],
                        "pathway_enrichment": [],
                        "validation_score": 0.75
                    }
                }
            ]
        }
        
        file_path = tmp_path / "test_results.json"
        with open(file_path, 'w') as f:
            json.dump(results, f)
        
        return file_path
    
    def test_initialization(self):
        """Test orchestrator initialization."""
        orchestrator = VisualizationOrchestrator(style='publication')
        assert len(orchestrator.visualizers) == 6
        assert orchestrator.style == 'publication'
    
    def test_detect_scenarios(self, mock_results_file):
        """Test scenario detection."""
        orchestrator = VisualizationOrchestrator()
        
        with open(mock_results_file, 'r') as f:
            results = json.load(f)
        
        scenarios = orchestrator._detect_scenarios(results)
        assert scenarios == [1]
    
    def test_extract_scenario_data(self, mock_results_file):
        """Test scenario data extraction."""
        orchestrator = VisualizationOrchestrator()
        
        with open(mock_results_file, 'r') as f:
            results = json.load(f)
        
        data = orchestrator._extract_scenario_data(results, 1)
        assert data is not None
        assert 'network_nodes' in data
        assert len(data['network_nodes']) == 2


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
