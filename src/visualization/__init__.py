"""
OmniTarget Visualization Suite

This module provides comprehensive visualization capabilities for all OmniTarget scenarios.
Supports both static (PNG, PDF, SVG) and interactive (HTML) output formats.

Usage:
    from src.visualization import VisualizationOrchestrator
    
    orchestrator = VisualizationOrchestrator()
    orchestrator.visualize_all_scenarios('results/analysis.json', 'results/figures')
"""

from src.visualization.base import BaseVisualizer
from src.visualization.orchestrator import VisualizationOrchestrator

__all__ = [
    'BaseVisualizer',
    'VisualizationOrchestrator',
]

__version__ = '1.0.0'

