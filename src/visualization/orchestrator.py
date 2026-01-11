"""
Visualization Orchestrator

Coordinates visualization generation across all scenarios,
manages output organization, and creates summary reports.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional

from src.visualization.base import BaseVisualizer
from src.visualization.data_adapter import VisualizationDataAdapter
from src.visualization.scenario_1_visualizer import DiseaseNetworkVisualizer
from src.visualization.scenario_2_visualizer import TargetAnalysisVisualizer
from src.visualization.scenario_3_visualizer import CancerAnalysisVisualizer
from src.visualization.scenario_4_visualizer import MRASimulationVisualizer
from src.visualization.scenario_5_visualizer import PathwayComparisonVisualizer
from src.visualization.scenario_6_visualizer import DrugRepurposingVisualizer

logger = logging.getLogger(__name__)


class VisualizationOrchestrator:
    """
    Orchestrates visualization generation for all OmniTarget scenarios.
    
    Manages:
    - Automatic scenario detection
    - Visualizer selection and execution
    - Output organization
    - Summary report generation
    """
    
    def __init__(self, style: str = 'publication'):
        """
        Initialize orchestrator.
        
        Args:
            style: Default visualization style preset
        """
        self.style = style
        self.visualizers = {
            1: DiseaseNetworkVisualizer(style),
            2: TargetAnalysisVisualizer(style),
            3: CancerAnalysisVisualizer(style),
            4: MRASimulationVisualizer(style),
            5: PathwayComparisonVisualizer(style),
            6: DrugRepurposingVisualizer(style),
        }
        logger.info(f"Initialized VisualizationOrchestrator with style: {style}")
    
    def visualize_scenario(
        self,
        scenario_id: int,
        json_path: str,
        output_dir: str = 'results/figures',
        interactive: bool = False,
        formats: List[str] = ['png']
    ) -> List[Path]:
        """
        Generate visualizations for a specific scenario.
        
        Args:
            scenario_id: Scenario ID (1-6)
            json_path: Path to JSON results file
            output_dir: Output directory for figures
            interactive: Generate interactive HTML versions
            formats: Output formats for static figures
            
        Returns:
            List of generated file paths
        """
        logger.info(f"Generating visualizations for Scenario {scenario_id}")
        
        # Validate scenario ID
        if scenario_id not in self.visualizers:
            logger.error(f"Invalid scenario ID: {scenario_id}")
            raise ValueError(f"Invalid scenario ID: {scenario_id}. Must be 1-6.")
        
        # Load results
        try:
            with open(json_path, 'r') as f:
                results = json.load(f)
        except FileNotFoundError:
            logger.error(f"Results file not found: {json_path}")
            raise
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in {json_path}: {e}")
            raise
        
        # Extract scenario data
        scenario_data = self._extract_scenario_data(results, scenario_id)
        
        if scenario_data is None:
            logger.warning(f"No data found for Scenario {scenario_id}")
            return []
        
        # Adapt data structure to match visualizer expectations
        adapted_data = VisualizationDataAdapter.adapt(scenario_id, scenario_data)
        
        # Generate visualizations
        visualizer = self.visualizers[scenario_id]
        generated_files = visualizer.visualize(
            adapted_data,
            output_dir,
            interactive=interactive,
            formats=formats
        )
        
        logger.info(f"Generated {len(generated_files)} visualizations for Scenario {scenario_id}")
        return generated_files
    
    def visualize_all_scenarios(
        self,
        json_path: str,
        output_dir: str = 'results/figures',
        interactive: bool = False,
        formats: List[str] = ['png']
    ) -> Dict[int, List[Path]]:
        """
        Generate visualizations for all scenarios in results file.
        
        Args:
            json_path: Path to JSON results file
            output_dir: Output directory for figures
            interactive: Generate interactive HTML versions
            formats: Output formats for static figures
            
        Returns:
            Dictionary mapping scenario IDs to lists of generated file paths
        """
        logger.info(f"Generating visualizations for all scenarios from {json_path}")
        
        # Load results
        try:
            with open(json_path, 'r') as f:
                results = json.load(f)
        except FileNotFoundError:
            logger.error(f"Results file not found: {json_path}")
            raise
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in {json_path}: {e}")
            raise
        
        # Detect available scenarios
        available_scenarios = self._detect_scenarios(results)
        logger.info(f"Found {len(available_scenarios)} scenarios: {available_scenarios}")
        
        # Generate visualizations for each scenario
        all_generated_files = {}
        
        for scenario_id in available_scenarios:
            try:
                scenario_data = self._extract_scenario_data(results, scenario_id)
                
                if scenario_data:
                    # Adapt data structure
                    adapted_data = VisualizationDataAdapter.adapt(scenario_id, scenario_data)
                    
                    visualizer = self.visualizers[scenario_id]
                    generated_files = visualizer.visualize(
                        adapted_data,
                        output_dir,
                        interactive=interactive,
                        formats=formats
                    )
                    all_generated_files[scenario_id] = generated_files
                    logger.info(f"Scenario {scenario_id}: Generated {len(generated_files)} files")
                else:
                    logger.warning(f"Scenario {scenario_id}: No data available")
                    all_generated_files[scenario_id] = []
            
            except Exception as e:
                logger.error(f"Failed to generate visualizations for Scenario {scenario_id}: {e}")
                all_generated_files[scenario_id] = []
        
        # Generate summary report
        try:
            self.generate_report(results, output_dir, all_generated_files)
        except Exception as e:
            logger.error(f"Failed to generate summary report: {e}")
        
        total_files = sum(len(files) for files in all_generated_files.values())
        logger.info(f"Total visualizations generated: {total_files}")
        
        return all_generated_files
    
    def generate_report(
        self,
        results: Dict[str, Any],
        output_dir: str,
        generated_files: Dict[int, List[Path]],
        format: str = 'html'
    ) -> Path:
        """
        Generate summary report with visualization index.
        
        Args:
            results: Full results dictionary
            output_dir: Output directory
            generated_files: Dictionary of generated files by scenario
            format: Report format ('html' or 'markdown')
            
        Returns:
            Path to generated report
        """
        logger.info("Generating visualization summary report")
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        if format == 'html':
            report_file = output_path / 'index.html'
            self._generate_html_report(results, generated_files, report_file)
        else:
            report_file = output_path / 'README.md'
            self._generate_markdown_report(results, generated_files, report_file)
        
        logger.info(f"Summary report generated: {report_file}")
        return report_file
    
    def _extract_scenario_data(
        self,
        results: Dict[str, Any],
        scenario_id: int
    ) -> Optional[Dict[str, Any]]:
        """Extract data for specific scenario from results."""
        if 'results' not in results:
            return None
        
        for scenario in results['results']:
            if isinstance(scenario, dict) and scenario.get('scenario_id') == scenario_id:
                return scenario.get('data', {})
        
        return None
    
    def _detect_scenarios(self, results: Dict[str, Any]) -> List[int]:
        """Detect which scenarios are present in results."""
        scenarios = []
        
        if 'results' not in results:
            return scenarios
        
        for scenario in results['results']:
            if isinstance(scenario, dict):
                scenario_id = scenario.get('scenario_id')
                if scenario_id and 1 <= scenario_id <= 6:
                    scenarios.append(scenario_id)
        
        return sorted(scenarios)
    
    def _generate_html_report(
        self,
        results: Dict[str, Any],
        generated_files: Dict[int, List[Path]],
        report_file: Path
    ):
        """Generate HTML summary report."""
        scenario_names = {
            1: "Disease Network Construction",
            2: "Target Analysis",
            3: "Cancer Analysis",
            4: "MRA Simulation",
            5: "Pathway Comparison",
            6: "Drug Repurposing"
        }
        
        html_content = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>OmniTarget Visualization Report</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }
        h1 {
            color: #2c3e50;
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
        }
        h2 {
            color: #34495e;
            margin-top: 30px;
        }
        .scenario {
            background-color: white;
            padding: 20px;
            margin: 20px 0;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .figures {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
            gap: 20px;
            margin-top: 20px;
        }
        .figure {
            border: 1px solid #ddd;
            border-radius: 4px;
            padding: 10px;
            text-align: center;
        }
        .figure img {
            max-width: 100%;
            height: auto;
            border-radius: 4px;
        }
        .figure-title {
            font-weight: bold;
            margin-top: 10px;
            color: #2c3e50;
        }
        .stats {
            background-color: #ecf0f1;
            padding: 15px;
            border-radius: 4px;
            margin: 20px 0;
        }
        .footer {
            margin-top: 40px;
            text-align: center;
            color: #7f8c8d;
            font-size: 0.9em;
        }
    </style>
</head>
<body>
    <h1>🎯 OmniTarget Visualization Report</h1>
"""
        
        # Add overview statistics
        total_figures = sum(len(files) for files in generated_files.values())
        html_content += f"""
    <div class="stats">
        <h3>Overview</h3>
        <p><strong>Total Scenarios:</strong> {len(generated_files)}</p>
        <p><strong>Total Visualizations:</strong> {total_figures}</p>
        <p><strong>Generated:</strong> {results.get('timestamp', 'N/A')}</p>
    </div>
"""
        
        # Add sections for each scenario
        for scenario_id in sorted(generated_files.keys()):
            files = generated_files[scenario_id]
            scenario_name = scenario_names.get(scenario_id, f"Scenario {scenario_id}")
            
            html_content += f"""
    <div class="scenario">
        <h2>Scenario {scenario_id}: {scenario_name}</h2>
        <p><strong>Figures Generated:</strong> {len(files)}</p>
"""
            
            if files:
                html_content += '        <div class="figures">\n'
                
                for file_path in files:
                    if file_path.suffix in ['.png', '.jpg', '.jpeg', '.svg']:
                        rel_path = file_path.relative_to(report_file.parent)
                        figure_name = file_path.stem.replace('_', ' ').title()
                        
                        html_content += f"""
            <div class="figure">
                <a href="{rel_path}" target="_blank">
                    <img src="{rel_path}" alt="{figure_name}">
                </a>
                <div class="figure-title">{figure_name}</div>
            </div>
"""
                    elif file_path.suffix == '.html':
                        rel_path = file_path.relative_to(report_file.parent)
                        figure_name = file_path.stem.replace('_', ' ').title()
                        
                        html_content += f"""
            <div class="figure">
                <a href="{rel_path}" target="_blank">
                    <div style="padding: 50px; background: #ecf0f1; border-radius: 4px;">
                        📊 Interactive
                    </div>
                </a>
                <div class="figure-title">{figure_name}</div>
            </div>
"""
                
                html_content += '        </div>\n'
            else:
                html_content += '        <p><em>No visualizations generated for this scenario.</em></p>\n'
            
            html_content += '    </div>\n'
        
        # Add footer
        html_content += """
    <div class="footer">
        <p>Generated by OmniTarget Visualization Suite</p>
    </div>
</body>
</html>
"""
        
        # Write report
        with open(report_file, 'w') as f:
            f.write(html_content)
    
    def _generate_markdown_report(
        self,
        results: Dict[str, Any],
        generated_files: Dict[int, List[Path]],
        report_file: Path
    ):
        """Generate Markdown summary report."""
        scenario_names = {
            1: "Disease Network Construction",
            2: "Target Analysis",
            3: "Cancer Analysis",
            4: "MRA Simulation",
            5: "Pathway Comparison",
            6: "Drug Repurposing"
        }
        
        md_content = "# OmniTarget Visualization Report\n\n"
        
        # Overview
        total_figures = sum(len(files) for files in generated_files.values())
        md_content += "## Overview\n\n"
        md_content += f"- **Total Scenarios:** {len(generated_files)}\n"
        md_content += f"- **Total Visualizations:** {total_figures}\n"
        md_content += f"- **Generated:** {results.get('timestamp', 'N/A')}\n\n"
        
        # Scenarios
        for scenario_id in sorted(generated_files.keys()):
            files = generated_files[scenario_id]
            scenario_name = scenario_names.get(scenario_id, f"Scenario {scenario_id}")
            
            md_content += f"## Scenario {scenario_id}: {scenario_name}\n\n"
            md_content += f"**Figures Generated:** {len(files)}\n\n"
            
            if files:
                for file_path in files:
                    rel_path = file_path.relative_to(report_file.parent)
                    figure_name = file_path.stem.replace('_', ' ').title()
                    
                    if file_path.suffix in ['.png', '.jpg', '.jpeg']:
                        md_content += f"### {figure_name}\n\n"
                        md_content += f"![{figure_name}]({rel_path})\n\n"
                    else:
                        md_content += f"- [{figure_name}]({rel_path})\n"
                
                md_content += "\n"
            else:
                md_content += "*No visualizations generated for this scenario.*\n\n"
        
        # Write report
        with open(report_file, 'w') as f:
            f.write(md_content)

