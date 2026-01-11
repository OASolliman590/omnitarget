"""
CLI Command for Visualization Generation

Provides command-line interface for generating visualizations
from OmniTarget analysis results.
"""

import logging
import sys
from pathlib import Path
from typing import List, Optional

import questionary

from src.visualization.orchestrator import VisualizationOrchestrator

logger = logging.getLogger(__name__)


def visualize_command(
    results_file: str,
    output_dir: str = 'results/figures',
    scenario: Optional[int] = None,
    interactive: bool = False,
    format: str = 'png',
    style: str = 'publication'
):
    """
    Generate visualizations from OmniTarget results.
    
    Args:
        results_file: Path to JSON results file
        output_dir: Output directory for figures
        scenario: Specific scenario ID (1-6) or None for all
        interactive: Generate interactive HTML versions
        format: Output format(s) - 'png', 'pdf', 'svg', 'all'
        style: Visualization style preset - 'publication', 'presentation', 'notebook'
    """
    logger.info("Starting visualization generation")
    
    # Validate inputs
    results_path = Path(results_file)
    if not results_path.exists():
        print(f"❌ Error: Results file not found: {results_file}")
        sys.exit(1)
    
    # Parse formats
    if format == 'all':
        formats = ['png', 'pdf', 'svg']
    else:
        formats = [f.strip() for f in format.split(',')]
    
    # Validate format choices
    valid_formats = ['png', 'pdf', 'svg']
    for fmt in formats:
        if fmt not in valid_formats:
            print(f"❌ Error: Invalid format '{fmt}'. Valid formats: {', '.join(valid_formats)}")
            sys.exit(1)
    
    # Validate style
    valid_styles = ['publication', 'presentation', 'notebook']
    if style not in valid_styles:
        print(f"❌ Error: Invalid style '{style}'. Valid styles: {', '.join(valid_styles)}")
        sys.exit(1)
    
    # Validate scenario if provided
    if scenario is not None and not (1 <= scenario <= 6):
        print(f"❌ Error: Invalid scenario ID '{scenario}'. Must be 1-6.")
        sys.exit(1)
    
    try:
        # Initialize orchestrator
        orchestrator = VisualizationOrchestrator(style=style)
        
        print(f"\n🎨 Generating visualizations...")
        print(f"   Results file: {results_file}")
        print(f"   Output directory: {output_dir}")
        print(f"   Style: {style}")
        print(f"   Formats: {', '.join(formats)}")
        print(f"   Interactive: {'Yes' if interactive else 'No'}")
        
        # Generate visualizations
        if scenario is not None:
            print(f"   Scenario: {scenario}\n")
            generated_files = orchestrator.visualize_scenario(
                scenario_id=scenario,
                json_path=str(results_path),
                output_dir=output_dir,
                interactive=interactive,
                formats=formats
            )
            
            if generated_files:
                print(f"\n✅ Generated {len(generated_files)} visualizations for Scenario {scenario}")
                print(f"   Output: {output_dir}/scenario_{scenario}/")
            else:
                print(f"\n⚠️  No visualizations generated for Scenario {scenario}")
                print("   Check if the scenario has valid data in the results file.")
        
        else:
            print(f"   Scenario: All\n")
            all_generated_files = orchestrator.visualize_all_scenarios(
                json_path=str(results_path),
                output_dir=output_dir,
                interactive=interactive,
                formats=formats
            )
            
            total_files = sum(len(files) for files in all_generated_files.values())
            
            if total_files > 0:
                print(f"\n✅ Generated {total_files} visualizations across {len(all_generated_files)} scenarios")
                
                for scenario_id, files in sorted(all_generated_files.items()):
                    if files:
                        print(f"   Scenario {scenario_id}: {len(files)} figures")
                
                print(f"\n📊 Summary report: {output_dir}/index.html")
            else:
                print(f"\n⚠️  No visualizations generated")
                print("   Check if the results file contains valid scenario data.")
        
        print(f"\n✨ Visualization generation complete!")
        
    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Visualization failed: {e}", exc_info=True)
        print(f"\n❌ Error generating visualizations: {e}")
        sys.exit(1)


def interactive_visualize():
    """Interactive mode for visualization generation."""
    print("\n" + "="*60)
    print("🎨 OmniTarget Visualization Generator")
    print("="*60 + "\n")
    
    # Get results file
    results_file = questionary.path(
        "Path to results JSON file:",
        default="results/",
        only_directories=False
    ).ask()
    
    if not results_file:
        print("Cancelled.")
        return
    
    if not Path(results_file).exists():
        print(f"❌ Error: File not found: {results_file}")
        return
    
    # Get output directory
    output_dir = questionary.text(
        "Output directory for figures:",
        default="results/figures"
    ).ask()
    
    if not output_dir:
        print("Cancelled.")
        return
    
    # Get scenario selection
    scenario_choice = questionary.select(
        "Generate visualizations for:",
        choices=[
            "All scenarios",
            "Scenario 1: Disease Network",
            "Scenario 2: Target Analysis",
            "Scenario 3: Cancer Analysis",
            "Scenario 4: MRA Simulation",
            "Scenario 5: Pathway Comparison",
            "Scenario 6: Drug Repurposing"
        ]
    ).ask()
    
    if not scenario_choice:
        print("Cancelled.")
        return
    
    if scenario_choice == "All scenarios":
        scenario = None
    else:
        scenario = int(scenario_choice.split(':')[0].split()[-1])
    
    # Get style
    style = questionary.select(
        "Visualization style:",
        choices=[
            "publication",
            "presentation",
            "notebook"
        ],
        default="publication"
    ).ask()
    
    if not style:
        print("Cancelled.")
        return
    
    # Get formats
    format_choices = questionary.checkbox(
        "Output formats (select multiple):",
        choices=["png", "pdf", "svg"]
    ).ask()
    
    if not format_choices:
        print("Cancelled.")
        return
    
    # Get interactive option
    interactive = questionary.confirm(
        "Generate interactive HTML versions?",
        default=False
    ).ask()
    
    # Generate visualizations
    visualize_command(
        results_file=results_file,
        output_dir=output_dir,
        scenario=scenario,
        interactive=interactive,
        format=','.join(format_choices),
        style=style
    )


if __name__ == '__main__':
    # Allow direct execution
    if len(sys.argv) > 1:
        import argparse
        
        parser = argparse.ArgumentParser(description='Generate OmniTarget visualizations')
        parser.add_argument('results_file', help='Path to JSON results file')
        parser.add_argument('--output-dir', '-o', default='results/figures',
                          help='Output directory for figures')
        parser.add_argument('--scenario', '-s', type=int, choices=[1,2,3,4,5,6],
                          help='Specific scenario ID (1-6)')
        parser.add_argument('--interactive', '-i', action='store_true',
                          help='Generate interactive HTML versions')
        parser.add_argument('--format', '-f', default='png',
                          help='Output formats: png, pdf, svg, all (comma-separated)')
        parser.add_argument('--style', default='publication',
                          choices=['publication', 'presentation', 'notebook'],
                          help='Visualization style preset')
        
        args = parser.parse_args()
        
        visualize_command(
            results_file=args.results_file,
            output_dir=args.output_dir,
            scenario=args.scenario,
            interactive=args.interactive,
            format=args.format,
            style=args.style
        )
    else:
        interactive_visualize()

