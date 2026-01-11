"""
OmniTarget CLI

Command-line interface for the OmniTarget pipeline.
Supports interactive wizard, YAML batch execution, and programmatic modes.
"""

import asyncio
import argparse
import sys
import logging
from pathlib import Path

# Import existing functionality
from ..core.mcp_client_manager import MCPClientManager


class ColoredFormatter(logging.Formatter):
    """User-friendly colored formatter for terminal output."""

    COLORS = {
        'DEBUG': '\033[36m',    # Cyan
        'INFO': '\033[32m',     # Green
        'WARNING': '\033[33m',  # Yellow
        'ERROR': '\033[31m',    # Red
        'CRITICAL': '\033[35m', # Magenta
    }
    RESET = '\033[0m'

    def format(self, record):
        # Simplify logger names for readability (keep only last component)
        record.name = record.name.split('.')[-1]

        # Add color to level name
        if record.levelname in self.COLORS:
            colored_level = f"{self.COLORS[record.levelname]}{record.levelname}{self.RESET}"
            record.levelname = colored_level

        return super().format(record)

# Lazy imports - only import modules when their commands are actually used
# This avoids requiring optional dependencies (questionary, pyyaml) for all commands


async def run_basic_example():
    """Run the basic usage example."""
    print("🧬 Running OmniTarget Basic Example...")
    
    config_path = "config/mcp_servers.json"
    manager = MCPClientManager(config_path)
    
    try:
        async with manager.session() as session:
            print("✅ MCP servers started successfully")
            
            # Test KEGG
            print("🔍 Testing KEGG...")
            diseases = await session.kegg.search_diseases("breast cancer", limit=1)
            print(f"Found {len(diseases.get('diseases', []))} diseases")
            
            # Test Reactome
            print("🛤️ Testing Reactome...")
            pathways = await session.reactome.search_pathways("transcription", limit=1)
            print(f"Found {len(pathways.get('pathways', []))} pathways")
            
            # Test STRING
            print("🕸️ Testing STRING...")
            network = await session.string.get_interaction_network(
                genes=["AXL"],  # Use AXL instead of hardcoded examples
                species=9606, 
                required_score=400
            )
            nodes = network.get('network', {}).get('nodes', [])
            print(f"Network has {len(nodes)} nodes")
            
            # Test HPA
            print("🧪 Testing HPA...")
            expression = await session.hpa.get_tissue_expression("TP53")
            tissues = list(expression.get('expression', {}).keys())[:3]
            print(f"Expression data for {len(tissues)} tissues")
            
            print("🎉 All tests completed successfully!")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1
    
    return 0


async def run_health_check():
    """Run health check on all MCP servers."""
    print("🏥 Running MCP Server Health Check...")
    
    config_path = "config/mcp_servers.json"
    manager = MCPClientManager(config_path)
    
    try:
        async with manager.session() as session:
            health_status = await session.health_check()
            
            print("\n📊 Health Check Results:")
            for server, status in health_status.items():
                status_icon = "✅" if status else "❌"
                print(f"  {server}: {status_icon}")
            
            all_healthy = all(health_status.values())
            if all_healthy:
                print("\n🎉 All servers are healthy!")
                return 0
            else:
                print("\n⚠️ Some servers are not responding")
                return 1
                
    except Exception as e:
        print(f"❌ Health check failed: {e}")
        return 1


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="OmniTarget Pipeline - Bioinformatics Workflow Integration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive wizard mode
  python -m src.cli interactive
  
  # YAML batch execution
  python -m src.cli yaml examples/yaml_configs/axl_breast_cancer.yaml
  
  # Generate visualizations
  python -m src.cli visualize results/analysis.json
  python -m src.cli visualize results/analysis.json --interactive --format all
  
  # Existing commands
  python -m src.cli example       Run basic usage example
  python -m src.cli health        Check MCP server health
  
  # Help
  python -m src.cli --help        Show this help message
        """
    )
    
    parser.add_argument(
        "command",
        choices=["example", "health", "interactive", "yaml", "visualize"],
        help="Command to execute"
    )
    
    parser.add_argument(
        "file_path",
        nargs="?",
        help="Path to file (YAML for 'yaml' command, JSON for 'visualize' command)"
    )
    
    parser.add_argument(
        "--config",
        default="config/mcp_servers.json",
        help="Path to MCP servers configuration file"
    )
    
    parser.add_argument(
        "--output-dir", "-o",
        default="results/figures",
        help="Output directory for visualizations (visualize command)"
    )
    
    parser.add_argument(
        "--scenario", "-s",
        type=int,
        choices=[1,2,3,4,5,6],
        help="Specific scenario ID (visualize command)"
    )
    
    parser.add_argument(
        "--interactive", "-i",
        action="store_true",
        help="Generate interactive HTML versions (visualize command)"
    )
    
    parser.add_argument(
        "--format", "-f",
        default="png",
        help="Output formats: png, pdf, svg, all (visualize command)"
    )
    
    parser.add_argument(
        "--style",
        default="publication",
        choices=['publication', 'presentation', 'notebook'],
        help="Visualization style preset (visualize command)"
    )
    
    parser.add_argument(
        "--auto-visualize",
        action="store_true",
        help="After YAML run, automatically generate visualizations from the output JSON (yaml command)"
    )

    parser.add_argument(
        "--viz-output-dir",
        default="results/figures",
        help="Visualization output directory for auto visualize (yaml command)"
    )

    parser.add_argument(
        "--viz-formats",
        default="png",
        help="Visualization formats for auto visualize: png, pdf, svg, all (yaml command)"
    )

    parser.add_argument(
        "--viz-style",
        default="publication",
        choices=['publication', 'presentation', 'notebook'],
        help="Visualization style for auto visualize (yaml command)"
    )

    parser.add_argument(
        "--viz-scenario",
        type=int,
        choices=[1, 2, 3, 4, 5, 6],
        help="Specific scenario ID to visualize automatically (yaml command)"
    )

    parser.add_argument(
        "--viz-interactive",
        action="store_true",
        help="Generate interactive HTML outputs during auto visualize (yaml command)"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output"
    )
    
    args = parser.parse_args()

    # Configure logging based on verbosity
    # Default: INFO level with clean format for user-friendly output
    # Verbose: DEBUG level with full details (timestamps, logger names)
    if args.verbose:
        log_level = logging.DEBUG
        log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    else:
        log_level = logging.INFO
        log_format = '%(levelname)s - %(message)s'  # Simpler for standard runs

    # Create handler with colored formatter
    handler = logging.StreamHandler()
    handler.setFormatter(ColoredFormatter(log_format))

    # Configure root logger
    logging.basicConfig(
        level=log_level,
        handlers=[handler]
    )

    # Get logger for this module
    logger = logging.getLogger(__name__)
    logger.info(f"Logging configured: level={logging.getLevelName(log_level)}")
    
    # Run the appropriate command
    try:
        if args.command == "example":
            return asyncio.run(run_basic_example())
        
        elif args.command == "health":
            return asyncio.run(run_health_check())
        
        elif args.command == "interactive":
            # Lazy import to avoid requiring questionary for non-interactive commands
            from .interactive import interactive_mode
            return asyncio.run(interactive_mode(args.config))
        
        elif args.command == "yaml":
            if not args.file_path:
                print("❌ Error: YAML file path is required for 'yaml' command")
                print("Usage: python -m src.cli yaml <yaml_file>")
                return 1

            # Lazy import to avoid requiring pyyaml for non-YAML commands
            from .yaml_runner import YAMLRunner
            runner = YAMLRunner(args.config)
            viz_options = {
                'output_dir': args.viz_output_dir,
                'formats': args.viz_formats,
                'style': args.viz_style,
                'scenario': args.viz_scenario,
                'interactive': args.viz_interactive,
            } if args.auto_visualize else None

            return asyncio.run(
                runner.run(
                    args.file_path,
                    visualize=args.auto_visualize,
                    visualize_options=viz_options,
                )
            )
        
        elif args.command == "visualize":
            if not args.file_path:
                # Run interactive mode (lazy import)
                from .visualize import interactive_visualize
                interactive_visualize()
                return 0
            
            # Run with arguments (lazy import)
            from .visualize import visualize_command
            visualize_command(
                results_file=args.file_path,
                output_dir=args.output_dir,
                scenario=args.scenario,
                interactive=args.interactive,
                format=args.format,
                style=args.style
            )
            return 0
        
        else:
            parser.print_help()
            return 1
            
    except KeyboardInterrupt:
        print("\n\n⚠️  Operation cancelled by user")
        return 1
    except Exception as e:
        print(f"\n❌ Error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
