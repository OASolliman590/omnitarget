"""
OmniTarget CLI

Command-line interface for the OmniTarget pipeline.
"""

import asyncio
import argparse
import sys
from pathlib import Path

from core.mcp_client_manager import MCPClientManager


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
  omnitarget example          Run basic usage example
  omnitarget health           Check MCP server health
  omnitarget --help           Show this help message
        """
    )
    
    parser.add_argument(
        "command",
        choices=["example", "health"],
        help="Command to execute"
    )
    
    parser.add_argument(
        "--config",
        default="config/mcp_servers.json",
        help="Path to MCP servers configuration file"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output"
    )
    
    args = parser.parse_args()
    
    # Set up logging if verbose
    if args.verbose:
        import logging
        logging.basicConfig(level=logging.INFO)
    
    # Run the appropriate command
    if args.command == "example":
        return asyncio.run(run_basic_example())
    elif args.command == "health":
        return asyncio.run(run_health_check())
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())
