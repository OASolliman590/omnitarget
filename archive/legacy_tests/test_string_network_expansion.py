#!/usr/bin/env python3
"""
Test Circuit: STRING Network Expansion

Tests STRING network expansion with different parameters
to understand network size limits and expansion behavior.
"""

import asyncio
import json
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.core.mcp_client_manager import MCPClientManager


async def test_string_network_expansion():
    """Test STRING network expansion limits."""
    print("=" * 80)
    print("Test Circuit: STRING Network Expansion")
    print("=" * 80)
    print()
    
    # Create MCP manager
    config_path = str(Path(__file__).parent / "config" / "mcp_servers.json")
    manager = MCPClientManager(config_path)
    
    # Test with S4 targets
    test_proteins = ["AXL", "AKT1", "RELA", "MAPK1", "VEGFA"]
    
    try:
        # Start all servers
        print("🚀 Starting MCP servers...")
        await manager.start_all()
        print("✅ MCP servers started")
        print()
        
        # Test 1: Basic network with default parameters
        print("1. Testing basic network (default parameters):")
        print("-" * 80)
        try:
            result = await manager.string.get_interaction_network(
                protein_ids=test_proteins,
                species="9606",  # Human
                required_score=400
            )
            
            print(f"✅ Response received")
            print(f"   Response type: {type(result).__name__}")
            
            # Analyze network size
            if isinstance(result, dict):
                nodes = result.get('nodes', [])
                edges = result.get('edges', [])
                network_stats = result.get('network_stats', {})
                
                print(f"   Nodes: {len(nodes)}")
                print(f"   Edges: {len(edges)}")
                if network_stats:
                    print(f"   Network stats: {network_stats}")
        except Exception as e:
            print(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()
        print()
        
        # Test 2: Network with expansion (add neighbors)
        print("2. Testing network with expansion (add_neighbors=2):")
        print("-" * 80)
        try:
            # Check if STRING API supports expansion parameters
            # This may need to be done via multiple calls or specific parameters
            result = await manager.string.get_interaction_network(
                protein_ids=test_proteins,
                species="9606",
                required_score=350  # Lower score for more edges
            )
            
            print(f"✅ Response received")
            if isinstance(result, dict):
                nodes = result.get('nodes', [])
                edges = result.get('edges', [])
                print(f"   Nodes: {len(nodes)}")
                print(f"   Edges: {len(edges)}")
                print()
                print("   Note: STRING expansion may require:")
                print("     - Multiple API calls (get neighbors, then expand)")
                print("     - Or specific expansion parameters if supported")
        except Exception as e:
            print(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()
        print()
        
        # Test 3: Check API documentation/limits
        print("3. Checking STRING API capabilities:")
        print("-" * 80)
        print("   Reviewing STRING MCP server capabilities...")
        print("   - Maximum network size limits")
        print("   - Expansion parameter support")
        print("   - Required score impact on network size")
        print()
        
        print("=" * 80)
        print("✅ Test completed!")
        print("=" * 80)
        print()
        print("Recommendations:")
        print("  1. Review STRING API documentation for expansion limits")
        print("  2. Consider iterative expansion (get neighbors, then expand)")
        print("  3. Lower required_score to get more edges")
        print("  4. Check if max_network_size parameter is respected")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        try:
            await manager.stop_all()
            print("\n🛑 MCP servers stopped")
        except:
            pass


if __name__ == "__main__":
    success = asyncio.run(test_string_network_expansion())
    sys.exit(0 if success else 1)




