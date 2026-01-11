#!/usr/bin/env python3
"""
Test Circuit: S2 Reactome Interactions

Tests Reactome pathway/interaction retrieval for AXL to verify
the fix for get_protein_interactions works correctly.
"""

import asyncio
import json
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.core.mcp_client_manager import MCPClientManager


async def test_s2_reactome_interactions():
    """Test Reactome pathway and interaction retrieval for AXL."""
    print("=" * 80)
    print("Test Circuit: S2 Reactome Interactions for AXL")
    print("=" * 80)
    print()
    
    # Create MCP manager
    config_path = str(Path(__file__).parent / "config" / "mcp_servers.json")
    manager = MCPClientManager(config_path)
    
    try:
        # Start all servers
        print("🚀 Starting MCP servers...")
        await manager.start_all()
        print("✅ MCP servers started")
        print()
        
        gene_symbol = "AXL"
        print(f"Testing Reactome interactions for gene: {gene_symbol}")
        print("-" * 80)
        
        # Step 1: Find pathways for AXL
        print("\n1. Finding pathways for AXL...")
        try:
            pathways_result = await manager.reactome.find_pathways_by_gene(gene_symbol)
            print(f"   ✅ Response type: {type(pathways_result)}")
            
            if isinstance(pathways_result, dict):
                pathways = pathways_result.get('pathways', [])
                print(f"   ✅ Found {len(pathways)} pathways")
                
                if pathways:
                    print(f"   📋 Pathway IDs:")
                    for i, pathway in enumerate(pathways[:5], 1):  # Show first 5
                        pathway_id = pathway.get('stId') or pathway.get('id') or 'Unknown'
                        pathway_name = pathway.get('displayName') or pathway.get('name') or 'Unknown'
                        print(f"      {i}. {pathway_id}: {pathway_name[:60]}")
                    
                    # Step 2: Get interactions for first pathway
                    if pathways:
                        first_pathway_id = pathways[0].get('stId') or pathways[0].get('id')
                        print(f"\n2. Getting protein interactions for pathway: {first_pathway_id}")
                        try:
                            interactions_result = await manager.reactome.get_protein_interactions(first_pathway_id)
                            print(f"   ✅ Response type: {type(interactions_result)}")
                            
                            if isinstance(interactions_result, dict):
                                interactions = interactions_result.get('interactions', [])
                                print(f"   ✅ Found {len(interactions)} interactions")
                                
                                if interactions:
                                    print(f"   📋 Sample interactions:")
                                    for i, interaction in enumerate(interactions[:3], 1):  # Show first 3
                                        print(f"      {i}. {interaction}")
                                    
                                    # Save to file
                                    with open("test_s2_interactions_response.json", "w") as f:
                                        json.dump(interactions_result, f, indent=2, default=str)
                                    print(f"   💾 Saved to: test_s2_interactions_response.json")
                                else:
                                    print("   ⚠️  No interactions found in response")
                            else:
                                print(f"   ⚠️  Unexpected response type: {type(interactions_result)}")
                        except Exception as e:
                            print(f"   ❌ Error getting interactions: {e}")
                            import traceback
                            traceback.print_exc()
                else:
                    print("   ⚠️  No pathways found - cannot test interactions")
            else:
                print(f"   ⚠️  Unexpected response type: {type(pathways_result)}")
                
        except Exception as e:
            print(f"   ❌ Error finding pathways: {e}")
            import traceback
            traceback.print_exc()
        
        # Summary
        print("\n" + "=" * 80)
        print("Summary")
        print("=" * 80)
        print(f"Gene: {gene_symbol}")
        if isinstance(pathways_result, dict) and pathways_result.get('pathways'):
            print(f"✅ Pathways found: {len(pathways_result['pathways'])}")
            if interactions_result and isinstance(interactions_result, dict):
                interactions = interactions_result.get('interactions', [])
                print(f"✅ Interactions found: {len(interactions)}")
            else:
                print("⚠️  Interactions: Not tested (no pathways or error)")
        else:
            print("⚠️  Pathways: Not found or error")
            print("   This is expected if Reactome is unavailable")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Stop all servers
        print("\n🛑 Stopping MCP servers...")
        await manager.stop_all()
        print("✅ MCP servers stopped")


if __name__ == "__main__":
    asyncio.run(test_s2_reactome_interactions())




