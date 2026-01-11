#!/usr/bin/env python3
"""
Test Circuit: KEGG find_related_entries Response Structure

Tests KEGG find_related_entries to capture actual response structure
and validate against documentation expectations.
"""

import asyncio
import json
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.core.mcp_client_manager import MCPClientManager


async def test_kegg_response_structure():
    """Test KEGG find_related_entries response structure."""
    print("=" * 80)
    print("Test Circuit: KEGG find_related_entries Response Structure")
    print("=" * 80)
    print()
    
    # Create MCP manager
    config_path = str(Path(__file__).parent / "config" / "mcp_servers.json")
    manager = MCPClientManager(config_path)
    
    # Test genes from user's logs
    test_genes = [
        "hsa:91464",  # AXL
        "hsa:2621",   # GAS6
        "hsa:84839",  # TYRO3
        "hsa:558",    # APEX1
    ]
    
    try:
        # Start all servers
        print("🚀 Starting MCP servers...")
        await manager.start_all()
        print("✅ MCP servers started")
        print()
        
        for gene_id in test_genes:
            print(f"Testing gene: {gene_id}")
            print("-" * 80)
            
            try:
                # Call find_related_entries directly
                result = await manager.kegg.find_related_entries(
                    source_entries=[gene_id],
                    source_db="gene",
                    target_db="pathway"
                )
                
                print(f"✅ Response received")
                print(f"   Response type: {type(result).__name__}")
                
                # Detailed structure analysis
                if isinstance(result, dict):
                    print(f"   Response keys: {list(result.keys())}")
                    print()
                    print("   Key analysis:")
                    for key in result.keys():
                        value = result[key]
                        print(f"     '{key}': type={type(value).__name__}, "
                              f"is_list={isinstance(value, list)}, "
                              f"is_dict={isinstance(value, dict)}")
                        if isinstance(value, list):
                            print(f"       List length: {len(value)}")
                            if len(value) > 0:
                                print(f"       First item type: {type(value[0]).__name__}")
                                if isinstance(value[0], str):
                                    print(f"       First item: {value[0][:100]}")
                                elif isinstance(value[0], dict):
                                    print(f"       First item keys: {list(value[0].keys())[:5]}")
                        elif isinstance(value, dict):
                            print(f"       Dict keys: {list(value.keys())[:5]}")
                        elif isinstance(value, str):
                            print(f"       String value: {value[:100]}")
                
                elif isinstance(result, list):
                    print(f"   List length: {len(result)}")
                    if len(result) > 0:
                        print(f"   First item type: {type(result[0]).__name__}")
                        if isinstance(result[0], str):
                            print(f"   First item: {result[0][:100]}")
                        elif isinstance(result[0], dict):
                            print(f"   First item keys: {list(result[0].keys())[:5]}")
                
                # Test extraction logic
                print()
                print("   Testing extraction logic:")
                pathways = []
                if isinstance(result, dict):
                    pathways = (result.get('pathways') or 
                               result.get('entries') or 
                               result.get('related_entries') or
                               result.get('results') or
                               result.get('data') or
                               [])
                    if not pathways:
                        # Check all values for pathway-like content
                        for key, value in result.items():
                            if isinstance(value, list) and len(value) > 0:
                                first_item = value[0]
                                if isinstance(first_item, str):
                                    if ('path:' in first_item.lower() or 
                                        'hsa' in first_item.lower() or 
                                        first_item.startswith('hsa') or
                                        'map' in first_item.lower()):
                                        pathways = value
                                        print(f"     Found pathways in key '{key}'")
                                        break
                elif isinstance(result, list):
                    pathways = [item for item in result if isinstance(item, str) and 
                               ('path:' in item.lower() or 'hsa' in item.lower() or 'map' in item.lower())]
                
                print(f"   Extracted pathways: {len(pathways)}")
                if pathways:
                    print(f"   First pathway: {pathways[0]}")
                else:
                    print("   ⚠️  No pathways extracted")
                
                print()
                
            except Exception as e:
                print(f"❌ Error: {e}")
                import traceback
                traceback.print_exc()
                print()
        
        print("=" * 80)
        print("✅ Test completed!")
        print("=" * 80)
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
    success = asyncio.run(test_kegg_response_structure())
    sys.exit(0 if success else 1)

