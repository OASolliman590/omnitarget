#!/usr/bin/env python3
"""
Test Circuit: KEGG links[gene_id] Structure

Tests KEGG find_related_entries to understand links dict structure
and identify why some genes return 0 pathways.
"""

import asyncio
import json
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.core.mcp_client_manager import MCPClientManager


async def test_kegg_links_structure():
    """Test KEGG find_related_entries links dict structure."""
    print("=" * 80)
    print("Test Circuit: KEGG links[gene_id] Structure")
    print("=" * 80)
    print()
    
    # Create MCP manager
    config_path = str(Path(__file__).parent / "config" / "mcp_servers.json")
    manager = MCPClientManager(config_path)
    
    # Test genes from user's logs - problematic and working ones
    test_genes = [
        {"id": "hsa:91464", "name": "AXL", "expected": "problematic"},
        {"id": "hsa:84839", "name": "TYRO3", "expected": "problematic"},
        {"id": "hsa:2621", "name": "GAS6", "expected": "working"},
        {"id": "hsa:558", "name": "APEX1", "expected": "working"},
    ]
    
    try:
        # Start all servers
        print("🚀 Starting MCP servers...")
        await manager.start_all()
        print("✅ MCP servers started")
        print()
        
        for gene_info in test_genes:
            gene_id = gene_info["id"]
            gene_name = gene_info["name"]
            expected = gene_info["expected"]
            print(f"Testing gene: {gene_name} ({gene_id}) - {expected}")
            print("-" * 80)
            
            try:
                # Call find_related_entries
                result = await manager.kegg.find_related_entries(
                    source_entries=[gene_id],
                    source_db="gene",
                    target_db="pathway"
                )
                
                print(f"✅ Response received")
                print(f"   Response type: {type(result).__name__}")
                
                # Save full response to file
                output_file = Path(__file__).parent / f"kegg_links_{gene_id.replace(':', '_')}.json"
                with open(output_file, 'w') as f:
                    json.dump(result, f, indent=2, default=str)
                print(f"   Full response saved to: {output_file}")
                print()
                
                # Detailed links dict analysis
                if isinstance(result, dict):
                    print("   Response structure:")
                    print(f"     Keys: {list(result.keys())}")
                    print()
                    
                    # Check link_count
                    link_count = result.get('link_count', 'N/A')
                    print(f"   link_count: {link_count}")
                    
                    # Analyze links dict
                    if 'links' in result:
                        links = result['links']
                        print(f"   links type: {type(links).__name__}")
                        
                        if isinstance(links, dict):
                            print(f"   links dict keys: {list(links.keys())[:10]}")
                            if len(links.keys()) > 10:
                                print(f"   ... and {len(links.keys()) - 10} more keys")
                            print()
                            
                            # Check if gene_id exists in links
                            print(f"   Checking for gene_id '{gene_id}' in links:")
                            if gene_id in links:
                                pathways = links[gene_id]
                                print(f"     ✅ Found! Pathways: {pathways}")
                            else:
                                print(f"     ❌ Not found!")
                                print(f"     Available keys in links:")
                                for key in list(links.keys())[:10]:
                                    print(f"       - '{key}' (type: {type(links[key]).__name__})")
                                
                                # Try variations
                                print()
                                print("     Trying key variations:")
                                variations = [
                                    gene_id,
                                    gene_id.replace(':', ''),
                                    gene_id.upper(),
                                    gene_id.lower(),
                                    gene_id.split(':')[1] if ':' in gene_id else None,
                                ]
                                for var in variations:
                                    if var and var in links:
                                        print(f"       ✅ Found with variation '{var}': {links[var]}")
                                        break
                                else:
                                    print(f"       ❌ No variation matched")
                            
                            # Show sample links structure
                            if len(links) > 0:
                                first_key = list(links.keys())[0]
                                first_value = links[first_key]
                                print()
                                print(f"   Sample link structure:")
                                print(f"     Key: '{first_key}'")
                                print(f"     Value type: {type(first_value).__name__}")
                                if isinstance(first_value, list):
                                    print(f"     Value length: {len(first_value)}")
                                    if len(first_value) > 0:
                                        print(f"     First pathway: {first_value[0]}")
                                elif isinstance(first_value, str):
                                    print(f"     Value: {first_value}")
                        else:
                            print(f"   ⚠️  links is not a dict: {type(links).__name__}")
                    
                    # Test extraction logic
                    print()
                    print("   Testing extraction logic:")
                    pathways = []
                    if isinstance(result, dict) and 'links' in result:
                        links = result['links']
                        if isinstance(links, dict):
                            gene_links = links.get(gene_id, [])
                            if isinstance(gene_links, list):
                                pathways = gene_links
                                print(f"     Direct lookup: {len(pathways)} pathways")
                            elif isinstance(gene_links, str):
                                pathways = [gene_links]
                                print(f"     Direct lookup (string): 1 pathway")
                    
                    if not pathways:
                        print(f"     ⚠️  No pathways found with direct lookup")
                        print(f"     link_count: {link_count}, but links['{gene_id}'] is empty")
                
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
    success = asyncio.run(test_kegg_links_structure())
    sys.exit(0 if success else 1)




