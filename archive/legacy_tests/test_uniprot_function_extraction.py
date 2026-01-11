#!/usr/bin/env python3
"""
Test Circuit: UniProt Function Extraction

Tests UniProt get_protein_info to capture actual response structure
and identify where function data is located.
"""

import asyncio
import json
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.core.mcp_client_manager import MCPClientManager


async def test_uniprot_function_extraction():
    """Test UniProt get_protein_info response structure for function data."""
    print("=" * 80)
    print("Test Circuit: UniProt Function Extraction")
    print("=" * 80)
    print()
    
    # Create MCP manager
    config_path = str(Path(__file__).parent / "config" / "mcp_servers.json")
    manager = MCPClientManager(config_path)
    
    # Test protein: AXL (P30530) - known to be in the pipeline
    test_proteins = [
        {"accession": "P30530", "gene": "AXL"},
        {"accession": "P04637", "gene": "TP53"},
    ]
    
    try:
        # Start all servers
        print("🚀 Starting MCP servers...")
        await manager.start_all()
        print("✅ MCP servers started")
        print()
        
        for protein in test_proteins:
            accession = protein["accession"]
            gene = protein["gene"]
            print(f"Testing protein: {gene} ({accession})")
            print("-" * 80)
            
            try:
                # Call get_protein_info
                result = await manager.uniprot.get_protein_info(accession)
                
                print(f"✅ Response received")
                print(f"   Response type: {type(result).__name__}")
                
                # Save full response to file
                output_file = Path(__file__).parent / f"uniprot_response_{accession}.json"
                with open(output_file, 'w') as f:
                    json.dump(result, f, indent=2, default=str)
                print(f"   Full response saved to: {output_file}")
                print()
                
                # Detailed structure analysis
                if isinstance(result, dict):
                    print("   Response keys:")
                    for key in list(result.keys())[:20]:  # First 20 keys
                        print(f"     - {key}")
                    if len(result.keys()) > 20:
                        print(f"     ... and {len(result.keys()) - 20} more keys")
                    print()
                    
                    # Look for function-related keys
                    print("   Function-related keys:")
                    function_keys = [k for k in result.keys() if 'function' in k.lower() or 
                                   'description' in k.lower() or 'name' in k.lower() or
                                   'comment' in k.lower()]
                    for key in function_keys:
                        value = result[key]
                        value_type = type(value).__name__
                        if isinstance(value, str):
                            preview = value[:100] + "..." if len(value) > 100 else value
                            print(f"     '{key}': {value_type} = {preview}")
                        elif isinstance(value, list):
                            print(f"     '{key}': {value_type} (length: {len(value)})")
                            if len(value) > 0:
                                first_item = value[0]
                                if isinstance(first_item, str):
                                    print(f"       First item: {first_item[:100]}")
                                elif isinstance(first_item, dict):
                                    print(f"       First item keys: {list(first_item.keys())[:5]}")
                        elif isinstance(value, dict):
                            print(f"     '{key}': {value_type} with keys: {list(value.keys())[:5]}")
                    print()
                    
                    # Test current extraction logic
                    print("   Testing current extraction logic:")
                    function = None
                    
                    # Current code logic
                    func_list = result.get('function', [])
                    if isinstance(func_list, list) and func_list:
                        function = ' '.join(str(f) for f in func_list if f)
                        print(f"     Found via 'function' key (list): {len(func_list)} items")
                    elif isinstance(func_list, str):
                        function = func_list
                        print(f"     Found via 'function' key (string): {function[:100]}")
                    
                    if not function:
                        # Fallback logic
                        function = (result.get('protein_name') or 
                                  result.get('description') or
                                  result.get('recommendedName', {}).get('fullName', {}).get('value') 
                                  if isinstance(result.get('recommendedName'), dict) else None)
                        if function:
                            print(f"     Found via fallback: {function[:100]}")
                    
                    if function:
                        print(f"   ✅ Function extracted: {function[:150]}...")
                    else:
                        print(f"   ⚠️  No function extracted with current logic")
                        print(f"   Available keys that might contain function:")
                        for key in result.keys():
                            if any(term in key.lower() for term in ['function', 'description', 'name', 'comment', 'summary']):
                                print(f"     - {key}")
                
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
    success = asyncio.run(test_uniprot_function_extraction())
    sys.exit(0 if success else 1)




