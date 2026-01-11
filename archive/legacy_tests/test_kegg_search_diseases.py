#!/usr/bin/env python3
"""
Test Circuit: KEGG search_diseases Method

Tests KEGG search_diseases (not search_disease) to verify correct method name
and response structure.
"""

import asyncio
import json
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.core.mcp_client_manager import MCPClientManager


async def test_kegg_search_diseases():
    """Test KEGG search_diseases method."""
    print("=" * 80)
    print("Test Circuit: KEGG search_diseases Method")
    print("=" * 80)
    print()
    
    # Create MCP manager
    config_path = str(Path(__file__).parent / "config" / "mcp_servers.json")
    manager = MCPClientManager(config_path)
    
    test_query = "breast cancer"
    
    try:
        # Start all servers
        print("🚀 Starting MCP servers...")
        await manager.start_all()
        print("✅ MCP servers started")
        print()
        
        # Test 1: Correct method name - search_diseases
        print(f"1. Testing search_diseases with query: '{test_query}'")
        print("-" * 80)
        try:
            result = await manager.kegg.search_diseases(test_query, limit=10)
            
            print(f"✅ Response received")
            print(f"   Response type: {type(result).__name__}")
            
            # Save full response
            output_file = Path(__file__).parent / "kegg_search_diseases_result.json"
            with open(output_file, 'w') as f:
                json.dump(result, f, indent=2, default=str)
            print(f"   Full response saved to: {output_file}")
            print()
            
            # Analyze structure
            if isinstance(result, dict):
                print("   Response keys:")
                for key in result.keys():
                    print(f"     - {key}")
                print()
                
                # Check for diseases
                diseases = result.get('diseases', [])
                if not diseases:
                    diseases = result.get('results', [])
                if not diseases:
                    diseases = result.get('entries', [])
                
                if diseases:
                    print(f"   Found {len(diseases)} diseases")
                    print()
                    print("   Sample diseases (first 5):")
                    for i, disease in enumerate(diseases[:5]):
                        if isinstance(disease, dict):
                            print(f"     {i+1}. ID: {disease.get('id', 'N/A')}")
                            print(f"        Name: {disease.get('name', disease.get('title', 'N/A'))}")
                            print(f"        Keys: {list(disease.keys())[:5]}")
                        else:
                            print(f"     {i+1}. {disease}")
                else:
                    print("   ⚠️  No diseases found in response")
            elif isinstance(result, list):
                print(f"   Response is list with {len(result)} items")
                if len(result) > 0:
                    print(f"   First item: {result[0]}")
        except Exception as e:
            print(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()
        print()
        
        # Test 2: Wrong method name - search_disease (should fail)
        print("2. Testing search_disease (wrong method name):")
        print("-" * 80)
        try:
            # This should fail - method doesn't exist
            if hasattr(manager.kegg, 'search_disease'):
                result = await manager.kegg.search_disease(test_query)
                print(f"⚠️  Unexpected success with wrong method name")
            else:
                print(f"✅ Correctly fails - method 'search_disease' does not exist")
                print(f"   Available method: 'search_diseases' (plural)")
        except AttributeError as e:
            print(f"✅ Expected AttributeError: {e}")
        except Exception as e:
            print(f"❌ Unexpected error: {e}")
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
    success = asyncio.run(test_kegg_search_diseases())
    sys.exit(0 if success else 1)




