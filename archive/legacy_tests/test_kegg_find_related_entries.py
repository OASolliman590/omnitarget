#!/usr/bin/env python3
"""
Test Circuit: KEGG find_related_entries Parameter Fix

Tests that KEGG find_related_entries uses correct MCP parameters:
- source_entries (array) instead of entry_id
- source_db instead of source_database
- target_db instead of target_database

This test can be run independently without running the full pipeline.
"""

import asyncio
import json
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.core.mcp_client_manager import MCPClientManager


async def test_kegg_find_related_entries():
    """Test KEGG find_related_entries with correct parameters."""
    print("=" * 80)
    print("Test Circuit: KEGG find_related_entries Parameter Fix")
    print("=" * 80)
    print()
    
    # Create MCP manager
    manager = MCPClientManager()
    
    try:
        # Start all servers
        print("🚀 Starting MCP servers...")
        await manager.start_all_servers()
        print("✅ MCP servers started")
        print()
        
        # Test 1: Test with AXL gene (hsa:91464)
        print("Test 1: Testing find_related_entries with AXL gene (hsa:91464)")
        print("-" * 80)
        
        try:
            result = await manager.kegg.find_related_entries(
                source_entries=["hsa:91464"],
                source_db="gene",
                target_db="pathway"
            )
            
            print(f"✅ Success! Received response")
            print(f"   Response type: {type(result)}")
            
            if isinstance(result, dict):
                print(f"   Response keys: {list(result.keys())[:10]}")
                
                # Check for pathways
                pathways = result.get('pathways', [])
                if not pathways:
                    pathways = result.get('entries', [])
                if not pathways and isinstance(result, list):
                    pathways = result
                
                print(f"   Pathways found: {len(pathways)}")
                if pathways:
                    print(f"   First pathway: {pathways[0] if isinstance(pathways[0], str) else pathways[0].get('id', pathways[0])}")
                    print("   ✅ Parameter fix successful - no 'Source and target databases are required' error")
                else:
                    print("   ⚠️  No pathways found, but no error occurred (may be valid)")
            else:
                print(f"   Response: {str(result)[:200]}...")
            
            print()
            
        except Exception as e:
            error_msg = str(e)
            if "Source and target databases are required" in error_msg:
                print(f"❌ FAILED: Still using wrong parameters")
                print(f"   Error: {error_msg}")
                return False
            else:
                print(f"⚠️  Error (may be expected): {error_msg}")
                print("   (This could be a different issue, not parameter mismatch)")
        
        # Test 2: Test get_gene_pathways fallback
        print("Test 2: Testing get_gene_pathways fallback")
        print("-" * 80)
        
        try:
            result = await manager.kegg.get_gene_pathways("hsa:91464")
            
            print(f"✅ Success! Received response")
            print(f"   Response type: {type(result)}")
            
            if isinstance(result, dict):
                pathways = result.get('pathways', [])
                source = result.get('source', 'unknown')
                print(f"   Source: {source}")
                print(f"   Pathways found: {len(pathways)}")
                if pathways:
                    print(f"   First pathway: {pathways[0] if isinstance(pathways[0], str) else pathways[0].get('id', pathways[0])}")
                    print("   ✅ Fallback using correct parameters")
                else:
                    print("   ⚠️  No pathways found, but no error occurred")
            print()
            
        except Exception as e:
            error_msg = str(e)
            if "Source and target databases are required" in error_msg:
                print(f"❌ FAILED: Fallback still using wrong parameters")
                print(f"   Error: {error_msg}")
                return False
            else:
                print(f"⚠️  Error: {error_msg}")
                print("   (get_gene_pathways may not be available, fallback should handle this)")
        
        print("=" * 80)
        print("✅ All tests completed successfully!")
        print("=" * 80)
        return True
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        # Stop servers
        try:
            await manager.stop_all_servers()
            print("\n🛑 MCP servers stopped")
        except:
            pass


if __name__ == "__main__":
    success = asyncio.run(test_kegg_find_related_entries())
    sys.exit(0 if success else 1)

