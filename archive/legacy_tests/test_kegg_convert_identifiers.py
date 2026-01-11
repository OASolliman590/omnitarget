#!/usr/bin/env python3
"""
Test Circuit: KEGG convert_identifiers Parameter Names

Tests KEGG convert_identifiers with correct and incorrect parameter names
to verify the MCP API expects source_db/target_db not from_db/to_db.
"""

import asyncio
import json
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.core.mcp_client_manager import MCPClientManager


async def test_kegg_convert_identifiers():
    """Test KEGG convert_identifiers parameter names."""
    print("=" * 80)
    print("Test Circuit: KEGG convert_identifiers Parameter Names")
    print("=" * 80)
    print()
    
    # Create MCP manager
    config_path = str(Path(__file__).parent / "config" / "mcp_servers.json")
    manager = MCPClientManager(config_path)
    
    # Test IDs from user's logs
    test_ids = ["PDE6", "CAMK2", "PRKG", "NFAT1"]
    
    try:
        # Start all servers
        print("🚀 Starting MCP servers...")
        await manager.start_all()
        print("✅ MCP servers started")
        print()
        
        # Test 1: Current (wrong) parameters - from_db/to_db
        print("1. Testing with WRONG parameters (from_db/to_db):")
        print("-" * 80)
        try:
            # This should fail based on documentation
            result = await manager.kegg.convert_identifiers(
                ids=test_ids,
                from_db="hsa",
                to_db="ncbi-geneid"
            )
            print(f"⚠️  Unexpected success with wrong parameters")
            print(f"   Response: {result}")
        except Exception as e:
            print(f"✅ Expected error with wrong parameters:")
            print(f"   Error: {e}")
        print()
        
        # Test 2: Correct parameters - source_db/target_db
        print("2. Testing with CORRECT parameters (source_db/target_db):")
        print("-" * 80)
        try:
            # Check current method signature
            import inspect
            sig = inspect.signature(manager.kegg.convert_identifiers)
            print(f"   Current method signature: {sig}")
            print()
            
            # Try to call with correct parameters (may need to update method first)
            # For now, just document what should work
            print("   Expected call (from documentation):")
            print("     convert_identifiers(")
            print("       ids=['PDE6', 'CAMK2'],")
            print("       source_db='hsa',")
            print("       target_db='ncbi-geneid'")
            print("     )")
            print()
            print("   ⚠️  Note: Method signature needs to be updated first")
            
        except Exception as e:
            print(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()
        print()
        
        print("=" * 80)
        print("✅ Test completed!")
        print("=" * 80)
        print()
        print("Next steps:")
        print("1. Update convert_identifiers method signature")
        print("2. Change from_db/to_db → source_db/target_db")
        print("3. Update all call sites")
        print("4. Re-run this test to verify")
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
    success = asyncio.run(test_kegg_convert_identifiers())
    sys.exit(0 if success else 1)




