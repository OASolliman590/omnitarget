#!/usr/bin/env python3
"""
Test ChEMBL integration with MCP Client Manager.
Verifies that ChEMBL client can be initialized and used through the manager.
"""
import asyncio
import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
os.chdir(project_root)

from src.core.mcp_client_manager import MCPClientManager


async def test_chembl_integration():
    """Test ChEMBL integration with manager."""

    print("=" * 80)
    print("ChEMBL Integration Test")
    print("=" * 80)

    config_path = "config/mcp_servers.json"

    try:
        # Initialize manager
        print("\n1. Initializing MCP Client Manager...")
        manager = MCPClientManager(config_path)

        # Check if ChEMBL is configured
        if 'chembl' not in manager.clients:
            print("   ❌ ChEMBL client not configured")
            return False

        print("   ✅ ChEMBL client configured")

        # Start all servers
        print("\n2. Starting all MCP servers...")
        async with manager.session():
            print("   ✅ All servers started")

            # Test ChEMBL client access
            print("\n3. Accessing ChEMBL client...")
            chembl = manager.chembl

            if chembl is None:
                print("   ❌ ChEMBL client is None")
                return False

            print(f"   ✅ ChEMBL client: {chembl.server_name}")

            # Test simple compound search
            print("\n4. Testing search_compounds('aspirin')...")
            result = await chembl.search_compounds("aspirin", limit=3)

            if 'molecules' in result:
                molecules = result['molecules']
                print(f"   ✅ Found {len(molecules)} compound(s)")

                for mol in molecules:
                    chembl_id = mol.get('molecule_chembl_id', 'N/A')
                    name = mol.get('pref_name', 'N/A')
                    print(f"      - {chembl_id}: {name}")
            else:
                print(f"   ⚠️  Unexpected response format: {result}")
                return False

            # Test compound info
            print("\n5. Testing get_compound_info('CHEMBL25')...")
            info = await chembl.get_compound_info("CHEMBL25")

            if 'molecule_chembl_id' in info:
                print(f"   ✅ Compound info retrieved")
                print(f"      Name: {info.get('pref_name', 'N/A')}")
                props = info.get('molecule_properties', {})
                if props:
                    mw = props.get('molecular_weight', 'N/A')
                    print(f"      MW: {mw}")
            else:
                print(f"   ⚠️  Unexpected response format: {info}")

            # Test health check
            print("\n6. Testing health check...")
            health = await manager.health_check()

            if 'chembl' in health:
                if health['chembl']:
                    print("   ✅ ChEMBL health check passed")
                else:
                    print("   ❌ ChEMBL health check failed")
                    return False
            else:
                print("   ⚠️  ChEMBL not in health check results")

        print("\n" + "=" * 80)
        print("✅ ALL INTEGRATION TESTS PASSED!")
        print("=" * 80)
        print("\nChEMBL is successfully integrated with OmniTarget pipeline!")
        return True

    except Exception as e:
        print(f"\n❌ Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = asyncio.run(test_chembl_integration())
    sys.exit(0 if success else 1)
