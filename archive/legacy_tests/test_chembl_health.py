#!/usr/bin/env python3
"""
Test ChEMBL MCP server connectivity and basic functionality
"""
import asyncio
import logging
from src.core.mcp_client_manager import MCPClientManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def test_chembl_connectivity():
    """Test ChEMBL MCP server connectivity"""

    print("=" * 80)
    print("CHEMBL MCP SERVER CONNECTIVITY TEST")
    print("=" * 80)

    try:
        # Initialize MCP manager
        manager = MCPClientManager('config/mcp_servers.json')

        async with manager.session() as session:
            # Test 1: Check if ChEMBL client is available
            print(f"\n✅ ChEMBL client initialized")

            # Test 2: Search for EGFR target
            print(f"\n🔍 Test 1: Search targets for 'EGFR'")
            try:
                result = await session.chembl.search_targets("EGFR", limit=5)
                targets = result.get('targets', [])
                print(f"   Found {len(targets)} targets")
                if targets:
                    print(f"   First target: {targets[0]}")
                else:
                    print(f"   ⚠️ No targets found!")
            except Exception as e:
                print(f"   ❌ Error: {e}")

            # Test 3: Search for compounds
            print(f"\n🔍 Test 2: Search compounds for 'imatinib'")
            try:
                result = await session.chembl.search_compounds("imatinib", limit=5)
                compounds = result.get('compounds', [])
                print(f"   Found {len(compounds)} compounds")
                if compounds:
                    print(f"   First compound: {compounds[0]}")
                else:
                    print(f"   ⚠️ No compounds found!")
            except Exception as e:
                print(f"   ❌ Error: {e}")

            # Test 4: Search activities
            print(f"\n🔍 Test 3: Search activities for EGFR")
            try:
                # First get EGFR target ID
                target_result = await session.chembl.search_targets("EGFR", limit=1)
                targets = target_result.get('targets', [])

                if targets:
                    target_id = targets[0].get('target_chembl_id')
                    print(f"   Using target ID: {target_id}")

                    activity_result = await session.chembl.search_activities(
                        target_chembl_id=target_id,
                        limit=10
                    )
                    activities = activity_result.get('activities', [])
                    print(f"   Found {len(activities)} activities")
                    if activities:
                        print(f"   First activity: {activities[0]}")
                else:
                    print(f"   ⚠️ No targets found for EGFR")
            except Exception as e:
                print(f"   ❌ Error: {e}")
                import traceback
                traceback.print_exc()

            # Test 5: Search drugs
            print(f"\n🔍 Test 4: Search drugs for 'cancer'")
            try:
                result = await session.chembl.search_drugs("cancer", limit=5)
                drugs = result.get('drugs', [])
                print(f"   Found {len(drugs)} drugs")
                if drugs:
                    print(f"   First drug: {drugs[0]}")
                else:
                    print(f"   ⚠️ No drugs found!")
            except Exception as e:
                print(f"   ❌ Error: {e}")

    except Exception as e:
        print(f"\n❌ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

    print(f"\n{'=' * 80}")
    print("TEST COMPLETE")
    print(f"{'=' * 80}")
    return True


if __name__ == "__main__":
    success = asyncio.run(test_chembl_connectivity())
    exit(0 if success else 1)
