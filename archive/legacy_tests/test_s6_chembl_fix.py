#!/usr/bin/env python3
"""
Test ChEMBL fix in Scenario 6
"""
import asyncio
import logging
from src.core.mcp_client_manager import MCPClientManager
from src.scenarios.scenario_6_drug_repurposing import DrugRepurposingScenario

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def test_s6_chembl():
    """Test Scenario 6 ChEMBL drug discovery"""

    print("=" * 80)
    print("TESTING SCENARIO 6 CHEMBL DRUG DISCOVERY FIX")
    print("=" * 80)

    try:
        # Initialize MCP manager
        manager = MCPClientManager('config/mcp_servers.json')

        async with manager.session() as session:
            # Create scenario instance
            scenario = DrugRepurposingScenario(manager)

            # Test gene
            test_gene = 'EGFR'

            print(f"\n🔬 Testing ChEMBL drug discovery for: {test_gene}")

            # Call the ChEMBL method directly
            compounds = await scenario._get_drugs_for_gene_chembl(test_gene)

            print(f"\n📊 RESULTS:")
            print(f"   ChEMBL compounds found: {len(compounds)}")

            if compounds:
                print(f"\n✅ SUCCESS! ChEMBL is now working!")
                print(f"   Top 5 compounds:")
                for i, compound in enumerate(compounds[:5]):
                    print(f"      {i+1}. {compound.chembl_id} - IC50: {getattr(compound, 'ic50_nm', 'N/A')} nM")
            else:
                print(f"\n❌ FAILED! No ChEMBL compounds found")

    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

    print(f"\n{'=' * 80}")
    return True


if __name__ == "__main__":
    success = asyncio.run(test_s6_chembl())
    exit(0 if success else 1)
