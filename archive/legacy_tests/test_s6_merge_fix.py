#!/usr/bin/env python3
"""
Test S6 drug merging with debug logging
"""
import asyncio
import logging
from src.core.mcp_client_manager import MCPClientManager
from src.scenarios.scenario_6_drug_repurposing import DrugRepurposingScenario

# Configure logging to see debug messages
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def test_s6_merge():
    """Test Scenario 6 drug merging"""

    print("=" * 80)
    print("TESTING SCENARIO 6 DRUG MERGING")
    print("=" * 80)

    try:
        # Initialize MCP manager
        manager = MCPClientManager('config/mcp_servers.json')

        async with manager.session() as session:
            # Create scenario instance
            scenario = DrugRepurposingScenario(manager)

            # Test genes - use AXL which we know has KEGG drugs from the analysis
            test_genes = ['AXL']  # From analysis: AXL has 39 KEGG drugs

            print(f"\n🔬 Testing drug discovery for: {test_genes}")

            # Run the step3_known_drugs method
            # First, get proteins for the genes
            from src.models.data_models import Protein

            proteins = []
            for gene in test_genes:
                protein = Protein(
                    gene_symbol=gene,
                    protein_id=f"test_{gene}",
                    protein_name=f"Test protein {gene}",
                    uniprot_id=f"TEST{gene}",
                    organism="Homo sapiens"
                )
                proteins.append(protein)

            # Run step 3 (known drugs)
            result = await scenario._step3_known_drugs(proteins)

            print(f"\n📊 RESULTS:")
            print(f"   Merged drugs: {len(result.get('merged_drugs', []))}")
            print(f"   Repurposing candidates: {len(result.get('repurposing_candidates', []))}")
            print(f"   KEGG count: {result.get('kegg_drug_count', 0)}")
            print(f"   ChEMBL count: {result.get('chembl_drug_count', 0)}")

            if result.get('merged_drugs'):
                print(f"\n✅ SUCCESS! Found {len(result['merged_drugs'])} merged drugs")
                for i, drug in enumerate(result['merged_drugs'][:5]):
                    print(f"   {i+1}. {drug.name} ({drug.drug_id})")
            else:
                print(f"\n❌ FAILED! No merged drugs found")

    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

    print(f"\n{'=' * 80}")
    return True


if __name__ == "__main__":
    success = asyncio.run(test_s6_merge())
    exit(0 if success else 1)
