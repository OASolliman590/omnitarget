#!/usr/bin/env python3
"""
Test Scenario 3 expression profile fix
"""
import asyncio
import json
import logging
from src.core.mcp_client_manager import MCPClientManager
from src.scenarios.scenario_3_cancer_analysis import CancerAnalysisScenario

# Configure logging to see debug messages
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def test_s3_expression():
    """Test Scenario 3 expression profile retrieval"""

    print("=" * 80)
    print("TESTING SCENARIO 3 EXPRESSION PROFILE FIX")
    print("=" * 80)

    try:
        # Initialize MCP manager
        manager = MCPClientManager('config/mcp_servers.json')

        async with manager.session() as session:
            # Create scenario instance
            scenario = CancerAnalysisScenario(manager)

            # Run Phase 4 (expression dysregulation) directly
            print(f"\n🔬 Running Phase 4: Expression Dysregulation")
            print(f"   Gene: AXL")
            print(f"   Tissue context: breast")

            # Get some seed genes from pathways
            try:
                reactome_search = await session.reactome.find_pathways_by_disease("breast cancer")
                seed_genes = set()

                if reactome_search.get('pathways'):
                    pathway_list = reactome_search['pathways'][:2]  # Just 2 pathways
                    for pathway_data in pathway_list:
                        pathway_id = pathway_data.get('stId') or pathway_data.get('id')
                        if pathway_id:
                            try:
                                participants = await session.reactome.get_pathway_participants(pathway_id)
                                if participants.get('participants'):
                                    for participant in participants['participants']:
                                        gene = (participant.get('gene_symbol') or
                                               participant.get('gene') or
                                               participant.get('displayName'))
                                        if gene and gene.strip() and len(gene) <= 10:
                                            seed_genes.add(gene.strip().upper())
                            except Exception as e:
                                logger.debug(f"Failed to get pathway genes: {e}")

                test_genes = list(seed_genes)[:5] if seed_genes else ['AXL', 'BRCA1']
                print(f"   Using genes: {test_genes}")

            except Exception as e:
                print(f"   ⚠️ Could not get seed genes: {e}")
                test_genes = ['AXL', 'BRCA1']

            # Run expression analysis
            expression_data = await scenario._phase4_expression_dysregulation(
                genes=test_genes,
                tissue_context='breast'
            )

            print(f"\n📊 RESULTS:")
            print(f"   Total genes processed: {len(test_genes)}")
            print(f"   Expression profiles found: {len(expression_data.get('profiles', []))}")
            print(f"   Dysregulation scores: {len(expression_data.get('concordance_scores', []))}")
            print(f"   Mean concordance: {expression_data.get('mean_concordance', 0.0):.3f}")

            if expression_data.get('profiles'):
                print(f"\n✅ SUCCESS! Found {len(expression_data['profiles'])} expression profiles")
                for i, profile in enumerate(expression_data['profiles'][:3]):
                    print(f"   Profile {i+1}: {profile.gene} in {profile.tissue} - {profile.expression_level}")
            else:
                print(f"\n❌ FAILED! No expression profiles found")

            if expression_data.get('concordance_scores'):
                print(f"\n📈 Dysregulation scores:")
                for score in expression_data['concordance_scores'][:3]:
                    print(f"   {score['gene']}: FC={score['fold_change']:.2f} ({score['dysregulation']})")

            # Save results
            output_file = f"/Users/omara.soliman/Desktop/Projects /My Projects/15-OmniTarget_/test_s3_results.json"
            with open(output_file, 'w') as f:
                json.dump({
                    'genes_processed': test_genes,
                    'expression_profiles': [
                        {
                            'gene': p.gene,
                            'tissue': p.tissue,
                            'expression_level': p.expression_level
                        }
                        for p in expression_data.get('profiles', [])
                    ],
                    'dysregulation_scores': expression_data.get('concordance_scores', []),
                    'mean_concordance': expression_data.get('mean_concordance', 0.0)
                }, f, indent=2, default=str)
            print(f"\n💾 Results saved to: {output_file}")

    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

    print(f"\n{'=' * 80}")
    return True


if __name__ == "__main__":
    success = asyncio.run(test_s3_expression())
    exit(0 if success else 1)
