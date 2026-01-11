#!/usr/bin/env python3
"""
Test Scenario 6 ChEMBL Enhancement
Validates multi-source drug discovery (KEGG + ChEMBL) integration.
"""
import asyncio
import sys
import logging
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.core.mcp_client_manager import MCPClientManager
from src.scenarios.scenario_6_drug_repurposing import DrugRepurposingScenario

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def test_scenario_6_chembl():
    """Test Scenario 6 with ChEMBL enhancement for breast cancer."""

    print("=" * 80)
    print("Scenario 6 ChEMBL Enhancement Test")
    print("=" * 80)
    print("\nTest: Drug Repurposing for Breast Cancer with Multi-Source Discovery")
    print("Expected: 50+ repurposing candidates from KEGG + ChEMBL")
    print("=" * 80)

    config_path = "config/mcp_servers.json"

    try:
        # Initialize manager
        print("\n1. Initializing MCP Client Manager...")
        manager = MCPClientManager(config_path)

        # Check if ChEMBL is configured
        chembl_configured = 'chembl' in manager.clients
        print(f"   ChEMBL configured: {chembl_configured}")

        if not chembl_configured:
            print("   ⚠️  ChEMBL not configured - test will only use KEGG")
            print("   To enable ChEMBL, update config/mcp_servers.json")

        # Start servers
        print("\n2. Starting MCP servers...")
        async with manager.session():
            print("   ✅ All servers started")

            # Initialize scenario
            print("\n3. Initializing Scenario 6...")
            scenario = DrugRepurposingScenario(manager)
            print("   ✅ Scenario initialized")

            # Execute drug repurposing
            print("\n4. Executing drug repurposing for breast cancer...")
            print("   (This will take several minutes...)")

            result = await scenario.execute(
                disease_query="breast cancer",
                tissue_context="breast",
                simulation_mode='simple'
            )

            # Analyze results
            print("\n" + "=" * 80)
            print("RESULTS")
            print("=" * 80)

            # Disease pathways
            num_pathways = len(result.disease_pathways)
            print(f"\n✅ Disease pathways: {num_pathways}")

            # Drug candidates
            num_candidates = len(result.candidate_drugs)
            print(f"✅ Drug candidates: {num_candidates}")

            # Repurposing candidates
            num_repurposing = len(result.repurposing_scores)
            print(f"✅ Repurposing candidates: {num_repurposing}")

            # Show top repurposing candidates
            if result.repurposing_scores:
                print("\n📊 Top 10 Repurposing Candidates:")
                sorted_candidates = sorted(
                    result.repurposing_scores.items(),
                    key=lambda x: x[1],
                    reverse=True
                )[:10]

                for i, (drug_id, score) in enumerate(sorted_candidates, 1):
                    # Find drug info
                    drug_info = next(
                        (d for d in result.candidate_drugs if d.get('drug_id') == drug_id),
                        None
                    )
                    drug_name = drug_info.get('name', drug_id) if drug_info else drug_id
                    print(f"   {i:2d}. {drug_id:15s} ({drug_name:30s}) - Score: {score:.3f}")

            # Network validation
            print(f"\n✅ Network validation: {result.network_validation}")

            # Off-target analysis
            off_target_count = len(result.off_target_analysis.get('off_target_analysis', []))
            print(f"✅ Off-target analysis: {off_target_count} drug-target pairs analyzed")

            # Expression validation
            expression_coverage = result.expression_validation.get('coverage', 0.0)
            print(f"✅ Expression validation coverage: {expression_coverage:.1%}")

            # Test assertions
            print("\n" + "=" * 80)
            print("VALIDATION")
            print("=" * 80)

            success = True

            # Check 1: Pathways found
            if num_pathways > 0:
                print("✅ PASS: Disease pathways identified")
            else:
                print("❌ FAIL: No disease pathways found")
                success = False

            # Check 2: Drug candidates found
            expected_min_candidates = 50 if chembl_configured else 5
            if num_candidates >= expected_min_candidates:
                print(f"✅ PASS: Found {num_candidates} drug candidates (expected: >={expected_min_candidates})")
            else:
                print(f"⚠️  WARNING: Found {num_candidates} drug candidates (expected: >={expected_min_candidates})")
                if chembl_configured:
                    success = False

            # Check 3: Repurposing candidates with scores
            if num_repurposing > 0:
                print(f"✅ PASS: Generated {num_repurposing} repurposing scores")
            else:
                print("❌ FAIL: No repurposing candidates scored")
                success = False

            # Check 4: Network validation performed
            if result.network_validation:
                print("✅ PASS: Network validation completed")
            else:
                print("⚠️  WARNING: Network validation missing")

            # Check 5: Expression validation performed
            if expression_coverage > 0:
                print(f"✅ PASS: Expression validation completed ({expression_coverage:.1%} coverage)")
            else:
                print("⚠️  WARNING: Expression validation incomplete")

            # ChEMBL-specific checks
            if chembl_configured:
                print("\n📊 ChEMBL Integration Checks:")

                # Check if any candidates have drug-likeness scores (ChEMBL feature)
                drugs_with_likeness = sum(
                    1 for d in result.candidate_drugs
                    if isinstance(d, dict) and d.get('drug_likeness_score') is not None
                )

                if drugs_with_likeness > 0:
                    print(f"✅ PASS: {drugs_with_likeness} drugs have drug-likeness scores (ChEMBL data)")
                else:
                    print("⚠️  INFO: No drugs with drug-likeness scores (may be KEGG-only)")

            # Final result
            print("\n" + "=" * 80)
            if success:
                print("✅ ALL TESTS PASSED!")
            else:
                print("⚠️  SOME TESTS FAILED OR WARNINGS PRESENT")

            print("=" * 80)

            return success

    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = asyncio.run(test_scenario_6_chembl())
    sys.exit(0 if success else 1)
