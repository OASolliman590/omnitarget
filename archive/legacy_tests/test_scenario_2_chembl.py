#!/usr/bin/env python3
"""
Test Scenario 2 ChEMBL Enhancement
Validates bioactivity-based druggability assessment with ChEMBL integration.
"""
import asyncio
import sys
import logging
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.core.mcp_client_manager import MCPClientManager
from src.scenarios.scenario_2_target_analysis import TargetAnalysisScenario

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def test_scenario_2_chembl():
    """Test Scenario 2 with ChEMBL bioactivity enhancement."""

    print("=" * 80)
    print("Scenario 2 ChEMBL Enhancement Test")
    print("=" * 80)
    print("\nTest: Target Analysis with Bioactivity-Based Druggability")
    print("Expected: +0.2 druggability score improvement with ChEMBL")
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
            print("   ⚠️  ChEMBL not configured - test will use KEGG-only scoring")
            print("   To enable ChEMBL, update config/mcp_servers.json")

        # Start servers
        print("\n2. Starting MCP servers...")
        async with manager.session():
            print("   ✅ All servers started")

            # Initialize scenario
            print("\n3. Initializing Scenario 2...")
            scenario = TargetAnalysisScenario(manager)
            print("   ✅ Scenario initialized")

            # Execute target analysis
            print("\n4. Executing target analysis for EGFR...")
            print("   (This will take several minutes...)")

            result = await scenario.execute(
                target_query="EGFR",
                tissue_context="lung"
            )

            # Analyze results
            print("\n" + "=" * 80)
            print("RESULTS")
            print("=" * 80)

            # Target info
            print(f"\n✅ Target: {result.target.gene_symbol}")
            print(f"   UniProt: {result.target.uniprot_id}")
            if result.target.description:
                print(f"   Description: {result.target.description}")

            # Pathways
            num_pathways = len(result.pathways)
            print(f"\n✅ Pathways: {num_pathways}")
            if num_pathways > 0:
                print(f"   Top pathways:")
                for i, pathway in enumerate(result.pathways[:3], 1):
                    print(f"   {i}. {pathway.id}: {pathway.name}")

            # Druggability score (KEY METRIC FOR ChEMBL)
            druggability_score = result.druggability_score
            print(f"\n✅ Druggability Score: {druggability_score:.3f}")

            # Test assertions
            print("\n" + "=" * 80)
            print("VALIDATION")
            print("=" * 80)

            success = True

            # Check 1: Target found
            if result.target and result.target.gene_symbol == "EGFR":
                print("✅ PASS: Target EGFR identified")
            else:
                print("❌ FAIL: Target not found")
                success = False

            # Check 2: Pathways found
            if num_pathways > 0:
                print(f"✅ PASS: Found {num_pathways} pathways")
            else:
                print("⚠️  WARNING: No pathways found")

            # Check 3: Interactors found
            if num_interactors >= 5:
                print(f"✅ PASS: Found {num_interactors} interactors (expected: >=5)")
            else:
                print(f"⚠️  WARNING: Found {num_interactors} interactors (expected: >=5)")

            # Check 4: Network metrics available
            if result.network_metrics and result.network_metrics.get('num_nodes', 0) > 0:
                print("✅ PASS: Network metrics calculated")
            else:
                print("⚠️  WARNING: Network metrics incomplete")

            # Check 5: Druggability score reasonable
            if 0.0 <= druggability_score <= 1.0:
                print(f"✅ PASS: Druggability score in valid range: {druggability_score:.3f}")
            else:
                print(f"❌ FAIL: Druggability score out of range: {druggability_score}")
                success = False

            # Check 6: High druggability for EGFR (well-known drug target)
            expected_min_score = 0.65 if chembl_configured else 0.45
            if druggability_score >= expected_min_score:
                print(f"✅ PASS: EGFR druggability score >= {expected_min_score:.2f} (found: {druggability_score:.3f})")
            else:
                print(f"⚠️  WARNING: EGFR druggability score < {expected_min_score:.2f} (found: {druggability_score:.3f})")
                print(f"   EGFR is a well-known drug target - score should be high")
                if chembl_configured:
                    success = False

            # ChEMBL-specific checks
            if chembl_configured:
                print("\n📊 ChEMBL Integration Checks:")

                # Check if bioactivity data was retrieved
                if hasattr(result, 'chembl_bioactivity') and result.chembl_bioactivity:
                    print(f"✅ PASS: ChEMBL bioactivity data retrieved")

                    bioactivity = result.chembl_bioactivity

                    # Check activity count
                    if bioactivity.activity_count > 0:
                        print(f"✅ PASS: {bioactivity.activity_count} bioactivities found")
                    else:
                        print(f"⚠️  WARNING: No bioactivities found for EGFR")

                    # Check IC50 data
                    if bioactivity.median_ic50:
                        print(f"✅ PASS: Median IC50 available: {bioactivity.median_ic50:.1f} nM")

                        # EGFR has potent inhibitors - IC50 should be < 100 nM
                        if bioactivity.median_ic50 < 100:
                            print(f"✅ PASS: Potent inhibitors found (IC50 < 100 nM)")
                        else:
                            print(f"⚠️  INFO: Median IC50 = {bioactivity.median_ic50:.1f} nM")

                    # Check bioactivity druggability score
                    if bioactivity.druggability_score >= 0.7:
                        print(f"✅ PASS: High bioactivity-based druggability: {bioactivity.druggability_score:.3f}")
                    else:
                        print(f"⚠️  INFO: Bioactivity druggability: {bioactivity.druggability_score:.3f}")

                else:
                    print("⚠️  WARNING: No ChEMBL bioactivity data in result")
                    print("   This may indicate ChEMBL integration issue")

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
    success = asyncio.run(test_scenario_2_chembl())
    sys.exit(0 if success else 1)
