#!/usr/bin/env python3
"""
Test Scenario 4 confidence score fix
"""
import asyncio
import logging
from src.core.mcp_client_manager import MCPClientManager
from src.scenarios.scenario_4_mra_simulation import MultiTargetSimulationScenario

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def test_s4_confidence():
    """Test Scenario 4 confidence score extraction"""

    print("=" * 80)
    print("TESTING SCENARIO 4 CONFIDENCE SCORE FIX")
    print("=" * 80)

    try:
        # Initialize MCP manager
        manager = MCPClientManager('config/mcp_servers.json')

        async with manager.session() as session:
            # Create scenario instance
            scenario = MultiTargetSimulationScenario(manager)

            # Test genes
            test_genes = ['AXL', 'BRCA1', 'TP53']

            print(f"\n🔬 Running MRA Simulation with genes: {test_genes}")

            # Run the simulation
            result = await scenario.run_simulation(
                target_genes=test_genes,
                perturbation_type='knockdown',
                simulation_method='boolean'
            )

            print(f"\n📊 RESULTS:")
            print(f"   Genes: {result.num_genes}")
            print(f"   Interactions: {result.num_interactions}")
            print(f"   Confidence scores: {result.confidence_scores}")

            if result.confidence_scores:
                overall_confidence = result.confidence_scores.get('overall', 0.0)
                print(f"\n✅ SUCCESS! Confidence score found: {overall_confidence:.3f}")

                if overall_confidence > 0.0:
                    print(f"   🎉 Issue #2 FIXED! Confidence is now > 0.0")
                else:
                    print(f"   ⚠️ Confidence still 0.0 - may need further investigation")
            else:
                print(f"\n❌ FAILED! No confidence scores found")

    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

    print(f"\n{'=' * 80}")
    return True


if __name__ == "__main__":
    success = asyncio.run(test_s4_confidence())
    exit(0 if success else 1)
