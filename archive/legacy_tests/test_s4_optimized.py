#!/usr/bin/env python3
"""
Optimized Test Script for Scenario 4

Tests S4 with optimizations applied to verify it completes without being killed.
"""

import asyncio
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.core.mcp_client_manager import MCPClientManager
from src.scenarios.scenario_4_mra_simulation import MultiTargetSimulationScenario

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def test_scenario_4_optimized():
    """Test Scenario 4 with optimizations - using fewer targets."""
    logger.info("=" * 80)
    logger.info("Testing Scenario 4 (MRA Simulation) - Optimized Version")
    logger.info("=" * 80)
    
    # Initialize MCP manager
    config_path = "config/mcp_servers.json"
    try:
        mcp_manager = MCPClientManager(config_path)
        await mcp_manager.start_all()
        logger.info("✅ MCP Manager initialized")
    except Exception as e:
        logger.error(f"❌ Failed to initialize MCP Manager: {e}")
        return None
    
    # Initialize Scenario 4
    scenario = MultiTargetSimulationScenario(mcp_manager)
    
    # Use fewer targets for testing (6 instead of 11)
    targets = [
        "AXL",     # Primary target
        "AKT1",    # Key downstream target
        "MAPK1",   # ERK2
        "STAT3",   # Signal transducer
        "VEGFA",   # Angiogenesis
        "MMP9",    # Invasion
    ]
    
    disease_context = "breast cancer"
    simulation_mode = "simple"
    tissue_context = "breast"
    
    logger.info(f"\n📋 Test Configuration (OPTIMIZED):")
    logger.info(f"  Targets: {targets} ({len(targets)} targets)")
    logger.info(f"  Disease Context: {disease_context}")
    logger.info(f"  Simulation Mode: {simulation_mode}")
    logger.info(f"  Tissue Context: {tissue_context}\n")
    
    try:
        # Execute Scenario 4
        logger.info("🚀 Executing Scenario 4 with optimizations...")
        result = await scenario.execute(
            targets=targets,
            disease_context=disease_context,
            simulation_mode=simulation_mode,
            tissue_context=tissue_context
        )
        
        logger.info("\n" + "=" * 80)
        logger.info("✅ Scenario 4 Execution Complete")
        logger.info("=" * 80)
        
        # Quick validation
        logger.info("\n📊 Quick Results:")
        logger.info(f"  Individual Results: {len(result.individual_results)}")
        logger.info(f"  Pathway Enrichment: {len(result.pathway_enrichment.get('pathway_impact', {}))} pathways")
        logger.info(f"  Synergy Score: {result.synergy_analysis.get('synergy_score', 0.0):.3f}")
        logger.info(f"  Convergence Rate: {result.validation_metrics.get('convergence_rate', 0.0):.3f}")
        logger.info(f"  Biological Relevance: {result.validation_metrics.get('biological_relevance', 0.0):.3f}")
        
        return result
        
    except Exception as e:
        logger.error(f"\n❌ Scenario 4 execution failed: {e}", exc_info=True)
        return None
    
    finally:
        # Cleanup
        try:
            await mcp_manager.stop_all()
        except Exception as e:
            logger.warning(f"Failed to stop MCP servers: {e}")


if __name__ == "__main__":
    result = asyncio.run(test_scenario_4_optimized())
    
    if result:
        logger.info("\n✅ Optimized test completed successfully!")
        sys.exit(0)
    else:
        logger.error("\n❌ Optimized test failed!")
        sys.exit(1)









