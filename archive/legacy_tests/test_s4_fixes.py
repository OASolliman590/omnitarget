#!/usr/bin/env python3
"""
Test Script for Scenario 4 (S4) Fixes

Tests the fixed S4 implementation to verify:
1. Network propagation is working
2. Synergy scores are calculated correctly
3. Pathway enrichment is populated
4. Convergence detection works
5. Biological relevance is improved
"""

import asyncio
import json
import logging
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.core.mcp_client_manager import MCPClientManager
from src.scenarios.scenario_4_mra_simulation import MultiTargetSimulationScenario

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def test_scenario_4():
    """Test Scenario 4 with fixes applied."""
    logger.info("=" * 80)
    logger.info("Testing Scenario 4 (MRA Simulation) with Fixes")
    logger.info("=" * 80)
    
    # Initialize MCP manager
    config_path = "config/mcp_servers.json"
    try:
        mcp_manager = MCPClientManager(config_path)
        await mcp_manager.start_all()
        logger.info("✅ MCP Manager initialized and started")
    except Exception as e:
        logger.error(f"❌ Failed to initialize MCP Manager: {e}")
        return None
    
    # Initialize Scenario 4
    scenario = MultiTargetSimulationScenario(mcp_manager)
    
    # Test parameters - same as comprehensive_axl_analysis.yaml
    targets = [
        "AXL",     # Primary target
        "AKT1",    # Key downstream target
        "MAPK1",   # ERK2
        "STAT3",   # Signal transducer
    ]
    
    disease_context = "breast cancer"
    simulation_mode = "simple"
    tissue_context = "breast"
    
    logger.info(f"\n📋 Test Configuration:")
    logger.info(f"  Targets: {targets}")
    logger.info(f"  Disease Context: {disease_context}")
    logger.info(f"  Simulation Mode: {simulation_mode}")
    logger.info(f"  Tissue Context: {tissue_context}\n")
    
    try:
        # Execute Scenario 4
        logger.info("🚀 Executing Scenario 4...")
        result = await scenario.execute(
            targets=targets,
            disease_context=disease_context,
            simulation_mode=simulation_mode,
            tissue_context=tissue_context
        )
        
        logger.info("\n" + "=" * 80)
        logger.info("✅ Scenario 4 Execution Complete")
        logger.info("=" * 80)
        
        # Analyze results
        logger.info("\n📊 RESULTS ANALYSIS:")
        logger.info("-" * 80)
        
        # 1. Check network propagation
        logger.info("\n1️⃣  Network Propagation:")
        propagation_working = False
        for individual_result in result.individual_results:
            target = individual_result.target_node
            affected_count = len(individual_result.affected_nodes)
            network_impact = individual_result.network_impact
            total_affected = network_impact.get('total_affected', 1)
            
            logger.info(f"  Target: {target}")
            logger.info(f"    - Affected nodes: {affected_count}")
            logger.info(f"    - Total affected: {total_affected}")
            logger.info(f"    - Direct targets: {len(individual_result.direct_targets)}")
            logger.info(f"    - Downstream nodes: {len(individual_result.downstream)}")
            logger.info(f"    - Upstream nodes: {len(individual_result.upstream)}")
            
            if total_affected > 1:
                propagation_working = True
        
        if propagation_working:
            logger.info("  ✅ Network propagation is WORKING (affected nodes > 1)")
        else:
            logger.warning("  ⚠️  Network propagation may not be working (all targets only affect themselves)")
        
        # 2. Check synergy scores
        logger.info("\n2️⃣  Synergy Analysis:")
        synergy_analysis = result.synergy_analysis
        synergy_score = synergy_analysis.get('synergy_score', 0.0)
        pairwise_synergy = synergy_analysis.get('pairwise_synergy', {})
        
        logger.info(f"  Overall Synergy Score: {synergy_score:.3f}")
        logger.info(f"  Pairwise Combinations: {len(pairwise_synergy)}")
        
        # Show top synergy pairs
        top_pairs = list(pairwise_synergy.items())[:5]
        if top_pairs:
            logger.info("  Top Synergy Pairs:")
            for pair_key, pair_data in top_pairs:
                score = pair_data.get('synergy_score', 0.0)
                interaction_type = pair_data.get('interaction_type', 'unknown')
                logger.info(f"    - {pair_key}: {score:.3f} ({interaction_type})")
        
        if synergy_score > 0.0 or any(v.get('synergy_score', 0.0) > 0.0 for v in pairwise_synergy.values()):
            logger.info("  ✅ Synergy calculation is WORKING (non-zero scores)")
        else:
            logger.warning("  ⚠️  All synergy scores are 0.0")
        
        # 3. Check pathway enrichment
        logger.info("\n3️⃣  Pathway Enrichment:")
        pathway_enrichment = result.pathway_enrichment
        enrichment_score = pathway_enrichment.get('enrichment_score', 0.0)
        pathway_impact = pathway_enrichment.get('pathway_impact', {})
        
        logger.info(f"  Enrichment Score: {enrichment_score:.3f}")
        logger.info(f"  Pathway Impact Count: {len(pathway_impact)}")
        
        if pathway_impact:
            logger.info("  Top Affected Pathways:")
            sorted_pathways = sorted(
                pathway_impact.items(),
                key=lambda x: x[1].get('impact_score', 0.0) if isinstance(x[1], dict) else 0.0,
                reverse=True
            )[:5]
            for pathway_id, impact_data in sorted_pathways:
                if isinstance(impact_data, dict):
                    impact_score = impact_data.get('impact_score', 0.0)
                    pathway_name = impact_data.get('pathway_name', pathway_id)
                    logger.info(f"    - {pathway_name}: {impact_score:.3f}")
        
        if len(pathway_impact) > 0:
            logger.info("  ✅ Pathway enrichment is WORKING (pathway_impact populated)")
        else:
            logger.warning("  ⚠️  Pathway enrichment is empty")
        
        # 4. Check convergence
        logger.info("\n4️⃣  Convergence Detection:")
        validation_metrics = result.validation_metrics
        convergence_rate = validation_metrics.get('convergence_rate', 0.0)
        
        logger.info(f"  Convergence Rate: {convergence_rate:.3f}")
        
        if convergence_rate > 0.0:
            logger.info(f"  ✅ Convergence detection is WORKING ({convergence_rate*100:.1f}% converged)")
        else:
            logger.warning("  ⚠️  Convergence rate is 0.0")
        
        # 5. Check biological relevance
        logger.info("\n5️⃣  Biological Relevance:")
        biological_relevance = validation_metrics.get('biological_relevance', 0.0)
        
        logger.info(f"  Biological Relevance: {biological_relevance:.3f} ({biological_relevance*100:.1f}%)")
        
        if biological_relevance > 0.04:
            logger.info(f"  ✅ Biological relevance is IMPROVED (increased from 4%)")
        else:
            logger.warning("  ⚠️  Biological relevance is still low")
        
        # 6. Combined effects
        logger.info("\n6️⃣  Combined Effects:")
        combined_effects = result.combined_effects
        logger.info(f"  Overall Effect: {combined_effects.get('overall_effect', 0.0):.3f}")
        logger.info(f"  Network Coverage: {combined_effects.get('network_coverage', 0.0):.3f}")
        logger.info(f"  Average Effect Strength: {combined_effects.get('average_effect_strength', 0.0):.3f}")
        logger.info(f"  Total Affected Nodes: {combined_effects.get('total_affected_nodes', 0)}")
        
        # Summary
        logger.info("\n" + "=" * 80)
        logger.info("📈 FIX VALIDATION SUMMARY:")
        logger.info("=" * 80)
        
        fixes_status = {
            "Network Propagation": propagation_working,
            "Synergy Calculation": synergy_score > 0.0 or any(v.get('synergy_score', 0.0) > 0.0 for v in pairwise_synergy.values()),
            "Pathway Enrichment": len(pathway_impact) > 0,
            "Convergence Detection": convergence_rate > 0.0,
            "Biological Relevance": biological_relevance > 0.04
        }
        
        for fix_name, status in fixes_status.items():
            status_symbol = "✅" if status else "❌"
            logger.info(f"  {status_symbol} {fix_name}: {'WORKING' if status else 'NOT WORKING'}")
        
        all_fixed = all(fixes_status.values())
        if all_fixed:
            logger.info("\n🎉 ALL FIXES ARE WORKING!")
        else:
            logger.warning(f"\n⚠️  {sum(1 for v in fixes_status.values() if not v)} fix(es) still need attention")
        
        # Save results
        output_path = Path("results") / "s4_test_results.json"
        output_path.parent.mkdir(exist_ok=True)
        
        # Convert result to dict for JSON serialization
        result_dict = {
            'targets': result.targets,
            'individual_results': [
                {
                    'target_node': r.target_node,
                    'mode': r.mode,
                    'affected_nodes_count': len(r.affected_nodes),
                    'total_affected': r.network_impact.get('total_affected', 1),
                    'direct_targets_count': len(r.direct_targets),
                    'downstream_count': len(r.downstream),
                    'upstream_count': len(r.upstream),
                    'confidence_scores': r.confidence_scores
                }
                for r in result.individual_results
            ],
            'combined_effects': result.combined_effects,
            'synergy_analysis': {
                'synergy_score': result.synergy_analysis.get('synergy_score', 0.0),
                'pairwise_count': len(result.synergy_analysis.get('pairwise_synergy', {})),
                'top_pairs': list(result.synergy_analysis.get('pairwise_synergy', {}).items())[:5]
            },
            'pathway_enrichment': {
                'enrichment_score': result.pathway_enrichment.get('enrichment_score', 0.0),
                'pathway_impact_count': len(result.pathway_enrichment.get('pathway_impact', {}))
            },
            'validation_metrics': result.validation_metrics
        }
        
        with open(output_path, 'w') as f:
            json.dump(result_dict, f, indent=2, default=str)
        
        logger.info(f"\n💾 Results saved to: {output_path}")
        
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
    result = asyncio.run(test_scenario_4())
    
    if result:
        logger.info("\n✅ Test completed successfully!")
        sys.exit(0)
    else:
        logger.error("\n❌ Test failed!")
        sys.exit(1)

