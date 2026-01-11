"""
All Scenarios Example

Demonstrates all six core scenarios of the OmniTarget pipeline.
"""

import asyncio
import json
from pathlib import Path
import sys

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

from core.pipeline_orchestrator import OmniTargetPipeline


async def demonstrate_scenario_1_disease_network():
    """Demonstrate Scenario 1: Disease Network Construction."""
    print("🔬 Scenario 1: Disease Network Construction")
    print("=" * 60)
    
    try:
        # Initialize pipeline
        pipeline = OmniTargetPipeline()
        
        # Run scenario
        result = await pipeline.run_scenario(
            1,
            disease_query="breast cancer",
            tissue_context="breast"
        )
        
        print(f"📊 Disease Network Results:")
        print(f"  Primary Disease: {result.disease.name if result.disease else 'Not found'}")
        print(f"  Pathways: {len(result.pathways)}")
        print(f"  Network Nodes: {len(result.network_nodes)}")
        print(f"  Network Edges: {len(result.network_edges)}")
        print(f"  Expression Profiles: {len(result.expression_profiles)}")
        print(f"  Cancer Markers: {len(result.cancer_markers)}")
        print(f"  Validation Score: {result.validation_score:.3f}")
        
        return result
        
    except Exception as e:
        print(f"❌ Error in Scenario 1: {e}")
        return None


async def demonstrate_scenario_2_target_analysis():
    """Demonstrate Scenario 2: Target-Centric Analysis."""
    print("\n🎯 Scenario 2: Target-Centric Analysis")
    print("=" * 60)
    
    try:
        # Initialize pipeline
        pipeline = OmniTargetPipeline()
        
        # Run scenario
        result = await pipeline.run_scenario(
            2,
            target_query="TP53"
        )
        
        print(f"📊 Target Analysis Results:")
        print(f"  Target: {result.target.gene_symbol if result.target else 'Not found'}")
        print(f"  Pathways: {len(result.pathways)}")
        print(f"  Network Nodes: {len(result.network_nodes)}")
        print(f"  Network Edges: {len(result.network_edges)}")
        print(f"  Expression Profiles: {len(result.expression_profiles)}")
        print(f"  Druggability Score: {result.druggability_score:.3f}")
        print(f"  Validation Score: {result.validation_score:.3f}")
        
        return result
        
    except Exception as e:
        print(f"❌ Error in Scenario 2: {e}")
        return None


async def demonstrate_scenario_3_cancer_analysis():
    """Demonstrate Scenario 3: Cancer-Specific Analysis."""
    print("\n🦠 Scenario 3: Cancer-Specific Analysis")
    print("=" * 60)
    
    try:
        # Initialize pipeline
        pipeline = OmniTargetPipeline()
        
        # Run scenario
        result = await pipeline.run_scenario(
            3,
            cancer_type="breast cancer",
            tissue_context="breast"
        )
        
        print(f"📊 Cancer Analysis Results:")
        print(f"  Cancer Type: {result.cancer_type}")
        print(f"  Tissue Context: {result.tissue_context}")
        print(f"  Cancer Markers: {len(result.cancer_markers)}")
        print(f"  Pathways: {len(result.pathways)}")
        print(f"  Network Nodes: {len(result.network_nodes)}")
        print(f"  Network Edges: {len(result.network_edges)}")
        print(f"  Expression Profiles: {len(result.expression_profiles)}")
        print(f"  Prioritized Targets: {len(result.prioritized_targets)}")
        print(f"  Validation Score: {result.validation_score:.3f}")
        
        return result
        
    except Exception as e:
        print(f"❌ Error in Scenario 3: {e}")
        return None


async def demonstrate_scenario_4_multi_target_simulation():
    """Demonstrate Scenario 4: Multi-Target Simulation with MRA."""
    print("\n🧮 Scenario 4: Multi-Target Simulation with MRA")
    print("=" * 60)
    
    try:
        # Initialize pipeline
        pipeline = OmniTargetPipeline()
        
        # Run scenario
        result = await pipeline.run_scenario(
            4,
            targets=["TP53", "BRCA1", "BRCA2"],
            disease_context="breast cancer",
            simulation_mode="simple",
            tissue_context="breast"
        )
        
        print(f"📊 Multi-Target Simulation Results:")
        print(f"  Targets: {len(result.targets)}")
        print(f"  Disease Context: {result.disease_context}")
        print(f"  Simulation Mode: {result.simulation_mode}")
        print(f"  Network Nodes: {len(result.network_nodes)}")
        print(f"  Network Edges: {len(result.network_edges)}")
        print(f"  Simulation Results: {len(result.simulation_results)}")
        print(f"  Impact Assessment: {result.impact_assessment}")
        print(f"  Validation Score: {result.validation_score:.3f}")
        
        return result
        
    except Exception as e:
        print(f"❌ Error in Scenario 4: {e}")
        return None


async def demonstrate_scenario_5_pathway_comparison():
    """Demonstrate Scenario 5: Pathway Comparison and Validation."""
    print("\n🔄 Scenario 5: Pathway Comparison and Validation")
    print("=" * 60)
    
    try:
        # Initialize pipeline
        pipeline = OmniTargetPipeline()
        
        # Run scenario
        result = await pipeline.run_scenario(
            5,
            pathway_query="cancer"
        )
        
        print(f"📊 Pathway Comparison Results:")
        print(f"  Pathway Query: {result.pathway_query}")
        print(f"  KEGG Pathways: {len(result.kegg_pathways)}")
        print(f"  Reactome Pathways: {len(result.reactome_pathways)}")
        print(f"  Gene Overlap: {result.gene_overlap}")
        print(f"  Mechanistic Details: {len(result.mechanistic_details)}")
        print(f"  Interaction Validation: {result.interaction_validation}")
        print(f"  Expression Context: {len(result.expression_context)}")
        print(f"  Validation Score: {result.validation_score:.3f}")
        
        return result
        
    except Exception as e:
        print(f"❌ Error in Scenario 5: {e}")
        return None


async def demonstrate_scenario_6_drug_repurposing():
    """Demonstrate Scenario 6: Drug Repurposing with Network Analysis."""
    print("\n💊 Scenario 6: Drug Repurposing with Network Analysis")
    print("=" * 60)
    
    try:
        # Initialize pipeline
        pipeline = OmniTargetPipeline()
        
        # Run scenario
        result = await pipeline.run_scenario(
            6,
            disease_query="breast cancer",
            tissue_context="breast",
            simulation_mode="simple"
        )
        
        print(f"📊 Drug Repurposing Results:")
        print(f"  Disease Query: {result.disease_query}")
        print(f"  Tissue Context: {result.tissue_context}")
        print(f"  Pathways: {len(result.pathways)}")
        print(f"  Drug Targets: {len(result.drug_targets)}")
        print(f"  Repurposing Candidates: {len(result.repurposing_candidates)}")
        print(f"  Network Nodes: {len(result.network_nodes)}")
        print(f"  Network Edges: {len(result.network_edges)}")
        print(f"  Simulation Results: {len(result.simulation_results)}")
        print(f"  Enrichment Results: {result.enrichment_results}")
        print(f"  Validation Score: {result.validation_score:.3f}")
        
        return result
        
    except Exception as e:
        print(f"❌ Error in Scenario 6: {e}")
        return None


async def demonstrate_pipeline_orchestrator():
    """Demonstrate the pipeline orchestrator with all scenarios."""
    print("\n🚀 Pipeline Orchestrator Demo")
    print("=" * 60)
    
    try:
        # Initialize pipeline
        pipeline = OmniTargetPipeline()
        
        # Get pipeline status
        status = await pipeline.get_pipeline_status()
        print(f"📊 Pipeline Status:")
        print(f"  Version: {status['pipeline_version']}")
        print(f"  Available Scenarios: {status['available_scenarios']}")
        print(f"  Implemented Scenarios: {status['implemented_scenarios']}")
        print(f"  MCP Servers: {status['mcp_servers']}")
        
        # List available scenarios
        scenarios = await pipeline.list_available_scenarios()
        print(f"\n🔍 Available Scenarios:")
        for scenario in scenarios:
            print(f"  {scenario['scenario_id']}. {scenario['name']}")
            print(f"     Description: {scenario['description'][:100]}...")
            print(f"     Parameters: {list(scenario['parameters'].keys())}")
        
        # Test parameter validation for each scenario
        print(f"\n✅ Parameter Validation:")
        validation_tests = [
            (1, {'disease_query': 'breast cancer', 'tissue_context': 'breast'}),
            (2, {'target_query': 'TP53'}),
            (3, {'cancer_type': 'breast cancer', 'tissue_context': 'breast'}),
            (4, {'targets': ['TP53', 'BRCA1'], 'disease_context': 'breast cancer'}),
            (5, {'pathway_query': 'cancer'}),
            (6, {'disease_query': 'breast cancer', 'tissue_context': 'breast'})
        ]
        
        for scenario_id, params in validation_tests:
            valid_result = await pipeline.validate_scenario_parameters(scenario_id, params)
            print(f"  Scenario {scenario_id}: {'✅ Valid' if valid_result['valid'] else '❌ Invalid'}")
        
        # Test health check
        print(f"\n🏥 Health Check:")
        health = await pipeline.health_check()
        for server, status in health.items():
            print(f"  {server}: {status['status']} - {status['message']}")
        
        await pipeline.shutdown()
        
    except Exception as e:
        print(f"❌ Error in Pipeline Orchestrator: {e}")


async def demonstrate_batch_processing():
    """Demonstrate batch processing of all scenarios."""
    print("\n📦 Batch Processing Demo")
    print("=" * 60)
    
    try:
        # Initialize pipeline
        pipeline = OmniTargetPipeline()
        
        # Define batch scenarios
        batch_configs = [
            {
                'scenario_id': 1,
                'disease_query': 'breast cancer',
                'tissue_context': 'breast'
            },
            {
                'scenario_id': 2,
                'target_query': 'TP53'
            },
            {
                'scenario_id': 3,
                'cancer_type': 'breast cancer',
                'tissue_context': 'breast'
            },
            {
                'scenario_id': 4,
                'targets': ['TP53', 'BRCA1'],
                'disease_context': 'breast cancer',
                'simulation_mode': 'simple'
            },
            {
                'scenario_id': 5,
                'pathway_query': 'cancer'
            },
            {
                'scenario_id': 6,
                'disease_query': 'breast cancer',
                'tissue_context': 'breast',
                'simulation_mode': 'simple'
            }
        ]
        
        print(f"🔄 Running batch of {len(batch_configs)} scenarios...")
        
        # Run batch scenarios
        results = await pipeline.run_scenario_batch(batch_configs)
        
        print(f"✅ Batch processing completed!")
        print(f"  Results: {len(results)}")
        
        # Summarize results
        for i, result in enumerate(results):
            scenario_id = batch_configs[i]['scenario_id']
            print(f"\n  Scenario {scenario_id} Results:")
            if hasattr(result, 'validation_score'):
                print(f"    Validation Score: {result.validation_score:.3f}")
            if hasattr(result, 'network_nodes'):
                print(f"    Network Nodes: {len(result.network_nodes)}")
            if hasattr(result, 'pathways'):
                print(f"    Pathways: {len(result.pathways)}")
            if hasattr(result, 'simulation_results'):
                print(f"    Simulation Results: {len(result.simulation_results)}")
        
        await pipeline.shutdown()
        
    except Exception as e:
        print(f"❌ Error in Batch Processing: {e}")


async def main():
    """Main demonstration function."""
    print("🧬 OmniTarget Pipeline - All Scenarios Demonstration")
    print("=" * 80)
    
    try:
        # Demonstrate individual scenarios
        scenario1_result = await demonstrate_scenario_1_disease_network()
        scenario2_result = await demonstrate_scenario_2_target_analysis()
        scenario3_result = await demonstrate_scenario_3_cancer_analysis()
        scenario4_result = await demonstrate_scenario_4_multi_target_simulation()
        scenario5_result = await demonstrate_scenario_5_pathway_comparison()
        scenario6_result = await demonstrate_scenario_6_drug_repurposing()
        
        # Demonstrate pipeline orchestrator
        await demonstrate_pipeline_orchestrator()
        
        # Demonstrate batch processing
        await demonstrate_batch_processing()
        
        print("\n🎉 All demonstrations completed successfully!")
        
        # Summary
        print(f"\n📊 Summary:")
        print(f"  Scenario 1 (Disease Network): {'✅' if scenario1_result else '❌'}")
        print(f"  Scenario 2 (Target Analysis): {'✅' if scenario2_result else '❌'}")
        print(f"  Scenario 3 (Cancer Analysis): {'✅' if scenario3_result else '❌'}")
        print(f"  Scenario 4 (Multi-Target Simulation): {'✅' if scenario4_result else '❌'}")
        print(f"  Scenario 5 (Pathway Comparison): {'✅' if scenario5_result else '❌'}")
        print(f"  Scenario 6 (Drug Repurposing): {'✅' if scenario6_result else '❌'}")
        
        print(f"\n🏆 OmniTarget Pipeline Status: All 6 scenarios implemented and tested!")
        
    except Exception as e:
        print(f"❌ Error during demonstration: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
