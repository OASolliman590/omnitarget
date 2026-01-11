"""
Scenarios Example

Demonstrates the first three core scenarios of the OmniTarget pipeline.
"""

import asyncio
import json
from pathlib import Path
import sys

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

from core.pipeline_orchestrator import OmniTargetPipeline, analyze_disease, analyze_target, analyze_cancer


async def demonstrate_scenario_1_disease_network():
    """Demonstrate Scenario 1: Disease Network Construction."""
    print("🔬 Scenario 1: Disease Network Construction")
    print("=" * 60)
    
    try:
        # Analyze breast cancer disease network
        result = await analyze_disease(
            disease_query="breast cancer",
            tissue_context="breast"
        )
        
        print(f"📊 Disease Analysis Results:")
        print(f"  Primary Disease: {result.disease.name if result.disease else 'Not found'}")
        print(f"  Pathways: {len(result.pathways)}")
        print(f"  Network Nodes: {len(result.network_nodes)}")
        print(f"  Network Edges: {len(result.network_edges)}")
        print(f"  Expression Profiles: {len(result.expression_profiles)}")
        print(f"  Cancer Markers: {len(result.cancer_markers)}")
        print(f"  Validation Score: {result.validation_score:.3f}")
        
        print(f"\n🔍 Top Pathways:")
        for i, pathway in enumerate(result.pathways[:3]):
            print(f"  {i+1}. {pathway.name} ({pathway.id})")
            print(f"     Genes: {len(pathway.genes)}")
            print(f"     Database: {pathway.database}")
        
        print(f"\n🧬 Top Network Nodes:")
        for i, node in enumerate(result.network_nodes[:5]):
            print(f"  {i+1}. {node.gene_symbol} ({node.id})")
            print(f"     Centrality: {node.centrality_measures}")
        
        print(f"\n📈 Expression Profiles:")
        for i, profile in enumerate(result.expression_profiles[:3]):
            print(f"  {i+1}. {profile.gene} in {profile.tissue}: {profile.expression_level}")
        
        return result
        
    except Exception as e:
        print(f"❌ Error in Scenario 1: {e}")
        return None


async def demonstrate_scenario_2_target_analysis():
    """Demonstrate Scenario 2: Target-Centric Analysis."""
    print("\n🎯 Scenario 2: Target-Centric Analysis")
    print("=" * 60)
    
    try:
        # Analyze TP53 target
        result = await analyze_target(target_query="TP53")
        
        print(f"📊 Target Analysis Results:")
        print(f"  Target: {result.target.gene_symbol if result.target else 'Not found'}")
        print(f"  Pathways: {len(result.pathways)}")
        print(f"  Network Nodes: {len(result.network_nodes)}")
        print(f"  Network Edges: {len(result.network_edges)}")
        print(f"  Expression Profiles: {len(result.expression_profiles)}")
        print(f"  Druggability Score: {result.druggability_score:.3f}")
        print(f"  Validation Score: {result.validation_score:.3f}")
        
        print(f"\n🔍 Target Pathways:")
        for i, pathway in enumerate(result.pathways[:3]):
            print(f"  {i+1}. {pathway.name} ({pathway.id})")
            print(f"     Genes: {len(pathway.genes)}")
            print(f"     Database: {pathway.database}")
        
        print(f"\n🧬 Interaction Network:")
        for i, node in enumerate(result.network_nodes[:5]):
            print(f"  {i+1}. {node.gene_symbol} ({node.id})")
            print(f"     Centrality: {node.centrality_measures}")
        
        print(f"\n💊 Drug Targets:")
        for i, drug in enumerate(result.drug_targets[:3]):
            print(f"  {i+1}. {drug.name} ({drug.id})")
            print(f"     Target: {drug.target}")
            print(f"     Confidence: {drug.confidence}")
        
        return result
        
    except Exception as e:
        print(f"❌ Error in Scenario 2: {e}")
        return None


async def demonstrate_scenario_3_cancer_analysis():
    """Demonstrate Scenario 3: Cancer-Specific Analysis."""
    print("\n🦠 Scenario 3: Cancer-Specific Analysis")
    print("=" * 60)
    
    try:
        # Analyze breast cancer
        result = await analyze_cancer(
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
        
        print(f"\n🔍 Cancer Markers:")
        for i, marker in enumerate(result.cancer_markers[:3]):
            print(f"  {i+1}. {marker.gene} ({marker.marker_type})")
            print(f"     Confidence: {marker.confidence_score:.3f}")
            print(f"     Prognostic Value: {marker.prognostic_value:.3f}")
        
        print(f"\n🧬 Cancer Pathways:")
        for i, pathway in enumerate(result.pathways[:3]):
            print(f"  {i+1}. {pathway.name} ({pathway.id})")
            print(f"     Genes: {len(pathway.genes)}")
            print(f"     Database: {pathway.database}")
        
        print(f"\n🎯 Prioritized Targets:")
        for i, target in enumerate(result.prioritized_targets[:5]):
            print(f"  {i+1}. {target.gene_symbol} ({target.target_id})")
            print(f"     Prioritization Score: {target.prioritization_score:.3f}")
            print(f"     Druggability: {target.druggability_score:.3f}")
            print(f"     Cancer Specificity: {target.cancer_specificity_score:.3f}")
            print(f"     Network Centrality: {target.network_centrality_score:.3f}")
        
        return result
        
    except Exception as e:
        print(f"❌ Error in Scenario 3: {e}")
        return None


async def demonstrate_pipeline_orchestrator():
    """Demonstrate the pipeline orchestrator."""
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
        
        # Test parameter validation
        print(f"\n✅ Parameter Validation:")
        valid_result = await pipeline.validate_scenario_parameters(
            1, {'disease_query': 'breast cancer', 'tissue_context': 'breast'}
        )
        print(f"  Valid parameters: {valid_result['valid']}")
        
        invalid_result = await pipeline.validate_scenario_parameters(
            1, {'tissue_context': 'breast'}  # Missing required parameter
        )
        print(f"  Invalid parameters: {invalid_result['valid']}")
        print(f"  Errors: {invalid_result['errors']}")
        
        # Test health check
        print(f"\n🏥 Health Check:")
        health = await pipeline.health_check()
        for server, status in health.items():
            print(f"  {server}: {status['status']} - {status['message']}")
        
        await pipeline.shutdown()
        
    except Exception as e:
        print(f"❌ Error in Pipeline Orchestrator: {e}")


async def demonstrate_batch_processing():
    """Demonstrate batch processing of multiple scenarios."""
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
        
        await pipeline.shutdown()
        
    except Exception as e:
        print(f"❌ Error in Batch Processing: {e}")


async def main():
    """Main demonstration function."""
    print("🧬 OmniTarget Pipeline - Scenarios Demonstration")
    print("=" * 80)
    
    try:
        # Demonstrate individual scenarios
        disease_result = await demonstrate_scenario_1_disease_network()
        target_result = await demonstrate_scenario_2_target_analysis()
        cancer_result = await demonstrate_scenario_3_cancer_analysis()
        
        # Demonstrate pipeline orchestrator
        await demonstrate_pipeline_orchestrator()
        
        # Demonstrate batch processing
        await demonstrate_batch_processing()
        
        print("\n🎉 All demonstrations completed successfully!")
        
        # Summary
        print(f"\n📊 Summary:")
        print(f"  Disease Network: {'✅' if disease_result else '❌'}")
        print(f"  Target Analysis: {'✅' if target_result else '❌'}")
        print(f"  Cancer Analysis: {'✅' if cancer_result else '❌'}")
        
    except Exception as e:
        print(f"❌ Error during demonstration: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
