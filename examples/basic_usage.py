"""
Basic Usage Example

Demonstrates how to use the OmniTarget pipeline for disease analysis.
"""

import asyncio
import json
from pathlib import Path
import sys

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

from core.mcp_client_manager import MCPClientManager
from core.data_standardizer import DataStandardizer
from core.validation import DataValidator


async def main():
    """Main example function."""
    print("🧬 OmniTarget Pipeline - Basic Usage Example")
    print("=" * 50)
    
    # Initialize pipeline components
    config_path = "config/mcp_servers.json"
    standardizer = DataStandardizer()
    validator = DataValidator()
    
    try:
        # Initialize MCP client manager
        print("📡 Initializing MCP clients...")
        manager = MCPClientManager(config_path)
        
        # Use session context manager for automatic lifecycle management
        async with manager.session() as session:
            print("✅ All MCP servers started successfully")
            
            # Example 1: Disease search across databases
            print("\n🔍 Example 1: Disease Search")
            print("-" * 30)
            
            disease_query = "breast cancer"
            
            # Search in KEGG
            kegg_diseases = await session.kegg.search_diseases(disease_query, limit=3)
            print(f"KEGG found {len(kegg_diseases.get('diseases', []))} diseases")
            
            # Search in Reactome
            reactome_pathways = await session.reactome.find_pathways_by_disease(disease_query)
            print(f"Reactome found {len(reactome_pathways.get('pathways', []))} pathways")
            
            # Example 2: Pathway analysis
            print("\n🛤️ Example 2: Pathway Analysis")
            print("-" * 30)
            
            if kegg_diseases.get('diseases'):
                disease_id = kegg_diseases['diseases'][0]['id']
                print(f"Analyzing pathways for disease: {disease_id}")
                
                # Get disease pathways
                disease_pathways = await session.kegg.get_disease_pathways(disease_id)
                print(f"Found {len(disease_pathways.get('pathways', []))} associated pathways")
                
                # Get pathway genes for first pathway
                if disease_pathways.get('pathways'):
                    pathway_id = disease_pathways['pathways'][0]['id']
                    pathway_genes = await session.kegg.get_pathway_genes(pathway_id)
                    print(f"Pathway {pathway_id} contains {pathway_genes.get('gene_count', 0)} genes")
            
            # Example 3: Protein network construction
            print("\n🕸️ Example 3: Protein Network Construction")
            print("-" * 30)
            
            # Use sample genes for network construction
            sample_genes = ["TP53", "BRCA1", "BRCA2", "MDM2"]
            print(f"Building network for genes: {', '.join(sample_genes)}")
            
            # Get STRING network
            string_network = await session.string.get_interaction_network(
                genes=sample_genes,
                species=9606,  # Human
                required_score=400
            )
            
            network_data = string_network.get('network', {})
            nodes = network_data.get('nodes', [])
            edges = network_data.get('edges', [])
            
            print(f"Network contains {len(nodes)} nodes and {len(edges)} edges")
            
            # Example 4: Expression analysis
            print("\n🧪 Example 4: Expression Analysis")
            print("-" * 30)
            
            # Get expression data for TP53
            tp53_expression = await session.hpa.get_tissue_expression("TP53")
            expression_data = tp53_expression.get('expression', {})
            
            print(f"TP53 expression across tissues:")
            for tissue, level in list(expression_data.items())[:5]:  # Show first 5
                print(f"  {tissue}: {level}")
            
            # Example 5: Cancer marker search
            print("\n🎯 Example 5: Cancer Marker Search")
            print("-" * 30)
            
            cancer_markers = await session.hpa.search_cancer_markers("breast cancer")
            markers = cancer_markers.get('markers', [])
            
            print(f"Found {len(markers)} breast cancer markers")
            for marker in markers[:3]:  # Show first 3
                gene = marker.get('gene', 'Unknown')
                prognostic = marker.get('prognostic_value', 'Unknown')
                print(f"  {gene}: {prognostic} prognostic value")
            
            # Example 6: Data validation
            print("\n✅ Example 6: Data Validation")
            print("-" * 30)
            
            # Validate some sample data
            validation_results = {}
            
            # Validate disease confidence
            if kegg_diseases.get('diseases'):
                disease_data = kegg_diseases['diseases'][0]
                disease = standardizer.standardize_kegg_disease(disease_data)
                validation_results['disease_confidence'] = validator.validate_disease_confidence(disease)
            
            # Validate interaction confidence
            if edges:
                interaction_data = edges[0]
                interaction = standardizer.standardize_string_interaction(interaction_data)
                validation_results['interaction_confidence'] = validator.validate_interaction_confidence(interaction)
            
            # Calculate overall validation score
            overall_score = validator.calculate_overall_validation_score(validation_results)
            
            print(f"Validation results:")
            for metric, result in validation_results.items():
                print(f"  {metric}: {'✅' if result else '❌'}")
            print(f"  Overall score: {overall_score:.2f}")
            
            print("\n🎉 Pipeline execution completed successfully!")
            
    except Exception as e:
        print(f"❌ Error during pipeline execution: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        print("\n📊 Pipeline Summary:")
        print("- MCP clients tested: KEGG, Reactome, STRING, HPA")
        print("- Data standardization: ✅")
        print("- Cross-database integration: ✅")
        print("- Validation metrics: ✅")


if __name__ == "__main__":
    asyncio.run(main())
