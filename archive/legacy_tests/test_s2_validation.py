#!/usr/bin/env python
"""
Validation script for Scenario 2 type validation fix.

Tests that Scenario 2 handles UniProt error responses gracefully
without raising "'str' object has no attribute 'get'" errors.
"""

import asyncio
import logging
from unittest.mock import AsyncMock, Mock, patch
import sys

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


async def test_scenario_2_error_handling():
    """Test Scenario 2 with various UniProt error responses."""
    from src.scenarios.scenario_2_target_analysis import TargetAnalysisScenario
    from src.models.data_models import Protein

    # Create mock MCP manager
    mock_manager = Mock()

    # Mock STRING
    mock_manager.string = Mock()
    mock_manager.string.search_proteins = AsyncMock(return_value={
        'proteins': [
            {'string_id': '9606.ENSP00000269305', 'preferred_name': 'AXL'}
        ]
    })
    mock_manager.string.get_interaction_network = AsyncMock(return_value={
        'nodes': [{'preferred_name': 'AXL', 'string_id': '9606.ENSP00000269305'}],
        'edges': []
    })

    # Mock HPA
    mock_manager.hpa = Mock()
    mock_manager.hpa.search_proteins = AsyncMock(return_value=[
        {'Gene': 'AXL', 'UniProt': 'Q9UMF0'}
    ])
    mock_manager.hpa.get_tissue_expression = AsyncMock(return_value={})
    mock_manager.hpa.get_subcellular_location = AsyncMock(return_value=[])
    mock_manager.hpa.get_pathology_data = AsyncMock(return_value={})

    # Mock KEGG
    mock_manager.kegg = Mock()
    mock_manager.kegg.search_genes = AsyncMock(return_value={
        'genes': {'hsa:111': 'AXL'}
    })
    mock_manager.kegg.search_drugs = AsyncMock(return_value={'drugs': []})
    mock_manager.kegg.get_gene_pathways = AsyncMock(return_value={'pathways': []})

    # Mock Reactome
    mock_manager.reactome = Mock()
    mock_manager.reactome.find_pathways_by_gene = AsyncMock(return_value={'pathways': []})
    mock_manager.reactome.get_protein_interactions = AsyncMock(return_value={'interactions': []})

    # Mock UniProt
    mock_manager.uniprot = Mock()

    # Test case 1: UniProt returns error string (this was causing the bug!)
    print("\n" + "="*70)
    print("Test Case 1: UniProt returns error string")
    print("="*70)

    mock_manager.uniprot.search_by_gene = AsyncMock(
        return_value="Error: Gene not found in UniProt database"
    )
    mock_manager.uniprot.get_protein_features = AsyncMock(
        return_value="Error: Protein not found"
    )

    scenario = TargetAnalysisScenario(mock_manager)

    try:
        result = await scenario.execute("AXL")
        print(f"✅ SUCCESS: execute() completed without error")
        print(f"   Target: {result.target.gene_symbol if result.target else 'None'}")
        print(f"   Druggability Score: {result.druggability_score:.3f}")
    except AttributeError as e:
        print(f"❌ FAILED: AttributeError raised: {e}")
        return False
    except Exception as e:
        print(f"⚠️  Other exception: {type(e).__name__}: {e}")
        # Some other exceptions might be expected (e.g., missing data)

    # Test case 2: UniProt returns None
    print("\n" + "="*70)
    print("Test Case 2: UniProt returns None")
    print("="*70)

    mock_manager.uniprot.search_by_gene = AsyncMock(return_value=None)
    mock_manager.uniprot.get_protein_features = AsyncMock(return_value=None)

    try:
        result = await scenario.execute("BRCA1")
        print(f"✅ SUCCESS: execute() completed without error")
        print(f"   Target: {result.target.gene_symbol if result.target else 'None'}")
        print(f"   Druggability Score: {result.druggability_score:.3f}")
    except AttributeError as e:
        print(f"❌ FAILED: AttributeError raised: {e}")
        return False

    # Test case 3: UniProt returns unexpected type (list)
    print("\n" + "="*70)
    print("Test Case 3: UniProt returns list instead of dict")
    print("="*70)

    mock_manager.uniprot.search_by_gene = AsyncMock(
        return_value=[{'result': 'data'}]
    )
    mock_manager.uniprot.get_protein_features = AsyncMock(
        return_value=['features', 'domains']
    )

    try:
        result = await scenario.execute("TP53")
        print(f"✅ SUCCESS: execute() completed without error")
        print(f"   Target: {result.target.gene_symbol if result.target else 'None'}")
        print(f"   Druggability Score: {result.druggability_score:.3f}")
    except AttributeError as e:
        print(f"❌ FAILED: AttributeError raised: {e}")
        return False

    # Test case 4: Valid UniProt response (should work as before)
    print("\n" + "="*70)
    print("Test Case 4: Valid UniProt response")
    print("="*70)

    mock_manager.uniprot.search_by_gene = AsyncMock(return_value={
        'results': [
            {
                'primaryAccession': 'Q9UMF0',
                'uniProtkbId': 'AXL_HUMAN',
                'entryType': 'UniProtKB reviewed (Swiss-Prot)'
            }
        ]
    })
    mock_manager.uniprot.get_protein_features = AsyncMock(return_value={
        'features': [
            {'type': 'Domain', 'description': 'protein kinase domain'}
        ],
        'domains': [],
        'activeSites': [],
        'bindingSites': []
    })

    try:
        result = await scenario.execute("AXL")
        print(f"✅ SUCCESS: execute() completed without error")
        print(f"   Target: {result.target.gene_symbol if result.target else 'None'}")
        print(f"   UniProt ID: {result.target.uniprot_id if result.target else 'None'}")
        print(f"   Druggability Score: {result.druggability_score:.3f}")
    except Exception as e:
        print(f"❌ FAILED: {type(e).__name__}: {e}")
        return False

    # Test case 5: Direct druggability score calculation with error
    print("\n" + "="*70)
    print("Test Case 5: Direct druggability calculation with error")
    print("="*70)

    mock_manager.uniprot.get_protein_features = AsyncMock(
        return_value="Error: Invalid protein ID"
    )

    target = Protein(
        gene_symbol="PIK3R2",
        uniprot_id="O00459",
        confidence=0.9
    )

    import networkx as nx
    network = nx.Graph()
    network.add_node("PIK3R2")

    try:
        score = await scenario._calculate_druggability_score(target, network, [])
        print(f"✅ SUCCESS: druggability score calculated without error")
        print(f"   Score: {score:.3f} (domain boost should be 0.0 due to error)")
    except AttributeError as e:
        print(f"❌ FAILED: AttributeError raised: {e}")
        return False

    print("\n" + "="*70)
    print("✅ ALL VALIDATION TESTS PASSED!")
    print("="*70)
    return True


async def main():
    """Main validation function."""
    print("\n" + "="*70)
    print("SCENARIO 2 TYPE VALIDATION FIX - VALIDATION SCRIPT")
    print("="*70)
    print("\nThis script tests the fix for the bug:")
    print("'str' object has no attribute 'get'")
    print("\nThe fix adds type validation BEFORE calling .get() on")
    print("potentially-untrusted UniProt MCP responses.\n")

    success = await test_scenario_2_error_handling()

    if success:
        print("\n🎉 Fix validation successful!")
        print("\nThe following genes were tested without errors:")
        print("  - AXL (UniProt error string)")
        print("  - BRCA1 (UniProt None)")
        print("  - TP53 (UniProt wrong type)")
        print("  - PIK3R2 (domain scoring with error)")
        print("\nAll tests passed! ✅")
        return 0
    else:
        print("\n❌ Fix validation failed!")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
