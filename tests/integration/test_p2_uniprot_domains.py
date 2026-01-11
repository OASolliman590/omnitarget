"""
Integration Tests for P2 UniProt Domain Enrichment (S1)

Tests Scenario 1 domain enrichment with ≥60% coverage requirement.
"""

import pytest
from unittest.mock import AsyncMock, Mock

pytestmark = pytest.mark.integration


class TestP2UniProtDomainEnrichment:
    """Integration tests for S1 domain enrichment (P2 enhancement)."""
    
    @pytest.fixture
    def mock_mcp_manager_with_uniprot(self):
        """Create mocked MCP manager with UniProt client."""
        manager = Mock()
        
        # Mock KEGG
        manager.kegg = Mock()
        manager.kegg.search_diseases = AsyncMock(return_value={
            'diseases': {'hsa05224': 'Breast cancer'}
        })
        manager.kegg.get_disease_info = AsyncMock(return_value={
            'description': 'Breast cancer',
            'pathway': {'hsa05224': 'Breast cancer'}
        })
        manager.kegg.get_pathway_info = AsyncMock(return_value={
            'entry': {'name': 'Breast cancer'}
        })
        manager.kegg.get_pathway_genes = AsyncMock(return_value={
            'genes': [
                {'name': 'TP53'},
                {'name': 'BRCA1'},
                {'name': 'BRCA2'}
            ]
        })
        
        # Mock Reactome
        manager.reactome = Mock()
        manager.reactome.find_pathways_by_disease = AsyncMock(return_value={
            'pathways': [
                {'id': 'R-HSA-73864', 'name': 'Cell Cycle'}
            ]
        })
        manager.reactome.get_pathway_details = AsyncMock(return_value={
            'participants': ['TP53', 'BRCA1']
        })
        manager.reactome.get_pathway_participants = AsyncMock(return_value={
            'participants': ['TP53', 'BRCA1']
        })
        
        # Mock HPA
        manager.hpa = Mock()
        manager.hpa.search_cancer_markers = AsyncMock(return_value={'markers': []})
        manager.hpa.get_subcellular_location = AsyncMock(return_value=[
            {'Gene': 'TP53', 'Subcellular main location': ['Nucleus'], 'Reliability (IF)': 'Enhanced'}
        ])
        manager.hpa.get_tissue_expression = AsyncMock(return_value=[
            {'Tissue RNA - breast [nTPM]': '25.7'}
        ])
        manager.hpa.get_protein_info = AsyncMock(return_value=[
            {'Uniprot': ['P04637']}
        ])
        
        # Mock STRING
        manager.string = Mock()
        manager.string.get_interaction_network = AsyncMock(return_value={
            'nodes': [
                {'preferred_name': 'TP53', 'string_id': '9606.ENSP00000269305', 'protein_name': 'TP53'},
                {'preferred_name': 'BRCA1', 'string_id': '9606.ENSP00000418960', 'protein_name': 'BRCA1'}
            ],
            'edges': [
                {'protein_a': 'TP53', 'protein_b': 'BRCA1', 'confidence_score': 0.8}
            ]
        })
        
        # Mock UniProt (P2 Enhancement)
        manager.uniprot = Mock()
        manager.uniprot.search_by_gene = AsyncMock(return_value={
            'results': [
                {'primaryAccession': 'P04637', 'uniProtkbId': 'P53_HUMAN'}
            ]
        })
        manager.uniprot.get_protein_features = AsyncMock(return_value={
            'features': [
                {'type': 'Domain', 'description': 'DNA binding', 'location': {'start': {'value': 102}, 'end': {'value': 292}}},
                {'type': 'Binding site', 'description': 'ATP binding', 'location': {'start': {'value': 1}, 'end': {'value': 100}}},
                {'type': 'Region', 'description': 'Catalytic domain'}
            ],
            'domains': [],
            'activeSites': [],
            'bindingSites': [
                {'type': 'Binding site', 'description': 'Protein binding'}
            ]
        })
        
        return manager
    
    @pytest.mark.asyncio
    async def test_s1_domain_enrichment_coverage(self, mock_mcp_manager_with_uniprot):
        """Test that S1 domain enrichment achieves ≥60% coverage."""
        from src.scenarios.scenario_1_disease_network import DiseaseNetworkScenario
        
        scenario = DiseaseNetworkScenario(mock_mcp_manager_with_uniprot)
        
        # Execute S1
        result = await scenario.execute("breast cancer")
        
        # Extract network nodes
        nodes = result.network_nodes
        
        assert len(nodes) > 0, "Should have network nodes"
        
        # Check domain coverage
        nodes_with_domains = [n for n in nodes if n.domains and len(n.domains) > 0]
        coverage = len(nodes_with_domains) / len(nodes) if nodes else 0
        
        print(f"\n📊 Domain Coverage: {len(nodes_with_domains)}/{len(nodes)} ({coverage*100:.1f}%)")
        
        assert coverage >= 0.60, f"Domain coverage {coverage*100:.1f}% is below 60% threshold"
        
        # Verify domain structure
        if nodes_with_domains:
            sample_node = nodes_with_domains[0]
            sample_domain = sample_node.domains[0]
            
            assert 'type' in sample_domain or isinstance(sample_domain, dict)
            assert 'description' in sample_domain or isinstance(sample_domain, dict)
    
    @pytest.mark.asyncio
    async def test_domain_feature_structure(self, mock_mcp_manager_with_uniprot):
        """Test that domain features have correct structure."""
        from src.scenarios.scenario_1_disease_network import DiseaseNetworkScenario
        
        scenario = DiseaseNetworkScenario(mock_mcp_manager_with_uniprot)
        result = await scenario.execute("breast cancer")
        
        nodes = result.network_nodes
        nodes_with_domains = [n for n in nodes if n.domains]
        
        if nodes_with_domains:
            for node in nodes_with_domains[:3]:  # Check first 3 nodes
                for domain in node.domains[:3]:  # Check first 3 domains per node
                    assert isinstance(domain, dict)
                    assert 'type' in domain or isinstance(domain.get('type'), str)
                    # Location fields are optional
                    if 'start' in domain:
                        assert isinstance(domain['start'], (int, type(None)))
                    if 'end' in domain:
                        assert isinstance(domain['end'], (int, type(None)))
    
    @pytest.mark.asyncio
    async def test_uniprot_id_resolution_in_s1(self, mock_mcp_manager_with_uniprot):
        """Test that UniProt IDs are resolved for domain enrichment."""
        from src.scenarios.scenario_1_disease_network import DiseaseNetworkScenario
        
        scenario = DiseaseNetworkScenario(mock_mcp_manager_with_uniprot)
        result = await scenario.execute("breast cancer")
        
        nodes = result.network_nodes
        
        # Check if nodes have UniProt IDs
        nodes_with_uniprot = [n for n in nodes if n.uniprot_id]
        uniprot_coverage = len(nodes_with_uniprot) / len(nodes) if nodes else 0
        
        print(f"\n📊 UniProt ID Coverage: {len(nodes_with_uniprot)}/{len(nodes)} ({uniprot_coverage*100:.1f}%)")
        
        # At least some nodes should have UniProt IDs
        assert len(nodes_with_uniprot) > 0, "Should have nodes with UniProt IDs"
        
        # Verify nodes with UniProt IDs have domains
        if nodes_with_uniprot:
            nodes_with_uniprot_and_domains = [n for n in nodes_with_uniprot if n.domains]
            domain_coverage_for_uniprot_nodes = len(nodes_with_uniprot_and_domains) / len(nodes_with_uniprot) if nodes_with_uniprot else 0
            
            print(f"\n📊 Domain Coverage (for UniProt nodes): {len(nodes_with_uniprot_and_domains)}/{len(nodes_with_uniprot)} ({domain_coverage_for_uniprot_nodes*100:.1f}%)")
            
            # Nodes with UniProt IDs should have high domain coverage
            assert domain_coverage_for_uniprot_nodes >= 0.80, f"Domain coverage for UniProt nodes {domain_coverage_for_uniprot_nodes*100:.1f}% is below 80%"
    
    @pytest.mark.asyncio
    async def test_domain_parsing_from_nodejs_format(self, mock_mcp_manager_with_uniprot):
        """Test parsing domains from Node.js UniProt MCP response format."""
        from src.scenarios.scenario_1_disease_network import DiseaseNetworkScenario
        
        scenario = DiseaseNetworkScenario(mock_mcp_manager_with_uniprot)
        result = await scenario.execute("breast cancer")
        
        nodes = result.network_nodes
        nodes_with_domains = [n for n in nodes if n.domains]
        
        if nodes_with_domains:
            # Check that domains were parsed correctly
            sample_node = nodes_with_domains[0]
            
            # Verify domain features include expected types
            domain_types = [d.get('type', '').lower() for d in sample_node.domains if isinstance(d, dict)]
            
            # Should have at least some domain-related features
            assert len(domain_types) > 0
            
            # Check for druggable feature types
            druggable_keywords = ['domain', 'binding', 'site', 'region']
            found_keywords = [kw for kw in druggable_keywords if any(kw in dt for dt in domain_types)]
            
            assert len(found_keywords) > 0, "Should find druggable domain keywords"

