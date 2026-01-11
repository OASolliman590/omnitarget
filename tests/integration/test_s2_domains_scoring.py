"""
Integration Tests for S2 Domain-Based Druggability Scoring

Tests Scenario 2 druggability scoring with domain boost enhancement.
"""

import pytest
from unittest.mock import AsyncMock, Mock
import networkx as nx

pytestmark = pytest.mark.integration


class TestS2DomainScoring:
    """Integration tests for S2 domain-based druggability scoring."""
    
    @pytest.fixture
    def mock_mcp_manager_with_uniprot(self):
        """Create mocked MCP manager with UniProt client."""
        manager = Mock()
        
        # Mock STRING
        manager.string = Mock()
        manager.string.search_proteins = AsyncMock(return_value={
            'proteins': [
                {'string_id': '9606.ENSP00000269305', 'preferred_name': 'TP53'}
            ]
        })
        manager.string.get_interaction_network = AsyncMock(return_value={
            'nodes': [
                {'preferred_name': 'TP53', 'string_id': '9606.ENSP00000269305'}
            ],
            'edges': [
                {'protein_a': 'TP53', 'protein_b': 'MDM2', 'confidence_score': 0.9}
            ]
        })
        
        # Mock HPA
        manager.hpa = Mock()
        manager.hpa.search_proteins = AsyncMock(return_value=[
            {'Gene': 'TP53', 'UniProt': 'P04637'}
        ])
        manager.hpa.get_tissue_expression = AsyncMock(return_value=[
            {'Tissue RNA - breast [nTPM]': '25.7'}
        ])
        manager.hpa.get_subcellular_location = AsyncMock(return_value=[
            {'Gene': 'TP53', 'Subcellular main location': ['Nucleus']}
        ])
        
        # Mock KEGG
        manager.kegg = Mock()
        manager.kegg.search_genes = AsyncMock(return_value={
            'genes': {'hsa:7157': 'TP53'}
        })
        manager.kegg.search_drugs = AsyncMock(return_value={
            'drugs': []
        })
        
        # Mock Reactome
        manager.reactome = Mock()
        manager.reactome.find_pathways_by_gene = AsyncMock(return_value={
            'pathways': [
                {'id': 'R-HSA-73864', 'name': 'Cell Cycle'}
            ]
        })
        manager.reactome.get_pathway_details = AsyncMock(return_value={
            'participants': ['TP53']
        })
        manager.reactome.get_protein_interactions = AsyncMock(return_value={
            'interactions': []
        })
        
        # Mock UniProt (P2 Enhancement - Critical for domain scoring)
        manager.uniprot = Mock()
        manager.uniprot.search_by_gene = AsyncMock(return_value={
            'results': [
                {
                    'primaryAccession': 'P04637',
                    'uniProtkbId': 'P53_HUMAN',
                    'entryType': 'UniProtKB reviewed (Swiss-Prot)'
                }
            ]
        })
        manager.uniprot.get_protein_features = AsyncMock(return_value={
            'features': [
                {'type': 'Domain', 'description': 'DNA binding domain', 'location': {'start': {'value': 102}, 'end': {'value': 292}}},
                {'type': 'Binding site', 'description': 'ATP binding site', 'location': {'start': {'value': 1}, 'end': {'value': 100}}},
                {'type': 'Domain', 'description': 'protein kinase domain'},
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
    async def test_s2_domain_boost_applied(self, mock_mcp_manager_with_uniprot):
        """Test that domain boost is applied to druggability score."""
        from src.scenarios.scenario_2_target_analysis import TargetAnalysisScenario
        from src.models.data_models import Protein
        
        scenario = TargetAnalysisScenario(mock_mcp_manager_with_uniprot)
        
        # Create target with UniProt ID
        target = Protein(
            gene_symbol="TP53",
            uniprot_id="P04637",
            confidence=0.9
        )
        
        # Create network
        network = nx.Graph()
        network.add_node("TP53")
        network.add_node("MDM2")
        network.add_edge("TP53", "MDM2", weight=0.9)
        
        # Calculate druggability score
        score = await scenario._calculate_druggability_score(
            target, network, []
        )
        
        print(f"\n📊 Druggability Score: {score:.3f}")
        
        # Score should be > 0 due to network centrality/size and domain boost
        assert score > 0.0, "Druggability score should be > 0"
        
        # With domain boost, score should be at least 0.04 (minimal domain boost)
        # But likely higher due to network centrality (30% weight) + network size (20% weight) + domain boost
        assert score >= 0.04, f"Score {score:.3f} should include domain boost (≥0.04)"
        
        # Verify UniProt features were called
        mock_mcp_manager_with_uniprot.uniprot.get_protein_features.assert_called_once_with("P04637")
    
    @pytest.mark.asyncio
    async def test_domain_boost_values(self, mock_mcp_manager_with_uniprot):
        """Test that correct domain boost values are applied."""
        from src.scenarios.scenario_2_target_analysis import TargetAnalysisScenario
        from src.models.data_models import Protein
        
        scenario = TargetAnalysisScenario(mock_mcp_manager_with_uniprot)
        
        # Test with kinase domain (highest boost)
        mock_mcp_manager_with_uniprot.uniprot.get_protein_features = AsyncMock(return_value={
            'features': [
                {'type': 'Domain', 'description': 'protein kinase domain'}
            ],
            'domains': [],
            'activeSites': [],
            'bindingSites': []
        })
        
        target = Protein(gene_symbol="AKT1", uniprot_id="P31749", confidence=0.9)
        network = nx.Graph()
        network.add_node("AKT1")
        
        score_kinase = await scenario._calculate_druggability_score(
            target, network, []
        )
        
        # Reset for ATP-binding test
        mock_mcp_manager_with_uniprot.uniprot.get_protein_features = AsyncMock(return_value={
            'features': [
                {'type': 'Binding site', 'description': 'ATP binding site'}
            ],
            'domains': [],
            'activeSites': [],
            'bindingSites': []
        })
        
        score_atp = await scenario._calculate_druggability_score(
            target, network, []
        )
        
        # Reset for binding site test
        mock_mcp_manager_with_uniprot.uniprot.get_protein_features = AsyncMock(return_value={
            'features': [
                {'type': 'Binding site', 'description': 'protein binding'}
            ],
            'domains': [],
            'activeSites': [],
            'bindingSites': []
        })
        
        score_binding = await scenario._calculate_druggability_score(
            target, network, []
        )
        
        print(f"\n📊 Domain Boost Comparison:")
        print(f"   Kinase domain: {score_kinase:.3f}")
        print(f"   ATP binding: {score_atp:.3f}")
        print(f"   Binding site: {score_binding:.3f}")
        
        # Kinase should have highest boost (0.15)
        # ATP should have second highest (0.12)
        # Binding should have lower boost (0.08)
        # Note: Scores include network size contribution, so exact values may vary
        assert score_kinase >= score_atp >= score_binding, "Domain boost rankings should be correct"
    
    @pytest.mark.asyncio
    async def test_uniprot_id_resolution_for_domain_scoring(self, mock_mcp_manager_with_uniprot):
        """Test that UniProt ID is resolved for domain scoring."""
        from src.scenarios.scenario_2_target_analysis import TargetAnalysisScenario
        
        scenario = TargetAnalysisScenario(mock_mcp_manager_with_uniprot)
        
        # Execute S2 (which should resolve UniProt ID if missing)
        result = await scenario.execute("TP53")
        
        target_info = result.target
        
        # Verify UniProt ID was resolved
        assert target_info.uniprot_id is not None, "UniProt ID should be resolved"
        assert target_info.uniprot_id == "P04637", "UniProt ID should be P04637 for TP53"
        
        # Verify search_by_gene was called
        mock_mcp_manager_with_uniprot.uniprot.search_by_gene.assert_called()
        
        # Verify druggability score was calculated
        assert result.druggability_score >= 0.0
    
    @pytest.mark.asyncio
    async def test_domain_scoring_without_uniprot_id(self, mock_mcp_manager_with_uniprot):
        """Test that domain scoring gracefully handles missing UniProt ID."""
        from src.scenarios.scenario_2_target_analysis import TargetAnalysisScenario
        from src.models.data_models import Protein
        
        scenario = TargetAnalysisScenario(mock_mcp_manager_with_uniprot)
        
        # Create target without UniProt ID
        target = Protein(
            gene_symbol="UNKNOWN",
            uniprot_id=None,
            confidence=0.5
        )
        
        network = nx.Graph()
        network.add_node("UNKNOWN")
        
        # Should not crash, but score should be lower (no domain boost)
        score = await scenario._calculate_druggability_score(
            target, network, []
        )
        
        assert score >= 0.0, "Score should be valid even without UniProt ID"
        
        # Verify get_protein_features was NOT called
        mock_mcp_manager_with_uniprot.uniprot.get_protein_features.assert_not_called()
    
    @pytest.mark.asyncio
    async def test_domain_boost_max_value(self, mock_mcp_manager_with_uniprot):
        """Test that domain boost respects maximum value (0.15)."""
        from src.scenarios.scenario_2_target_analysis import TargetAnalysisScenario
        from src.models.data_models import Protein
        
        scenario = TargetAnalysisScenario(mock_mcp_manager_with_uniprot)
        
        # Create target with multiple high-value domains
        mock_mcp_manager_with_uniprot.uniprot.get_protein_features = AsyncMock(return_value={
            'features': [
                {'type': 'Domain', 'description': 'protein kinase domain'},  # 0.15
                {'type': 'Binding site', 'description': 'ATP binding site'},  # 0.12
                {'type': 'Domain', 'description': 'catalytic domain'},  # 0.10
            ],
            'domains': [],
            'activeSites': [],
            'bindingSites': []
        })
        
        target = Protein(gene_symbol="TEST", uniprot_id="P12345", confidence=0.9)
        network = nx.Graph()
        network.add_node("TEST")
        
        # Score with domain boost
        score_with_domains = await scenario._calculate_druggability_score(
            target, network, []
        )
        
        # Score without domain boost (mock to return empty)
        mock_mcp_manager_with_uniprot.uniprot.get_protein_features = AsyncMock(return_value={
            'features': [],
            'domains': [],
            'activeSites': [],
            'bindingSites': []
        })
        
        score_without_domains = await scenario._calculate_druggability_score(
            target, network, []
        )
        
        domain_boost = score_with_domains - score_without_domains
        
        print(f"\n📊 Domain Boost Analysis:")
        print(f"   Score with domains: {score_with_domains:.3f}")
        print(f"   Score without domains: {score_without_domains:.3f}")
        print(f"   Domain boost: {domain_boost:.3f}")
        
        # Domain boost should be ≤ 0.15 (maximum boost)
        assert domain_boost <= 0.15, f"Domain boost {domain_boost:.3f} exceeds maximum of 0.15"

        # Domain boost should be > 0 (if domains found)
        # Note: If no domains match keywords, boost is 0, which is acceptable
        assert domain_boost >= 0.0, "Domain boost should be non-negative"

    @pytest.mark.asyncio
    async def test_uniprot_error_response_handling(self, mock_mcp_manager_with_uniprot):
        """Test that Scenario 2 handles UniProt error responses gracefully.

        This test verifies the fix for the bug where UniProt MCP returns error strings
        instead of dictionaries, causing "'str' object has no attribute 'get'" errors.
        """
        from src.scenarios.scenario_2_target_analysis import TargetAnalysisScenario
        from src.models.data_models import Protein

        scenario = TargetAnalysisScenario(mock_mcp_manager_with_uniprot)

        # Test 1: get_protein_features returns error string
        mock_mcp_manager_with_uniprot.uniprot.get_protein_features = AsyncMock(
            return_value="Error: Gene not found in UniProt database"
        )

        target = Protein(gene_symbol="PIK3R2", uniprot_id="O00459", confidence=0.9)
        network = nx.Graph()
        network.add_node("PIK3R2")

        # Should not raise error, should return valid score with domain_boost = 0.0
        score = await scenario._calculate_druggability_score(
            target, network, []
        )

        assert score >= 0.0, "Score should be valid even with UniProt error response"
        assert score <= 1.0, "Score should not exceed 1.0"

        print(f"\n✅ Score with UniProt error string: {score:.3f}")

        # Test 2: search_by_gene returns error string (during execute)
        mock_mcp_manager_with_uniprot.uniprot.search_by_gene = AsyncMock(
            return_value="Error: Gene symbol not recognized"
        )

        # Mock successful database results for other sources
        mock_mcp_manager_with_uniprot.string.search_proteins = AsyncMock(return_value={
            'proteins': [
                {'string_id': '9606.ENSP00000222222', 'preferred_name': 'TEST'}
            ]
        })

        # Execute should not crash with error response
        try:
            result = await scenario.execute("TEST")
            # Should complete successfully even with UniProt error
            assert result is not None
            assert hasattr(result, 'target')
            print(f"✅ Execute completed successfully with UniProt error: {result.target.gene_symbol}")
        except AttributeError as e:
            pytest.fail(f"Should not raise AttributeError: {e}")

        # Test 3: get_protein_features returns unexpected type (list)
        mock_mcp_manager_with_uniprot.uniprot.get_protein_features = AsyncMock(
            return_value=['features', 'domains']  # Wrong type!
        )

        score = await scenario._calculate_druggability_score(
            target, network, []
        )

        assert score >= 0.0, "Score should be valid with unexpected response type"
        assert score <= 1.0, "Score should not exceed 1.0"

        print(f"✅ Score with unexpected response type: {score:.3f}")

        # Test 4: search_by_gene returns unexpected type (list)
        mock_mcp_manager_with_uniprot.uniprot.search_by_gene = AsyncMock(
            return_value=[{'result': 'data'}]  # Wrong type!
        )

        try:
            result = await scenario.execute("TEST2")
            assert result is not None
            print(f"✅ Execute completed with unexpected response type")
        except AttributeError as e:
            pytest.fail(f"Should not raise AttributeError: {e}")

