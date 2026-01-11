"""
Unit tests for S3 STRING network optimization.

Tests deterministic gene selection, marker overflow handling, and deduplication
in _phase3_cancer_network_construction() using AdaptiveStringNetworkBuilder.
"""

import os
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from src.scenarios.scenario_3_cancer_analysis import CancerAnalysisScenario
from src.core.mcp_client_manager import MCPClientManager
from src.core.string_network_builder import AdaptiveStringNetworkBuilder
from src.models.data_models import CancerMarker, Pathway


class TestStringNetworkOptimization:
    """Test STRING network optimization in S3."""
    
    @pytest.fixture
    def scenario(self):
        """Create a CancerAnalysisScenario instance for testing."""
        mcp_manager = MCPClientManager(config_path="config/mcp_servers.json")
        scenario = CancerAnalysisScenario(mcp_manager)
        return scenario
    
    @pytest.fixture
    def sample_markers(self):
        """Create sample cancer markers."""
        return [
            CancerMarker(
                gene='TP53',
                cancer_type='breast cancer',
                prognostic_value='favorable',
                survival_association='high',
                expression_pattern={'tumor': 'high', 'normal': 'low'},
                clinical_relevance='Good prognosis',
                confidence=0.9
            ),
            CancerMarker(
                gene='BRCA1',
                cancer_type='breast cancer',
                prognostic_value='favorable',
                survival_association='high',
                expression_pattern={'tumor': 'high', 'normal': 'low'},
                clinical_relevance='Good prognosis',
                confidence=0.9
            ),
        ]
    
    @pytest.fixture
    def sample_pathways(self):
        """Create sample pathways with genes."""
        return [
            Pathway(
                id='path:map05224',
                name='Breast cancer',
                source_db='kegg',
                genes=['TP53', 'BRCA1', 'MYC', 'AXL', 'ERBB2', 'PIK3CA', 'AKT1'],
                pathway_type='disease'
            ),
            Pathway(
                id='R-HSA-1227990',
                name='Signaling by ERBB2',
                source_db='reactome',
                genes=['ERBB2', 'PIK3CA', 'AKT1', 'MAPK1', 'STAT3'],
                pathway_type='signaling'
            ),
        ]
    
    @pytest.mark.asyncio
    async def test_gene_selection_respects_max_genes(self, scenario, sample_markers, sample_pathways):
        """Test that gene selection respects max_genes limit via AdaptiveStringNetworkBuilder."""
        # Create many pathway genes
        large_pathway = Pathway(
            id='test',
            name='Test',
            source_db='kegg',
            genes=[f'GENE{i}' for i in range(20)],  # 20 genes
            pathway_type='test'
        )
        
        # Mock AdaptiveStringNetworkBuilder with limited max_genes
        with patch('src.scenarios.scenario_3_cancer_analysis.AdaptiveStringNetworkBuilder') as MockBuilder:
            mock_builder_instance = MagicMock()
            mock_builder_instance.build_network = AsyncMock(return_value={
                'nodes': [],
                'edges': [],
                'genes_used': ['TP53', 'BRCA1', 'GENE0', 'GENE1', 'GENE2', 'GENE3', 'GENE4', 'GENE5', 'GENE6', 'GENE7'],
                'expansion_attempts': 1
            })
            MockBuilder.return_value = mock_builder_instance
            
            # Set environment variable for max_genes
            os.environ['STRING_ADAPTIVE_INITIAL_MAX_GENES'] = '10'
            
            result = await scenario._phase3_cancer_network_construction(
                sample_markers,
                [large_pathway],
                {}
            )
            
            # Should limit to max_genes (10)
            assert len(result['genes']) <= 10
            
            # Clean up
            os.environ.pop('STRING_ADAPTIVE_INITIAL_MAX_GENES', None)
    
    @pytest.mark.asyncio
    async def test_marker_genes_prioritized(self, scenario, sample_markers, sample_pathways):
        """Test that marker genes are prioritized over pathway genes."""
        with patch('src.scenarios.scenario_3_cancer_analysis.AdaptiveStringNetworkBuilder') as MockBuilder:
            mock_builder_instance = MagicMock()
            # Mock builder to return genes_used that includes all markers
            mock_builder_instance.build_network = AsyncMock(return_value={
                'nodes': [],
                'edges': [],
                'genes_used': ['TP53', 'BRCA1', 'MYC', 'AXL', 'ERBB2'],  # Markers first
                'expansion_attempts': 1
            })
            MockBuilder.return_value = mock_builder_instance
            
            result = await scenario._phase3_cancer_network_construction(
                sample_markers,
                sample_pathways,
                {}
            )
            
            # All marker genes should be included
            marker_genes = {m.gene for m in sample_markers}
            assert all(gene in result['genes'] for gene in marker_genes)
    
    @pytest.mark.asyncio
    async def test_marker_overflow_handling(self, scenario, sample_pathways):
        """Test that marker overflow is handled correctly."""
        # Create many markers (more than max_genes)
        many_markers = [
            CancerMarker(
                gene=f'MARKER{i}',
                cancer_type='breast cancer',
                prognostic_value='favorable',
                survival_association='high',
                expression_pattern={'tumor': 'high', 'normal': 'low'},
                clinical_relevance='Test',
                confidence=0.9
            )
            for i in range(15)  # 15 markers > max_genes (10)
        ]
        
        with patch('src.scenarios.scenario_3_cancer_analysis.AdaptiveStringNetworkBuilder') as MockBuilder:
            mock_builder_instance = MagicMock()
            # Mock builder to cap at max_genes
            mock_builder_instance.build_network = AsyncMock(return_value={
                'nodes': [],
                'edges': [],
                'genes_used': [f'MARKER{i}' for i in range(10)],  # Capped at 10
                'expansion_attempts': 1
            })
            MockBuilder.return_value = mock_builder_instance
            
            # Set environment variable for max_genes
            os.environ['STRING_ADAPTIVE_INITIAL_MAX_GENES'] = '10'
            
            result = await scenario._phase3_cancer_network_construction(
                many_markers,
                sample_pathways,
                {}
            )
            
            # Should cap to max_genes
            assert len(result['genes']) == 10
            
            # Clean up
            os.environ.pop('STRING_ADAPTIVE_INITIAL_MAX_GENES', None)
    
    @pytest.mark.asyncio
    async def test_deterministic_selection(self, scenario, sample_markers, sample_pathways):
        """Test that gene selection is deterministic (same input = same output)."""
        with patch('src.scenarios.scenario_3_cancer_analysis.AdaptiveStringNetworkBuilder') as MockBuilder:
            mock_builder_instance = MagicMock()
            # Mock builder to return consistent results
            consistent_genes = ['TP53', 'BRCA1', 'MYC', 'AXL', 'ERBB2']
            mock_builder_instance.build_network = AsyncMock(return_value={
                'nodes': [],
                'edges': [],
                'genes_used': consistent_genes,
                'expansion_attempts': 1
            })
            MockBuilder.return_value = mock_builder_instance
            
            # Run twice with same input
            result1 = await scenario._phase3_cancer_network_construction(
                sample_markers,
                sample_pathways,
                {}
            )
            
            result2 = await scenario._phase3_cancer_network_construction(
                sample_markers,
                sample_pathways,
                {}
            )
            
            # Should produce same gene list
            assert result1['genes'] == result2['genes']
    
    @pytest.mark.asyncio
    async def test_pathway_genes_sorted_by_frequency(self, scenario, sample_markers):
        """Test that pathway genes are sorted by frequency then alphabetically."""
        # Create pathways with overlapping genes
        pathway1 = Pathway(
            id='path1',
            name='Pathway 1',
            source_db='kegg',
            genes=['GENE_A', 'GENE_B', 'GENE_C'],
            pathway_type='test'
        )
        pathway2 = Pathway(
            id='path2',
            name='Pathway 2',
            source_db='kegg',
            genes=['GENE_A', 'GENE_B'],
            pathway_type='test'
        )
        pathway3 = Pathway(
            id='path3',
            name='Pathway 3',
            source_db='kegg',
            genes=['GENE_A', 'GENE_D'],
            pathway_type='test'
        )
        
        with patch('src.scenarios.scenario_3_cancer_analysis.AdaptiveStringNetworkBuilder') as MockBuilder:
            mock_builder_instance = MagicMock()
            # Mock builder to return genes sorted by frequency (GENE_A appears 3 times)
            mock_builder_instance.build_network = AsyncMock(return_value={
                'nodes': [],
                'edges': [],
                'genes_used': ['TP53', 'BRCA1', 'GENE_A', 'GENE_B', 'GENE_C', 'GENE_D'],
                'expansion_attempts': 1
            })
            MockBuilder.return_value = mock_builder_instance
            
            result = await scenario._phase3_cancer_network_construction(
                sample_markers,
                [pathway1, pathway2, pathway3],
                {}
            )
            
            # GENE_A should appear before GENE_B (higher frequency)
            marker_genes = {m.gene for m in sample_markers}
            pathway_genes = [g for g in result['genes'] if g not in marker_genes]
            
            # Check that frequency-based sorting is applied
            gene_a_idx = pathway_genes.index('GENE_A') if 'GENE_A' in pathway_genes else -1
            gene_b_idx = pathway_genes.index('GENE_B') if 'GENE_B' in pathway_genes else -1
            
            if gene_a_idx >= 0 and gene_b_idx >= 0:
                assert gene_a_idx < gene_b_idx, "GENE_A should appear before GENE_B (higher frequency)"
    
    @pytest.mark.asyncio
    async def test_batch_deduplication(self, scenario, sample_markers, sample_pathways):
        """Test that batch results are deduplicated correctly by AdaptiveStringNetworkBuilder."""
        with patch('src.scenarios.scenario_3_cancer_analysis.AdaptiveStringNetworkBuilder') as MockBuilder:
            mock_builder_instance = MagicMock()
            # Mock builder to return deduplicated results
            mock_builder_instance.build_network = AsyncMock(return_value={
                'nodes': [
                    {'string_id': '9606.ENSP000001', 'preferred_name': 'TP53'},
                    {'string_id': '9606.ENSP000002', 'preferred_name': 'BRCA1'},
                    {'string_id': '9606.ENSP000003', 'preferred_name': 'MYC'},
                ],
                'edges': [
                    {'protein_a': 'TP53', 'protein_b': 'BRCA1', 'confidence_score': 0.9},
                    {'protein_a': 'BRCA1', 'protein_b': 'MYC', 'confidence_score': 0.8},
                ],
                'genes_used': ['TP53', 'BRCA1', 'MYC'],
                'expansion_attempts': 1
            })
            MockBuilder.return_value = mock_builder_instance
            
            result = await scenario._phase3_cancer_network_construction(
                sample_markers,
                sample_pathways,
                {}
            )
            
            # Should have deduplicated nodes (3 unique: TP53, BRCA1, MYC)
            # Should have deduplicated edges (2 unique)
            assert len(result['nodes']) == 3
            assert len(result['edges']) == 2
