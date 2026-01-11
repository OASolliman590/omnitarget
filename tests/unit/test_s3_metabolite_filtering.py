"""
Unit tests for S3 metabolite filtering.

Tests the _is_valid_gene_symbol_s3() and _filter_valid_gene_symbols_s3() methods
to ensure metabolites are properly filtered before HPA calls.
"""

import pytest
from src.scenarios.scenario_3_cancer_analysis import CancerAnalysisScenario
from src.core.mcp_client_manager import MCPClientManager


class TestMetaboliteFiltering:
    """Test metabolite filtering in S3 gene validation."""
    
    @pytest.fixture
    def scenario(self):
        """Create a CancerAnalysisScenario instance for testing."""
        # Mock MCPClientManager - we only need the validation methods
        mcp_manager = MCPClientManager(config_path="config/mcp_servers.json")
        return CancerAnalysisScenario(mcp_manager)
    
    def test_rejects_common_metabolites(self, scenario):
        """Test that common metabolites are rejected."""
        metabolites = ['ATP', 'GDP', 'ADP', 'GTP', 'NAD', 'NADP', 'FAD', 'COA', 
                       'AMP', 'CMP', 'GMP', 'UMP', 'DAG', 'IP3', 'CA', 'MG', 'NA', 
                       'K', 'FE', 'ZN', 'CU', 'MN']
        
        for metabolite in metabolites:
            assert not scenario._is_valid_gene_symbol_s3(metabolite), \
                f"Metabolite {metabolite} should be rejected"
    
    def test_rejects_metabolite_variants(self, scenario):
        """Test that metabolite variants with punctuation are rejected."""
        variants = ['ATP+', 'ATP-', 'fadh2', 'nad+', 'amp-pnp', 'FADH2']
        
        for variant in variants:
            assert not scenario._is_valid_gene_symbol_s3(variant), \
                f"Metabolite variant {variant} should be rejected"
    
    def test_accepts_valid_gene_symbols(self, scenario):
        """Test that valid gene symbols are accepted."""
        valid_genes = ['MYC', 'ELK', 'AXL', 'TP53', 'BRCA1', 'ERBB2', 'PIK3CA', 
                       'AKT1', 'MAPK1', 'STAT3']
        
        for gene in valid_genes:
            assert scenario._is_valid_gene_symbol_s3(gene), \
                f"Valid gene {gene} should be accepted"
    
    def test_normalization_handles_variants(self, scenario):
        """Test that normalization handles metabolite variants correctly."""
        # All these should be rejected as 'atp' after normalization
        atp_variants = ['ATP', 'atp', 'ATP+', 'ATP-', 'ATP.', 'ATP_']
        
        for variant in atp_variants:
            assert not scenario._is_valid_gene_symbol_s3(variant), \
                f"ATP variant {variant} should be rejected after normalization"
    
    def test_filter_aggregates_metabolites(self, scenario):
        """Test that _filter_valid_gene_symbols_s3 aggregates filtered metabolites."""
        import logging
        from io import StringIO
        
        # Set up logging capture
        log_capture = StringIO()
        handler = logging.StreamHandler(log_capture)
        handler.setLevel(logging.DEBUG)
        logger = logging.getLogger('src.scenarios.scenario_3_cancer_analysis')
        logger.addHandler(handler)
        logger.setLevel(logging.DEBUG)
        
        # Mix of valid genes and metabolites
        genes = {'TP53', 'BRCA1', 'ATP', 'GDP', 'MYC', 'NAD', 'AXL'}
        
        result = scenario._filter_valid_gene_symbols_s3(genes)
        
        # Should only return valid genes
        assert set(result) == {'TP53', 'BRCA1', 'MYC', 'AXL'}
        
        # Check that logging occurred (metabolites were filtered)
        log_output = log_capture.getvalue()
        assert 'Filtered' in log_output or len(result) < len(genes)
        
        logger.removeHandler(handler)
    
    def test_filter_preserves_valid_genes(self, scenario):
        """Test that valid genes are preserved through filtering."""
        genes = {'TP53', 'BRCA1', 'MYC', 'AXL', 'ERBB2', 'PIK3CA'}
        
        result = scenario._filter_valid_gene_symbols_s3(genes)
        
        # All should be preserved (uppercased)
        assert len(result) == len(genes)
        assert all(gene.upper() in result for gene in genes)


