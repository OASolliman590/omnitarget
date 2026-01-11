"""
Unit tests for gene validation in Scenario 3 - Cancer Analysis.
"""
import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, MagicMock

from src.scenarios.scenario_3_cancer_analysis import CancerAnalysisScenario
from src.core.mcp_client_manager import MCPClientManager


class TestGeneValidation:
    """Test gene symbol validation."""

    @pytest.fixture
    def mcp_manager(self):
        """Create mock MCP manager."""
        return Mock(spec=MCPClientManager)

    @pytest.fixture
    def cancer_scenario(self, mcp_manager):
        """Create cancer analysis scenario with mocked MCP manager."""
        return CancerAnalysisScenario(mcp_manager)

    def test_validate_gene_symbol_valid(self, cancer_scenario):
        """Test validation of valid gene symbols."""
        # Valid genes - standard format
        assert cancer_scenario._validate_gene_symbol("AXL") == (True, "AXL")
        assert cancer_scenario._validate_gene_symbol("BRCA1") == (True, "BRCA1")
        assert cancer_scenario._validate_gene_symbol("TP53") == (True, "TP53")
        assert cancer_scenario._validate_gene_symbol("PIK3R2") == (True, "PIK3R2")
        assert cancer_scenario._validate_gene_symbol("APC2") == (True, "APC2")
        assert cancer_scenario._validate_gene_symbol("FZD5") == (True, "FZD5")
        assert cancer_scenario._validate_gene_symbol("FOS") == (True, "FOS")
        assert cancer_scenario._validate_gene_symbol("FGF1") == (True, "FGF1")
        assert cancer_scenario._validate_gene_symbol("NOTCH3") == (True, "NOTCH3")
        assert cancer_scenario._validate_gene_symbol("GADD45A") == (True, "GADD45A")

    def test_validate_gene_symbol_normalization(self, cancer_scenario):
        """Test gene symbol normalization."""
        # Whitespace normalization
        assert cancer_scenario._validate_gene_symbol("  tp53  ") == (True, "TP53")
        assert cancer_scenario._validate_gene_symbol("axl ") == (True, "AXL")
        assert cancer_scenario._validate_gene_symbol("  BRCA1") == (True, "BRCA1")

        # Case normalization
        assert cancer_scenario._validate_gene_symbol("axl") == (True, "AXL")
        assert cancer_scenario._validate_gene_symbol("bRca1") == (True, "BRCA1")
        assert cancer_scenario._validate_gene_symbol("tp53") == (True, "TP53")

    def test_validate_gene_symbol_invalid(self, cancer_scenario):
        """Test rejection of invalid gene symbols."""
        # Empty/None
        assert cancer_scenario._validate_gene_symbol("") == (False, "")
        assert cancer_scenario._validate_gene_symbol("   ") == (False, "")

        # Too short
        assert cancer_scenario._validate_gene_symbol("A") == (False, "A")
        assert cancer_scenario._validate_gene_symbol("B") == (False, "B")

        # Too long
        assert cancer_scenario._validate_gene_symbol("12345678901234567890") == (False, "12345678901234567890")
        assert cancer_scenario._validate_gene_symbol("VERY_LONG_GENE_NAME") == (False, "VERY_LONG_GENE_NAME")

        # Invalid characters (non-alphanumeric except underscore)
        assert cancer_scenario._validate_gene_symbol("GENE-1") == (False, "GENE-1")
        assert cancer_scenario._validate_gene_symbol("GENE.1") == (False, "GENE.1")
        assert cancer_scenario._validate_gene_symbol("GENE@1") == (False, "GENE@1")
        assert cancer_scenario._validate_gene_symbol("GENE!1") == (False, "GENE!1")

    def test_validate_gene_symbol_edge_cases(self, cancer_scenario):
        """Test edge cases for gene validation."""
        # Underscores allowed
        assert cancer_scenario._validate_gene_symbol("GENE_1") == (True, "GENE_1")
        assert cancer_scenario._validate_gene_symbol("GENE_NAME") == (True, "GENE_NAME")

        # Mixed alphanumeric with underscores
        assert cancer_scenario._validate_gene_symbol("GENE_123") == (True, "GENE_123")
        assert cancer_scenario._validate_gene_symbol("123_GENE") == (True, "123_GENE")

        # Boundary lengths
        assert cancer_scenario._validate_gene_symbol("AB") == (True, "AB")  # Min length
        assert cancer_scenario._validate_gene_symbol("ABCDEFGHIJKLMNO") == (True, "ABCDEFGHIJKLMNO")  # Max length (15)


class TestPathologyWithFallback:
    """Test pathology data retrieval with fallback."""

    @pytest.fixture
    def mcp_manager(self):
        """Create mock MCP manager with HPA client."""
        manager = Mock(spec=MCPClientManager)
        manager.hpa = Mock()
        return manager

    @pytest.fixture
    def cancer_scenario(self, mcp_manager):
        """Create cancer analysis scenario."""
        return CancerAnalysisScenario(mcp_manager)

    @pytest.mark.asyncio
    async def test_hpa_success(self, cancer_scenario, mcp_manager):
        """Test successful HPA response."""
        # Mock HPA response
        mcp_manager.hpa.get_pathology_data = AsyncMock(return_value={
            "gene": "TP53",
            "markers": [{"gene": "TP53", "prognostic": "unfavorable"}]
        })

        result = await cancer_scenario._get_pathology_with_fallback("TP53")

        assert result is not None
        assert result["gene"] == "TP53"
        assert len(result["markers"]) > 0

    @pytest.mark.asyncio
    async def test_invalid_gene_format(self, cancer_scenario, mcp_manager):
        """Test invalid gene format handling."""
        result = await cancer_scenario._get_pathology_with_fallback("")

        assert result is None

    @pytest.mark.asyncio
    async def test_hpa_invalid_gene_error(self, cancer_scenario, mcp_manager):
        """Test HPA 'Invalid gene arguments' error handling."""
        # Mock HPA error
        mcp_manager.hpa.get_pathology_data = AsyncMock(
            side_effect=Exception("Invalid gene arguments (error -32602)")
        )

        result = await cancer_scenario._get_pathology_with_fallback("INVALID_GENE")

        assert result is None

    @pytest.mark.asyncio
    async def test_hpa_error_code_32602(self, cancer_scenario, mcp_manager):
        """Test HPA error code -32602 handling."""
        mcp_manager.hpa.get_pathology_data = AsyncMock(
            side_effect=Exception("Error -32602: Invalid gene arguments")
        )

        result = await cancer_scenario._get_pathology_with_fallback("PIK3R2")

        assert result is None

    @pytest.mark.asyncio
    async def test_hpa_fallback_to_tcga(self, cancer_scenario, mcp_manager):
        """Test fallback from HPA to TCGA."""
        # Mock HPA to fail
        mcp_manager.hpa.get_pathology_data = AsyncMock(
            side_effect=Exception("Invalid gene arguments")
        )

        # Mock TCGA to succeed
        mcp_manager.tcga = Mock()
        mcp_manager.tcga.get_cancer_markers = AsyncMock(return_value={
            "markers": [{"gene": "PIK3R2", "prognostic": "unfavorable"}]
        })

        result = await cancer_scenario._get_pathology_with_fallback("PIK3R2")

        assert result is not None
        assert result["gene"] == "PIK3R2"
        assert result["source"] == "TCGA"
        assert len(result["markers"]) > 0

    @pytest.mark.asyncio
    async def test_tcga_not_available(self, cancer_scenario, mcp_manager):
        """Test graceful handling when TCGA is not available."""
        # Mock HPA to fail
        mcp_manager.hpa.get_pathology_data = AsyncMock(
            side_effect=Exception("Invalid gene arguments")
        )

        # TCGA not available (no tcga attribute)
        result = await cancer_scenario._get_pathology_with_fallback("PIK3R2")

        assert result is None

    @pytest.mark.asyncio
    async def test_both_hpa_and_tcga_fail(self, cancer_scenario, mcp_manager):
        """Test when both HPA and TCGA fail."""
        # Mock HPA to fail
        mcp_manager.hpa.get_pathology_data = AsyncMock(
            side_effect=Exception("Invalid gene arguments")
        )

        # Mock TCGA to fail
        mcp_manager.tcga = Mock()
        mcp_manager.tcga.get_cancer_markers = AsyncMock(
            side_effect=Exception("TCGA error")
        )

        result = await cancer_scenario._get_pathology_with_fallback("PIK3R2")

        assert result is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
