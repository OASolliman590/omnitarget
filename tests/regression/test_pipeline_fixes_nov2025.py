"""
Regression tests for pipeline fixes from November 2025 forensic analysis.

These tests prevent regression of critical bugs identified in run_latest.txt:
- P0-1: Scenario 1 UnboundLocalError in edge weight calculation
- P1-2: HPA chunk size limit errors
- P0-2: Scenario 4 simulation exception handling
"""

import pytest
import asyncio
import networkx as nx
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from src.scenarios.scenario_1_disease_network import DiseaseNetworkScenario
from src.scenarios.scenario_4_mra_simulation import MultiTargetSimulationScenario
from src.mcp_clients.hpa_client import HPAClient
from src.models.data_models import Protein


class TestScenario1EdgeWeightFix:
    """Test fix for Scenario 1 UnboundLocalError (P0-1)."""

    @pytest.mark.asyncio
    async def test_edge_weight_initialization_with_missing_nodes(self):
        """
        Test that edge weight calculation handles missing nodes gracefully.

        Regression test for bug where 'weight' variable was used before assignment
        when nodes were missing from the graph.
        """
        # Create mock MCP manager
        mock_mcp = Mock()
        scenario = DiseaseNetworkScenario(mcp_manager=mock_mcp)

        # Create a graph with only some nodes
        G = nx.Graph()
        G.add_node("GENE1", name="GENE1")
        G.add_node("GENE2", name="GENE2")
        # GENE3 is intentionally missing

        # Create edge data including an edge with missing target node
        edges = [
            {
                "protein_a": "GENE1",
                "protein_b": "GENE2",
                "confidence_score": 0.8
            },
            {
                "protein_a": "GENE1",
                "protein_b": "GENE3",  # This node doesn't exist in G
                "confidence_score": 0.9
            }
        ]

        # Mock the pathway context weight method
        scenario._get_pathway_context_weight = Mock(return_value=1.0)

        # This should NOT raise UnboundLocalError anymore
        # The fix moves weight calculation inside the if block
        result = await scenario._phase3_network_construction(
            G,
            edges,
            {},  # pathways
            []   # disease_genes
        )

        # Verify the edge with valid nodes was added
        assert result.has_edge("GENE1", "GENE2"), "Valid edge should be added"

        # Verify the edge with missing node was NOT added (no UnboundLocalError)
        assert not result.has_edge("GENE1", "GENE3"), "Edge with missing node should not be added"

        # Verify graph has correct number of edges (only 1, not 2)
        assert result.number_of_edges() == 1, "Only edges with valid nodes should be added"


class TestHPAChunkLimitFix:
    """Test fix for HPA chunk size limit errors (P1-2)."""

    @pytest.mark.asyncio
    async def test_hpa_chunk_limit_fallback_separator_not_found(self):
        """
        Test HPA gracefully handles 'separator is not found' chunk limit error.

        Regression test for HPA genes that exceed JSON-RPC chunk limit.
        """
        client = HPAClient("/fake/path")
        client.call_tool_with_retry = AsyncMock()
        client.get_expression_summary = AsyncMock(return_value={
            "gene": "AN",
            "tissues": [{"tissue": "brain", "level": "high"}]
        })

        # Simulate chunk limit error with "separator is not found" pattern
        client.call_tool_with_retry.side_effect = Exception(
            "Programming error in request/response handling: ValueError: "
            "Separator is not found, and chunk exceed the limit"
        )

        # Should fallback to expression summary, not raise
        result = await client.get_tissue_expression("AN")

        assert result is not None, "Should return result, not raise"
        assert result.get("fallback_used") == True, "Should indicate fallback was used"
        assert "AN" in result.get("gene", ""), "Should return correct gene"
        assert len(result.get("tissues", [])) > 0, "Should have tissue data from fallback"

    @pytest.mark.asyncio
    async def test_hpa_chunk_limit_fallback_chunk_longer_than_limit(self):
        """
        Test HPA handles 'chunk is longer than limit' error.
        """
        client = HPAClient("/fake/path")
        client.call_tool_with_retry = AsyncMock()
        client.get_expression_summary = AsyncMock(return_value={
            "gene": "LARGE_GENE",
            "tissues": [{"tissue": "liver", "level": "medium"}]
        })

        # Simulate the other chunk limit error pattern
        client.call_tool_with_retry.side_effect = Exception(
            "Separator is found, but chunk is longer than limit"
        )

        result = await client.get_tissue_expression("LARGE_GENE")

        assert result.get("fallback_used") == True
        assert "LARGE_GENE" in result.get("gene", "")

    @pytest.mark.asyncio
    async def test_hpa_double_fallback_failure_returns_minimal_structure(self):
        """
        Test that when both primary and fallback fail, minimal structure is returned.

        This allows the pipeline to continue instead of crashing.
        """
        client = HPAClient("/fake/path")
        client.call_tool_with_retry = AsyncMock()

        # Both primary and fallback fail
        client.call_tool_with_retry.side_effect = Exception("Chunk exceed the limit")
        client.get_expression_summary = AsyncMock(
            side_effect=Exception("Summary also failed")
        )

        result = await client.get_tissue_expression("PROBLEMATIC_GENE")

        # Should return minimal structure, not raise
        assert result is not None
        assert result.get("gene") == "PROBLEMATIC_GENE"
        assert result.get("tissues") == []
        assert result.get("fallback_failed") == True
        assert "error" in result


class TestScenario4MemoryTracking:
    """Test Scenario 4 enhanced exception handling and memory tracking."""

    @pytest.mark.asyncio
    async def test_scenario4_logs_memory_during_simulation(self):
        """
        Test that Scenario 4 tracks memory usage during simulation.

        This helps diagnose pipeline truncation issues.
        """
        mock_mcp = Mock()
        scenario = MultiTargetSimulationScenario(mcp_manager=mock_mcp)

        # Create minimal test data
        network = nx.Graph()
        network.add_edge("GENE1", "GENE2", weight=0.8)

        targets = [
            Protein(gene_symbol="GENE1", uniprot_id="P12345", name="Gene 1")
        ]

        # Mock the simulation methods
        scenario._simulate_network_propagation = Mock(return_value={
            "affected_nodes": {"GENE2": 0.5},
            "network_impact": 0.6,
            "confidence_scores": {"GENE2": 0.7},
            "downstream": ["GENE2"],
            "upstream": [],
            "feedback_loops": [],
            "direct_targets": ["GENE2"],
            "execution_time": 0.1
        })
        scenario._get_biological_annotation = Mock(return_value={})
        scenario._get_drug_annotations = Mock(return_value={})
        scenario._calculate_synergy_matrix = Mock(return_value={
            "pairwise_synergy": {},
            "overall_synergy": 0.0,
            "top_synergistic_pairs": []
        })
        scenario._assess_network_wide_impact = Mock(return_value={})
        scenario._calculate_simulation_convergence = Mock(return_value=0.95)

        # Run simulation - should log memory without crashing
        with patch('src.scenarios.scenario_4_mra_simulation.logger') as mock_logger:
            result = await scenario._step7_simulation(
                network,
                targets,
                simulation_mode="simple",
                tissue_context="breast"
            )

            # Verify memory logging occurred
            memory_logs = [
                call for call in mock_logger.info.call_args_list
                if 'memory' in str(call).lower()
            ]
            assert len(memory_logs) > 0, "Should log memory usage"

            # Verify simulation completed
            assert "simulation_results" in result
            assert len(result["simulation_results"]) == 1

    @pytest.mark.asyncio
    async def test_scenario4_logs_full_exception_traceback(self):
        """
        Test that exceptions in Step 7 are logged with full stack trace.

        Regression test for enhanced exception logging (exc_info=True).
        """
        mock_mcp = Mock()
        scenario = MultiTargetSimulationScenario(mcp_manager=mock_mcp)

        network = nx.Graph()
        network.add_edge("GENE1", "GENE2")

        targets = [
            Protein(gene_symbol="GENE1", uniprot_id="P12345", name="Gene 1")
        ]

        # Simulate an exception during propagation
        test_exception = RuntimeError("Test simulation error")
        scenario._simulate_network_propagation = Mock(side_effect=test_exception)
        scenario._fallback_simulation = Mock(return_value={
            "target": "GENE1",
            "affected_nodes": {},
            "direct_targets": [],
            "downstream": [],
            "upstream": [],
            "feedback_loops": [],
            "network_impact": {},
            "confidence_scores": {},
            "biological_context": {},
            "drug_info": {},
            "execution_time": 0.0
        })
        scenario._calculate_synergy_matrix = Mock(return_value={
            "pairwise_synergy": {},
            "overall_synergy": 0.0,
            "top_synergistic_pairs": []
        })
        scenario._assess_network_wide_impact = Mock(return_value={})
        scenario._calculate_simulation_convergence = Mock(return_value=0.0)

        # Should not raise, should use fallback
        with patch('src.scenarios.scenario_4_mra_simulation.logger') as mock_logger:
            result = await scenario._step7_simulation(
                network,
                targets,
                simulation_mode="simple",
                tissue_context="breast"
            )

            # Verify error was logged with exc_info=True
            error_logs = [
                call for call in mock_logger.error.call_args_list
                if 'Enhanced simulation failed' in str(call)
            ]
            assert len(error_logs) > 0, "Should log error"

            # Verify exc_info=True was used
            for call in error_logs:
                _, kwargs = call
                assert kwargs.get('exc_info') == True, "Should log with exc_info=True"

            # Verify fallback was used
            assert scenario._fallback_simulation.called, "Should use fallback"


class TestReactomeTimeoutIncrease:
    """Test Reactome timeout increase (P1-1)."""

    def test_reactome_client_has_90s_timeout(self):
        """
        Test that ReactomeClient is initialized with 90s timeout.

        Regression test for increased timeout to handle large disease queries.
        """
        from src.mcp_clients.reactome_client import ReactomeClient

        client = ReactomeClient("/fake/path")

        # Verify timeout was increased to 90s
        assert client.timeout == 90, "ReactomeClient should have 90s timeout"


class TestKEGGClientOptimization:
    """Test KEGG client no longer tries non-existent tool (P2-1)."""

    @pytest.mark.asyncio
    async def test_kegg_uses_fallback_directly(self):
        """
        Test that get_gene_pathways uses find_related_entries directly.

        Regression test to ensure we don't waste time trying non-existent tool.
        """
        from src.mcp_clients.kegg_client import KEGGClient

        client = KEGGClient("/fake/path")
        client.find_related_entries = AsyncMock(return_value={
            "source_db": "gene",
            "target_db": "pathway",
            "link_count": 1,
            "links": {"hsa:1234": ["path:hsa00010"]}
        })

        # Should call find_related_entries without trying get_gene_pathways first
        result = await client.get_gene_pathways("hsa:1234")

        # Verify find_related_entries was called
        assert client.find_related_entries.called

        # Verify result structure
        assert result["gene_id"] == "hsa:1234"
        assert "pathways" in result
        assert result["source"] == "kegg_fallback_find_related"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
