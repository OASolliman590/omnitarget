"""
Production Tests with Real MCP Servers

Tests that validate actual MCP server functionality and scientific accuracy.
These tests require real MCP servers to be running and may take 10-60 minutes.
"""

import pytest
import asyncio
import time
import os
from typing import Dict, List, Any, Optional

from src.core.mcp_client_manager import MCPClientManager
from src.core.data_standardizer import DataStandardizer
from src.core.validation import DataValidator
from src.utils.id_mapping import IDMapper
from src.scenarios.scenario_1_disease_network import DiseaseNetworkScenario
from src.scenarios.scenario_2_target_analysis import TargetAnalysisScenario
from src.scenarios.scenario_3_cancer_analysis import CancerAnalysisScenario
from src.core.pipeline_orchestrator import OmniTargetPipeline


class TestRealMCPServerHealth:
    """Test real MCP server health and connectivity."""
    
    @pytest.fixture(scope="class")
    async def real_mcp_manager(self):
        """Create real MCP manager with actual servers."""
        config_path = "config/mcp_servers.json"
        manager = MCPClientManager(config_path)
        await manager.initialize()
        yield manager
        await manager.cleanup()
    
    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_kegg_server_health(self, real_mcp_manager):
        """Test KEGG MCP server health and basic functionality."""
        start_time = time.time()
        
        try:
            # Test server connection
            kegg_client = await real_mcp_manager.get_client("kegg")
            assert kegg_client is not None
            
            # Test basic API call
            result = await kegg_client.search_diseases("breast cancer")
            assert result is not None
            assert isinstance(result, list)
            assert len(result) > 0
            
            # Test pathway genes retrieval
            pathway_result = await kegg_client.get_pathway_genes("hsa05224")
            assert pathway_result is not None
            assert "genes" in pathway_result
            assert len(pathway_result["genes"]) > 0
            
            execution_time = time.time() - start_time
            print(f"KEGG server test completed in {execution_time:.2f}s")
            assert execution_time < 30.0  # Should complete in <30 seconds
            
        except Exception as e:
            pytest.fail(f"KEGG server health check failed: {e}")
    
    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_reactome_server_health(self, real_mcp_manager):
        """Test Reactome MCP server health and basic functionality."""
        start_time = time.time()
        
        try:
            # Test server connection
            reactome_client = await real_mcp_manager.get_client("reactome")
            assert reactome_client is not None
            
            # Test pathway search
            result = await reactome_client.find_pathways_by_disease("breast cancer")
            assert result is not None
            assert isinstance(result, list)
            assert len(result) > 0
            
            # Test gene pathway search
            gene_result = await reactome_client.find_pathways_by_gene("TP53")
            assert gene_result is not None
            assert isinstance(gene_result, list)
            
            execution_time = time.time() - start_time
            print(f"Reactome server test completed in {execution_time:.2f}s")
            assert execution_time < 30.0  # Should complete in <30 seconds
            
        except Exception as e:
            pytest.fail(f"Reactome server health check failed: {e}")
    
    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_string_server_health(self, real_mcp_manager):
        """Test STRING MCP server health and basic functionality."""
        start_time = time.time()
        
        try:
            # Test server connection
            string_client = await real_mcp_manager.get_client("string")
            assert string_client is not None
            
            # Test protein info retrieval
            protein_info = await string_client.get_protein_info("TP53")
            assert protein_info is not None
            assert isinstance(protein_info, list)
            assert len(protein_info) > 0
            
            # Test interaction network
            network_result = await string_client.get_interaction_network(["TP53", "BRCA1"])
            assert network_result is not None
            assert "nodes" in network_result
            assert "edges" in network_result
            assert len(network_result["nodes"]) > 0
            
            execution_time = time.time() - start_time
            print(f"STRING server test completed in {execution_time:.2f}s")
            assert execution_time < 30.0  # Should complete in <30 seconds
            
        except Exception as e:
            pytest.fail(f"STRING server health check failed: {e}")
    
    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_hpa_server_health(self, real_mcp_manager):
        """Test Human Protein Atlas MCP server health and basic functionality."""
        start_time = time.time()
        
        try:
            # Test server connection
            hpa_client = await real_mcp_manager.get_client("hpa")
            assert hpa_client is not None
            
            # Test tissue expression
            expression_result = await hpa_client.get_tissue_expression(["TP53"], "breast")
            assert expression_result is not None
            assert isinstance(expression_result, list)
            
            # Test cancer markers
            cancer_markers = await hpa_client.search_cancer_markers("TP53")
            assert cancer_markers is not None
            assert isinstance(cancer_markers, list)
            
            execution_time = time.time() - start_time
            print(f"HPA server test completed in {execution_time:.2f}s")
            assert execution_time < 30.0  # Should complete in <30 seconds
            
        except Exception as e:
            pytest.fail(f"HPA server health check failed: {e}")


class TestScientificValidation:
    """Test scientific validation against known biological data."""
    
    @pytest.fixture(scope="class")
    async def production_pipeline(self):
        """Create production pipeline with real MCP servers."""
        config_path = "config/mcp_servers.json"
        pipeline = OmniTargetPipeline(config_path)
        await pipeline.initialize()
        yield pipeline
        await pipeline.cleanup()
    
    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_breast_cancer_gene_validation(self, production_pipeline):
        """Validate breast cancer analysis against known genes."""
        start_time = time.time()
        
        # Known breast cancer genes from literature
        known_breast_cancer_genes = {
            "TP53", "BRCA1", "BRCA2", "ESR1", "ERBB2", "PIK3CA", 
            "AKT1", "PTEN", "CDKN2A", "RB1", "MYC", "CCND1",
            "ATM", "CHEK2", "PALB2", "BARD1", "BRIP1", "RAD51C"
        }
        
        try:
            # Run real scenario
            result = await production_pipeline.execute_scenario(
                1, {"disease_query": "breast cancer", "tissue_context": "breast"}
            )
            
            # Extract genes from result
            pipeline_genes = set()
            if hasattr(result, 'network') and result.network:
                for node in result.network.nodes():
                    if hasattr(node, 'gene_symbol') and node.gene_symbol:
                        pipeline_genes.add(node.gene_symbol)
            
            # Calculate overlap
            overlap = len(known_breast_cancer_genes & pipeline_genes) / len(known_breast_cancer_genes)
            
            # Validate overlap
            assert overlap >= 0.60, f"Low gene overlap: {overlap:.2f} (expected ≥0.60)"
            print(f"Gene overlap: {overlap:.2f} ({len(known_breast_cancer_genes & pipeline_genes)}/{len(known_breast_cancer_genes)})")
            
            execution_time = time.time() - start_time
            print(f"Breast cancer validation completed in {execution_time:.2f}s")
            assert execution_time < 300.0  # Should complete in <5 minutes
            
        except Exception as e:
            pytest.fail(f"Breast cancer validation failed: {e}")
    
    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_tp53_target_validation(self, production_pipeline):
        """Validate TP53 target analysis against known interactions."""
        start_time = time.time()
        
        # Known TP53 interactions
        known_tp53_interactions = {
            "MDM2", "MDM4", "ATM", "ATR", "CHEK1", "CHEK2", "BRCA1", "BRCA2",
            "BAX", "BAK1", "PUMA", "NOXA", "CDKN1A", "CDKN2A", "RB1"
        }
        
        try:
            # Run real scenario
            result = await production_pipeline.execute_scenario(
                2, {"target_query": "TP53"}
            )
            
            # Extract interactions from result
            pipeline_interactions = set()
            if hasattr(result, 'interactions') and result.interactions:
                for interaction in result.interactions:
                    if hasattr(interaction, 'protein_a') and interaction.protein_a == "TP53":
                        pipeline_interactions.add(interaction.protein_b)
                    elif hasattr(interaction, 'protein_b') and interaction.protein_b == "TP53":
                        pipeline_interactions.add(interaction.protein_a)
            
            # Calculate overlap
            overlap = len(known_tp53_interactions & pipeline_interactions) / len(known_tp53_interactions)
            
            # Validate overlap
            assert overlap >= 0.50, f"Low interaction overlap: {overlap:.2f} (expected ≥0.50)"
            print(f"Interaction overlap: {overlap:.2f} ({len(known_tp53_interactions & pipeline_interactions)}/{len(known_tp53_interactions)})")
            
            execution_time = time.time() - start_time
            print(f"TP53 target validation completed in {execution_time:.2f}s")
            assert execution_time < 300.0  # Should complete in <5 minutes
            
        except Exception as e:
            pytest.fail(f"TP53 target validation failed: {e}")
    
    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_cancer_marker_validation(self, production_pipeline):
        """Validate cancer marker analysis against known markers."""
        start_time = time.time()
        
        # Known cancer markers
        known_cancer_markers = {
            "TP53", "BRCA1", "BRCA2", "ERBB2", "ESR1", "PIK3CA", "PTEN",
            "MYC", "CCND1", "RB1", "CDKN2A", "ATM", "CHEK2"
        }
        
        try:
            # Run real scenario
            result = await production_pipeline.execute_scenario(
                3, {"cancer_type": "breast cancer", "tissue_context": "breast"}
            )
            
            # Extract markers from result
            pipeline_markers = set()
            if hasattr(result, 'prognostic_markers') and result.prognostic_markers:
                for marker in result.prognostic_markers:
                    if hasattr(marker, 'gene') and marker.gene:
                        pipeline_markers.add(marker.gene)
            
            # Calculate overlap
            overlap = len(known_cancer_markers & pipeline_markers) / len(known_cancer_markers)
            
            # Validate overlap
            assert overlap >= 0.40, f"Low marker overlap: {overlap:.2f} (expected ≥0.40)"
            print(f"Marker overlap: {overlap:.2f} ({len(known_cancer_markers & pipeline_markers)}/{len(known_cancer_markers)})")
            
            execution_time = time.time() - start_time
            print(f"Cancer marker validation completed in {execution_time:.2f}s")
            assert execution_time < 300.0  # Should complete in <5 minutes
            
        except Exception as e:
            pytest.fail(f"Cancer marker validation failed: {e}")


class TestCrossDatabaseValidation:
    """Test cross-database concordance and validation."""
    
    @pytest.fixture(scope="class")
    async def production_pipeline(self):
        """Create production pipeline with real MCP servers."""
        config_path = "config/mcp_servers.json"
        pipeline = OmniTargetPipeline(config_path)
        await pipeline.initialize()
        yield pipeline
        await pipeline.cleanup()
    
    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_kegg_reactome_concordance(self, production_pipeline):
        """Test concordance between KEGG and Reactome databases."""
        start_time = time.time()
        
        try:
            # Get data from both databases
            kegg_result = await production_pipeline.mcp_manager.get_client("kegg").search_diseases("breast cancer")
            reactome_result = await production_pipeline.mcp_manager.get_client("reactome").find_pathways_by_disease("breast cancer")
            
            # Extract pathway information
            kegg_pathways = set()
            for disease in kegg_result:
                if "pathways" in disease:
                    kegg_pathways.update(disease["pathways"])
            
            reactome_pathways = set()
            for pathway in reactome_result:
                if "id" in pathway:
                    reactome_pathways.add(pathway["id"])
            
            # Calculate concordance
            if kegg_pathways and reactome_pathways:
                intersection = len(kegg_pathways & reactome_pathways)
                union = len(kegg_pathways | reactome_pathways)
                concordance = intersection / union if union > 0 else 0
                
                # Validate concordance
                assert concordance >= 0.20, f"Low cross-database concordance: {concordance:.2f} (expected ≥0.20)"
                print(f"KEGG-Reactome concordance: {concordance:.2f} ({intersection}/{union})")
            else:
                print("Warning: No pathway data found for concordance calculation")
            
            execution_time = time.time() - start_time
            print(f"Cross-database validation completed in {execution_time:.2f}s")
            assert execution_time < 180.0  # Should complete in <3 minutes
            
        except Exception as e:
            pytest.fail(f"Cross-database validation failed: {e}")
    
    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_expression_data_quality(self, production_pipeline):
        """Test expression data quality and coverage."""
        start_time = time.time()
        
        try:
            # Test expression data for known genes
            test_genes = ["TP53", "BRCA1", "BRCA2", "ESR1", "ERBB2"]
            
            # Get expression data
            expression_result = await production_pipeline.mcp_manager.get_client("hpa").get_tissue_expression(
                test_genes, "breast"
            )
            
            # Validate expression data
            assert expression_result is not None
            assert isinstance(expression_result, list)
            assert len(expression_result) > 0
            
            # Check data quality
            valid_expression_levels = ["Not detected", "Low", "Medium", "High"]
            valid_reliability_levels = ["Approved", "Supported", "Uncertain"]
            
            for expr_data in expression_result:
                if "expression_level" in expr_data:
                    assert expr_data["expression_level"] in valid_expression_levels
                if "reliability" in expr_data:
                    assert expr_data["reliability"] in valid_reliability_levels
            
            # Check tissue coverage
            tissues = set()
            for expr_data in expression_result:
                if "tissue" in expr_data:
                    tissues.add(expr_data["tissue"])
            
            assert len(tissues) >= 3, f"Low tissue coverage: {len(tissues)} (expected ≥3)"
            print(f"Tissue coverage: {len(tissues)} tissues")
            
            execution_time = time.time() - start_time
            print(f"Expression data validation completed in {execution_time:.2f}s")
            assert execution_time < 120.0  # Should complete in <2 minutes
            
        except Exception as e:
            pytest.fail(f"Expression data validation failed: {e}")
    
    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_network_quality_metrics(self, production_pipeline):
        """Test network quality and biological relevance."""
        start_time = time.time()
        
        try:
            # Run disease network scenario
            result = await production_pipeline.execute_scenario(
                1, {"disease_query": "breast cancer", "tissue_context": "breast"}
            )
            
            # Validate network properties
            if hasattr(result, 'network') and result.network:
                network = result.network
                
                # Check network size
                assert network.number_of_nodes() > 10, f"Network too small: {network.number_of_nodes()} nodes"
                assert network.number_of_edges() > 5, f"Network too sparse: {network.number_of_edges()} edges"
                
                # Check connectivity
                if network.number_of_nodes() > 0:
                    connected_components = list(network.connected_components())
                    largest_component_size = len(max(connected_components, key=len))
                    connectivity_ratio = largest_component_size / network.number_of_nodes()
                    
                    assert connectivity_ratio >= 0.50, f"Low network connectivity: {connectivity_ratio:.2f} (expected ≥0.50)"
                    print(f"Network connectivity: {connectivity_ratio:.2f} ({largest_component_size}/{network.number_of_nodes()})")
                
                # Check edge weights
                edge_weights = [data.get('weight', 0) for _, _, data in network.edges(data=True)]
                if edge_weights:
                    mean_weight = sum(edge_weights) / len(edge_weights)
                    assert mean_weight >= 0.3, f"Low mean edge weight: {mean_weight:.2f} (expected ≥0.30)"
                    print(f"Mean edge weight: {mean_weight:.2f}")
            
            execution_time = time.time() - start_time
            print(f"Network quality validation completed in {execution_time:.2f}s")
            assert execution_time < 300.0  # Should complete in <5 minutes
            
        except Exception as e:
            pytest.fail(f"Network quality validation failed: {e}")


class TestProductionPerformance:
    """Test production performance with real MCP servers."""
    
    @pytest.fixture(scope="class")
    async def production_pipeline(self):
        """Create production pipeline with real MCP servers."""
        config_path = "config/mcp_servers.json"
        pipeline = OmniTargetPipeline(config_path)
        await pipeline.initialize()
        yield pipeline
        await pipeline.cleanup()
    
    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_scenario_execution_times(self, production_pipeline):
        """Test scenario execution times with real servers."""
        scenarios = [
            (1, {"disease_query": "breast cancer", "tissue_context": "breast"}, 300),  # 5 min max
            (2, {"target_query": "TP53"}, 180),  # 3 min max
            (3, {"cancer_type": "breast cancer", "tissue_context": "breast"}, 300),  # 5 min max
        ]
        
        for scenario_id, params, max_time in scenarios:
            start_time = time.time()
            
            try:
                result = await production_pipeline.execute_scenario(scenario_id, params)
                execution_time = time.time() - start_time
                
                assert result is not None
                assert execution_time < max_time, f"Scenario {scenario_id} exceeded {max_time}s: {execution_time:.2f}s"
                print(f"Scenario {scenario_id}: {execution_time:.2f}s (max: {max_time}s)")
                
            except Exception as e:
                pytest.fail(f"Scenario {scenario_id} execution failed: {e}")
    
    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_concurrent_production_scenarios(self, production_pipeline):
        """Test concurrent scenario execution with real servers."""
        start_time = time.time()
        
        # Define concurrent scenarios
        scenarios = [
            (1, {"disease_query": "lung cancer", "tissue_context": "lung"}),
            (2, {"target_query": "BRCA1"}),
            (1, {"disease_query": "prostate cancer", "tissue_context": "prostate"}),
        ]
        
        try:
            # Run scenarios concurrently
            tasks = []
            for scenario_id, params in scenarios:
                task = asyncio.create_task(
                    production_pipeline.execute_scenario(scenario_id, params)
                )
                tasks.append(task)
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            execution_time = time.time() - start_time
            
            # Validate results
            assert len(results) == 3
            assert all(not isinstance(result, Exception) for result in results)
            assert execution_time < 600.0  # Should complete in <10 minutes
            print(f"Concurrent scenarios execution time: {execution_time:.2f}s")
            
        except Exception as e:
            pytest.fail(f"Concurrent scenario execution failed: {e}")
    
    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_memory_usage_production(self, production_pipeline):
        """Test memory usage during production scenarios."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        try:
            # Run scenario
            result = await production_pipeline.execute_scenario(
                1, {"disease_query": "breast cancer", "tissue_context": "breast"}
            )
            
            peak_memory = process.memory_info().rss / 1024 / 1024  # MB
            memory_increase = peak_memory - initial_memory
            
            # Validate memory usage
            assert memory_increase < 1000  # Should use <1GB additional memory
            assert peak_memory < 4000  # Total memory should be <4GB
            print(f"Memory increase: {memory_increase:.1f}MB, Peak: {peak_memory:.1f}MB")
            
        except Exception as e:
            pytest.fail(f"Memory usage test failed: {e}")


class TestProductionReliability:
    """Test production reliability and error handling."""
    
    @pytest.fixture(scope="class")
    async def production_pipeline(self):
        """Create production pipeline with real MCP servers."""
        config_path = "config/mcp_servers.json"
        pipeline = OmniTargetPipeline(config_path)
        await pipeline.initialize()
        yield pipeline
        await pipeline.cleanup()
    
    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_error_handling_invalid_queries(self, production_pipeline):
        """Test error handling for invalid queries."""
        invalid_queries = [
            (1, {"disease_query": "nonexistent_disease_xyz123"}),
            (2, {"target_query": "INVALID_GENE_XYZ"}),
            (3, {"cancer_type": "invalid_cancer_type"}),
        ]
        
        for scenario_id, params in invalid_queries:
            try:
                result = await production_pipeline.execute_scenario(scenario_id, params)
                
                # Should either return empty result or handle gracefully
                assert result is not None
                print(f"Invalid query handled gracefully for scenario {scenario_id}")
                
            except Exception as e:
                # Should handle errors gracefully, not crash
                print(f"Error handled for scenario {scenario_id}: {e}")
                assert "timeout" in str(e).lower() or "not found" in str(e).lower()
    
    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_timeout_handling(self, production_pipeline):
        """Test timeout handling for slow queries."""
        start_time = time.time()
        
        try:
            # Use a query that might be slow
            result = await production_pipeline.execute_scenario(
                1, {"disease_query": "cancer", "tissue_context": "all"}
            )
            
            execution_time = time.time() - start_time
            
            # Should complete within reasonable time
            assert execution_time < 600.0  # <10 minutes
            print(f"Timeout handling test completed in {execution_time:.2f}s")
            
        except Exception as e:
            # Should handle timeouts gracefully
            print(f"Timeout handled gracefully: {e}")
            assert "timeout" in str(e).lower() or "connection" in str(e).lower()
    
    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_data_validation_production(self, production_pipeline):
        """Test data validation in production scenarios."""
        try:
            # Run scenario
            result = await production_pipeline.execute_scenario(
                1, {"disease_query": "breast cancer", "tissue_context": "breast"}
            )
            
            # Validate result structure
            assert hasattr(result, 'disease') or hasattr(result, 'network')
            
            # Validate data quality
            if hasattr(result, 'validation_score'):
                assert 0.0 <= result.validation_score <= 1.0
                print(f"Validation score: {result.validation_score:.2f}")
            
            print("Data validation passed")
            
        except Exception as e:
            pytest.fail(f"Data validation failed: {e}")
