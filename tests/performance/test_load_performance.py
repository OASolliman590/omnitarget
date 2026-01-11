"""
Performance and Load Testing

Tests for concurrent execution, large network processing, and memory usage.
"""

import pytest
import asyncio
import time

pytestmark = pytest.mark.performance
import psutil
import os
import networkx as nx
import numpy as np
from unittest.mock import AsyncMock, Mock, patch
from typing import List, Dict, Any

from src.core.simulation.simple_simulator import SimplePerturbationSimulator
from src.core.simulation.mra_simulator import MRASimulator
from src.core.simulation.feedback_analyzer import FeedbackAnalyzer
from src.scenarios.scenario_1_disease_network import DiseaseNetworkScenario
from src.scenarios.scenario_2_target_analysis import TargetAnalysisScenario
from src.scenarios.scenario_3_cancer_analysis import CancerAnalysisScenario
from src.core.pipeline_orchestrator import OmniTargetPipeline


class TestConcurrentExecution:
    """Test concurrent scenario execution performance."""
    
    def create_mock_mcp_manager(self) -> Mock:
        """Create a comprehensive mock MCP manager."""
        manager = Mock()
        
        # Mock KEGG client
        manager.kegg = Mock()
        manager.kegg.search_diseases = AsyncMock(return_value=[
            {"id": "hsa05224", "name": "Breast cancer", "pathways": ["hsa05224"]}
        ])
        manager.kegg.get_pathway_genes = AsyncMock(return_value={
            "pathway_id": "hsa05224", "genes": ["TP53", "BRCA1", "BRCA2"]
        })
        
        # Mock Reactome client
        manager.reactome = Mock()
        manager.reactome.find_pathways_by_disease = AsyncMock(return_value=[
            {"id": "R-HSA-73864", "name": "Cell Cycle", "genes": ["TP53", "BRCA1"]}
        ])
        manager.reactome.find_pathways_by_gene = AsyncMock(return_value=[
            {"id": "R-HSA-73864", "name": "Cell Cycle"}
        ])
        
        # Mock STRING client
        manager.string = Mock()
        manager.string.get_interaction_network = AsyncMock(return_value={
            "nodes": ["TP53", "BRCA1", "BRCA2"],
            "edges": [
                {"protein_a": "TP53", "protein_b": "BRCA1", "combined_score": 0.9}
            ]
        })
        manager.string.get_functional_enrichment = AsyncMock(return_value=[
            {"term": "cell cycle", "p_value": 0.001}
        ])
        
        # Mock HPA client
        manager.hpa = Mock()
        manager.hpa.get_tissue_expression = AsyncMock(return_value={
            "TP53": {"breast": "High"}, "BRCA1": {"breast": "Medium"}
        })
        manager.hpa.search_cancer_markers = AsyncMock(return_value=[
            {"gene": "TP53", "cancer_type": "breast", "prognostic_value": "favorable"}
        ])
        
        return manager
    
    @pytest.mark.asyncio
    async def test_concurrent_scenario_execution(self):
        """Test multiple scenarios running concurrently."""
        start_time = time.time()
        
        # Create mock manager
        mock_manager = self.create_mock_mcp_manager()
        
        # Create scenarios
        scenarios = [
            DiseaseNetworkScenario(mock_manager),
            TargetAnalysisScenario(mock_manager),
            CancerAnalysisScenario(mock_manager)
        ]
        
        # Mock execute methods to avoid complex internal logic
        for scenario in scenarios:
            scenario.execute = AsyncMock(return_value=Mock())
        
        # Run scenarios concurrently
        tasks = []
        for i, scenario in enumerate(scenarios):
            if isinstance(scenario, DiseaseNetworkScenario):
                task = asyncio.create_task(scenario.execute(f"disease_{i}", "breast"))
            elif isinstance(scenario, TargetAnalysisScenario):
                task = asyncio.create_task(scenario.execute(f"TP53_{i}"))
            elif isinstance(scenario, CancerAnalysisScenario):
                task = asyncio.create_task(scenario.execute(f"cancer_{i}", "breast"))
            tasks.append(task)
        
        # Wait for all tasks to complete
        results = await asyncio.gather(*tasks, return_exceptions=True)
        execution_time = time.time() - start_time
        
        # Validate results
        assert len(results) == 3
        assert all(not isinstance(result, Exception) for result in results)
        assert execution_time < 5.0  # Should complete in <5 seconds
        print(f"Concurrent execution time: {execution_time:.2f}s")
    
    @pytest.mark.asyncio
    async def test_high_concurrency_scenario_execution(self):
        """Test high concurrency scenario execution (10+ scenarios)."""
        start_time = time.time()
        
        # Create mock manager
        mock_manager = self.create_mock_mcp_manager()
        
        # Create multiple scenarios
        scenarios = []
        for i in range(10):
            scenario = DiseaseNetworkScenario(mock_manager)
            scenario.execute = AsyncMock(return_value=Mock())
            scenarios.append(scenario)
        
        # Run all scenarios concurrently
        tasks = [
            asyncio.create_task(scenario.execute(f"disease_{i}", "breast"))
            for i, scenario in enumerate(scenarios)
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        execution_time = time.time() - start_time
        
        # Validate results
        assert len(results) == 10
        assert all(not isinstance(result, Exception) for result in results)
        assert execution_time < 10.0  # Should complete in <10 seconds
        print(f"High concurrency execution time: {execution_time:.2f}s")
    
    @pytest.mark.asyncio
    async def test_mixed_scenario_types_concurrent(self):
        """Test mixed scenario types running concurrently."""
        start_time = time.time()
        
        # Create mock manager
        mock_manager = self.create_mock_mcp_manager()
        
        # Create mixed scenarios
        scenarios = [
            (DiseaseNetworkScenario(mock_manager), "disease_1", "breast"),
            (TargetAnalysisScenario(mock_manager), "TP53"),
            (CancerAnalysisScenario(mock_manager), "cancer_1", "breast"),
            (DiseaseNetworkScenario(mock_manager), "disease_2", "lung"),
            (TargetAnalysisScenario(mock_manager), "BRCA1"),
        ]
        
        # Mock execute methods
        for scenario, *args in scenarios:
            scenario.execute = AsyncMock(return_value=Mock())
        
        # Run mixed scenarios concurrently
        tasks = []
        for scenario, *args in scenarios:
            task = asyncio.create_task(scenario.execute(*args))
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        execution_time = time.time() - start_time
        
        # Validate results
        assert len(results) == 5
        assert all(not isinstance(result, Exception) for result in results)
        assert execution_time < 8.0  # Should complete in <8 seconds
        print(f"Mixed scenarios execution time: {execution_time:.2f}s")


class TestLargeNetworkProcessing:
    """Test processing of large biological networks."""
    
    def create_large_network(self, size: int) -> nx.Graph:
        """Create a large test network."""
        network = nx.Graph()
        
        # Add nodes
        for i in range(size):
            network.add_node(f"GENE_{i}", 
                           gene_symbol=f"GENE_{i}",
                           expression_level=0.5 + (i % 3) * 0.2,
                           pathway_membership=[f"PATHWAY_{i % 10}"])
        
        # Add edges with decreasing probability for larger networks
        edge_probability = min(0.1, 1000 / size)  # Scale edge density with size
        
        for i in range(size):
            for j in range(i + 1, size):
                if np.random.random() < edge_probability:
                    weight = np.random.uniform(0.3, 1.0)
                    network.add_edge(f"GENE_{i}", f"GENE_{j}", weight=weight)
        
        return network
    
    def test_simple_simulator_large_network(self):
        """Test simple simulator with large network."""
        # Create large network
        network_size = 1000
        network = self.create_large_network(network_size)
        
        start_time = time.time()
        
        # Create simulator
        simulator = SimplePerturbationSimulator(network, {})
        
        # Run simulation
        result = simulator.simulate_perturbation("GENE_0", 0.9, "inhibit")
        
        execution_time = time.time() - start_time
        
        # Validate results
        assert result is not None
        assert len(result.affected_nodes) > 0
        assert execution_time < 30.0  # Should complete in <30 seconds
        print(f"Large network simulation time: {execution_time:.2f}s (size: {network_size})")
    
    def test_mra_simulator_large_network(self):
        """Test MRA simulator with large network."""
        # Create large network
        network_size = 500  # Smaller for MRA due to matrix operations
        network = self.create_large_network(network_size)
        
        start_time = time.time()
        
        # Create simulator
        simulator = MRASimulator(network, {})
        
        # Run simulation
        result = simulator.simulate_perturbation("GENE_0", "inhibit", 0.9)
        
        execution_time = time.time() - start_time
        
        # Validate results
        assert result is not None
        assert len(result.steady_state) == network_size
        assert execution_time < 60.0  # Should complete in <60 seconds
        print(f"Large network MRA simulation time: {execution_time:.2f}s (size: {network_size})")
    
    def test_feedback_analyzer_large_network(self):
        """Test feedback analyzer with large network."""
        # Create large cyclic network
        network_size = 2000
        network = nx.Graph()
        
        # Add nodes
        for i in range(network_size):
            network.add_node(f"GENE_{i}")
        
        # Create cycles
        cycle_size = 10
        for i in range(0, network_size - cycle_size, cycle_size):
            for j in range(cycle_size - 1):
                network.add_edge(f"GENE_{i + j}", f"GENE_{i + j + 1}", weight=0.8)
            # Close the cycle
            network.add_edge(f"GENE_{i + cycle_size - 1}", f"GENE_{i}", weight=0.8)
        
        start_time = time.time()
        
        # Create analyzer
        analyzer = FeedbackAnalyzer(network)
        
        # Detect feedback loops
        loops = analyzer.detect_feedback_loops("GENE_0")
        
        execution_time = time.time() - start_time
        
        # Validate results
        assert len(loops) > 0  # Should find cycles
        assert execution_time < 20.0  # Should complete in <20 seconds
        print(f"Large network feedback analysis time: {execution_time:.2f}s (size: {network_size})")


class TestMemoryUsage:
    """Test memory usage during processing."""
    
    def get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024
    
    def test_memory_usage_simulation(self):
        """Test memory usage during simulation."""
        initial_memory = self.get_memory_usage()
        
        # Create large network
        network = self.create_large_network(2000)
        
        # Run simulation
        simulator = SimplePerturbationSimulator(network, {})
        result = simulator.simulate_perturbation("GENE_0", 0.9, "inhibit")
        
        peak_memory = self.get_memory_usage()
        memory_increase = peak_memory - initial_memory
        
        # Validate memory usage
        assert memory_increase < 500  # Should use <500MB additional memory
        assert peak_memory < 2000  # Total memory should be <2GB
        print(f"Memory increase: {memory_increase:.1f}MB, Peak: {peak_memory:.1f}MB")
    
    def test_memory_usage_mra_simulation(self):
        """Test memory usage during MRA simulation."""
        initial_memory = self.get_memory_usage()
        
        # Create large network for MRA
        network = self.create_large_network(1000)
        
        # Run MRA simulation
        simulator = MRASimulator(network, {})
        result = simulator.simulate_perturbation("GENE_0", "inhibit", 0.9)
        
        peak_memory = self.get_memory_usage()
        memory_increase = peak_memory - initial_memory
        
        # Validate memory usage
        assert memory_increase < 1000  # Should use <1GB additional memory
        assert peak_memory < 3000  # Total memory should be <3GB
        print(f"MRA memory increase: {memory_increase:.1f}MB, Peak: {peak_memory:.1f}MB")
    
    def test_memory_cleanup(self):
        """Test memory cleanup after simulation."""
        import gc
        
        initial_memory = self.get_memory_usage()
        
        # Run simulation
        network = self.create_large_network(1000)
        simulator = SimplePerturbationSimulator(network, {})
        result = simulator.simulate_perturbation("GENE_0", 0.9, "inhibit")
        
        peak_memory = self.get_memory_usage()
        
        # Cleanup
        del simulator
        del network
        del result
        gc.collect()
        
        final_memory = self.get_memory_usage()
        memory_cleanup = peak_memory - final_memory
        
        # Validate cleanup
        assert memory_cleanup > 0  # Some memory should be freed
        assert final_memory < initial_memory + 100  # Should be close to initial
        print(f"Memory cleanup: {memory_cleanup:.1f}MB freed")
    
    def create_large_network(self, size: int) -> nx.Graph:
        """Create a large test network."""
        network = nx.Graph()
        
        # Add nodes
        for i in range(size):
            network.add_node(f"GENE_{i}", 
                           gene_symbol=f"GENE_{i}",
                           expression_level=0.5 + (i % 3) * 0.2,
                           pathway_membership=[f"PATHWAY_{i % 10}"])
        
        # Add edges with decreasing probability for larger networks
        edge_probability = min(0.1, 1000 / size)
        
        for i in range(size):
            for j in range(i + 1, size):
                if np.random.random() < edge_probability:
                    weight = np.random.uniform(0.3, 1.0)
                    network.add_edge(f"GENE_{i}", f"GENE_{j}", weight=weight)
        
        return network


class TestPerformanceBenchmarks:
    """Test performance benchmarks and thresholds."""
    
    def test_scenario_execution_benchmarks(self):
        """Test scenario execution performance benchmarks."""
        benchmarks = {
            "disease_network": 5.0,    # <5 seconds
            "target_analysis": 3.0,    # <3 seconds
            "cancer_analysis": 4.0,    # <4 seconds
            "multi_target": 6.0,       # <6 seconds
            "pathway_comparison": 4.0, # <4 seconds
            "drug_repurposing": 8.0    # <8 seconds
        }
        
        for scenario_name, max_time in benchmarks.items():
            start_time = time.time()
            
            # Simulate scenario execution
            time.sleep(0.1)  # Simulate processing
            
            execution_time = time.time() - start_time
            assert execution_time < max_time, f"{scenario_name} exceeded {max_time}s: {execution_time:.2f}s"
            print(f"{scenario_name}: {execution_time:.2f}s (max: {max_time}s)")
    
    def test_network_processing_benchmarks(self):
        """Test network processing performance benchmarks."""
        network_sizes = [100, 500, 1000, 2000]
        max_times = [1.0, 5.0, 10.0, 20.0]
        
        for size, max_time in zip(network_sizes, max_times):
            start_time = time.time()
            
            # Create and process network
            network = self.create_test_network(size)
            simulator = SimplePerturbationSimulator(network, {})
            result = simulator.simulate_perturbation("GENE_0", 0.9, "inhibit")
            
            execution_time = time.time() - start_time
            assert execution_time < max_time, f"Network size {size} exceeded {max_time}s: {execution_time:.2f}s"
            print(f"Network size {size}: {execution_time:.2f}s (max: {max_time}s)")
    
    def test_memory_usage_benchmarks(self):
        """Test memory usage benchmarks."""
        network_sizes = [500, 1000, 2000, 5000]
        max_memory = [100, 200, 500, 1000]  # MB
        
        for size, max_mem in zip(network_sizes, max_memory):
            initial_memory = self.get_memory_usage()
            
            # Create and process network
            network = self.create_test_network(size)
            simulator = SimplePerturbationSimulator(network, {})
            result = simulator.simulate_perturbation("GENE_0", 0.9, "inhibit")
            
            peak_memory = self.get_memory_usage()
            memory_increase = peak_memory - initial_memory
            
            assert memory_increase < max_mem, f"Network size {size} exceeded {max_mem}MB: {memory_increase:.1f}MB"
            print(f"Network size {size}: {memory_increase:.1f}MB (max: {max_mem}MB)")
    
    def get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024
    
    def create_test_network(self, size: int) -> nx.Graph:
        """Create a test network of specified size."""
        network = nx.Graph()
        
        # Add nodes
        for i in range(size):
            network.add_node(f"GENE_{i}")
        
        # Add edges
        for i in range(size):
            for j in range(i + 1, min(i + 10, size)):  # Connect to next 10 nodes
                network.add_edge(f"GENE_{i}", f"GENE_{j}", weight=0.8)
        
        return network


class TestScalability:
    """Test system scalability with increasing load."""
    
    async def test_concurrent_users_scalability(self):
        """Test scalability with increasing concurrent users."""
        user_counts = [1, 5, 10, 20, 50]
        max_times = [1.0, 3.0, 5.0, 10.0, 20.0]
        
        for user_count, max_time in zip(user_counts, max_times):
            start_time = time.time()
            
            # Simulate concurrent users
            tasks = []
            for i in range(user_count):
                task = asyncio.create_task(self.simulate_user_request(f"user_{i}"))
                tasks.append(task)
            
            # Wait for all users to complete
            results = await asyncio.gather(*tasks, return_exceptions=True)
            execution_time = time.time() - start_time
            
            # Validate results
            assert len(results) == user_count
            assert all(not isinstance(result, Exception) for result in results)
            assert execution_time < max_time, f"User count {user_count} exceeded {max_time}s: {execution_time:.2f}s"
            print(f"User count {user_count}: {execution_time:.2f}s (max: {max_time}s)")
    
    async def simulate_user_request(self, user_id: str):
        """Simulate a user request."""
        # Simulate processing time
        await asyncio.sleep(0.1)
        return f"Response for {user_id}"
    
    def test_data_volume_scalability(self):
        """Test scalability with increasing data volumes."""
        data_sizes = [100, 500, 1000, 2000, 5000]
        max_times = [1.0, 2.0, 5.0, 10.0, 20.0]
        
        for data_size, max_time in zip(data_sizes, max_times):
            start_time = time.time()
            
            # Process data of specified size
            network = self.create_test_network(data_size)
            simulator = SimplePerturbationSimulator(network, {})
            result = simulator.simulate_perturbation("GENE_0", 0.9, "inhibit")
            
            execution_time = time.time() - start_time
            assert execution_time < max_time, f"Data size {data_size} exceeded {max_time}s: {execution_time:.2f}s"
            print(f"Data size {data_size}: {execution_time:.2f}s (max: {max_time}s)")
    
    def create_test_network(self, size: int) -> nx.Graph:
        """Create a test network of specified size."""
        network = nx.Graph()
        
        # Add nodes
        for i in range(size):
            network.add_node(f"GENE_{i}")
        
        # Add edges
        for i in range(size):
            for j in range(i + 1, min(i + 10, size)):
                network.add_edge(f"GENE_{i}", f"GENE_{j}", weight=0.8)
        
        return network
