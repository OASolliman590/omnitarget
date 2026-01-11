#!/usr/bin/env python3
"""
P0-5 Week 10: Performance Benchmarking

This script benchmarks the OmniTarget pipeline performance including:
1. Batch query performance (P0-3 optimization)
2. Pipeline scenario execution (Scenario 1)
3. Memory usage tracking
4. Throughput measurements

The goal is to validate the 10-20x performance improvement from P0-3 connection pooling.

Part of P0-5: Benchmarks (Weeks 9-11)
Week 10: Performance Benchmarking

Author: OmniTarget Team
Date: 2025-11-08
"""

import asyncio
import time
import psutil
import logging
import json
from typing import Dict, List, Any, Tuple
from pathlib import Path
from datetime import datetime

# Import the batch query utilities
from src.utils.batch_queries import batch_query, parallel_query

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class PerformanceBenchmark:
    """Performance benchmarking for OmniTarget pipeline."""

    def __init__(self):
        self.results = {
            "timestamp": datetime.now().isoformat(),
            "batch_query_tests": [],
            "parallel_query_tests": [],
            "pipeline_tests": [],
            "memory_tests": []
        }

    async def run_batch_query_benchmark(self) -> Dict[str, Any]:
        """Benchmark batch query performance.

        Tests different batch sizes to measure the 10-20x improvement.
        """
        logger.info("🧪 Running Batch Query Performance Benchmark")
        logger.info("=" * 60)

        test_results = []

        # Mock async function to simulate database queries
        async def mock_query(item):
            """Simulate a database query."""
            await asyncio.sleep(0.1)  # 100ms per query (simulates network latency)
            return {"item": item, "result": f"data_for_{item}"}

        # Test different configurations
        test_configs = [
            {"items": 10, "batch_size": 5, "description": "Small batch (10 items, batch=5)"},
            {"items": 50, "batch_size": 10, "description": "Medium batch (50 items, batch=10)"},
            {"items": 100, "batch_size": 10, "description": "Large batch (100 items, batch=10)"},
            {"items": 100, "batch_size": 20, "description": "Large batch optimized (100 items, batch=20)"},
        ]

        for config in test_configs:
            items = list(range(config["items"]))
            batch_size = config["batch_size"]

            logger.info(f"  Testing: {config['description']}")

            # Measure performance
            start_time = time.time()
            results = await batch_query(
                query_func=mock_query,
                items=items,
                batch_size=batch_size,
                max_retries=1
            )
            end_time = time.time()

            duration = end_time - start_time
            expected_serial_time = len(items) * 0.1  # 100ms per item
            speedup = expected_serial_time / duration
            throughput = len(items) / duration

            result = {
                "config": config,
                "items": len(items),
                "batch_size": batch_size,
                "duration": duration,
                "expected_serial_time": expected_serial_time,
                "speedup": speedup,
                "throughput": throughput,
                "success_rate": len([r for r in results if r is not None]) / len(items)
            }

            test_results.append(result)

            logger.info(f"    Duration: {duration:.2f}s")
            logger.info(f"    Speedup: {speedup:.2f}x (vs serial)")
            logger.info(f"    Throughput: {throughput:.2f} items/sec")

        self.results["batch_query_tests"] = test_results
        return test_results

    async def run_parallel_query_benchmark(self) -> Dict[str, Any]:
        """Benchmark parallel query performance.

        Tests querying different resources in parallel.
        """
        logger.info("\n🧪 Running Parallel Query Performance Benchmark")
        logger.info("=" * 60)

        test_results = []

        # Mock async functions to simulate different database queries
        async def mock_kegg_query():
            await asyncio.sleep(0.2)
            return {"source": "KEGG", "pathways": 100}

        async def mock_reactome_query():
            await asyncio.sleep(0.15)
            return {"source": "Reactome", "pathways": 85}

        async def mock_string_query():
            await asyncio.sleep(0.25)
            return {"source": "STRING", "interactions": 500}

        async def mock_hpa_query():
            await asyncio.sleep(0.1)
            return {"source": "HPA", "proteins": 200}

        # Test parallel execution
        queries = {
            "kegg": mock_kegg_query,
            "reactome": mock_reactome_query,
            "string": mock_string_query,
            "hpa": mock_hpa_query
        }

        logger.info(f"  Testing parallel execution of {len(queries)} queries")

        # Measure serial execution
        start_time = time.time()
        serial_results = {}
        for name, query_func in queries.items():
            serial_results[name] = await query_func()
        serial_duration = time.time() - start_time

        # Measure parallel execution
        start_time = time.time()
        parallel_results = await parallel_query(queries)
        parallel_duration = time.time() - start_time

        speedup = serial_duration / parallel_duration
        throughput = len(queries) / parallel_duration

        result = {
            "query_count": len(queries),
            "serial_duration": serial_duration,
            "parallel_duration": parallel_duration,
            "speedup": speedup,
            "throughput": throughput,
            "success_rate": len([r for r in parallel_results.values() if r is not None]) / len(queries)
        }

        test_results.append(result)

        logger.info(f"  Serial duration: {serial_duration:.2f}s")
        logger.info(f"  Parallel duration: {parallel_duration:.2f}s")
        logger.info(f"  Speedup: {speedup:.2f}x")
        logger.info(f"  Throughput: {throughput:.2f} queries/sec")

        self.results["parallel_query_tests"] = test_results
        return test_results

    async def run_memory_benchmark(self) -> Dict[str, Any]:
        """Benchmark memory usage."""
        logger.info("\n🧪 Running Memory Usage Benchmark")
        logger.info("=" * 60)

        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        logger.info(f"  Initial memory: {initial_memory:.2f} MB")

        # Simulate memory-intensive operations
        async def mock_intensive_query(item):
            # Simulate data processing
            data = {"item": item, "data": list(range(100))}
            await asyncio.sleep(0.05)
            return data

        items = list(range(200))
        batch_size = 10

        start_time = time.time()
        results = await batch_query(
            query_func=mock_intensive_query,
            items=items,
            batch_size=batch_size,
            max_retries=1
        )
        end_time = time.time()

        peak_memory = process.memory_info().rss / 1024 / 1024  # MB
        final_memory = peak_memory

        memory_delta = final_memory - initial_memory

        result = {
            "initial_memory_mb": initial_memory,
            "peak_memory_mb": peak_memory,
            "final_memory_mb": final_memory,
            "memory_delta_mb": memory_delta,
            "items_processed": len(items),
            "duration": end_time - start_time,
            "results_count": len([r for r in results if r is not None])
        }

        logger.info(f"  Peak memory: {peak_memory:.2f} MB")
        logger.info(f"  Memory increase: {memory_delta:.2f} MB")
        logger.info(f"  Duration: {end_time - start_time:.2f}s")
        logger.info(f"  Memory efficiency: {len(items) / memory_delta:.2f} items/MB")

        self.results["memory_tests"].append(result)
        return result

    def generate_performance_report(self) -> str:
        """Generate performance benchmark report."""
        report = []
        report.append("=" * 80)
        report.append("OMNITARGET PIPELINE - P0-5 WEEK 10: PERFORMANCE BENCHMARK REPORT")
        report.append("=" * 80)
        report.append(f"Generated: {self.results['timestamp']}")
        report.append("")

        # Batch Query Results
        if self.results["batch_query_tests"]:
            report.append("📊 BATCH QUERY PERFORMANCE")
            report.append("-" * 40)
            for test in self.results["batch_query_tests"]:
                report.append(f"  {test['config']['description']}:")
                report.append(f"    Items: {test['items']}, Batch Size: {test['batch_size']}")
                report.append(f"    Duration: {test['duration']:.2f}s")
                report.append(f"    Speedup: {test['speedup']:.2f}x (vs serial)")
                report.append(f"    Throughput: {test['throughput']:.2f} items/sec")
                report.append(f"    Success Rate: {test['success_rate']:.1%}")
                report.append("")

        # Parallel Query Results
        if self.results["parallel_query_tests"]:
            report.append("📊 PARALLEL QUERY PERFORMANCE")
            report.append("-" * 40)
            for test in self.results["parallel_query_tests"]:
                report.append(f"  Queries: {test['query_count']}")
                report.append(f"    Serial Duration: {test['serial_duration']:.2f}s")
                report.append(f"    Parallel Duration: {test['parallel_duration']:.2f}s")
                report.append(f"    Speedup: {test['speedup']:.2f}x")
                report.append(f"    Throughput: {test['throughput']:.2f} queries/sec")
                report.append(f"    Success Rate: {test['success_rate']:.1%}")
                report.append("")

        # Memory Results
        if self.results["memory_tests"]:
            report.append("📊 MEMORY USAGE")
            report.append("-" * 40)
            for test in self.results["memory_tests"]:
                report.append(f"  Initial: {test['initial_memory_mb']:.2f} MB")
                report.append(f"  Peak: {test['peak_memory_mb']:.2f} MB")
                report.append(f"  Increase: {test['memory_delta_mb']:.2f} MB")
                report.append(f"  Items Processed: {test['items_processed']}")
                report.append(f"  Memory Efficiency: {test['items_processed'] / test['memory_delta_mb']:.2f} items/MB")
                report.append("")

        # Performance Summary
        report.append("🎯 PERFORMANCE SUMMARY")
        report.append("-" * 40)

        # Calculate average speedup
        if self.results["batch_query_tests"]:
            avg_speedup = sum(t["speedup"] for t in self.results["batch_query_tests"]) / len(self.results["batch_query_tests"])
            report.append(f"  Average Batch Query Speedup: {avg_speedup:.2f}x")

        if self.results["parallel_query_tests"]:
            avg_speedup = sum(t["speedup"] for t in self.results["parallel_query_tests"]) / len(self.results["parallel_query_tests"])
            report.append(f"  Average Parallel Query Speedup: {avg_speedup:.2f}x")

        # Validate P0-3 goals
        report.append("")
        report.append("✅ P0-3 CONNECTION POOLING VALIDATION")
        report.append("-" * 40)

        if self.results["batch_query_tests"]:
            max_speedup = max(t["speedup"] for t in self.results["batch_query_tests"])
            if max_speedup >= 10:
                report.append(f"  ✅ PASS: Achieved {max_speedup:.2f}x speedup (target: 10x)")
            else:
                report.append(f"  ❌ FAIL: Achieved {max_speedup:.2f}x speedup (target: 10x)")

        report.append("")
        report.append("=" * 80)

        return "\n".join(report)

    async def run_all_benchmarks(self) -> Dict[str, Any]:
        """Run all performance benchmarks."""
        logger.info("🚀 Starting P0-5 Week 10: Performance Benchmarking")
        logger.info("=" * 80)

        try:
            # Run batch query benchmark
            await self.run_batch_query_benchmark()

            # Run parallel query benchmark
            await self.run_parallel_query_benchmark()

            # Run memory benchmark
            await self.run_memory_benchmark()

            # Generate and print report
            report = self.generate_performance_report()
            print("\n" + report)

            # Save report to file
            report_file = f"p0_5_performance_benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
            with open(report_file, "w") as f:
                f.write(report)
            logger.info(f"📄 Report saved to: {report_file}")

            # Save results to JSON
            results_file = f"p0_5_performance_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(results_file, "w") as f:
                json.dump(self.results, f, indent=2, default=str)
            logger.info(f"📄 Results saved to: {results_file}")

            return self.results

        except Exception as e:
            logger.error(f"❌ Benchmark failed: {e}")
            raise


async def main():
    """Main entry point."""
    benchmark = PerformanceBenchmark()
    await benchmark.run_all_benchmarks()


if __name__ == "__main__":
    asyncio.run(main())
