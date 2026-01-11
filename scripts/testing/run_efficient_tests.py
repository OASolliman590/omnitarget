#!/usr/bin/env python3
"""
OmniTarget Pipeline - Efficient Test Runner

Runs different test suites based on development phase and requirements.
"""

import argparse
import subprocess
import sys
import time
import os
from typing import List, Dict, Any


class TestRunner:
    """Efficient test runner for OmniTarget pipeline."""
    
    def __init__(self):
        self.test_categories = {
            "comprehensive": {
                "path": "tests/",
                "description": "Curated end-to-end suite for core functionality (fast, mocked)",
                "timeout": 300,
                "markers": ["comprehensive"]
            },
            "unit": {
                "path": "tests/unit/",
                "description": "Fast unit tests including optimization components (<1s each)",
                "timeout": 60,
                "markers": ["unit"]
            },
            "fast": {
                "path": "tests/fast/",
                "description": "Quick development tests with optimization integration (<1s each)",
                "timeout": 30,
                "markers": ["fast"]
            },
            "optimization": {
                "path": "tests/unit/test_optimization_components.py",
                "description": "Optimization component tests (<1s each)",
                "timeout": 30,
                "markers": ["optimization"]
            },
            "integration": {
                "path": "tests/integration/",
                "description": "Integration tests with mocks and optimizations (1-5s each)",
                "timeout": 300,
                "markers": ["integration"]
            },
            "performance": {
                "path": "tests/performance/",
                "description": "Performance and load tests with optimizations (1-10min each)",
                "timeout": 1800,
                "markers": ["performance"]
            },
            "benchmark": {
                "path": "tests/benchmark/",
                "description": "Scientific benchmark validation tests (10-60min each)",
                "timeout": 3600,
                "markers": ["benchmark", "slow"]
            },
            "production": {
                "path": "tests/production/",
                "description": "Production tests with real MCP servers and optimizations (10-60min each)",
                "timeout": 7200,
                "markers": ["production", "slow"]
            }
        }
    
    def run_tests(self, categories: List[str], verbose: bool = True, 
                  max_failures: int = 5, parallel: bool = False) -> Dict[str, Any]:
        """Run specified test categories."""
        results = {}
        
        for category in categories:
            if category not in self.test_categories:
                print(f"Warning: Unknown test category '{category}'")
                continue
            
            config = self.test_categories[category]
            print(f"\n{'='*60}")
            print(f"Running {category.upper()} tests: {config['description']}")
            print(f"{'='*60}")
            
            start_time = time.time()
            
            try:
                # Build pytest command
                cmd = [
                    "python", "-m", "pytest",
                    config["path"],
                    "-v" if verbose else "",
                    f"--maxfail={max_failures}",
                    "--tb=short",
                    "--disable-warnings"
                ]
                
                if parallel and category in ["unit", "fast", "integration"]:
                    cmd.extend(["-n", "auto"])  # Use pytest-xdist for parallel execution
                
                # Add markers
                if config["markers"]:
                    markers = " or ".join(config["markers"])
                    cmd.extend(["-m", markers])
                
                # Remove empty strings
                cmd = [arg for arg in cmd if arg]
                
                print(f"Command: {' '.join(cmd)}")
                print()
                
                # Run tests
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=config["timeout"])
                
                execution_time = time.time() - start_time
                
                # Parse results
                results[category] = {
                    "success": result.returncode == 0,
                    "returncode": result.returncode,
                    "execution_time": execution_time,
                    "stdout": result.stdout,
                    "stderr": result.stderr
                }
                
                # Print results
                if result.returncode == 0:
                    print(f"✅ {category.upper()} tests PASSED in {execution_time:.2f}s")
                else:
                    print(f"❌ {category.upper()} tests FAILED in {execution_time:.2f}s")
                    print(f"Return code: {result.returncode}")
                
                if verbose and result.stdout:
                    print("\nSTDOUT:")
                    print(result.stdout)
                
                if result.stderr:
                    print("\nSTDERR:")
                    print(result.stderr)
                
            except subprocess.TimeoutExpired:
                execution_time = time.time() - start_time
                print(f"⏰ {category.upper()} tests TIMEOUT after {execution_time:.2f}s")
                results[category] = {
                    "success": False,
                    "returncode": -1,
                    "execution_time": execution_time,
                    "timeout": True
                }
            
            except Exception as e:
                execution_time = time.time() - start_time
                print(f"💥 {category.upper()} tests ERROR: {e}")
                results[category] = {
                    "success": False,
                    "returncode": -2,
                    "execution_time": execution_time,
                    "error": str(e)
                }
        
        return results
    
    def run_development_cycle(self) -> bool:
        """Run fast development cycle tests."""
        print("🚀 Running Development Cycle Tests")
        print("=" * 50)
        
        results = self.run_tests(["unit", "fast"], verbose=True, max_failures=3)
        
        # Check if all passed
        all_passed = all(result["success"] for result in results.values())
        
        if all_passed:
            print("\n✅ Development cycle tests PASSED - Ready for commit!")
        else:
            print("\n❌ Development cycle tests FAILED - Fix issues before commit!")
            for category, result in results.items():
                if not result["success"]:
                    print(f"  - {category}: FAILED")
        
        return all_passed
    
    def run_integration_tests(self) -> bool:
        """Run integration tests."""
        print("🔗 Running Integration Tests")
        print("=" * 50)
        
        results = self.run_tests(["unit", "fast", "integration"], verbose=True, max_failures=5)
        
        # Check if all passed
        all_passed = all(result["success"] for result in results.values())
        
        if all_passed:
            print("\n✅ Integration tests PASSED - Ready for deployment!")
        else:
            print("\n❌ Integration tests FAILED - Fix issues before deployment!")
            for category, result in results.items():
                if not result["success"]:
                    print(f"  - {category}: FAILED")
        
        return all_passed

    def run_comprehensive_tests(self) -> bool:
        """Run the curated comprehensive test suite."""
        print("Running Comprehensive Test Suite")
        print("=" * 50)

        results = self.run_tests(["comprehensive"], verbose=True, max_failures=3)

        all_passed = all(result["success"] for result in results.values())

        if all_passed:
            print("\nComprehensive test suite PASSED - Core functionality validated!")
        else:
            print("\nComprehensive test suite FAILED - Fix issues before release!")
            for category, result in results.items():
                if not result["success"]:
                    print(f"  - {category}: FAILED")

        return all_passed
    
    def run_performance_tests(self) -> bool:
        """Run performance tests."""
        print("⚡ Running Performance Tests")
        print("=" * 50)
        
        results = self.run_tests(["unit", "fast", "integration", "performance"], 
                               verbose=True, max_failures=3)
        
        # Check if all passed
        all_passed = all(result["success"] for result in results.values())
        
        if all_passed:
            print("\n✅ Performance tests PASSED - System performance validated!")
        else:
            print("\n❌ Performance tests FAILED - Performance issues detected!")
            for category, result in results.items():
                if not result["success"]:
                    print(f"  - {category}: FAILED")
        
        return all_passed
    
    def run_production_tests(self) -> bool:
        """Run production tests with real MCP servers."""
        print("🔬 Running Production Tests")
        print("=" * 50)
        print("⚠️  WARNING: These tests require real MCP servers and may take 1+ hours!")
        print("⚠️  Make sure all MCP servers are running before proceeding.")
        
        # Check if MCP servers are available
        if not self.check_mcp_servers():
            print("❌ MCP servers not available - skipping production tests")
            return False
        
        results = self.run_tests(["unit", "fast", "integration", "performance", "production"], 
                               verbose=True, max_failures=2)
        
        # Check if all passed
        all_passed = all(result["success"] for result in results.values())
        
        if all_passed:
            print("\n✅ Production tests PASSED - System ready for scientific use!")
        else:
            print("\n❌ Production tests FAILED - System not ready for production!")
            for category, result in results.items():
                if not result["success"]:
                    print(f"  - {category}: FAILED")
        
        return all_passed
    
    def check_mcp_servers(self) -> bool:
        """Check if MCP servers are available."""
        try:
            # Check if config file exists
            config_path = "config/mcp_servers.json"
            if not os.path.exists(config_path):
                print(f"❌ MCP config file not found: {config_path}")
                return False
            
            # Check if MCP server directories exist
            mcp_base_path = "/Users/omara.soliman/Documents/mcp/"
            if not os.path.exists(mcp_base_path):
                print(f"❌ MCP servers directory not found: {mcp_base_path}")
                return False
            
            # Check individual server directories
            servers = ["kegg", "reactome", "string", "proteinatlas"]
            for server in servers:
                server_path = os.path.join(mcp_base_path, server)
                if not os.path.exists(server_path):
                    print(f"❌ MCP server not found: {server_path}")
                    return False
            
            print("✅ MCP servers appear to be available")
            return True
            
        except Exception as e:
            print(f"❌ Error checking MCP servers: {e}")
            return False
    
    def print_summary(self, results: Dict[str, Any]):
        """Print test summary."""
        print("\n" + "="*60)
        print("TEST SUMMARY")
        print("="*60)
        
        total_time = sum(result.get("execution_time", 0) for result in results.values())
        
        for category, result in results.items():
            status = "✅ PASSED" if result["success"] else "❌ FAILED"
            time_str = f"{result.get('execution_time', 0):.2f}s"
            print(f"{category.upper():<15} {status:<10} {time_str}")
        
        print(f"\nTotal execution time: {total_time:.2f}s")
        
        # Overall status
        all_passed = all(result["success"] for result in results.values())
        if all_passed:
            print("\n🎉 ALL TESTS PASSED!")
        else:
            print("\n💥 SOME TESTS FAILED!")
            failed_categories = [cat for cat, result in results.items() if not result["success"]]
            print(f"Failed categories: {', '.join(failed_categories)}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="OmniTarget Pipeline Test Runner")
    parser.add_argument("--mode", choices=["dev", "integration", "performance", "production", "comprehensive", "all"], 
                       default="dev", help="Test mode to run")
    parser.add_argument("--categories", nargs="+", 
                       choices=["comprehensive", "unit", "fast", "optimization", "integration", "performance", "benchmark", "production"],
                       help="Specific test categories to run")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--max-failures", type=int, default=5, help="Maximum test failures")
    parser.add_argument("--parallel", action="store_true", help="Run tests in parallel")
    parser.add_argument("--check-mcp", action="store_true", help="Check MCP server availability")
    
    args = parser.parse_args()
    
    runner = TestRunner()
    
    # Check MCP servers if requested
    if args.check_mcp:
        if runner.check_mcp_servers():
            print("✅ MCP servers are available")
            sys.exit(0)
        else:
            print("❌ MCP servers are not available")
            sys.exit(1)
    
    # Run tests based on mode
    if args.mode == "dev":
        success = runner.run_development_cycle()
    elif args.mode == "integration":
        success = runner.run_integration_tests()
    elif args.mode == "performance":
        success = runner.run_performance_tests()
    elif args.mode == "production":
        success = runner.run_production_tests()
    elif args.mode == "comprehensive":
        success = runner.run_comprehensive_tests()
    elif args.mode == "all":
        success = runner.run_production_tests()
    else:
        # Run specific categories
        if args.categories:
            results = runner.run_tests(args.categories, args.verbose, args.max_failures, args.parallel)
            runner.print_summary(results)
            success = all(result["success"] for result in results.values())
        else:
            print("❌ No test categories specified")
            sys.exit(1)
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
