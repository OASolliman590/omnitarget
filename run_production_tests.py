#!/usr/bin/env python3
"""
Production Test Runner

Run production validation tests with real MCP servers and optimization components.
"""

import asyncio
import argparse
import logging
import time
import json
from pathlib import Path
from typing import Dict, List, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('production_test_results.log')
    ]
)

logger = logging.getLogger(__name__)


class ProductionTestRunner:
    """Production test runner for OmniTarget pipeline."""
    
    def __init__(self):
        self.test_categories = {
            "validation": {
                "path": "tests/production/test_production_validation.py",
                "description": "Production validation tests with real MCP servers",
                "timeout": 1800,  # 30 minutes
                "markers": ["production"]
            },
            "readiness": {
                "path": "tests/production/test_production_validation.py::TestProductionReadiness",
                "description": "Production readiness checklist",
                "timeout": 300,  # 5 minutes
                "markers": ["production"]
            }
        }
    
    def run_tests(self, categories: List[str], verbose: bool = True, 
                  max_failures: int = 3) -> Dict[str, Any]:
        """Run specified test categories."""
        results = {}
        
        for category in categories:
            if category not in self.test_categories:
                logger.warning(f"Unknown test category: {category}")
                continue
            
            config = self.test_categories[category]
            logger.info(f"🚀 Running {category.upper()} tests: {config['description']}")
            
            start_time = time.time()
            
            try:
                # Build pytest command
                cmd = [
                    "python", "-m", "pytest",
                    config["path"],
                    "-v" if verbose else "",
                    f"--maxfail={max_failures}",
                    "--tb=short",
                    "--disable-warnings",
                    f"-m {' and '.join(config['markers'])}"
                ]
                
                # Remove empty strings
                cmd = [arg for arg in cmd if arg]
                
                # Run tests
                import subprocess
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=config["timeout"])
                
                execution_time = time.time() - start_time
                
                if result.returncode == 0:
                    logger.info(f"✅ {category.upper()} tests PASSED in {execution_time:.2f}s")
                    results[category] = {
                        "status": "PASSED",
                        "time": execution_time,
                        "output": result.stdout
                    }
                else:
                    logger.error(f"❌ {category.upper()} tests FAILED in {execution_time:.2f}s")
                    results[category] = {
                        "status": "FAILED",
                        "time": execution_time,
                        "output": result.stdout,
                        "error": result.stderr
                    }
                
            except subprocess.TimeoutExpired:
                execution_time = time.time() - start_time
                logger.error(f"⏰ {category.upper()} tests TIMEOUT after {execution_time:.2f}s")
                results[category] = {
                    "status": "TIMEOUT",
                    "time": execution_time,
                    "error": f"Timeout after {config['timeout']}s"
                }
            except Exception as e:
                execution_time = time.time() - start_time
                logger.error(f"💥 {category.upper()} tests ERROR: {e}")
                results[category] = {
                    "status": "ERROR",
                    "time": execution_time,
                    "error": str(e)
                }
        
        return results
    
    def generate_report(self, results: Dict[str, Any]) -> str:
        """Generate production test report."""
        report = []
        report.append("=" * 80)
        report.append("OMNITARGET PIPELINE - PRODUCTION VALIDATION REPORT")
        report.append("=" * 80)
        report.append(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # Summary
        total_tests = len(results)
        passed_tests = sum(1 for r in results.values() if r["status"] == "PASSED")
        failed_tests = sum(1 for r in results.values() if r["status"] == "FAILED")
        timeout_tests = sum(1 for r in results.values() if r["status"] == "TIMEOUT")
        error_tests = sum(1 for r in results.values() if r["status"] == "ERROR")
        
        report.append("📊 SUMMARY")
        report.append("-" * 40)
        report.append(f"Total Test Categories: {total_tests}")
        report.append(f"✅ Passed: {passed_tests}")
        report.append(f"❌ Failed: {failed_tests}")
        report.append(f"⏰ Timeout: {timeout_tests}")
        report.append(f"💥 Error: {error_tests}")
        report.append("")
        
        # Detailed results
        report.append("📋 DETAILED RESULTS")
        report.append("-" * 40)
        
        for category, result in results.items():
            status_icon = {
                "PASSED": "✅",
                "FAILED": "❌", 
                "TIMEOUT": "⏰",
                "ERROR": "💥"
            }.get(result["status"], "❓")
            
            report.append(f"{status_icon} {category.upper()}: {result['status']} ({result['time']:.2f}s)")
            
            if result["status"] != "PASSED" and "error" in result:
                report.append(f"   Error: {result['error']}")
        
        report.append("")
        
        # Production readiness assessment
        if passed_tests >= total_tests * 0.8:  # 80% pass rate
            report.append("🎉 PRODUCTION READY")
            report.append("The pipeline is ready for production deployment.")
        elif passed_tests >= total_tests * 0.6:  # 60% pass rate
            report.append("⚠️  PRODUCTION READY WITH CAUTIONS")
            report.append("The pipeline is mostly ready but has some issues to address.")
        else:
            report.append("🚫 NOT PRODUCTION READY")
            report.append("The pipeline requires significant fixes before production deployment.")
        
        report.append("")
        report.append("=" * 80)
        
        return "\n".join(report)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="OmniTarget Pipeline Production Test Runner")
    parser.add_argument("--categories", nargs="+", 
                       choices=["validation", "readiness", "all"],
                       default=["all"],
                       help="Test categories to run")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--max-failures", type=int, default=3, help="Maximum test failures")
    parser.add_argument("--report", action="store_true", help="Generate detailed report")
    
    args = parser.parse_args()
    
    # Expand "all" category
    if "all" in args.categories:
        args.categories = ["validation", "readiness"]
    
    # Initialize runner
    runner = ProductionTestRunner()
    
    # Run tests
    logger.info("🚀 Starting Production Validation Tests")
    logger.info("=" * 60)
    
    results = runner.run_tests(
        categories=args.categories,
        verbose=args.verbose,
        max_failures=args.max_failures
    )
    
    # Generate report
    if args.report:
        report = runner.generate_report(results)
        print("\n" + report)
        
        # Save report to file
        with open("production_validation_report.txt", "w") as f:
            f.write(report)
        logger.info("📄 Report saved to: production_validation_report.txt")
    
    # Determine exit code
    failed_tests = sum(1 for r in results.values() if r["status"] in ["FAILED", "TIMEOUT", "ERROR"])
    exit_code = 0 if failed_tests == 0 else 1
    
    if exit_code == 0:
        logger.info("🎉 All production tests passed!")
    else:
        logger.error(f"💥 {failed_tests} production test categories failed!")
    
    return exit_code


if __name__ == "__main__":
    exit(main())
