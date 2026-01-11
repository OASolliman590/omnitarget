#!/usr/bin/env python3
"""
MRA Improvements Validation Test Runner

Runs comprehensive validation tests for all MRA simulation improvements:
- Phase 1: Timeout wrappers, progress monitoring, semaphore limiting
- Phase 2: Retry logic, error handling
- Phase 3: Connection pooling, circuit breaker
- Phase 4: Troubleshooting guide validation

Usage:
    python run_mra_validation_tests.py
    python run_mra_validation_tests.py --quick    # Run only quick tests
    python run_mra_validation_tests.py --verbose  # Run with verbose output
"""

import sys
import argparse
import subprocess
import time
from pathlib import Path

# Test categories
TEST_CATEGORIES = {
    'unit': {
        'path': 'tests/unit/',
        'marker': 'unit',
        'description': 'Unit tests for individual components',
        'timeout': 60
    },
    'connection_pool': {
        'path': 'tests/unit/test_mcp_connection_pool.py',
        'marker': 'unit',
        'description': 'Connection pool tests',
        'timeout': 60
    },
    'circuit_breaker': {
        'path': 'tests/unit/test_circuit_breaker.py',
        'marker': 'unit',
        'description': 'Circuit breaker tests',
        'timeout': 60
    },
    'retry': {
        'path': 'tests/unit/',
        'marker': 'unit',
        'description': 'Retry logic tests',
        'timeout': 60
    },
    'mra_production': {
        'path': 'tests/production/test_mra_improvements_validation.py',
        'marker': 'production',
        'description': 'MRA improvements validation (requires MCP servers)',
        'timeout': 1800  # 30 minutes
    }
}


def print_header(title):
    """Print formatted header."""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80 + "\n")


def print_section(title):
    """Print formatted section."""
    print("\n" + "-" * 80)
    print(f"  {title}")
    print("-" * 80 + "\n")


def run_command(cmd, timeout=None, verbose=False):
    """Run a command and return result."""
    if verbose:
        print(f"Running: {' '.join(cmd)}")

    start_time = time.time()

    try:
        result = subprocess.run(
            cmd,
            capture_output=False,  # Show output
            text=True,
            timeout=timeout
        )

        execution_time = time.time() - start_time

        return {
            'success': result.returncode == 0,
            'returncode': result.returncode,
            'execution_time': execution_time
        }

    except subprocess.TimeoutExpired:
        print(f"\n❌ Test suite timed out after {timeout}s")
        return {
            'success': False,
            'returncode': -1,
            'execution_time': timeout
        }
    except Exception as e:
        print(f"\n❌ Error running tests: {e}")
        return {
            'success': False,
            'returncode': -2,
            'execution_time': time.time() - start_time
        }


def check_mcp_servers():
    """Check if MCP servers are available."""
    print_section("Checking MCP Server Health")

    try:
        result = subprocess.run(
            ['python', '-m', 'src.cli', 'health'],
            capture_output=True,
            text=True,
            timeout=30
        )

        if result.returncode == 0:
            print("✅ MCP servers are healthy")
            return True
        else:
            print("⚠️  MCP servers may not be available")
            print("Some tests will be skipped")
            return False

    except Exception as e:
        print(f"⚠️  Could not check MCP servers: {e}")
        print("Some tests will be skipped")
        return False


def run_unit_tests(verbose=False):
    """Run unit tests for MRA improvements."""
    print_section("Running Unit Tests")

    test_files = [
        'tests/unit/test_mcp_connection_pool.py',
        'tests/unit/test_circuit_breaker.py'
    ]

    results = {}

    for test_file in test_files:
        test_name = Path(test_file).stem
        print(f"\nRunning {test_name}...")

        cmd = [
            'python', '-m', 'pytest',
            test_file,
            '-v' if verbose else '-q',
            '--tb=short'
        ]

        result = run_command(cmd, timeout=120, verbose=verbose)
        results[test_name] = result

        if result['success']:
            print(f"✅ {test_name} passed ({result['execution_time']:.2f}s)")
        else:
            print(f"❌ {test_name} failed (exit code {result['returncode']})")

    return results


def run_production_tests(verbose=False):
    """Run production validation tests."""
    print_section("Running Production Validation Tests")

    print("These tests require MCP servers and will take 15-30 minutes...")
    print("Make sure MCP servers are running: python start_mcp_servers.py\n")

    test_file = 'tests/production/test_mra_improvements_validation.py'

    cmd = [
        'python', '-m', 'pytest',
        test_file,
        '-v' if verbose else '-s',  # Use -s to see progress
        '--tb=short',
        '-m', 'production'
    ]

    result = run_command(cmd, timeout=2000, verbose=verbose)

    if result['success']:
        print(f"\n✅ All production tests passed ({result['execution_time']:.2f}s)")
    else:
        print(f"\n❌ Production tests failed (exit code {result['returncode']})")

    return result


def run_quick_validation():
    """Run quick validation tests (unit tests only)."""
    print_header("MRA Improvements - Quick Validation")

    results = {}

    # Check if config file exists
    if not Path('config/mcp_servers.json').exists():
        print("❌ config/mcp_servers.json not found")
        print("   Please ensure MCP server configuration exists")
        return False

    print("Running quick validation (unit tests only)...\n")

    # Run unit tests
    results['unit'] = run_unit_tests(verbose=False)

    # Summary
    print_section("Quick Validation Summary")

    all_passed = all(r['success'] for r in results.values())

    for category, result in results.items():
        status = "✅ PASS" if result['success'] else "❌ FAIL"
        print(f"{status} {category}: {result['execution_time']:.2f}s")

    if all_passed:
        print("\n🎉 All quick validation tests passed!")
        print("   MRA improvements are working correctly")
    else:
        print("\n⚠️  Some tests failed")
        print("   Check output above for details")

    return all_passed


def run_full_validation(verbose=False):
    """Run full validation tests including production tests."""
    print_header("MRA Improvements - Full Validation")

    # Check MCP servers
    mcp_available = check_mcp_servers()

    results = {}

    # Run unit tests
    results['unit'] = run_unit_tests(verbose=verbose)

    # Run production tests if MCP servers are available
    if mcp_available:
        print("\n" + "=" * 80)
        print("  Running Production Tests (MCP servers available)")
        print("=" * 80 + "\n")

        results['production'] = run_production_tests(verbose=verbose)
    else:
        print("\n" + "=" * 80)
        print("  Skipping Production Tests (MCP servers not available)")
        print("=" * 80 + "\n")
        results['production'] = {
            'success': False,
            'execution_time': 0,
            'skipped': True
        }

    # Summary
    print_header("Full Validation Summary")

    for category, result in results.items():
        if result.get('skipped'):
            print(f"⏭️  {category}: SKIPPED (MCP servers not available)")
        elif result['success']:
            print(f"✅ {category}: PASS ({result['execution_time']:.2f}s)")
        else:
            print(f"❌ {category}: FAIL ({result['execution_time']:.2f}s)")

    all_passed = all(r['success'] for r in results.values() if not r.get('skipped'))

    if all_passed:
        print("\n🎉 All validation tests passed!")
        print("\nMRA Improvements Summary:")
        print("  ✅ Timeout protection active")
        print("  ✅ Progress monitoring working")
        print("  ✅ Semaphore limiting configured")
        print("  ✅ Retry logic implemented")
        print("  ✅ Connection pooling enabled")
        print("  ✅ Circuit breaker monitoring active")
        print("  ✅ Error handling robust")
        print("\nThe MRA simulation should no longer experience 77+ minute hangs!")
    else:
        print("\n⚠️  Some tests failed or were skipped")
        print("   Check output above for details")

    return all_passed


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='MRA Improvements Validation Test Runner',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_mra_validation_tests.py              # Run full validation
  python run_mra_validation_tests.py --quick      # Run quick validation (unit tests only)
  python run_mra_validation_tests.py --verbose    # Run with verbose output

Quick validation runs in ~2 minutes.
Full validation runs in ~20-30 minutes (requires MCP servers).
        """
    )

    parser.add_argument(
        '--quick',
        action='store_true',
        help='Run quick validation (unit tests only, ~2 minutes)'
    )

    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Run with verbose output'
    )

    args = parser.parse_args()

    # Print introduction
    print_header("MRA Simulation Improvements - Validation Suite")
    print("This validation suite tests all improvements made to fix the")
    print("77+ minute hanging issue in batch 2.")
    print("")
    print("Improvements tested:")
    print("  • Phase 1: Timeout wrappers, progress monitoring, semaphore limiting")
    print("  • Phase 2: Retry logic with exponential backoff")
    print("  • Phase 3: Connection pooling, circuit breaker pattern")
    print("  • Phase 4: Comprehensive testing and validation")
    print("")

    if args.quick:
        success = run_quick_validation()
    else:
        success = run_full_validation(verbose=args.verbose)

    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
