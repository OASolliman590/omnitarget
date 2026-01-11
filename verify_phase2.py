#!/usr/bin/env python3
"""
Phase 2 Verification Script

Analyzes the latest pipeline results to verify tracking infrastructure is working.
"""

import json
from pathlib import Path
from typing import Dict, List, Any

def find_latest_result() -> Path:
    """Find the most recent results JSON file."""
    results_dir = Path('results')
    if not results_dir.exists():
        raise FileNotFoundError("results/ directory not found")

    json_files = list(results_dir.glob('*.json'))
    if not json_files:
        raise FileNotFoundError("No JSON files found in results/")

    return max(json_files, key=lambda p: p.stat().st_mtime)

def analyze_scenario(scenario_num: int, result: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze a single scenario result."""
    scenario_name = result.get('scenario_name', f'Scenario {scenario_num}')
    scenario_data = result.get('data', {})
    sources = scenario_data.get('data_sources')
    val_score = scenario_data.get('validation_score', 0)

    analysis = {
        'name': scenario_name,
        'has_tracking': sources is not None,
        'validation_score': val_score,
        'total_calls': 0,
        'successful_calls': 0,
        'failed_calls': 0,
        'sources': {},
        'issues': []
    }

    if sources:
        for s in sources:
            req = s.get('requested', 0)
            succ = s.get('successful', 0)
            fail = s.get('failed', 0)
            rate = s.get('success_rate', 0)
            errors = s.get('error_types', [])

            analysis['total_calls'] += req
            analysis['successful_calls'] += succ
            analysis['failed_calls'] += fail

            analysis['sources'][s['source_name']] = {
                'requested': req,
                'successful': succ,
                'failed': fail,
                'success_rate': rate,
                'errors': errors
            }

        # Check for issues
        if analysis['total_calls'] == 0:
            analysis['issues'].append("No MCP calls tracked (counters not incrementing)")
        elif analysis['failed_calls'] > analysis['total_calls'] * 0.5:
            analysis['issues'].append(f"High failure rate ({analysis['failed_calls']/analysis['total_calls']:.1%})")
    else:
        analysis['issues'].append("No tracking data (data_sources is None)")

    return analysis

def print_scenario_analysis(scenario_num: int, analysis: Dict[str, Any]):
    """Print detailed analysis for a scenario."""
    print(f"{'='*80}")
    print(f"Scenario {scenario_num}: {analysis['name']}")
    print(f"{'='*80}")

    if analysis['has_tracking']:
        print(f"✅ Tracking Active")
        print(f"   Total MCP Calls: {analysis['total_calls']}")
        print(f"   Successful: {analysis['successful_calls']} ({analysis['successful_calls']/max(analysis['total_calls'],1):.1%})")
        print(f"   Failed: {analysis['failed_calls']}")
        print(f"   Validation Score: {analysis['validation_score']:.3f}")
        print(f"\n   Database Breakdown:")

        for source_name, stats in analysis['sources'].items():
            req = stats['requested']
            succ = stats['successful']
            fail = stats['failed']
            rate = stats['success_rate']
            errors = stats['errors']

            if req > 0:
                status = "✅" if succ == req else "⚠️" if succ > 0 else "❌"
                print(f"     {status} {source_name:10s}: {succ}/{req} ({rate:.1%}) | {fail} failed")
                if errors:
                    print(f"        Errors: {', '.join(errors)}")
            elif source_name in ['chembl', 'uniprot']:
                print(f"     ⚪ {source_name:10s}: Not used in this scenario")
            else:
                print(f"     ⚠️  {source_name:10s}: 0 calls (unexpected)")

        if analysis['issues']:
            print(f"\n   ⚠️  WARNINGS:")
            for issue in analysis['issues']:
                print(f"     - {issue}")
    else:
        print(f"❌ NO TRACKING DATA")
        print(f"   Validation Score: {analysis['validation_score']:.3f}")
        print(f"\n   ⚠️  ERROR: {analysis['issues'][0]}")

    print()

def print_summary(analyses: List[Dict[str, Any]], total_mcp_calls: int):
    """Print overall summary."""
    scenarios_with_tracking = sum(1 for a in analyses if a['has_tracking'])
    all_pass = all(len(a['issues']) == 0 for a in analyses)

    print("="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Scenarios with tracking: {scenarios_with_tracking}/6")
    print(f"Total MCP calls tracked: {total_mcp_calls}")
    print(f"\nPhase 2 Status: ", end="")

    if scenarios_with_tracking == 6 and total_mcp_calls > 0:
        print("✅ COMPLETE - All scenarios tracking")
    elif scenarios_with_tracking == 6:
        print("⚠️  PARTIAL - Infrastructure present but counters at 0")
    else:
        print("❌ INCOMPLETE - Some scenarios missing tracking")

    print(f"Verification: {'✅ PASSED' if all_pass else '❌ ISSUES FOUND'}")
    print()

def print_comparison(analyses: List[Dict[str, Any]]):
    """Print comparison with Run 8 expectations."""
    expected_changes = [
        ("S1", 0.020, (0.45, 0.65), "Tracking fixed"),
        ("S2", 0.033, (0.40, 0.60), "Tracking fixed"),
        ("S3", 0.680, (0.55, 0.75), "Penalty will apply"),
        ("S4", 0.480, (0.40, 0.60), "Penalty will apply"),
        ("S5", 0.370, (0.35, 0.55), "Penalty will apply"),
        ("S6", 1.000, (0.60, 0.80), "False perfect fixed"),
    ]

    print("="*80)
    print("COMPARISON WITH RUN 8 EXPECTATIONS")
    print("="*80)

    print(f"\n{'Scenario':<8} {'Run 8':<8} {'Expected':<12} {'Actual':<8} {'Status':<10} {'Notes'}")
    print("-"*80)

    for i, (name, run8_score, (low, high), notes) in enumerate(expected_changes):
        actual_score = analyses[i]['validation_score']
        in_range = low <= actual_score <= high
        status = "✅" if in_range else "⚠️"
        expected_str = f"{low}-{high}"

        print(f"{name:<8} {run8_score:<8.3f} {expected_str:<12} {actual_score:<8.3f} {status:<10} {notes}")

    print()

def print_next_steps(analyses: List[Dict[str, Any]], total_mcp_calls: int):
    """Print recommended next steps."""
    scenarios_with_tracking = sum(1 for a in analyses if a['has_tracking'])
    all_pass = all(len(a['issues']) == 0 for a in analyses)

    print("="*80)
    print("NEXT STEPS")
    print("="*80)

    if all_pass and scenarios_with_tracking == 6:
        print("✅ Phase 2 verification PASSED!")
        print("\nReady to proceed with:")
        print("  1. Update RUN8_TODO.md - mark Phase 2 complete")
        print("  2. Begin Phase 3 - Fix validation penalty logic")
        print("  3. Document results in PHASE2_TEST_RESULTS.md")
    else:
        print("⚠️  Phase 2 verification found issues")
        print("\nAction items:")
        if scenarios_with_tracking < 6:
            print("  - Check scenarios with no tracking data")
            print("  - Verify data_sources passed to result models")
        if total_mcp_calls == 0:
            print("  - Debug why counters are not incrementing")
            print("  - Check _call_with_tracking is being called")
            print("  - Verify data_sources dict is initialized")
        if any(len(a['issues']) > 0 for a in analyses):
            print("  - Review scenarios with warnings")
            print("  - Run with --verbose to see [DATA_TRACKING] logs")

    print("="*80)

def main():
    """Main verification function."""
    try:
        # Find latest result
        latest_result = find_latest_result()
        print(f"Analyzing: {latest_result.name}")
        print("="*80)

        # Load data
        with open(latest_result) as f:
            data = json.load(f)

        print(f"\nTotal scenarios: {len(data['results'])}")
        print(f"Execution: {data['execution_metadata']['timestamp']}\n")

        # Analyze each scenario
        analyses = []
        total_mcp_calls = 0

        for i, result in enumerate(data['results'], 1):
            analysis = analyze_scenario(i, result)
            analyses.append(analysis)
            total_mcp_calls += analysis['total_calls']
            print_scenario_analysis(i, analysis)

        # Print summary
        print_summary(analyses, total_mcp_calls)

        # Print comparison
        print_comparison(analyses)

        # Print next steps
        print_next_steps(analyses, total_mcp_calls)

    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        print("\nMake sure to run the pipeline first:")
        print("  python -m src.cli yaml examples/yaml_configs/axl_breast_cancer_all_6_scenarios.yaml")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()
