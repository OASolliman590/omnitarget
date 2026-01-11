#!/usr/bin/env python3
"""
Phase 3 Verification Script

Analyzes pipeline results to verify validation penalty logic fixes are working correctly.
Compares Phase 3 scores with Phase 2 baseline and expected ranges.
"""

import json
from pathlib import Path
from typing import Dict, List, Any, Tuple

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

def print_scenario_analysis(scenario_num: int, analysis: Dict[str, Any], phase2_score: float):
    """Print detailed analysis for a scenario with Phase 2 comparison."""
    print(f"{'='*80}")
    print(f"Scenario {scenario_num}: {analysis['name']}")
    print(f"{'='*80}")

    if analysis['has_tracking']:
        print(f"✅ Tracking Active")
        print(f"   Total MCP Calls: {analysis['total_calls']}")
        print(f"   Successful: {analysis['successful_calls']} ({analysis['successful_calls']/max(analysis['total_calls'],1):.1%})")
        print(f"   Failed: {analysis['failed_calls']}")

        # Score comparison
        score_change = analysis['validation_score'] - phase2_score
        change_symbol = "📈" if score_change > 0 else "📉" if score_change < 0 else "➡️"
        print(f"\n   Validation Score: {analysis['validation_score']:.3f} (Phase 2: {phase2_score:.3f}) {change_symbol} {score_change:+.3f}")

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
        print(f"   Validation Score: {analysis['validation_score']:.3f} (Phase 2: {phase2_score:.3f})")
        print(f"\n   ⚠️  ERROR: {analysis['issues'][0]}")

    print()

def print_comparison(analyses: List[Dict[str, Any]], phase2_scores: List[float]):
    """Print comparison with Phase 2 results and expected ranges."""
    expected_ranges = [
        ("S1", 0.116, (0.45, 0.65), "Penalty logic fixed"),
        ("S2", 0.333, (0.40, 0.60), "Penalty logic fixed"),
        ("S3", 0.618, (0.55, 0.75), "Should remain stable"),
        ("S4", 0.480, (0.40, 0.60), "Should remain stable"),
        ("S5", 0.370, (0.35, 0.55), "Should remain stable"),
        ("S6", 1.000, (0.60, 0.80), "False perfect fixed"),
    ]

    print("="*80)
    print("PHASE 3 RESULTS vs PHASE 2 BASELINE")
    print("="*80)

    print(f"\n{'Scenario':<8} {'Phase 2':<10} {'Phase 3':<10} {'Change':<10} {'Expected':<12} {'Status':<10} {'Notes'}")
    print("-"*80)

    for i, (name, phase2_score, (low, high), notes) in enumerate(expected_ranges):
        actual_score = analyses[i]['validation_score']
        change = actual_score - phase2_score
        in_range = low <= actual_score <= high
        status = "✅" if in_range else "⚠️"
        expected_str = f"{low}-{high}"
        change_str = f"{change:+.3f}"

        print(f"{name:<8} {phase2_score:<10.3f} {actual_score:<10.3f} {change_str:<10} {expected_str:<12} {status:<10} {notes}")

    print()

def print_phase3_specific_checks(analyses: List[Dict[str, Any]]):
    """Print Phase 3 specific verification checks."""
    print("="*80)
    print("PHASE 3 SPECIFIC CHECKS")
    print("="*80)
    print()

    checks = []

    # Check 1: S6 no longer false perfect
    s6_score = analyses[5]['validation_score']
    if s6_score < 1.0:
        checks.append(("✅", "S6 false perfect fixed", f"Score dropped from 1.000 to {s6_score:.3f}"))
    else:
        checks.append(("❌", "S6 still showing false perfect", f"Score remains at {s6_score:.3f}"))

    # Check 2: S1/S2 scores improved
    s1_improved = analyses[0]['validation_score'] > 0.116
    s2_improved = analyses[1]['validation_score'] > 0.333
    if s1_improved or s2_improved:
        checks.append(("✅", "S1/S2 scores improved", f"S1: {s1_improved}, S2: {s2_improved}"))
    else:
        checks.append(("⚠️", "S1/S2 scores not improved", "May need further investigation"))

    # Check 3: All scenarios have tracking
    all_tracking = all(a['has_tracking'] for a in analyses)
    if all_tracking:
        checks.append(("✅", "All scenarios have tracking data", "6/6 scenarios"))
    else:
        no_tracking = sum(1 for a in analyses if not a['has_tracking'])
        checks.append(("❌", f"{no_tracking} scenarios missing tracking", "Critical issue"))

    # Check 4: S3/S4/S5 remain stable
    s3_stable = 0.55 <= analyses[2]['validation_score'] <= 0.75
    s4_stable = 0.40 <= analyses[3]['validation_score'] <= 0.60
    s5_stable = 0.35 <= analyses[4]['validation_score'] <= 0.55
    if s3_stable and s4_stable and s5_stable:
        checks.append(("✅", "S3/S4/S5 remain in expected ranges", "No unintended changes"))
    else:
        unstable = [i+3 for i, stable in enumerate([s3_stable, s4_stable, s5_stable]) if not stable]
        checks.append(("⚠️", f"S{unstable} scores shifted unexpectedly", "Review needed"))

    # Check 5: Penalty logic working
    # HPA failures in S6 should cause penalty
    s6_hpa_failures = analyses[5]['sources'].get('hpa', {}).get('failed', 0)
    if s6_hpa_failures > 0 and s6_score < 1.0:
        checks.append(("✅", "Penalty logic applying for HPA failures", f"{s6_hpa_failures} failures detected, score < 1.0"))
    elif s6_hpa_failures > 0 and s6_score == 1.0:
        checks.append(("❌", "Penalty logic not working", f"{s6_hpa_failures} failures but score = 1.0"))
    else:
        checks.append(("⚪", "No HPA failures to test penalty logic", "Cannot verify"))

    # Print checks
    for status, check_name, detail in checks:
        print(f"{status} {check_name}")
        print(f"   {detail}")
        print()

def print_summary(analyses: List[Dict[str, Any]], total_mcp_calls: int, phase2_scores: List[float]):
    """Print overall summary."""
    scenarios_in_range = sum(1 for i, a in enumerate(analyses)
                              if check_score_in_range(i, a['validation_score']))

    print("="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Scenarios with tracking: {sum(1 for a in analyses if a['has_tracking'])}/6")
    print(f"Total MCP calls tracked: {total_mcp_calls}")
    print(f"Scenarios in expected ranges: {scenarios_in_range}/6")
    print(f"\nPhase 3 Status: ", end="")

    if scenarios_in_range == 6 and total_mcp_calls > 0:
        print("✅ COMPLETE - All scores in expected ranges")
    elif scenarios_in_range >= 4:
        print("⚠️  PARTIAL - Most scenarios in range, some need review")
    else:
        print("❌ INCOMPLETE - Multiple scenarios outside expected ranges")

    # Calculate average score change
    score_changes = [analyses[i]['validation_score'] - phase2_scores[i] for i in range(6)]
    avg_change = sum(score_changes) / len(score_changes)
    print(f"\nAverage score change from Phase 2: {avg_change:+.3f}")

    print()

def check_score_in_range(scenario_idx: int, score: float) -> bool:
    """Check if score is in expected range for scenario."""
    ranges = [
        (0.45, 0.65),  # S1
        (0.40, 0.60),  # S2
        (0.55, 0.75),  # S3
        (0.40, 0.60),  # S4
        (0.35, 0.55),  # S5
        (0.60, 0.80),  # S6
    ]
    low, high = ranges[scenario_idx]
    return low <= score <= high

def print_next_steps(analyses: List[Dict[str, Any]], total_mcp_calls: int):
    """Print recommended next steps."""
    scenarios_in_range = sum(1 for i, a in enumerate(analyses)
                              if check_score_in_range(i, a['validation_score']))

    print("="*80)
    print("NEXT STEPS")
    print("="*80)

    if scenarios_in_range == 6:
        print("✅ Phase 3 verification PASSED!")
        print("\nReady to proceed with:")
        print("  1. Update RUN8_TODO.md - confirm Phase 3 complete")
        print("  2. Create PHASE3_TEST_RESULTS.md - document results")
        print("  3. Begin Phase 4 - Completeness metrics coverage")
    elif scenarios_in_range >= 4:
        print("⚠️  Phase 3 mostly successful, some adjustments needed")
        print("\nAction items:")
        out_of_range = [i+1 for i, a in enumerate(analyses)
                        if not check_score_in_range(i, a['validation_score'])]
        print(f"  - Review scenarios {out_of_range} - scores outside expected ranges")
        print("  - Analyze penalty calculation logic for these scenarios")
        print("  - May need to adjust expected ranges or penalty weights")
    else:
        print("❌ Phase 3 verification found issues")
        print("\nAction items:")
        print("  - Review validation penalty calculation logic")
        print("  - Check if penalty is being applied correctly")
        print("  - Verify data_sources are being passed correctly")
        print("  - May need to debug with --verbose flag")

    print("="*80)

def main():
    """Main verification function."""
    try:
        # Phase 2 baseline scores (from previous run)
        phase2_scores = [0.116, 0.333, 0.618, 0.480, 0.370, 1.000]

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
            print_scenario_analysis(i, analysis, phase2_scores[i-1])

        # Print Phase 3 specific checks
        print_phase3_specific_checks(analyses)

        # Print comparison
        print_comparison(analyses, phase2_scores)

        # Print summary
        print_summary(analyses, total_mcp_calls, phase2_scores)

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
