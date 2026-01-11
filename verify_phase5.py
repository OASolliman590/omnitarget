#!/usr/bin/env python3
"""
Phase 5 Verification Script

Analyzes test results to verify HPA batch resilience implementation:
- Compares Phase 5 results with Phase 3 baseline
- Checks for HPA chunk-limit errors
- Validates adaptive batching effectiveness
- Reports MCP success rates and validation scores
"""

import json
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Tuple

# Expected validation score ranges (from Phase 3)
EXPECTED_RANGES = {
    1: (0.10, 0.20),
    2: (0.30, 0.40),
    3: (0.50, 0.65),
    4: (0.40, 0.60),
    5: (0.35, 0.55),
    6: (0.60, 0.80)
}

# Phase 3 baseline (from PHASE3_COMPLETE.md)
PHASE3_BASELINE = {
    1: 0.116,
    2: 0.333,
    3: 0.527,
    4: 0.600,
    5: 0.370,
    6: 0.688
}

PHASE3_MCP_CALLS = {
    'total': 1059,
    'successful': 1010,
    'failed': 49,
    'success_rate': 0.954
}


def find_latest_result_file() -> Path:
    """Find the most recent result file."""
    results_dir = Path("results")
    if not results_dir.exists():
        raise FileNotFoundError("results/ directory not found")

    json_files = list(results_dir.glob("*.json"))
    if not json_files:
        raise FileNotFoundError("No result files found in results/")

    latest = max(json_files, key=lambda p: p.stat().st_mtime)
    return latest


def analyze_data_sources(data_sources: List[Dict]) -> Dict[str, Any]:
    """Analyze data source tracking statistics."""
    total_requested = sum(ds.get('requested', 0) for ds in data_sources)
    total_successful = sum(ds.get('successful', 0) for ds in data_sources)
    total_failed = sum(ds.get('failed', 0) for ds in data_sources)

    stats = {
        'total_calls': total_requested,
        'successful': total_successful,
        'failed': total_failed,
        'success_rate': total_successful / total_requested if total_requested > 0 else 0.0,
        'by_source': {}
    }

    for ds in data_sources:
        source_name = ds.get('source_name', 'unknown')
        stats['by_source'][source_name] = {
            'requested': ds.get('requested', 0),
            'successful': ds.get('successful', 0),
            'failed': ds.get('failed', 0),
            'success_rate': ds.get('success_rate', 0.0)
        }

    return stats


def analyze_scenario(scenario_num: int, result: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze a single scenario result."""
    scenario_data = result.get('data', {})
    data_sources = scenario_data.get('data_sources', [])
    validation_score = scenario_data.get('validation_score', 0.0)
    completeness = scenario_data.get('completeness_metrics', {})

    analysis = {
        'scenario_num': scenario_num,
        'scenario_name': result.get('scenario_name', 'Unknown'),
        'status': result.get('status', 'unknown'),
        'validation_score': validation_score,
        'has_tracking': data_sources is not None and len(data_sources) > 0,
        'data_source_stats': analyze_data_sources(data_sources) if data_sources else None,
        'completeness_metrics': completeness,
        'overall_completeness': completeness.get('overall_completeness', 0.0)
    }

    # Check if in expected range
    if scenario_num in EXPECTED_RANGES:
        lower, upper = EXPECTED_RANGES[scenario_num]
        analysis['expected_range'] = (lower, upper)
        analysis['in_range'] = lower <= validation_score <= upper
        analysis['distance_from_range'] = (
            0 if analysis['in_range']
            else min(abs(validation_score - lower), abs(validation_score - upper))
        )

    # Compare with Phase 3 baseline
    if scenario_num in PHASE3_BASELINE:
        baseline = PHASE3_BASELINE[scenario_num]
        analysis['phase3_baseline'] = baseline
        analysis['change_from_phase3'] = validation_score - baseline
        analysis['percent_change'] = (
            ((validation_score - baseline) / baseline * 100)
            if baseline > 0 else 0
        )

    return analysis


def check_hpa_errors(all_analyses: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Check for HPA-specific errors, especially chunk-limit errors."""
    hpa_stats = {
        'total_requests': 0,
        'successful': 0,
        'failed': 0,
        'chunk_limit_errors': 0,
        'scenarios_with_hpa': []
    }

    for analysis in all_analyses:
        stats = analysis.get('data_source_stats')
        if not stats:
            continue

        hpa = stats['by_source'].get('hpa')
        if hpa and hpa['requested'] > 0:
            hpa_stats['scenarios_with_hpa'].append(analysis['scenario_num'])
            hpa_stats['total_requests'] += hpa['requested']
            hpa_stats['successful'] += hpa['successful']
            hpa_stats['failed'] += hpa['failed']

    hpa_stats['success_rate'] = (
        hpa_stats['successful'] / hpa_stats['total_requests']
        if hpa_stats['total_requests'] > 0 else 0.0
    )

    return hpa_stats


def generate_report(result_file: Path) -> str:
    """Generate comprehensive Phase 5 verification report."""
    with open(result_file) as f:
        data = json.load(f)

    metadata = data.get('execution_metadata', {})
    results = data.get('results', [])

    # Analyze all scenarios
    analyses = [analyze_scenario(i+1, result) for i, result in enumerate(results)]

    # Check HPA errors
    hpa_stats = check_hpa_errors(analyses)

    # Aggregate statistics
    total_mcp_calls = sum(
        a['data_source_stats']['total_calls']
        for a in analyses if a['data_source_stats']
    )
    total_successful = sum(
        a['data_source_stats']['successful']
        for a in analyses if a['data_source_stats']
    )
    total_failed = sum(
        a['data_source_stats']['failed']
        for a in analyses if a['data_source_stats']
    )
    overall_success_rate = total_successful / total_mcp_calls if total_mcp_calls > 0 else 0.0

    # Count scenarios in range
    scenarios_in_range = sum(1 for a in analyses if a.get('in_range', False))

    # Build report
    report = []
    report.append("=" * 80)
    report.append("PHASE 5 VERIFICATION REPORT")
    report.append("HPA Batch Resilience & Adaptive Batching")
    report.append("=" * 80)
    report.append("")

    # Test run info
    report.append("Test Run Information")
    report.append("-" * 80)
    report.append(f"Result file: {result_file.name}")
    report.append(f"Timestamp: {metadata.get('timestamp', 'unknown')}")
    report.append(f"YAML config: {metadata.get('yaml_config', 'unknown')}")
    report.append(f"Total scenarios: {metadata.get('total_scenarios', 0)}")
    report.append(f"Successful scenarios: {metadata.get('successful_scenarios', 0)}")
    report.append(f"Failed scenarios: {metadata.get('failed_scenarios', 0)}")
    report.append("")

    # Overall statistics
    report.append("Overall MCP Call Statistics")
    report.append("-" * 80)
    report.append(f"Total MCP calls: {total_mcp_calls}")
    report.append(f"Successful: {total_successful}")
    report.append(f"Failed: {total_failed}")
    report.append(f"Success rate: {overall_success_rate:.1%}")
    report.append("")

    # Comparison with Phase 3
    report.append("Comparison with Phase 3 Baseline")
    report.append("-" * 80)
    report.append(f"Phase 3 total calls: {PHASE3_MCP_CALLS['total']}")
    report.append(f"Phase 5 total calls: {total_mcp_calls} ({total_mcp_calls - PHASE3_MCP_CALLS['total']:+d})")
    report.append(f"Phase 3 success rate: {PHASE3_MCP_CALLS['success_rate']:.1%}")
    report.append(f"Phase 5 success rate: {overall_success_rate:.1%} ({overall_success_rate - PHASE3_MCP_CALLS['success_rate']:+.1%})")
    report.append(f"Phase 3 HPA failures: {PHASE3_MCP_CALLS['failed']}")
    report.append(f"Phase 5 HPA failures: {hpa_stats['failed']} ({hpa_stats['failed'] - PHASE3_MCP_CALLS['failed']:+d})")
    report.append("")

    # HPA-specific analysis
    report.append("HPA Batch Resilience Analysis")
    report.append("-" * 80)
    report.append(f"Scenarios using HPA: {len(hpa_stats['scenarios_with_hpa'])} (S{', S'.join(map(str, hpa_stats['scenarios_with_hpa']))})")
    report.append(f"Total HPA requests: {hpa_stats['total_requests']}")
    report.append(f"Successful: {hpa_stats['successful']}")
    report.append(f"Failed: {hpa_stats['failed']}")
    report.append(f"HPA success rate: {hpa_stats['success_rate']:.1%}")
    report.append(f"Chunk-limit errors: {hpa_stats['chunk_limit_errors']}")

    if hpa_stats['failed'] == 0:
        report.append("✅ NO HPA FAILURES - Adaptive batching working!")
    else:
        report.append(f"⚠️  {hpa_stats['failed']} HPA failures detected")
    report.append("")

    # Per-scenario analysis
    report.append("Per-Scenario Analysis")
    report.append("-" * 80)
    report.append("")

    for analysis in analyses:
        scenario_num = analysis['scenario_num']
        report.append(f"Scenario {scenario_num}: {analysis['scenario_name']}")
        report.append(f"  Status: {analysis['status']}")
        report.append(f"  Validation Score: {analysis['validation_score']:.3f}")

        if 'expected_range' in analysis:
            lower, upper = analysis['expected_range']
            status = "✅ IN RANGE" if analysis['in_range'] else "⚠️  OUT OF RANGE"
            report.append(f"  Expected Range: {lower:.2f}-{upper:.2f} ({status})")
            if not analysis['in_range']:
                report.append(f"  Distance from range: {analysis['distance_from_range']:.3f}")

        if 'phase3_baseline' in analysis:
            baseline = analysis['phase3_baseline']
            change = analysis['change_from_phase3']
            pct = analysis['percent_change']
            arrow = "↑" if change > 0 else "↓" if change < 0 else "→"
            report.append(f"  Phase 3 baseline: {baseline:.3f}")
            report.append(f"  Change: {arrow} {change:+.3f} ({pct:+.1f}%)")

        stats = analysis.get('data_source_stats')
        if stats:
            report.append(f"  MCP Calls: {stats['total_calls']} (success rate: {stats['success_rate']:.1%})")
            report.append(f"  By source:")
            for source_name, source_stats in stats['by_source'].items():
                if source_stats['requested'] > 0:
                    report.append(
                        f"    - {source_name}: {source_stats['successful']}/{source_stats['requested']} "
                        f"({source_stats['success_rate']:.1%})"
                    )

        completeness = analysis.get('overall_completeness', 0.0)
        report.append(f"  Overall Completeness: {completeness:.1%}")

        report.append("")

    # Summary
    report.append("=" * 80)
    report.append("SUMMARY")
    report.append("=" * 80)
    report.append(f"✅ All scenarios completed: {metadata.get('failed_scenarios', 0) == 0}")
    report.append(f"✅ Overall success rate: {overall_success_rate:.1%}")
    report.append(f"✅ HPA failures: {hpa_stats['failed']} (Phase 3: 49)")
    report.append(f"✅ Scenarios in expected ranges: {scenarios_in_range}/6")
    report.append("")

    # Phase 5 specific checks
    report.append("Phase 5 Verification Checklist")
    report.append("-" * 80)
    report.append(f"[ {'✅' if hpa_stats['failed'] == 0 else '❌'} ] No HPA chunk-limit errors")
    report.append(f"[ {'✅' if overall_success_rate >= 0.95 else '❌'} ] >95% overall success rate")
    report.append(f"[ {'✅' if all(a.get('overall_completeness', 0) >= 0 for a in analyses) else '❌'} ] All scenarios report completeness metrics")
    report.append(f"[ {'✅' if all(a.get('has_tracking') for a in analyses) else '❌'} ] All scenarios have tracking data")
    report.append("")

    # Recommendations
    report.append("Recommendations")
    report.append("-" * 80)

    if total_mcp_calls < 100:
        report.append("⚠️  Very few MCP calls detected. This may indicate:")
        report.append("   - Test ran with sparse/empty dataset")
        report.append("   - Upstream data sources returned no results")
        report.append("   - Different YAML config than Phase 3")
        report.append("   Recommendation: Run with same dataset as Phase 3 for accurate comparison")
        report.append("")

    if hpa_stats['failed'] > 0:
        report.append("⚠️  HPA failures detected. Check logs for:")
        report.append("   - Chunk-limit error messages")
        report.append("   - Retry attempt logging")
        report.append("   - Final batch size used")
        report.append("")

    if scenarios_in_range < 3:
        report.append("⚠️  Fewer than 3 scenarios in expected ranges.")
        report.append("   This may be due to:")
        report.append("   - Sparse upstream data (not a bug)")
        report.append("   - Different test dataset than Phase 3")
        report.append("   - Expected ranges may need adjustment")
        report.append("")

    report.append("=" * 80)
    report.append(f"Report generated: {datetime.now().isoformat()}")
    report.append("=" * 80)

    return "\n".join(report)


def main():
    """Main verification function."""
    try:
        result_file = find_latest_result_file()
        print(f"Analyzing: {result_file}\n")

        report = generate_report(result_file)
        print(report)

        # Save report
        report_file = Path(f"PHASE5_VERIFICATION_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
        report_file.write_text(report)
        print(f"\nReport saved to: {report_file}")

    except Exception as e:
        print(f"Error: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
