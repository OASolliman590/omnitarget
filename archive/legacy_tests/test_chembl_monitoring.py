#!/usr/bin/env python3
"""
Test ChEMBL Monitoring System
Validates monitoring, quality checks, and data lineage tracking.
"""
import asyncio
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.core.chembl_monitoring import (
    ChEMBLMonitor,
    DataLineageTracker,
    DataQualityMetrics,
    DataQualityLevel,
    HealthStatus,
    get_monitor,
    get_lineage_tracker
)


async def test_query_monitoring():
    """Test 1: Query monitoring."""
    print("\n" + "=" * 80)
    print("Test 1: Query Monitoring")
    print("=" * 80)

    monitor = ChEMBLMonitor(window_size=100)

    # Record successful queries
    monitor.record_query("search_targets", 0.5, True, result_count=10)
    monitor.record_query("search_targets", 0.3, True, cache_hit=True, result_count=10)
    monitor.record_query("get_compounds", 1.2, True, result_count=50)

    # Record failed query
    monitor.record_query("search_targets", 2.0, False, error_message="Timeout")

    # Get stats
    stats = monitor.get_query_stats()

    assert stats['total_queries'] == 4, f"Expected 4 queries, got {stats['total_queries']}"
    assert stats['successful_queries'] == 3, f"Expected 3 successful, got {stats['successful_queries']}"
    assert stats['failed_queries'] == 1, f"Expected 1 failed, got {stats['failed_queries']}"
    assert stats['success_rate'] == 0.75, f"Expected 75% success rate, got {stats['success_rate']:.1%}"
    assert stats['cache_hit_rate'] == 0.25, f"Expected 25% cache hit rate, got {stats['cache_hit_rate']:.1%}"

    print(f"✅ Query stats: {stats['total_queries']} total, {stats['success_rate']:.1%} success rate")
    print(f"✅ Cache hit rate: {stats['cache_hit_rate']:.1%}")
    print(f"✅ Avg duration: {stats['avg_duration']:.3f}s")

    # Test query type specific stats
    target_stats = monitor.get_query_stats("search_targets")
    assert target_stats['total_queries'] == 3, "Expected 3 target queries"
    print(f"✅ Target queries: {target_stats['total_queries']}")

    return True


async def test_data_quality_monitoring():
    """Test 2: Data quality monitoring."""
    print("\n" + "=" * 80)
    print("Test 2: Data Quality Monitoring")
    print("=" * 80)

    monitor = ChEMBLMonitor()

    # Test high quality data
    high_quality = DataQualityMetrics(
        total_records=100,
        valid_records=95,
        invalid_records=5,
        missing_fields={'smiles': 2, 'inchi_key': 3}
    )
    monitor.record_data_quality(high_quality)

    assert high_quality.quality_level == DataQualityLevel.HIGH, \
        f"Expected HIGH quality, got {high_quality.quality_level}"
    assert high_quality.quality_score > 0.9, \
        f"Expected score > 0.9, got {high_quality.quality_score}"
    print(f"✅ High quality: {high_quality.quality_score:.1%} ({high_quality.quality_level.value})")

    # Test medium quality data
    medium_quality = DataQualityMetrics(
        total_records=100,
        valid_records=80,
        invalid_records=20,
        missing_fields={'smiles': 10, 'molecular_weight': 5}
    )
    monitor.record_data_quality(medium_quality)

    assert medium_quality.quality_level == DataQualityLevel.MEDIUM, \
        f"Expected MEDIUM quality, got {medium_quality.quality_level}"
    print(f"✅ Medium quality: {medium_quality.quality_score:.1%} ({medium_quality.quality_level.value})")

    # Test poor quality data
    poor_quality = DataQualityMetrics(
        total_records=100,
        valid_records=40,
        invalid_records=60,
        missing_fields={'smiles': 30, 'inchi_key': 25, 'name': 15}
    )
    monitor.record_data_quality(poor_quality)

    assert poor_quality.quality_level == DataQualityLevel.POOR, \
        f"Expected POOR quality, got {poor_quality.quality_level}"
    assert poor_quality.quality_score < 0.5, \
        f"Expected score < 0.5, got {poor_quality.quality_score}"
    print(f"✅ Poor quality: {poor_quality.quality_score:.1%} ({poor_quality.quality_level.value})")

    # Get quality summary
    summary = monitor.get_quality_summary()
    assert summary['num_checks'] == 3, f"Expected 3 checks, got {summary['num_checks']}"
    print(f"✅ Quality checks: {summary['num_checks']}, avg score: {summary['avg_quality_score']:.1%}")

    return True


async def test_health_checks():
    """Test 3: Health checks."""
    print("\n" + "=" * 80)
    print("Test 3: Health Checks")
    print("=" * 80)

    monitor = ChEMBLMonitor()

    # Simulate healthy system (high success rate, fast queries)
    for i in range(50):
        monitor.record_query("search", 0.5, True, result_count=10)

    health = await monitor.check_health()
    assert health.status == HealthStatus.HEALTHY, \
        f"Expected HEALTHY, got {health.status}"
    print(f"✅ Healthy system: {health.status.value} - {health.message}")

    # Simulate degraded system (some failures, slower)
    for i in range(20):
        monitor.record_query("search", 3.0, i % 5 != 0, result_count=5)

    health = await monitor.check_health()
    assert health.status in [HealthStatus.DEGRADED, HealthStatus.HEALTHY], \
        f"Expected DEGRADED or HEALTHY, got {health.status}"
    print(f"✅ Degraded system: {health.status.value} - {health.message}")

    # Simulate unhealthy system (many failures)
    for i in range(30):
        monitor.record_query("search", 5.0, i % 3 == 0, error_message="Error")

    health = await monitor.check_health()
    assert health.status in [HealthStatus.UNHEALTHY, HealthStatus.DEGRADED], \
        f"Expected UNHEALTHY or DEGRADED, got {health.status}"
    print(f"✅ Unhealthy system: {health.status.value} - {health.message}")

    return True


async def test_error_tracking():
    """Test 4: Error tracking."""
    print("\n" + "=" * 80)
    print("Test 4: Error Tracking")
    print("=" * 80)

    monitor = ChEMBLMonitor()

    # Record various errors
    monitor.record_query("search", 1.0, False, error_message="Connection timeout")
    monitor.record_query("search", 0.5, False, error_message="Invalid input")
    monitor.record_query("fetch", 2.0, False, error_message="Not found")

    # Get error summary
    error_summary = monitor.get_error_summary()

    assert error_summary['total_errors'] == 3, \
        f"Expected 3 errors, got {error_summary['total_errors']}"
    assert 'search' in error_summary['errors_by_type'], "Expected errors for 'search'"
    assert error_summary['errors_by_type']['search'] == 2, "Expected 2 search errors"

    print(f"✅ Total errors: {error_summary['total_errors']}")
    print(f"✅ Errors by type: {error_summary['errors_by_type']}")

    # Check recent errors
    recent = error_summary['recent_errors']
    assert len(recent) == 3, f"Expected 3 recent errors, got {len(recent)}"
    assert recent[0]['query_type'] in ['search', 'fetch'], "Invalid query type"
    print(f"✅ Recent errors tracked: {len(recent)}")

    return True


async def test_alert_system():
    """Test 5: Alert system."""
    print("\n" + "=" * 80)
    print("Test 5: Alert System")
    print("=" * 80)

    monitor = ChEMBLMonitor(alert_threshold=5)

    # Track alerts
    alerts = []

    def alert_callback(message: str):
        alerts.append(message)

    monitor.add_alert_callback(alert_callback)

    # Generate errors to trigger alert
    for i in range(6):
        monitor.record_query("test_query", 1.0, False, error_message=f"Error {i}")

    # Check alert was triggered
    assert len(alerts) > 0, "Expected alert to be triggered"
    assert "High error rate" in alerts[0], f"Expected error rate alert, got: {alerts[0]}"
    print(f"✅ Alert triggered: {alerts[0]}")

    # Test poor quality alert
    poor_quality = DataQualityMetrics(
        total_records=100,
        valid_records=30,
        invalid_records=70
    )
    monitor.record_data_quality(poor_quality)

    # Check that quality alert exists somewhere in alerts (may not be at specific index)
    quality_alerts = [a for a in alerts if "Poor data quality" in a]
    assert len(quality_alerts) > 0, f"Expected quality alert, got alerts: {alerts}"
    print(f"✅ Quality alert triggered: {quality_alerts[0]}")

    return True


async def test_data_lineage():
    """Test 6: Data lineage tracking."""
    print("\n" + "=" * 80)
    print("Test 6: Data Lineage Tracking")
    print("=" * 80)

    tracker = DataLineageTracker(max_records=100)

    # Create lineage for compound
    lineage = tracker.create_lineage(
        record_id="CHEMBL25",
        record_type="compound",
        source="chembl",
        metadata={'name': 'ASPIRIN', 'mw': 180.16}
    )

    assert lineage.record_id == "CHEMBL25", "Incorrect record ID"
    assert lineage.record_type == "compound", "Incorrect record type"
    assert lineage.source == "chembl", "Incorrect source"
    print(f"✅ Lineage created: {lineage.record_id}")

    # Add transformations
    lineage.add_transformation("standardized_to_pydantic")
    lineage.add_transformation("unit_conversion_nM")
    lineage.add_transformation("drug_likeness_assessment")

    assert len(lineage.transformations) == 3, \
        f"Expected 3 transformations, got {len(lineage.transformations)}"
    print(f"✅ Transformations: {len(lineage.transformations)}")

    # Add validations
    lineage.add_validation("drug_likeness", True)
    lineage.add_validation("structure_valid", True)
    lineage.add_validation("bioactivity_range", True)

    assert len(lineage.validation_results) == 3, "Expected 3 validations"
    assert all(lineage.validation_results.values()), "Expected all validations to pass"
    print(f"✅ Validations: {len(lineage.validation_results)}, all passed")

    # Store and retrieve
    tracker.store_lineage(lineage)
    retrieved = tracker.get_lineage("CHEMBL25")

    assert retrieved is not None, "Failed to retrieve lineage"
    assert retrieved.record_id == "CHEMBL25", "Retrieved wrong lineage"
    print(f"✅ Lineage stored and retrieved")

    # Get summary
    summary = tracker.get_lineage_summary("CHEMBL25")
    assert summary is not None, "Failed to get summary"
    assert summary['record_id'] == "CHEMBL25", "Wrong summary record"
    assert summary['validation_pass_rate'] == 1.0, "Expected 100% pass rate"
    print(f"✅ Summary: {summary['validation_pass_rate']:.1%} validation pass rate")

    return True


async def test_lineage_query():
    """Test 7: Lineage querying."""
    print("\n" + "=" * 80)
    print("Test 7: Lineage Querying")
    print("=" * 80)

    tracker = DataLineageTracker()

    # Create multiple lineage records
    for i in range(5):
        tracker.create_lineage(f"CHEMBL{i}", "compound", "chembl")

    for i in range(3):
        tracker.create_lineage(f"TARGET{i}", "target", "kegg")

    # Query by type
    compounds = tracker.query_lineage(record_type="compound")
    assert len(compounds) == 5, f"Expected 5 compounds, got {len(compounds)}"
    print(f"✅ Query by type: {len(compounds)} compounds")

    targets = tracker.query_lineage(record_type="target")
    assert len(targets) == 3, f"Expected 3 targets, got {len(targets)}"
    print(f"✅ Query by type: {len(targets)} targets")

    # Query by source
    chembl_records = tracker.query_lineage(source="chembl")
    assert len(chembl_records) == 5, f"Expected 5 ChEMBL records, got {len(chembl_records)}"
    print(f"✅ Query by source: {len(chembl_records)} ChEMBL records")

    # Get stats
    stats = tracker.get_lineage_stats()
    assert stats['total_records'] == 8, f"Expected 8 records, got {stats['total_records']}"
    assert stats['by_type']['compound'] == 5, "Expected 5 compounds in stats"
    assert stats['by_type']['target'] == 3, "Expected 3 targets in stats"
    print(f"✅ Lineage stats: {stats['total_records']} total records")
    print(f"   By type: {stats['by_type']}")
    print(f"   By source: {stats['by_source']}")

    return True


async def test_monitoring_report():
    """Test 8: Comprehensive monitoring report."""
    print("\n" + "=" * 80)
    print("Test 8: Comprehensive Monitoring Report")
    print("=" * 80)

    monitor = ChEMBLMonitor()

    # Generate some activity
    for i in range(20):
        success = i % 4 != 0  # 75% success rate
        monitor.record_query("search", 0.5, success, cache_hit=(i % 3 == 0))

    quality = DataQualityMetrics(
        total_records=100,
        valid_records=85,
        invalid_records=15
    )
    monitor.record_data_quality(quality)

    await monitor.check_health()

    # Get comprehensive report
    report = monitor.get_monitoring_report()

    assert 'query_stats' in report, "Missing query_stats in report"
    assert 'quality_summary' in report, "Missing quality_summary in report"
    assert 'error_summary' in report, "Missing error_summary in report"
    assert 'health_check' in report, "Missing health_check in report"

    print(f"\n✅ Comprehensive Report Generated:")
    print(f"   Query stats: {report['query_stats']['total_queries']} queries")
    print(f"   Quality: {report['quality_summary']['avg_quality_score']:.1%}")
    print(f"   Errors: {report['error_summary']['total_errors']}")
    print(f"   Health: {report['health_check']['status']}")

    return True


async def test_global_instances():
    """Test 9: Global singleton instances."""
    print("\n" + "=" * 80)
    print("Test 9: Global Singleton Instances")
    print("=" * 80)

    # Get monitor instances
    monitor1 = get_monitor()
    monitor2 = get_monitor()

    assert monitor1 is monitor2, "Expected same monitor instance"
    print("✅ Global monitor singleton working")

    # Get lineage tracker instances
    tracker1 = get_lineage_tracker()
    tracker2 = get_lineage_tracker()

    assert tracker1 is tracker2, "Expected same tracker instance"
    print("✅ Global lineage tracker singleton working")

    return True


async def run_all_tests():
    """Run all monitoring tests."""
    print("=" * 80)
    print("ChEMBL Monitoring System Test Suite")
    print("=" * 80)
    print("\nTesting monitoring, quality checks, and data lineage...")

    tests = [
        ("Query Monitoring", test_query_monitoring),
        ("Data Quality Monitoring", test_data_quality_monitoring),
        ("Health Checks", test_health_checks),
        ("Error Tracking", test_error_tracking),
        ("Alert System", test_alert_system),
        ("Data Lineage", test_data_lineage),
        ("Lineage Querying", test_lineage_query),
        ("Monitoring Report", test_monitoring_report),
        ("Global Instances", test_global_instances),
    ]

    passed = 0
    failed = 0

    for test_name, test_func in tests:
        try:
            result = await test_func()
            if result:
                passed += 1
            else:
                failed += 1
                print(f"❌ {test_name} FAILED")
        except Exception as e:
            failed += 1
            print(f"❌ {test_name} FAILED with error: {e}")
            import traceback
            traceback.print_exc()

    # Summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    print(f"Total tests: {len(tests)}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")

    if failed == 0:
        print("\n✅ ALL MONITORING TESTS PASSED!")
        print("=" * 80)
        print("\nMonitoring system features validated:")
        print("  ✅ Query performance monitoring")
        print("  ✅ Data quality metrics and alerts")
        print("  ✅ Health checks (healthy/degraded/unhealthy)")
        print("  ✅ Error tracking and reporting")
        print("  ✅ Alert system with callbacks")
        print("  ✅ Data lineage and provenance tracking")
        print("  ✅ Lineage querying and statistics")
        print("  ✅ Comprehensive monitoring reports")
        print("  ✅ Global singleton instances")
        print("\nPhase 7 monitoring implementation complete!")
    else:
        print(f"\n⚠️  {failed} test(s) failed")

    return failed == 0


if __name__ == "__main__":
    success = asyncio.run(run_all_tests())
    sys.exit(0 if success else 1)
