import math

from src.core.validation import DataValidator
from src.models.data_models import DataSourceStatus, CompletenessMetrics


def make_status(source_name, requested, successful, failed):
    success_rate = successful / requested if requested else 0.0
    return DataSourceStatus(
        source_name=source_name,
        requested=requested,
        successful=successful,
        failed=failed,
        success_rate=success_rate,
        error_types=[],
    )


def test_penalty_missing_data_sources():
    validator = DataValidator()
    penalty, details = validator.calculate_data_completeness_penalty(None, None)

    assert penalty == 0.3
    assert details["penalty_breakdown"]["missing_tracking"] == 0.3


def test_penalty_empty_data_sources():
    validator = DataValidator()
    penalty, details = validator.calculate_data_completeness_penalty([], None)

    assert penalty == 0.3
    assert details["penalty_breakdown"]["empty_tracking"] == 0.3


def test_penalty_all_calls_fail_capped():
    validator = DataValidator()
    data_sources = [
        make_status("kegg", 10, 0, 10),
        make_status("hpa", 20, 0, 20),
        make_status("string", 15, 0, 15),
    ]

    penalty, _ = validator.calculate_data_completeness_penalty(data_sources, None)

    assert math.isclose(penalty, 0.3)


def test_penalty_partial_success_proportional():
    validator = DataValidator()
    data_sources = [
        make_status("kegg", 10, 5, 5),   # 0.5 success rate, threshold 0.8
        make_status("hpa", 20, 15, 5),   # 0.75 success rate, threshold 0.5
    ]

    penalty, details = validator.calculate_data_completeness_penalty(data_sources, None)

    assert 0.1 <= penalty <= 0.15
    assert "kegg" in details["penalty_breakdown"]
    assert "hpa" not in details["penalty_breakdown"]


def test_penalty_all_success_zero():
    validator = DataValidator()
    data_sources = [
        make_status("kegg", 10, 10, 0),
        make_status("hpa", 20, 20, 0),
        make_status("string", 5, 5, 0),
    ]

    penalty, _ = validator.calculate_data_completeness_penalty(data_sources, None)
    assert penalty == 0.0


def test_penalty_cap_with_many_failures():
    validator = DataValidator()
    data_sources = [
        make_status("kegg", 50, 0, 50),
        make_status("hpa", 50, 0, 50),
        make_status("string", 50, 0, 50),
        make_status("chembl", 50, 0, 50),
    ]

    penalty, _ = validator.calculate_data_completeness_penalty(data_sources, None)
    assert penalty == 0.3


def test_penalty_zero_requests_handled():
    validator = DataValidator()
    data_sources = [
        make_status("kegg", 0, 0, 0),
    ]

    penalty, _ = validator.calculate_data_completeness_penalty(data_sources, None)

    # Zero requests should still incur a deficit penalty (0.8 threshold -> 0.2 penalty)
    assert math.isclose(penalty, 0.2)


def test_penalty_completeness_metrics_included():
    validator = DataValidator()
    data_sources = [
        make_status("kegg", 10, 10, 0),
    ]
    completeness = CompletenessMetrics(
        expression_data=0.3,
        pathology_data=None,
        network_data=0.6,
        pathway_data=0.7,
        drug_data=None,
        overall_completeness=0.0,
    )

    penalty, details = validator.calculate_data_completeness_penalty(data_sources, completeness)

    assert penalty > 0.0
    assert "expression_completeness" in details["penalty_breakdown"]
    assert "network_completeness" in details["penalty_breakdown"]
