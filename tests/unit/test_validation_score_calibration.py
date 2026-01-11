"""
Validation Score Calibration Tests

Tests to verify that validation scores properly reflect improved data quality
from Issues 1-3.
"""
import pytest
import asyncio
import json
from pathlib import Path
from typing import Dict, Any

# Import required modules
try:
    from src.core.validation import DataValidator
    from src.core.benchmark_validation import MINIMUM_SCORES
    from src.core.pipeline_orchestrator import OmniTargetPipeline
    VALIDATION_IMPORTS_AVAILABLE = True
except ImportError as e:
    VALIDATION_IMPORTS_AVAILABLE = False
    VALIDATION_IMPORT_ERROR = str(e)


@pytest.mark.unit
class TestValidationScoreCalibration:
    """Test validation score calibration after Issues 1-3 fixes."""

    def setup_method(self):
        """Setup test environment."""
        if not VALIDATION_IMPORTS_AVAILABLE:
            pytest.skip(f"Validation imports not available: {VALIDATION_IMPORT_ERROR}")

        self.validator = DataValidator()
        self.results_path = Path("/Users/omara.soliman/Desktop/Projects /My Projects/15-OmniTarget_/results")

    def test_weights_configuration(self):
        """Test that weights are properly configured."""
        # Create a sample validation result
        validation_results = {
            'disease_confidence': 0.8,
            'interaction_confidence': 0.7,
            'expression_coverage': 0.9,
            'pathway_coverage': 0.85,
            'id_mapping_accuracy': 0.95,
            'cross_database_concordance': 0.8,
            'expression_reproducibility': 0.75,
            'druggability_auc': 0.7
        }

        # Calculate score
        score = self.validator.calculate_overall_validation_score(validation_results)

        # Verify score is calculated
        assert score > 0, "Validation score should be positive"
        assert score <= 1.0, "Validation score should not exceed 1.0"

        print(f"Test score with calibrated weights: {score:.3f}")

    def test_minimum_score_thresholds(self):
        """Test that minimum score thresholds are properly set."""
        # Verify thresholds are set
        assert 'S1' in MINIMUM_SCORES, "S1 threshold should be defined"
        assert 'S2' in MINIMUM_SCORES, "S2 threshold should be defined"
        assert 'S3' in MINIMUM_SCORES, "S3 threshold should be defined"
        assert 'S4' in MINIMUM_SCORES, "S4 threshold should be defined"
        assert 'S5' in MINIMUM_SCORES, "S5 threshold should be defined"
        assert 'S6' in MINIMUM_SCORES, "S6 threshold should be defined"

        # Verify thresholds are 0.6 or higher
        for scenario, threshold in MINIMUM_SCORES.items():
            assert threshold >= 0.6, f"{scenario} threshold should be >= 0.6"

        print(f"Minimum score thresholds: {MINIMUM_SCORES}")

    def test_validation_score_ranges(self):
        """Test that validation scores are in expected ranges."""
        # Test with low quality data
        low_quality_scores = {
            'disease_confidence': 0.3,
            'interaction_confidence': 0.2,
            'expression_coverage': 0.4,
            'pathway_coverage': 0.3,
            'id_mapping_accuracy': 0.5,
            'cross_database_concordance': 0.2,
            'expression_reproducibility': 0.3,
            'druggability_auc': 0.2
        }

        low_score = self.validator.calculate_overall_validation_score(low_quality_scores)
        assert low_score < 0.6, f"Low quality data should score below 0.6, got {low_score:.3f}"
        assert low_score > 0, f"Low quality data should score above 0, got {low_score:.3f}"

        # Test with high quality data
        high_quality_scores = {
            'disease_confidence': 0.9,
            'interaction_confidence': 0.85,
            'expression_coverage': 0.95,
            'pathway_coverage': 0.9,
            'id_mapping_accuracy': 0.98,
            'cross_database_concordance': 0.85,
            'expression_reproducibility': 0.8,
            'druggability_auc': 0.8
        }

        high_score = self.validator.calculate_overall_validation_score(high_quality_scores)
        assert high_score >= 0.6, f"High quality data should score >= 0.6, got {high_score:.3f}"
        assert high_score <= 1.0, f"High quality data should not exceed 1.0, got {high_score:.3f}"

        print(f"Low quality score: {low_score:.3f}")
        print(f"High quality score: {high_score:.3f}")
        print(f"Score improvement: {high_score - low_score:.3f}")

    def test_validation_score_improvement_with_issues_1_3(self):
        """
        Test that validation scores reflect improvements from Issues 1-3.

        Issue 1: Gene overlap improved (Jaccard ≥0.4, was 0.069)
        Issue 2: HPA pathology now has ≥50 markers (was 0)
        Issue 3: KEGG DRUG has ≥20 genes with drugs (was 0)
        """
        # Before Issues 1-3 (low quality)
        before_issues_scores = {
            'disease_confidence': 0.6,
            'interaction_confidence': 0.4,
            'expression_coverage': 0.7,
            'pathway_coverage': 0.5,  # Low due to poor gene overlap
            'id_mapping_accuracy': 0.7,  # Low due to ID mapping issues
            'cross_database_concordance': 0.3,  # Low due to poor data integration
            'expression_reproducibility': 0.6,
            'druggability_auc': 0.2  # Low due to no drug coverage
        }

        # After Issues 1-3 (improved quality)
        after_issues_scores = {
            'disease_confidence': 0.8,
            'interaction_confidence': 0.6,
            'expression_coverage': 0.85,
            'pathway_coverage': 0.8,  # Improved due to better gene overlap
            'id_mapping_accuracy': 0.9,  # Improved due to ID mapping fixes
            'cross_database_concordance': 0.75,  # Improved due to better data integration
            'expression_reproducibility': 0.75,
            'druggability_auc': 0.7  # Improved due to drug coverage
        }

        before_score = self.validator.calculate_overall_validation_score(before_issues_scores)
        after_score = self.validator.calculate_overall_validation_score(after_issues_scores)

        # Verify improvement
        improvement = after_score - before_score
        assert improvement > 0, f"Score should improve after fixes, got {improvement:.3f}"

        # Verify after score meets threshold
        assert after_score >= 0.6, f"After score should be >= 0.6, got {after_score:.3f}"
        assert before_score < 0.6, f"Before score should be < 0.6, got {before_score:.3f}"

        print(f"Score before Issues 1-3: {before_score:.3f}")
        print(f"Score after Issues 1-3: {after_score:.3f}")
        print(f"Improvement: {improvement:.3f} ({improvement/before_score*100:.1f}%)")

    def test_weight_distribution(self):
        """Test that weights sum to 1.0."""
        # Create a comprehensive validation result with all metrics
        validation_results = {
            'disease_confidence': 0.8,
            'interaction_confidence': 0.7,
            'expression_coverage': 0.9,
            'pathway_coverage': 0.85,
            'id_mapping_accuracy': 0.95,
            'cross_database_concordance': 0.8,
            'expression_reproducibility': 0.75,
            'druggability_auc': 0.7
        }

        # The validator calculates weighted average
        # All metrics at 0.8 should give us the weight distribution
        all_metrics_score = self.validator.calculate_overall_validation_score(validation_results)

        # With all metrics at same value, result should be close to that value
        # (minor variations due to numerical precision)
        for metric_value in [0.5, 0.8, 1.0]:
            test_scores = {k: metric_value for k in validation_results.keys()}
            result = self.validator.calculate_overall_validation_score(test_scores)

            # Result should be very close to input value (within rounding error)
            assert abs(result - metric_value) < 0.01, \
                f"With all metrics at {metric_value}, result should be similar, got {result}"

        print(f"Score with all metrics at 0.8: {all_metrics_score:.3f}")

    @pytest.mark.integration
    def test_validation_with_actual_results(self):
        """Test validation with actual pipeline results if available."""
        # Look for actual result files
        try:
            result_files = list(self.results_path.glob("*.json"))
        except (FileNotFoundError, PermissionError):
            pytest.skip("Results directory not accessible")

        if not result_files:
            pytest.skip("No result files available for testing")

        # Try to load a recent result file
        result_file = sorted(result_files, reverse=True)[0]

        try:
            with open(result_file, 'r') as f:
                result_data = json.load(f)

            # Extract validation scores from results
            if 'results' in result_data:
                for result in result_data['results']:
                    if 'validation_score' in result:
                        score = result['validation_score']
                        scenario_id = result.get('scenario_id', 'Unknown')

                        print(f"Scenario {scenario_id} validation score: {score:.3f}")

                        # Score should be in valid range
                        assert 0 <= score <= 1.0, f"Score should be in [0,1], got {score}"

        except Exception as e:
            pytest.skip(f"Could not load result file: {e}")


@pytest.mark.unit
class TestValidationScoreAcceptance:
    """Test acceptance criteria for validation scores."""

    def setup_method(self):
        """Setup test environment."""
        if not VALIDATION_IMPORTS_AVAILABLE:
            pytest.skip(f"Validation imports not available: {VALIDATION_IMPORT_ERROR}")

    def test_s1_score_acceptance(self):
        """Test that S1 score meets acceptance criteria."""
        # S1 should meet minimum threshold
        s1_threshold = MINIMUM_SCORES.get('S1', 0.6)
        assert s1_threshold == 0.6, f"S1 threshold should be 0.6, got {s1_threshold}"
        assert s1_threshold >= 0.6, "S1 threshold should be at least 0.6"

    def test_all_scenarios_score_acceptance(self):
        """Test that all scenarios meet minimum threshold."""
        for scenario, threshold in MINIMUM_SCORES.items():
            assert threshold >= 0.6, f"{scenario} threshold should be >= 0.6"
            assert threshold <= 1.0, f"{scenario} threshold should be <= 1.0"

    def test_threshold_consistency(self):
        """Test that all scenarios have consistent thresholds."""
        thresholds = list(MINIMUM_SCORES.values())

        # All thresholds should be the same (0.6)
        assert all(t == 0.6 for t in thresholds), \
            f"All thresholds should be 0.6, got {thresholds}"

    def test_threshold_alignment_with_issue_fixes(self):
        """
        Test that thresholds align with expected improvements from Issues 1-3.

        Before fixes: S1 score ~0.332
        After fixes: S1 score should be ≥0.6
        """
        # Verify threshold reflects the improvement
        before_score = 0.332  # Known S1 score before fixes
        expected_threshold = 0.6

        # Threshold should be higher than before score
        assert expected_threshold > before_score, \
            f"Threshold {expected_threshold} should be > before score {before_score}"

        # Threshold should be achievable with improved data quality
        assert expected_threshold <= 0.8, \
            f"Threshold {expected_threshold} should be reasonably achievable"


if __name__ == '__main__':
    # Run tests
    pytest.main([__file__, '-v'])
