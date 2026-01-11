"""
Statistical Utilities for OmniTarget Pipeline

Provides statistical significance testing, confidence intervals, and multiple testing correction
for rigorous scientific validation of pipeline results.

Author: OmniTarget Team
Date: 2025-01-06
"""

from typing import Callable, Dict, List, Optional, Tuple, Any, Union
import numpy as np
from scipy import stats
from scipy.stats import mannwhitneyu, wilcoxon, pearsonr, spearmanr
from statsmodels.stats.multitest import multipletests
import logging
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class TestAlternative(Enum):
    """Alternative hypothesis for statistical tests."""
    TWO_SIDED = 'two-sided'
    GREATER = 'greater'
    LESS = 'less'


class CorrectionMethod(Enum):
    """Multiple testing correction methods."""
    FDR_BH = 'fdr_bh'  # Benjamini-Hochberg FDR
    FDR_BY = 'fdr_by'  # Benjamini-Yekutieli FDR
    BONFERRONI = 'bonferroni'
    HOLM = 'holm'
    SIDAK = 'sidak'
    NONE = 'none'


@dataclass
class StatisticalTestResult:
    """Results from a statistical significance test."""
    test_name: str
    statistic: float
    p_value: float
    is_significant: bool
    alpha: float
    alternative: str
    effect_size: Optional[float] = None
    confidence_interval: Optional[Tuple[float, float]] = None
    n_permutations: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'test_name': self.test_name,
            'statistic': self.statistic,
            'p_value': self.p_value,
            'is_significant': self.is_significant,
            'alpha': self.alpha,
            'alternative': self.alternative,
            'effect_size': self.effect_size,
            'confidence_interval': self.confidence_interval,
            'n_permutations': self.n_permutations
        }


@dataclass
class MultipleTestingResult:
    """Results from multiple testing correction."""
    original_pvalues: np.ndarray
    corrected_pvalues: np.ndarray
    rejected: np.ndarray
    method: str
    alpha: float
    n_significant_original: int
    n_significant_corrected: int

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'method': self.method,
            'alpha': self.alpha,
            'n_tests': len(self.original_pvalues),
            'n_significant_original': self.n_significant_original,
            'n_significant_corrected': self.n_significant_corrected,
            'original_pvalues': self.original_pvalues.tolist(),
            'corrected_pvalues': self.corrected_pvalues.tolist(),
            'rejected': self.rejected.tolist()
        }


class StatisticalUtils:
    """Statistical utilities for OmniTarget pipeline."""

    @staticmethod
    def permutation_test(
        observed_statistic: float,
        data: Union[np.ndarray, List],
        statistic_func: Callable[[np.ndarray], float],
        n_permutations: int = 10000,
        alternative: TestAlternative = TestAlternative.TWO_SIDED,
        alpha: float = 0.05,
        random_seed: Optional[int] = None
    ) -> StatisticalTestResult:
        """
        Perform permutation test for statistical significance.

        Args:
            observed_statistic: The observed test statistic
            data: Data to permute (e.g., scores, measurements)
            statistic_func: Function that computes test statistic from data
            n_permutations: Number of permutations (default: 10000)
            alternative: Alternative hypothesis
            alpha: Significance level (default: 0.05)
            random_seed: Random seed for reproducibility

        Returns:
            StatisticalTestResult with p-value and significance

        Example:
            >>> data = np.array([0.8, 0.9, 0.7, 0.85, 0.95])
            >>> observed = np.mean(data)
            >>> result = StatisticalUtils.permutation_test(
            ...     observed, data, np.mean, n_permutations=10000
            ... )
            >>> print(f"p-value: {result.p_value}, significant: {result.is_significant}")
        """
        if random_seed is not None:
            np.random.seed(random_seed)

        data = np.asarray(data)

        # Generate null distribution
        null_statistics = []
        for _ in range(n_permutations):
            permuted_data = np.random.permutation(data)
            null_stat = statistic_func(permuted_data)
            null_statistics.append(null_stat)

        null_statistics = np.array(null_statistics)

        # Calculate p-value based on alternative hypothesis
        if alternative == TestAlternative.TWO_SIDED:
            p_value = np.mean(np.abs(null_statistics) >= np.abs(observed_statistic))
        elif alternative == TestAlternative.GREATER:
            p_value = np.mean(null_statistics >= observed_statistic)
        elif alternative == TestAlternative.LESS:
            p_value = np.mean(null_statistics <= observed_statistic)
        else:
            raise ValueError(f"Unknown alternative: {alternative}")

        is_significant = p_value < alpha

        return StatisticalTestResult(
            test_name='permutation_test',
            statistic=observed_statistic,
            p_value=float(p_value),
            is_significant=is_significant,
            alpha=alpha,
            alternative=alternative.value,
            n_permutations=n_permutations
        )

    @staticmethod
    def bootstrap_confidence_interval(
        data: Union[np.ndarray, List],
        statistic_func: Callable[[np.ndarray], float],
        confidence_level: float = 0.95,
        n_bootstrap: int = 10000,
        method: str = 'percentile',
        random_seed: Optional[int] = None
    ) -> Tuple[float, Tuple[float, float]]:
        """
        Calculate bootstrap confidence interval for a statistic.

        Args:
            data: Data to bootstrap
            statistic_func: Function that computes statistic (e.g., np.mean)
            confidence_level: Confidence level (default: 0.95 for 95% CI)
            n_bootstrap: Number of bootstrap samples (default: 10000)
            method: Method for CI calculation ('percentile' or 'bca')
            random_seed: Random seed for reproducibility

        Returns:
            Tuple of (point_estimate, (lower_bound, upper_bound))

        Example:
            >>> data = np.array([0.8, 0.9, 0.7, 0.85, 0.95])
            >>> estimate, ci = StatisticalUtils.bootstrap_confidence_interval(
            ...     data, np.mean, confidence_level=0.95
            ... )
            >>> print(f"Mean: {estimate:.3f}, 95% CI: [{ci[0]:.3f}, {ci[1]:.3f}]")
        """
        if random_seed is not None:
            np.random.seed(random_seed)

        data = np.asarray(data)
        n = len(data)

        # Calculate observed statistic
        observed_stat = statistic_func(data)

        # Generate bootstrap distribution
        bootstrap_stats = []
        for _ in range(n_bootstrap):
            bootstrap_sample = np.random.choice(data, size=n, replace=True)
            bootstrap_stat = statistic_func(bootstrap_sample)
            bootstrap_stats.append(bootstrap_stat)

        bootstrap_stats = np.array(bootstrap_stats)

        # Calculate confidence interval
        alpha = 1 - confidence_level

        if method == 'percentile':
            lower_percentile = (alpha / 2) * 100
            upper_percentile = (1 - alpha / 2) * 100
            ci_lower = np.percentile(bootstrap_stats, lower_percentile)
            ci_upper = np.percentile(bootstrap_stats, upper_percentile)
        elif method == 'bca':
            # Bias-corrected and accelerated (BCa) bootstrap
            # Calculate bias correction
            n_less = np.sum(bootstrap_stats < observed_stat)
            z0 = stats.norm.ppf(n_less / n_bootstrap)

            # Calculate acceleration
            jackknife_stats = []
            for i in range(n):
                jackknife_sample = np.delete(data, i)
                jackknife_stat = statistic_func(jackknife_sample)
                jackknife_stats.append(jackknife_stat)

            jackknife_stats = np.array(jackknife_stats)
            jackknife_mean = np.mean(jackknife_stats)
            numerator = np.sum((jackknife_mean - jackknife_stats) ** 3)
            denominator = 6 * (np.sum((jackknife_mean - jackknife_stats) ** 2) ** 1.5)
            a = numerator / denominator if denominator != 0 else 0

            # Calculate adjusted percentiles
            z_alpha_2 = stats.norm.ppf(alpha / 2)
            z_1_alpha_2 = stats.norm.ppf(1 - alpha / 2)

            lower_percentile = stats.norm.cdf(z0 + (z0 + z_alpha_2) / (1 - a * (z0 + z_alpha_2))) * 100
            upper_percentile = stats.norm.cdf(z0 + (z0 + z_1_alpha_2) / (1 - a * (z0 + z_1_alpha_2))) * 100

            ci_lower = np.percentile(bootstrap_stats, lower_percentile)
            ci_upper = np.percentile(bootstrap_stats, upper_percentile)
        else:
            raise ValueError(f"Unknown method: {method}")

        return observed_stat, (float(ci_lower), float(ci_upper))

    @staticmethod
    def correct_multiple_testing(
        p_values: List[float],
        method: CorrectionMethod = CorrectionMethod.FDR_BH,
        alpha: float = 0.05
    ) -> MultipleTestingResult:
        """
        Apply multiple testing correction to p-values.

        Args:
            p_values: List of p-values to correct
            method: Correction method (default: FDR Benjamini-Hochberg)
            alpha: Family-wise error rate or FDR level (default: 0.05)

        Returns:
            MultipleTestingResult with corrected p-values and rejection decisions

        Example:
            >>> p_values = [0.01, 0.04, 0.03, 0.1, 0.5]
            >>> result = StatisticalUtils.correct_multiple_testing(
            ...     p_values, method=CorrectionMethod.FDR_BH, alpha=0.05
            ... )
            >>> print(f"Original significant: {result.n_significant_original}")
            >>> print(f"Corrected significant: {result.n_significant_corrected}")
        """
        p_values_array = np.array(p_values)

        # Count original significant results
        n_significant_original = np.sum(p_values_array < alpha)

        if method == CorrectionMethod.NONE:
            # No correction
            rejected = p_values_array < alpha
            corrected_pvalues = p_values_array.copy()
        else:
            # Apply correction using statsmodels
            rejected, corrected_pvalues, _, _ = multipletests(
                p_values_array,
                alpha=alpha,
                method=method.value
            )

        n_significant_corrected = np.sum(rejected)

        return MultipleTestingResult(
            original_pvalues=p_values_array,
            corrected_pvalues=corrected_pvalues,
            rejected=rejected,
            method=method.value,
            alpha=alpha,
            n_significant_original=int(n_significant_original),
            n_significant_corrected=int(n_significant_corrected)
        )

    @staticmethod
    def mann_whitney_u_test(
        group1: Union[np.ndarray, List],
        group2: Union[np.ndarray, List],
        alternative: TestAlternative = TestAlternative.TWO_SIDED,
        alpha: float = 0.05
    ) -> StatisticalTestResult:
        """
        Perform Mann-Whitney U test (non-parametric test for comparing two groups).

        Args:
            group1: First group of values
            group2: Second group of values
            alternative: Alternative hypothesis
            alpha: Significance level

        Returns:
            StatisticalTestResult with U statistic and p-value

        Example:
            >>> group1 = [0.8, 0.9, 0.7, 0.85]
            >>> group2 = [0.6, 0.65, 0.7, 0.55]
            >>> result = StatisticalUtils.mann_whitney_u_test(group1, group2)
            >>> print(f"p-value: {result.p_value}, significant: {result.is_significant}")
        """
        group1 = np.asarray(group1)
        group2 = np.asarray(group2)

        statistic, p_value = mannwhitneyu(
            group1, group2,
            alternative=alternative.value
        )

        # Calculate effect size (rank biserial correlation)
        n1, n2 = len(group1), len(group2)
        effect_size = 1 - (2 * statistic) / (n1 * n2)

        is_significant = p_value < alpha

        return StatisticalTestResult(
            test_name='mann_whitney_u',
            statistic=float(statistic),
            p_value=float(p_value),
            is_significant=is_significant,
            alpha=alpha,
            alternative=alternative.value,
            effect_size=float(effect_size)
        )

    @staticmethod
    def wilcoxon_test(
        x: Union[np.ndarray, List],
        y: Optional[Union[np.ndarray, List]] = None,
        alternative: TestAlternative = TestAlternative.TWO_SIDED,
        alpha: float = 0.05
    ) -> StatisticalTestResult:
        """
        Perform Wilcoxon signed-rank test (non-parametric paired test).

        Args:
            x: First set of observations
            y: Second set of observations (if None, test x against zero)
            alternative: Alternative hypothesis
            alpha: Significance level

        Returns:
            StatisticalTestResult with test statistic and p-value

        Example:
            >>> before = [0.7, 0.65, 0.8, 0.75]
            >>> after = [0.8, 0.75, 0.85, 0.9]
            >>> result = StatisticalUtils.wilcoxon_test(before, after)
            >>> print(f"p-value: {result.p_value}, significant: {result.is_significant}")
        """
        x = np.asarray(x)

        if y is not None:
            y = np.asarray(y)
            statistic, p_value = wilcoxon(x, y, alternative=alternative.value)
        else:
            statistic, p_value = wilcoxon(x, alternative=alternative.value)

        is_significant = p_value < alpha

        return StatisticalTestResult(
            test_name='wilcoxon',
            statistic=float(statistic),
            p_value=float(p_value),
            is_significant=is_significant,
            alpha=alpha,
            alternative=alternative.value
        )

    @staticmethod
    def correlation_test(
        x: Union[np.ndarray, List],
        y: Union[np.ndarray, List],
        method: str = 'pearson',
        alpha: float = 0.05
    ) -> StatisticalTestResult:
        """
        Test correlation between two variables.

        Args:
            x: First variable
            y: Second variable
            method: Correlation method ('pearson' or 'spearman')
            alpha: Significance level

        Returns:
            StatisticalTestResult with correlation coefficient and p-value

        Example:
            >>> x = [0.7, 0.8, 0.9, 0.85, 0.75]
            >>> y = [0.65, 0.75, 0.85, 0.8, 0.7]
            >>> result = StatisticalUtils.correlation_test(x, y, method='pearson')
            >>> print(f"r={result.statistic:.3f}, p={result.p_value:.3f}")
        """
        x = np.asarray(x)
        y = np.asarray(y)

        if method == 'pearson':
            statistic, p_value = pearsonr(x, y)
            test_name = 'pearson_correlation'
        elif method == 'spearman':
            statistic, p_value = spearmanr(x, y)
            test_name = 'spearman_correlation'
        else:
            raise ValueError(f"Unknown method: {method}")

        is_significant = p_value < alpha

        return StatisticalTestResult(
            test_name=test_name,
            statistic=float(statistic),
            p_value=float(p_value),
            is_significant=is_significant,
            alpha=alpha,
            alternative='two-sided',
            effect_size=float(statistic)  # Correlation coefficient is the effect size
        )

    @staticmethod
    def calculate_effect_size_cohens_d(
        group1: Union[np.ndarray, List],
        group2: Union[np.ndarray, List]
    ) -> float:
        """
        Calculate Cohen's d effect size for two groups.

        Args:
            group1: First group of values
            group2: Second group of values

        Returns:
            Cohen's d effect size

        Interpretation:
            - Small effect: d ≈ 0.2
            - Medium effect: d ≈ 0.5
            - Large effect: d ≈ 0.8

        Example:
            >>> group1 = [0.8, 0.9, 0.7, 0.85, 0.95]
            >>> group2 = [0.6, 0.65, 0.7, 0.55, 0.6]
            >>> d = StatisticalUtils.calculate_effect_size_cohens_d(group1, group2)
            >>> print(f"Cohen's d: {d:.3f}")
        """
        group1 = np.asarray(group1)
        group2 = np.asarray(group2)

        n1, n2 = len(group1), len(group2)
        mean1, mean2 = np.mean(group1), np.mean(group2)
        var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)

        # Pooled standard deviation
        pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))

        # Cohen's d
        d = (mean1 - mean2) / pooled_std

        return float(d)

    @staticmethod
    def validate_score_significance(
        score: float,
        threshold: float,
        baseline_scores: Optional[List[float]] = None,
        n_permutations: int = 10000,
        alpha: float = 0.05
    ) -> Dict[str, Any]:
        """
        Validate if a score is statistically significantly above a threshold.

        This is useful for validating pipeline scores (e.g., validation scores,
        quality scores) against baselines or thresholds.

        Args:
            score: Observed score to validate
            threshold: Threshold to test against
            baseline_scores: Optional baseline scores for comparison
            n_permutations: Number of permutations for permutation test
            alpha: Significance level

        Returns:
            Dictionary with validation results including p-value and significance

        Example:
            >>> score = 0.85
            >>> threshold = 0.7
            >>> baseline = [0.65, 0.7, 0.68, 0.72, 0.69]
            >>> result = StatisticalUtils.validate_score_significance(
            ...     score, threshold, baseline
            ... )
            >>> print(f"Significant: {result['is_significant']}")
        """
        result = {
            'score': score,
            'threshold': threshold,
            'passes_threshold': score >= threshold,
            'alpha': alpha
        }

        if baseline_scores is not None and len(baseline_scores) > 0:
            baseline_scores = np.asarray(baseline_scores)

            # Test if score is significantly higher than baseline
            # Use permutation test
            all_scores = np.append(baseline_scores, score)

            def mean_diff(data):
                # Last value is the test score
                return data[-1] - np.mean(data[:-1])

            observed_diff = mean_diff(all_scores)

            perm_result = StatisticalUtils.permutation_test(
                observed_diff,
                all_scores,
                mean_diff,
                n_permutations=n_permutations,
                alternative=TestAlternative.GREATER,
                alpha=alpha
            )

            result['permutation_test'] = perm_result.to_dict()
            result['is_significant'] = perm_result.is_significant
            result['p_value'] = perm_result.p_value

            # Calculate bootstrap CI for the score
            _, ci = StatisticalUtils.bootstrap_confidence_interval(
                baseline_scores,
                np.mean,
                confidence_level=1 - alpha
            )
            result['baseline_ci'] = ci
            result['score_above_baseline_ci'] = score > ci[1]
        else:
            # No baseline, just check threshold
            result['is_significant'] = score >= threshold
            result['p_value'] = None

        return result


# Convenience functions for common use cases

def validate_score_with_statistics(
    validation_score: float,
    threshold: float = 0.7,
    baseline_scores: Optional[List[float]] = None,
    alpha: float = 0.05
) -> Dict[str, Any]:
    """
    Validate if a score is statistically significant.

    This is a convenience wrapper around validate_score_significance
    specifically for pipeline validation scores.

    Args:
        validation_score: The validation score to validate
        threshold: Minimum acceptable score (default: 0.7)
        baseline_scores: Optional baseline scores for comparison
        alpha: Significance level (default: 0.05)

    Returns:
        Dictionary with validation results
    """
    return StatisticalUtils.validate_score_significance(
        validation_score,
        threshold,
        baseline_scores,
        alpha=alpha
    )


def compare_scenario_results(
    scores1: List[float],
    scores2: List[float],
    alpha: float = 0.05
) -> Dict[str, Any]:
    """
    Compare results from two scenarios or methods.

    Args:
        scores1: Scores from first scenario/method
        scores2: Scores from second scenario/method
        alpha: Significance level

    Returns:
        Dictionary with comparison results including statistical tests
    """
    # Mann-Whitney U test (non-parametric)
    mw_result = StatisticalUtils.mann_whitney_u_test(
        scores1, scores2, alpha=alpha
    )

    # Effect size
    effect_size = StatisticalUtils.calculate_effect_size_cohens_d(scores1, scores2)

    return {
        'mean_scores1': float(np.mean(scores1)),
        'mean_scores2': float(np.mean(scores2)),
        'mann_whitney_u': mw_result.to_dict(),
        'cohens_d': effect_size,
        'is_significant': mw_result.is_significant
    }
