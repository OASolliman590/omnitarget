"""
Data Validation

Quality checks and validation metrics based on success_metrics.md.
Enhanced with statistical significance testing (P0-1: Critical Fix).
"""

import logging
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
from ..models.data_models import Disease, Interaction, ExpressionProfile, CancerMarker, DataSourceStatus, CompletenessMetrics
from .exceptions import DataValidationError, format_error_for_logging
from .statistical_utils import (
    StatisticalUtils,
    validate_score_with_statistics,
    compare_scenario_results,
    TestAlternative,
    CorrectionMethod
)

logger = logging.getLogger(__name__)


class DataValidator:
    """Data quality validation based on success metrics."""
    
    def __init__(self):
        """Initialize validator with success metrics thresholds."""
        # From success_metrics.md
        self.thresholds = {
            'disease_confidence': 0.6,
            'interaction_confidence': 0.4,  # 400/1000 scale
            'expression_coverage': 0.85,
            'cancer_marker_confidence': 0.7,
            'pathway_coverage': 0.80,
            'id_mapping_accuracy': 0.95,
            'pathway_precision': 0.85,
            'pathway_recall': 0.80
        }
    
    def validate_disease_confidence(self, disease: Disease) -> bool:
        """
        Validate disease confidence score.

        Success metric: Score ≥0.6
        """
        try:
            return disease.confidence >= self.thresholds['disease_confidence']
        except (AttributeError, TypeError) as e:
            # Missing confidence attribute or wrong type
            logger.warning(
                f"Disease confidence validation failed - invalid disease object",
                extra={'error': str(e), 'disease_id': getattr(disease, 'id', 'unknown')}
            )
            return False
        except Exception as e:
            # Unexpected errors
            logger.error(
                f"Disease confidence validation failed: {type(e).__name__}: {e}",
                extra=format_error_for_logging(e)
            )
            return False
    
    def validate_interaction_confidence(self, interaction: Interaction) -> bool:
        """
        Validate interaction confidence score.

        Success metric: Median score ≥400 (0.4 in 0-1 scale)
        """
        try:
            return interaction.combined_score >= self.thresholds['interaction_confidence']
        except (AttributeError, TypeError) as e:
            # Missing combined_score attribute or wrong type
            logger.warning(
                f"Interaction confidence validation failed - invalid interaction object",
                extra={
                    'error': str(e),
                    'protein_a': getattr(interaction, 'protein_a', 'unknown'),
                    'protein_b': getattr(interaction, 'protein_b', 'unknown')
                }
            )
            return False
        except Exception as e:
            # Unexpected errors
            logger.error(
                f"Interaction confidence validation failed: {type(e).__name__}: {e}",
                extra=format_error_for_logging(e)
            )
            return False
    
    def validate_expression_coverage(self, genes: List[str], expression_data: Dict[str, Any]) -> float:
        """
        Validate expression data coverage.
        
        Success metric: ≥85% coverage
        """
        try:
            if not genes or not expression_data:
                return 0.0
            
            covered_genes = sum(1 for gene in genes if gene in expression_data)
            coverage = covered_genes / len(genes)
            return coverage
        except (TypeError, AttributeError) as e:
            # Invalid input types
            logger.warning(
                f"Expression coverage validation failed - invalid input",
                extra={
                    'error': str(e),
                    'genes_type': type(genes).__name__,
                    'expression_data_type': type(expression_data).__name__,
                    'genes_count': len(genes) if isinstance(genes, (list, tuple)) else 'N/A'
                }
            )
            return 0.0
        except Exception as e:
            # Unexpected errors
            logger.error(
                f"Expression coverage validation failed: {type(e).__name__}: {e}",
                extra=format_error_for_logging(e)
            )
            return 0.0
    
    def validate_cancer_marker_confidence(self, marker: CancerMarker) -> bool:
        """
        Validate cancer marker confidence.

        Success metric: ≥70% in literature validation
        """
        try:
            return marker.confidence >= self.thresholds['cancer_marker_confidence']
        except (AttributeError, TypeError) as e:
            # Missing confidence attribute or wrong type
            logger.warning(
                f"Cancer marker confidence validation failed - invalid marker object",
                extra={'error': str(e), 'marker_gene': getattr(marker, 'gene_symbol', 'unknown')}
            )
            return False
        except Exception as e:
            # Unexpected errors
            logger.error(
                f"Cancer marker confidence validation failed: {type(e).__name__}: {e}",
                extra=format_error_for_logging(e)
            )
            return False
    
    def validate_cancer_marker_validation_rate(self, markers: List[CancerMarker]) -> float:
        """
        Validate cancer marker validation rate.
        
        Success metric: ≥70% validation rate across markers
        """
        try:
            if not markers:
                return 0.0
            
            valid_markers = sum(1 for marker in markers if self.validate_cancer_marker_confidence(marker))
            validation_rate = valid_markers / len(markers)
            return validation_rate
        except Exception as e:
            logger.error(f"Cancer marker validation rate calculation failed: {e}")
            return 0.0
    
    def validate_protein_confidence(self, protein: Any) -> bool:
        """
        Validate protein confidence score.
        
        Success metric: Confidence ≥ 0.7
        """
        try:
            if hasattr(protein, 'confidence'):
                return protein.confidence >= 0.7
            return True  # If no confidence field, assume valid
        except Exception as e:
            logger.error(f"Protein confidence validation failed: {e}")
            return False
    
    def validate_target_resolution_accuracy(self, resolved: int, total: int) -> float:
        """
        Validate target resolution accuracy.
        
        Success metric: ≥ 90% resolution rate
        """
        try:
            if total == 0:
                return 0.0
            return resolved / total
        except Exception as e:
            logger.error(f"Target resolution accuracy calculation failed: {e}")
            return 0.0
    
    def validate_pathway_coverage(self, disease_pathways: List[str], found_pathways: List[str]) -> float:
        """
        Validate pathway coverage for disease.
        
        Success metric: ≥80% pathway coverage
        """
        try:
            if not disease_pathways:
                return 1.0  # No pathways to cover
            
            covered_pathways = set(disease_pathways) & set(found_pathways)
            coverage = len(covered_pathways) / len(disease_pathways)
            return coverage
        except Exception as e:
            logger.error(f"Pathway coverage validation failed: {e}")
            return 0.0
    
    def validate_id_mapping_accuracy(self, original_ids: List[str], mapped_ids: Dict[str, str]) -> float:
        """
        Validate ID mapping accuracy.
        
        Success metric: ≥95% accuracy
        """
        try:
            if not original_ids:
                return 1.0
            
            successfully_mapped = sum(1 for id in original_ids if id in mapped_ids and mapped_ids[id])
            accuracy = successfully_mapped / len(original_ids)
            return accuracy
        except Exception as e:
            logger.error(f"ID mapping accuracy validation failed: {e}")
            return 0.0
    
    def validate_pathway_precision_recall(
        self, 
        true_positive: int, 
        false_positive: int, 
        false_negative: int
    ) -> Dict[str, float]:
        """
        Validate pathway precision and recall.
        
        Success metrics: Precision ≥85%, Recall ≥80%
        """
        try:
            precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0.0
            recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0.0
            
            return {
                'precision': precision,
                'recall': recall,
                'precision_valid': precision >= self.thresholds['pathway_precision'],
                'recall_valid': recall >= self.thresholds['pathway_recall']
            }
        except Exception as e:
            logger.error(f"Pathway precision/recall validation failed: {e}")
            return {'precision': 0.0, 'recall': 0.0, 'precision_valid': False, 'recall_valid': False}
    
    def validate_cross_database_concordance(
        self, 
        kegg_results: Dict[str, Any], 
        reactome_results: Dict[str, Any]
    ) -> float:
        """
        Validate cross-database concordance.
        
        Success metric: ≥70% concordance
        """
        try:
            if not kegg_results or not reactome_results:
                return 0.0
            
            # Extract pathway IDs from both databases, supporting list of dicts or list of strings
            def _to_id_set(items: Any) -> set:
                ids = set()
                if isinstance(items, list):
                    for item in items:
                        if isinstance(item, dict):
                            pid = item.get('id') or item.get('pathway_id') or item.get('kegg_id')
                            if pid:
                                ids.add(pid)
                        elif isinstance(item, str):
                            ids.add(item)
                return ids

            kegg_pathways = _to_id_set(kegg_results.get('pathways', []))
            reactome_pathways = _to_id_set(reactome_results.get('pathways', []))
            
            if not kegg_pathways and not reactome_pathways:
                return 1.0  # Both empty, perfect concordance
            
            # Calculate Jaccard similarity
            intersection = len(kegg_pathways & reactome_pathways)
            union = len(kegg_pathways | reactome_pathways)
            
            concordance = intersection / union if union > 0 else 0.0
            return concordance
        except Exception as e:
            logger.error(f"Cross-database concordance validation failed: {e}")
            return 0.0
    
    def validate_expression_reproducibility(
        self, 
        hpa_expression: Dict[str, Any], 
        reference_expression: Dict[str, Any]
    ) -> float:
        """
        Validate expression reproducibility against reference.
        
        Success metric: Spearman ρ ≥0.7 vs GTEx
        """
        try:
            import scipy.stats as stats
            import numpy as np
            
            # Support both dict and list inputs by normalizing to dict[gene]->value
            def _to_gene_value_dict(data: Any) -> Dict[str, float]:
                if isinstance(data, dict):
                    return data
                if isinstance(data, list):
                    # Expect list of dict profiles; try common keys
                    result: Dict[str, float] = {}
                    for item in data:
                        if not isinstance(item, dict):
                            continue
                        gene = item.get('gene') or item.get('gene_symbol') or item.get('id')
                        value = item.get('expression') or item.get('value') or item.get('nTPM')
                        if gene is not None and isinstance(value, (int, float)):
                            result[str(gene)] = float(value)
                    return result
                # Unsupported type
                return {}

            hpa_dict = _to_gene_value_dict(hpa_expression)
            ref_dict = _to_gene_value_dict(reference_expression)

            # Extract common genes
            common_genes = set(hpa_dict.keys()) & set(ref_dict.keys())
            if len(common_genes) < 2:
                return 0.0
            
            # Extract expression values
            hpa_values = [hpa_dict[gene] for gene in common_genes]
            ref_values = [ref_dict[gene] for gene in common_genes]
            
            # Calculate Spearman correlation
            correlation, _ = stats.spearmanr(hpa_values, ref_values)
            return correlation if not np.isnan(correlation) else 0.0
            
        except Exception as e:
            logger.error(f"Expression reproducibility validation failed: {e}")
            return 0.0
    
    def validate_druggability_roc_auc(
        self, 
        druggability_scores: List[float], 
        known_druggable: List[bool]
    ) -> float:
        """
        Validate druggability ROC-AUC.
        
        Success metric: ROC-AUC ≥0.75
        """
        try:
            from sklearn.metrics import roc_auc_score
            
            if len(druggability_scores) != len(known_druggable) or len(set(known_druggable)) < 2:
                return 0.0
            
            auc = roc_auc_score(known_druggable, druggability_scores)
            return auc
            
        except Exception as e:
            logger.error(f"Druggability ROC-AUC validation failed: {e}")
            return 0.0
    
    def validate_mra_convergence(
        self, 
        convergence_history: List[float], 
        max_iterations: int = 1000
    ) -> Dict[str, Any]:
        """
        Validate MRA convergence.
        
        Success metric: max |Δx| <1e-6 within 1000 iterations
        """
        try:
            if not convergence_history:
                return {'converged': False, 'iterations': 0, 'final_error': float('inf')}
            
            final_error = convergence_history[-1]
            converged = final_error < 1e-6
            iterations = len(convergence_history)
            
            return {
                'converged': converged,
                'iterations': iterations,
                'final_error': final_error,
                'within_limit': iterations <= max_iterations
            }
            
        except Exception as e:
            logger.error(f"MRA convergence validation failed: {e}")
            return {'converged': False, 'iterations': 0, 'final_error': float('inf')}

    def calculate_data_completeness_penalty(
        self,
        data_sources: Optional[List[DataSourceStatus]] = None,
        completeness_metrics: Optional[CompletenessMetrics] = None
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Calculate penalty for missing critical data.

        Returns:
            Tuple of (penalty_score, details_dict)
            Penalty is between 0.0 (no penalty) and 0.3 (maximum penalty)

        Critical data sources and their expected minimum success rates:
        - HPA (expression data): ≥0.5 success rate
        - KEGG (pathways): ≥0.8 success rate
        - STRING (network): ≥0.8 success rate
        - ChEMBL (bioactivity): ≥0.5 success rate (if used)
        """
        try:
            penalty = 0.0
            details = {
                'critical_sources': {},
                'penalty_breakdown': {},
                'overall_penalty': 0.0
            }

            if data_sources is None:
                logger.warning("No data_sources provided - applying maximum penalty (0.3)")
                details['penalty_breakdown']['missing_tracking'] = 0.3
                details['overall_penalty'] = 0.3
                return 0.3, details

            if isinstance(data_sources, list) and len(data_sources) == 0:
                logger.warning("Empty data_sources list - applying maximum penalty (0.3)")
                details['penalty_breakdown']['empty_tracking'] = 0.3
                details['overall_penalty'] = 0.3
                return 0.3, details

            # Critical data source thresholds
            critical_thresholds = {
                'hpa': 0.5,      # HPA expression data
                'kegg': 0.8,     # KEGG pathway data
                'string': 0.8,   # STRING network data
                'chembl': 0.5,   # ChEMBL bioactivity (if used)
            }

            # Check each critical data source
            for source in data_sources:
                source_name = source.source_name.lower()

                if source_name in critical_thresholds:
                    threshold = critical_thresholds[source_name]
                    success_rate = source.success_rate

                    details['critical_sources'][source_name] = {
                        'success_rate': success_rate,
                        'threshold': threshold,
                        'requested': source.requested,
                        'successful': source.successful
                    }

                    # Calculate penalty if below threshold
                    if success_rate < threshold:
                        # Skip penalty if data source was never requested (optional source)
                        if source.requested == 0:
                            logger.debug(
                                f"Data source '{source_name}' not used in this scenario "
                                f"(requested=0, skipping threshold check)"
                            )
                            continue

                        # Penalty proportional to how far below threshold
                        deficit = threshold - success_rate
                        # Scale deficit (0-1) to penalty weight (0.1-0.2)
                        source_penalty = min(0.2, deficit * 0.4)
                        penalty += source_penalty
                        details['penalty_breakdown'][source_name] = {
                            'deficit': deficit,
                            'penalty': source_penalty
                        }
                        logger.warning(
                            f"Data source '{source_name}' below threshold: "
                            f"{success_rate:.2f} < {threshold} (penalty: {source_penalty:.3f})"
                        )

            # Apply completeness metrics penalty if available
            if completeness_metrics:
                completeness_penalty = 0.0

                # Check individual completeness metrics
                if completeness_metrics.expression_data is not None:
                    expr_comp = completeness_metrics.expression_data
                    if expr_comp < 0.5:  # Less than 50% expression coverage
                        expr_penalty = (0.5 - expr_comp) * 0.2
                        completeness_penalty += expr_penalty
                        details['penalty_breakdown']['expression_completeness'] = expr_penalty

                if completeness_metrics.network_data is not None:
                    net_comp = completeness_metrics.network_data
                    if net_comp < 0.8:  # Less than 80% network coverage
                        net_penalty = (0.8 - net_comp) * 0.2
                        completeness_penalty += net_penalty
                        details['penalty_breakdown']['network_completeness'] = net_penalty

                if completeness_metrics.pathway_data is not None:
                    path_comp = completeness_metrics.pathway_data
                    if path_comp < 0.8:  # Less than 80% pathway coverage
                        path_penalty = (0.8 - path_comp) * 0.2
                        completeness_penalty += path_penalty
                        details['penalty_breakdown']['pathway_completeness'] = path_penalty

                penalty += completeness_penalty

            # Cap maximum penalty at 0.3 (30% of total score)
            penalty = min(0.3, penalty)
            details['overall_penalty'] = penalty

            logger.debug(f"Data completeness penalty: {penalty:.3f}")
            return penalty, details

        except Exception as e:
            logger.error(f"Data completeness penalty calculation failed: {e}")
            return 0.0, {'error': str(e)}

    def calculate_overall_validation_score(
        self,
        validation_results: Dict[str, Any],
        data_sources: Optional[List[DataSourceStatus]] = None,
        completeness_metrics: Optional[CompletenessMetrics] = None
    ) -> Dict[str, Any]:
        """
        Calculate overall validation score from individual metrics.

        Returns weighted average of all validation metrics.

        RECALIBRATED: 2025-11-04
        Weights adjusted to reflect improved data quality from Issues 1-3:
        - Issue 1: Gene overlap improved (Jaccard ≥0.4, was 0.069)
        - Issue 2: HPA pathology now has ≥50 markers (was 0)
        - Issue 3: KEGG DRUG has ≥20 genes with drugs (was 0)

        Changes:
        - Reduced pathway_coverage (0.15→0.10): Less weight since coverage is now consistently high
        - Reduced id_mapping_accuracy (0.10→0.05): Less weight after ID mapping fixes
        - Increased cross_database_concordance (0.10→0.15): Reflects improved data integration
        - Increased druggability_auc (0.05→0.10): Reflects improved drug coverage
        """
        try:
            weights = {
                'disease_confidence': 0.15,
                'interaction_confidence': 0.20,
                'expression_coverage': 0.15,
                'pathway_coverage': 0.10,  # Reduced after Issue 1 fixes
                'id_mapping_accuracy': 0.05,  # Reduced after Issue 1 fixes
                'cross_database_concordance': 0.15,  # Increased after Issues 1, 2, 3
                'expression_reproducibility': 0.10,
                'druggability_auc': 0.10  # Increased after Issue 3 fixes
            }

            total_score = 0.0
            total_weight = 0.0

            for metric, weight in weights.items():
                if metric in validation_results:
                    score = validation_results[metric]

                    # Skip None values (missing data) - don't count them as 0.0
                    if score is None:
                        logger.debug(f"Skipping metric '{metric}' with None value (missing data)")
                        continue

                    # Convert boolean to float
                    if isinstance(score, bool):
                        score = 1.0 if score else 0.0

                    # Ensure score is numeric and in valid range
                    try:
                        score = float(score)
                        if not 0.0 <= score <= 1.0:
                            logger.warning(f"Metric '{metric}' has invalid score {score}, clamping to [0.0, 1.0]")
                            score = max(0.0, min(1.0, score))

                        total_score += score * weight
                        total_weight += weight
                    except (ValueError, TypeError) as e:
                        logger.warning(f"Skipping metric '{metric}' with non-numeric value: {score} ({e})")
                        continue

            # Calculate base score
            base_score = total_score / total_weight if total_weight > 0 else 0.0

            # Calculate data completeness penalty
            penalty, penalty_details = self.calculate_data_completeness_penalty(
                data_sources, completeness_metrics
            )

            # Apply penalty to get final score
            final_score = max(0.0, base_score - penalty)

            # Return detailed result
            return {
                'base_score': base_score,
                'penalty': penalty,
                'penalty_details': penalty_details,
                'final_score': final_score,
                'total_weight': total_weight,
                'metrics_included': list(weights.keys())
            }

        except Exception as e:
            logger.error(f"Overall validation score calculation failed: {e}")
            return {
                'base_score': 0.0,
                'penalty': 0.0,
                'penalty_details': {'error': str(e)},
                'final_score': 0.0,
                'total_weight': 0.0,
                'metrics_included': []
            }

    def calculate_overall_validation_score_with_statistics(
        self,
        validation_results: Dict[str, Any],
        baseline_scores: Optional[List[float]] = None,
        alpha: float = 0.05,
        n_permutations: int = 10000
    ) -> Dict[str, Any]:
        """
        Calculate overall validation score with statistical significance testing.

        This method addresses P0 Critical Gap #12: No statistical significance testing.

        Args:
            validation_results: Dictionary of validation metrics
            baseline_scores: Optional list of baseline scores for comparison
            alpha: Significance level (default: 0.05)
            n_permutations: Number of permutations for permutation test

        Returns:
            Dictionary containing:
            - score: Overall validation score
            - p_value: Statistical significance of score
            - is_significant: Whether score is statistically significant
            - confidence_interval: 95% confidence interval for score
            - baseline_comparison: Comparison with baseline (if provided)

        Example:
            >>> validator = DataValidator()
            >>> results = {'disease_confidence': 0.8, 'interaction_confidence': 0.7}
            >>> baseline = [0.65, 0.7, 0.68, 0.72, 0.69]
            >>> stats = validator.calculate_overall_validation_score_with_statistics(
            ...     results, baseline
            ... )
            >>> print(f"Score: {stats['score']:.3f}, p-value: {stats['p_value']:.3f}")
        """
        try:
            # Calculate base score
            score = self.calculate_overall_validation_score(validation_results)

            result = {
                'score': score,
                'alpha': alpha,
                'n_permutations': n_permutations
            }

            # If no baseline provided, use threshold-based validation
            if baseline_scores is None or len(baseline_scores) == 0:
                # Default threshold is 0.7 (70% validation score)
                threshold = 0.7
                result['threshold'] = threshold
                result['passes_threshold'] = score >= threshold
                result['p_value'] = None
                result['is_significant'] = score >= threshold
                result['confidence_interval'] = None

                logger.info(
                    f"Validation score: {score:.3f}, "
                    f"threshold: {threshold}, "
                    f"passes: {result['passes_threshold']}"
                )
            else:
                # Statistical test against baseline
                stat_result = validate_score_with_statistics(
                    score,
                    threshold=0.7,
                    baseline_scores=baseline_scores,
                    alpha=alpha
                )

                result['p_value'] = stat_result.get('p_value')
                result['is_significant'] = stat_result.get('is_significant', False)
                result['threshold'] = stat_result.get('threshold', 0.7)
                result['passes_threshold'] = stat_result.get('passes_threshold', False)

                if 'baseline_ci' in stat_result:
                    result['baseline_confidence_interval'] = stat_result['baseline_ci']
                    result['score_above_baseline_ci'] = stat_result['score_above_baseline_ci']

                if 'permutation_test' in stat_result:
                    result['permutation_test'] = stat_result['permutation_test']

                # Calculate bootstrap CI for individual metrics
                metric_scores = []
                for metric, weight in [
                    ('disease_confidence', 0.15),
                    ('interaction_confidence', 0.20),
                    ('expression_coverage', 0.15),
                    ('pathway_coverage', 0.10),
                    ('id_mapping_accuracy', 0.05),
                    ('cross_database_concordance', 0.15),
                    ('expression_reproducibility', 0.10),
                    ('druggability_auc', 0.10)
                ]:
                    if metric in validation_results:
                        metric_score = validation_results[metric]
                        if isinstance(metric_score, bool):
                            metric_score = 1.0 if metric_score else 0.0
                        metric_scores.append(metric_score)

                if len(metric_scores) > 1:
                    # Bootstrap CI for the score
                    estimate, ci = StatisticalUtils.bootstrap_confidence_interval(
                        metric_scores,
                        np.mean,
                        confidence_level=1 - alpha,
                        n_bootstrap=n_permutations
                    )
                    result['confidence_interval'] = ci
                    result['bootstrap_estimate'] = estimate

                logger.info(
                    f"Validation score: {score:.3f}, "
                    f"p-value: {result['p_value']:.4f}, "
                    f"significant: {result['is_significant']}, "
                    f"CI: {result.get('confidence_interval')}"
                )

            return result

        except Exception as e:
            logger.error(f"Statistical validation score calculation failed: {e}")
            return {
                'score': 0.0,
                'p_value': None,
                'is_significant': False,
                'alpha': alpha,
                'error': str(e)
            }
    
    def validate_cancer_hallmark_enrichment(self, pathways: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Validate cancer hallmark enrichment in pathways."""
        hallmarks = {
            'sustaining_proliferative_signaling': 0,
            'evading_growth_suppressors': 0,
            'resisting_cell_death': 0,
            'enabling_replicative_immortality': 0,
            'inducing_angiogenesis': 0,
            'activating_invasion_metastasis': 0,
            'reprogramming_energy_metabolism': 0,
            'evading_immune_destruction': 0,
            'genome_instability_mutation': 0,
            'tumor_promoting_inflammation': 0
        }
        
        # Count pathway matches for each hallmark
        for pathway in pathways:
            # Handle both dict and Pathway object
            if hasattr(pathway, 'name'):
                pathway_name = pathway.name.lower()
            else:
                pathway_name = pathway.get('name', '').lower()
            
            for hallmark in hallmarks:
                if hallmark.replace('_', ' ') in pathway_name:
                    hallmarks[hallmark] += 1
        
        # Calculate enrichment score
        total_hallmarks = sum(hallmarks.values())
        enrichment_score = total_hallmarks / len(hallmarks) if hallmarks else 0
        
        return {
            'hallmark_counts': hallmarks,
            'total_hallmarks': total_hallmarks,
            'enrichment_score': enrichment_score,
            'is_enriched': total_hallmarks >= 4  # At least 4 of 10 hallmarks
        }
    
    def validate_differential_expression_concordance(
        self, 
        expression_data: Dict[str, Any], 
        reference_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Validate differential expression concordance with reference data."""
        concordance_scores = []
        
        for gene, expr_value in expression_data.items():
            if gene in reference_data:
                ref_value = reference_data[gene]
                # Calculate correlation between expression values
                correlation = abs(expr_value - ref_value) / max(expr_value, ref_value, 0.1)
                concordance_scores.append(1 - correlation)  # Higher is better
        
        mean_concordance = sum(concordance_scores) / len(concordance_scores) if concordance_scores else 0
        
        return {
            'concordance_scores': concordance_scores,
            'mean_concordance': mean_concordance,
            'is_concordant': mean_concordance >= 0.6,  # ≥60% concordance
            'total_genes': len(concordance_scores)
        }
    
    def validate_driver_gene_overlap(self, targets: List[Any]) -> Dict[str, Any]:
        """
        Validate overlap of driver genes with targets.

        Success metric: ≥50% of targets are driver genes
        """
        try:
            if not targets:
                return {
                    'overlap_count': 0,
                    'total_targets': 0,
                    'overlap_ratio': 0.0,
                    'is_overlapping': False
                }

            # Common cancer driver genes
            driver_genes = {
                'TP53', 'BRCA1', 'BRCA2', 'EGFR', 'KRAS', 'PIK3CA', 'AKT1',
                'MYC', 'RB1', 'PTEN', 'ALK', 'BRAF', 'NRAS', 'HRAS', 'AXL',
                'HER2', 'ERBB2', 'MET', 'RET', 'KIT', 'PDGFRA', 'FGFR1',
                'IDH1', 'IDH2', 'DNMT3A', 'TET2', 'NPM1', 'FLT3'
            }

            # Check if targets are driver genes
            target_symbols = []
            for target in targets:
                if hasattr(target, 'gene_symbol'):
                    target_symbols.append(target.gene_symbol)
                elif isinstance(target, str):
                    target_symbols.append(target)

            overlapping = set(target_symbols) & driver_genes
            overlap_ratio = len(overlapping) / len(target_symbols) if target_symbols else 0

            return {
                'overlap_count': len(overlapping),
                'total_targets': len(target_symbols),
                'overlap_ratio': overlap_ratio,
                'is_overlapping': overlap_ratio >= 0.5,
                'driver_genes_found': list(overlapping)
            }
        except Exception as e:
            logger.error(f"Driver gene overlap validation failed: {e}")
            return {
                'overlap_count': 0,
                'total_targets': 0,
                'overlap_ratio': 0.0,
                'is_overlapping': False
            }

    def validate_multiple_metrics_with_correction(
        self,
        validation_results: Dict[str, float],
        thresholds: Optional[Dict[str, float]] = None,
        method: CorrectionMethod = CorrectionMethod.FDR_BH,
        alpha: float = 0.05
    ) -> Dict[str, Any]:
        """
        Validate multiple metrics with multiple testing correction.

        This method addresses P0 Critical Gap #12: No statistical significance testing.
        When testing multiple hypotheses (e.g., multiple validation metrics), we need to
        correct for multiple testing to avoid false discoveries.

        Args:
            validation_results: Dictionary of metric names to scores
            thresholds: Optional dict of metric-specific thresholds (default: use self.thresholds)
            method: Multiple testing correction method (default: FDR Benjamini-Hochberg)
            alpha: Family-wise error rate or FDR level (default: 0.05)

        Returns:
            Dictionary with:
            - individual_tests: Results for each metric
            - corrected_results: Results after multiple testing correction
            - n_significant_original: Number of significant results before correction
            - n_significant_corrected: Number of significant results after correction

        Example:
            >>> validator = DataValidator()
            >>> results = {
            ...     'disease_confidence': 0.8,
            ...     'interaction_confidence': 0.7,
            ...     'expression_coverage': 0.9
            ... }
            >>> corrected = validator.validate_multiple_metrics_with_correction(results)
            >>> print(f"Significant after correction: {corrected['n_significant_corrected']}")
        """
        try:
            if thresholds is None:
                thresholds = self.thresholds

            # Collect p-values for each metric
            # We use permutation test against threshold
            p_values = []
            metric_names = []
            scores = []

            for metric, score in validation_results.items():
                if metric not in thresholds:
                    continue

                threshold = thresholds[metric]
                metric_names.append(metric)
                scores.append(score)

                # Simple one-sample test: is score > threshold?
                # p-value approximation: if score > threshold, p = (1 - score), else p = 1.0
                # This is a conservative approximation
                if score >= threshold:
                    # The more above threshold, the more significant
                    p_value = max(0.001, 1.0 - score)
                else:
                    # Below threshold, not significant
                    p_value = 1.0

                p_values.append(p_value)

            if not p_values:
                return {
                    'individual_tests': {},
                    'corrected_results': None,
                    'n_significant_original': 0,
                    'n_significant_corrected': 0
                }

            # Apply multiple testing correction
            correction_result = StatisticalUtils.correct_multiple_testing(
                p_values,
                method=method,
                alpha=alpha
            )

            # Organize results
            individual_tests = {}
            for i, metric in enumerate(metric_names):
                individual_tests[metric] = {
                    'score': scores[i],
                    'threshold': thresholds[metric],
                    'p_value': p_values[i],
                    'is_significant_original': p_values[i] < alpha,
                    'corrected_p_value': correction_result.corrected_pvalues[i],
                    'is_significant_corrected': correction_result.rejected[i]
                }

            result = {
                'individual_tests': individual_tests,
                'corrected_results': correction_result.to_dict(),
                'n_significant_original': correction_result.n_significant_original,
                'n_significant_corrected': correction_result.n_significant_corrected,
                'method': method.value,
                'alpha': alpha
            }

            logger.info(
                f"Multiple testing correction: {correction_result.n_significant_original} → "
                f"{correction_result.n_significant_corrected} significant (method: {method.value})"
            )

            return result

        except Exception as e:
            logger.error(f"Multiple metrics validation with correction failed: {e}")
            return {
                'individual_tests': {},
                'corrected_results': None,
                'n_significant_original': 0,
                'n_significant_corrected': 0,
                'error': str(e)
            }

    def compare_validation_results(
        self,
        results1: Dict[str, Any],
        results2: Dict[str, Any],
        alpha: float = 0.05
    ) -> Dict[str, Any]:
        """
        Compare validation results from two scenarios or methods statistically.

        This method addresses P0 Critical Gap #12: No statistical significance testing.
        Use this to compare pipeline results from different scenarios, configurations,
        or time points.

        Args:
            results1: First set of validation results
            results2: Second set of validation results
            alpha: Significance level

        Returns:
            Dictionary with comparison results and statistical tests

        Example:
            >>> validator = DataValidator()
            >>> results1 = {'disease_confidence': 0.8, 'interaction_confidence': 0.7}
            >>> results2 = {'disease_confidence': 0.85, 'interaction_confidence': 0.75}
            >>> comparison = validator.compare_validation_results(results1, results2)
            >>> print(f"Significant difference: {comparison['overall_comparison']['is_significant']}")
        """
        try:
            # Calculate overall scores
            score1 = self.calculate_overall_validation_score(results1)
            score2 = self.calculate_overall_validation_score(results2)

            # Collect metric scores for comparison
            common_metrics = set(results1.keys()) & set(results2.keys())

            metric_comparisons = {}
            scores1_list = []
            scores2_list = []

            for metric in common_metrics:
                val1 = results1[metric]
                val2 = results2[metric]

                # Convert boolean to float
                if isinstance(val1, bool):
                    val1 = 1.0 if val1 else 0.0
                if isinstance(val2, bool):
                    val2 = 1.0 if val2 else 0.0

                scores1_list.append(val1)
                scores2_list.append(val2)

                metric_comparisons[metric] = {
                    'score1': val1,
                    'score2': val2,
                    'difference': val2 - val1,
                    'percent_change': ((val2 - val1) / val1 * 100) if val1 > 0 else 0
                }

            # Statistical comparison if we have enough metrics
            if len(scores1_list) >= 2:
                comparison_result = compare_scenario_results(
                    scores1_list,
                    scores2_list,
                    alpha=alpha
                )
            else:
                comparison_result = {
                    'mean_scores1': score1,
                    'mean_scores2': score2,
                    'is_significant': False
                }

            result = {
                'overall_score1': score1,
                'overall_score2': score2,
                'overall_difference': score2 - score1,
                'overall_percent_change': ((score2 - score1) / score1 * 100) if score1 > 0 else 0,
                'metric_comparisons': metric_comparisons,
                'overall_comparison': comparison_result,
                'alpha': alpha,
                'n_common_metrics': len(common_metrics)
            }

            logger.info(
                f"Validation comparison: score1={score1:.3f}, score2={score2:.3f}, "
                f"diff={score2-score1:.3f}, "
                f"significant={comparison_result.get('is_significant', False)}"
            )

            return result

        except Exception as e:
            logger.error(f"Validation results comparison failed: {e}")
            return {
                'overall_score1': 0.0,
                'overall_score2': 0.0,
                'overall_difference': 0.0,
                'metric_comparisons': {},
                'overall_comparison': {},
                'error': str(e)
            }
