"""
Validation utilities for data quality and consistency checks.
"""

from typing import Dict, List, Any, Optional
import numpy as np
from src.models.data_models import Disease, Interaction, Protein, Pathway


class ValidationUtils:
    """Utility class for data validation and quality checks."""
    
    @staticmethod
    def validate_disease_confidence(disease: Disease) -> bool:
        """
        Validate disease confidence score.
        
        Args:
            disease: Disease object to validate
            
        Returns:
            True if confidence score meets criteria
        """
        return disease.confidence >= 0.6
    
    @staticmethod
    def validate_interaction_confidence(interaction: Interaction) -> bool:
        """
        Validate interaction confidence score.
        
        Args:
            interaction: Interaction object to validate
            
        Returns:
            True if confidence score meets criteria
        """
        return interaction.score is not None and interaction.score >= 0.4
    
    @staticmethod
    def validate_expression_coverage(genes: List[str], expression_data: Dict) -> float:
        """
        Validate expression data coverage.
        
        Args:
            genes: List of gene symbols
            expression_data: Expression data dictionary
            
        Returns:
            Coverage percentage (0.0 to 1.0)
        """
        if not genes:
            return 0.0
        
        covered_genes = set(expression_data.keys())
        total_genes = set(genes)
        
        return len(covered_genes & total_genes) / len(total_genes)
    
    @staticmethod
    def validate_pathway_coverage(target_genes: List[str], pathway_genes: List[str]) -> float:
        """
        Validate pathway coverage for target genes.
        
        Args:
            target_genes: List of target gene symbols
            pathway_genes: List of pathway gene symbols
            
        Returns:
            Coverage percentage (0.0 to 1.0)
        """
        if not target_genes:
            return 0.0
        
        target_set = set(target_genes)
        pathway_set = set(pathway_genes)
        
        return len(target_set & pathway_set) / len(target_set)
    
    @staticmethod
    def calculate_jaccard_similarity(set1: set, set2: set) -> float:
        """
        Calculate Jaccard similarity between two sets.
        
        Args:
            set1: First set
            set2: Second set
            
        Returns:
            Jaccard similarity (0.0 to 1.0)
        """
        if not set1 and not set2:
            return 1.0
        
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        
        return intersection / union if union > 0 else 0.0
    
    @staticmethod
    def validate_cross_database_concordance(
        kegg_results: List[Any], 
        reactome_results: List[Any]
    ) -> float:
        """
        Validate cross-database concordance.
        
        Args:
            kegg_results: Results from KEGG database
            reactome_results: Results from Reactome database
            
        Returns:
            Concordance score (0.0 to 1.0)
        """
        if not kegg_results and not reactome_results:
            return 0.0
        
        # Extract gene symbols from both databases
        kegg_genes = set()
        reactome_genes = set()
        
        for result in kegg_results:
            if hasattr(result, 'genes'):
                kegg_genes.update(result.genes)
        
        for result in reactome_results:
            if hasattr(result, 'genes'):
                reactome_genes.update(result.genes)
        
        return ValidationUtils.calculate_jaccard_similarity(kegg_genes, reactome_genes)
    
    @staticmethod
    def validate_network_connectivity(network: Any) -> Dict[str, float]:
        """
        Validate network connectivity metrics.
        
        Args:
            network: NetworkX graph object
            
        Returns:
            Dictionary with connectivity metrics
        """
        if not hasattr(network, 'nodes') or not hasattr(network, 'edges'):
            return {'connectivity': 0.0, 'density': 0.0}
        
        n_nodes = len(network.nodes())
        n_edges = len(network.edges())
        
        # Calculate density
        max_edges = n_nodes * (n_nodes - 1) / 2 if n_nodes > 1 else 0
        density = n_edges / max_edges if max_edges > 0 else 0.0
        
        # Calculate connectivity (simplified)
        connectivity = min(1.0, n_edges / n_nodes) if n_nodes > 0 else 0.0
        
        return {
            'connectivity': connectivity,
            'density': density,
            'n_nodes': n_nodes,
            'n_edges': n_edges
        }
    
    @staticmethod
    def validate_simulation_convergence(
        results: Dict[str, float], 
        threshold: float = 1e-6
    ) -> bool:
        """
        Validate simulation convergence.
        
        Args:
            results: Simulation results dictionary
            threshold: Convergence threshold
            
        Returns:
            True if simulation converged
        """
        if not results:
            return False
        
        # Check if all values are within threshold
        max_change = max(abs(value) for value in results.values())
        return max_change < threshold
    
    @staticmethod
    def calculate_overall_validation_score(scores: Dict[str, float]) -> float:
        """
        Calculate overall validation score from individual scores.
        
        Args:
            scores: Dictionary of individual validation scores
            
        Returns:
            Overall validation score (0.0 to 1.0)
        """
        if not scores:
            return 0.0
        
        # Weighted average of scores
        weights = {
            'resolution_accuracy': 0.25,
            'pathway_coverage': 0.20,
            'interaction_confidence': 0.20,
            'expression_coverage': 0.15,
            'simulation_convergence': 0.20
        }
        
        weighted_sum = 0.0
        total_weight = 0.0
        
        for key, score in scores.items():
            weight = weights.get(key, 0.1)  # Default weight
            weighted_sum += score * weight
            total_weight += weight
        
        return weighted_sum / total_weight if total_weight > 0 else 0.0
