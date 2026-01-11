"""
Simulation Models

Pydantic models for perturbation simulation and MRA analysis.
"""

from typing import Dict, List, Optional, Literal, Any, Union
from pydantic import BaseModel, Field
import numpy as np
from .data_models import Pathway, Protein, Interaction, NetworkNode, NetworkEdge, DataSourceStatus, CompletenessMetrics


class DrugInfo(BaseModel):
    """Drug information model."""
    drug_id: str = Field(..., description="Drug identifier")
    drug_name: str = Field(..., description="Drug name")
    target_protein: str = Field(..., description="Target protein")
    repurposing_score: float = Field(..., description="Repurposing score")
    safety_profile: Dict[str, Any] = Field(default_factory=dict, description="Safety profile")
    efficacy_prediction: float = Field(..., description="Efficacy prediction score")
    bioactivity_nm: Optional[float] = Field(None, description="Median bioactivity value (nM)")
    drug_likeness_score: Optional[float] = Field(
        None, ge=0.0, le=1.0, description="Drug-likeness score (0-1)"
    )


class SimulationConfig(BaseModel):
    """Configuration for simulation engines."""
    mode: Literal['inhibit', 'activate', 'knockdown', 'overexpress'] = Field(
        ..., description="Simulation mode"
    )
    depth: int = Field(ge=1, le=10, default=3, description="Propagation depth")
    propagation_factor: float = Field(ge=0.0, le=1.0, default=0.7, description="Propagation strength")
    confidence_threshold: float = Field(ge=0.0, le=1.0, default=0.4, description="Confidence threshold")
    max_iterations: int = Field(ge=10, le=10000, default=1000, description="Maximum iterations for MRA")
    convergence_threshold: float = Field(ge=1e-8, le=1e-3, default=1e-6, description="Convergence threshold")
    tissue_context: Optional[str] = Field(None, description="Tissue context for expression filtering")


class NetworkImpactMetrics(BaseModel):
    """
    Standardized network impact metrics for all simulators.

    Ensures consistent schema across SimplePerturbationSimulator, MRASimulator,
    and custom BFS propagation engines.
    """
    # Core metrics (required by all engines)
    total_affected: int = Field(..., ge=0, description="Number of affected nodes (excluding target)")
    mean_effect: float = Field(..., ge=0.0, description="Mean absolute effect strength")
    max_effect: float = Field(..., ge=0.0, le=1.0, description="Maximum effect strength")
    network_coverage: float = Field(..., ge=0.0, le=1.0, description="Fraction of network affected")

    # Extended metrics (optional, for advanced analysis)
    network_centrality: Optional[float] = Field(None, ge=0.0, le=1.0, description="Target node degree centrality")
    betweenness_centrality: Optional[float] = Field(None, ge=0.0, le=1.0, description="Target node betweenness centrality")
    propagation_depth: Optional[int] = Field(None, ge=0, le=10, description="Maximum propagation depth reached")
    perturbation_magnitude: Optional[float] = Field(None, ge=0.0, le=1.0, description="Overall perturbation magnitude")

    # Effect distribution (optional, for detailed analysis)
    positive_effects: Optional[int] = Field(None, ge=0, description="Number of nodes with positive effects")
    negative_effects: Optional[int] = Field(None, ge=0, description="Number of nodes with negative effects")
    effect_ratio: Optional[float] = Field(None, ge=0.0, description="Ratio of positive to negative effects")


class SimulationResult(BaseModel):
    """Result from simplified perturbation simulation."""
    target_node: str = Field(..., description="Target node identifier")
    mode: Literal['inhibit', 'activate'] = Field(..., description="Simulation mode")
    affected_nodes: Dict[str, float] = Field(
        ..., description="Node ID -> effect strength mapping"
    )
    direct_targets: List[str] = Field(..., description="Direct interaction targets")
    downstream: List[str] = Field(..., description="Downstream affected nodes")
    upstream: List[str] = Field(default_factory=list, description="Upstream regulators")
    feedback_loops: List[str] = Field(default_factory=list, description="Feedback loop nodes")
    network_impact: Dict[str, Any] = Field(..., description="Network-level impact metrics")
    confidence_scores: Dict[str, float] = Field(
        ..., description="Confidence scores for predictions"
    )
    execution_time: float = Field(ge=0.0, description="Simulation execution time in seconds")
    biological_context: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Biological annotations (functions, pathways, feedback loops)"
    )
    drug_info: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Known drug annotations for the simulated target"
    )


class MRASimulationResult(BaseModel):
    """Result from full MRA simulation."""
    target_node: str = Field(..., description="Target node identifier")
    mode: str = Field(..., description="Simulation mode")
    steady_state: Dict[str, float] = Field(
        ..., description="Steady-state node values"
    )
    local_response_matrix: List[List[float]] = Field(
        ..., description="Local response coefficient matrix"
    )
    global_response_matrix: List[List[float]] = Field(
        ..., description="Global response matrix"
    )
    convergence_info: Dict[str, Any] = Field(
        ..., description="Convergence analysis results"
    )
    feedback_loops: List['FeedbackLoop'] = Field(
        ..., description="Detected feedback loops"
    )
    upstream_classification: Dict[str, str] = Field(
        ..., description="Upstream node classification"
    )
    downstream_classification: Dict[str, str] = Field(
        ..., description="Downstream node classification"
    )
    tissue_specificity: Dict[str, float] = Field(
        ..., description="Tissue-specific effect scores"
    )
    execution_time: float = Field(..., description="MRA execution time in seconds")


class FeedbackLoop(BaseModel):
    """Feedback loop information."""
    nodes: List[str] = Field(..., description="Nodes in feedback loop")
    loop_type: Literal['positive', 'negative'] = Field(..., description="Loop type")
    strength: float = Field(ge=0.0, le=1.0, description="Loop strength")
    pathway_context: Optional[str] = Field(None, description="Pathway context")
    biological_function: Optional[str] = Field(None, description="Biological function")


class MultiTargetSimulationResult(BaseModel):
    """Result from multi-target simulation."""
    targets: List[str] = Field(..., description="Target node identifiers")
    individual_results: List[SimulationResult] = Field(
        ..., description="Individual target simulation results"
    )
    combined_effects: Dict[str, float] = Field(
        ..., description="Combined effect strengths"
    )
    synergy_analysis: Dict[str, Any] = Field(
        ..., description="Synergy/antagonism analysis"
    )
    network_perturbation: Dict[str, Any] = Field(
        ..., description="Overall network perturbation"
    )
    pathway_enrichment: Dict[str, Any] = Field(
        ..., description="Affected pathway enrichment"
    )
    validation_metrics: Dict[str, float] = Field(
        ..., description="Cross-validation metrics"
    )
    validation_score: float = Field(..., description="Cross-validation score")
    data_sources: Optional[List[DataSourceStatus]] = Field(
        None, description="Data source status and success rates"
    )
    completeness_metrics: Optional[CompletenessMetrics] = Field(
        None, description="Data completeness metrics"
    )


class PathwayComparisonResult(BaseModel):
    """Result from Scenario 5: Pathway Comparison."""
    pathway_query: str = Field(..., description="Original pathway query")
    kegg_pathways: List[Dict[str, Any]] = Field(..., description="KEGG pathway results")
    reactome_pathways: List[Dict[str, Any]] = Field(..., description="Reactome pathway results")
    pathway_overlap: Dict[str, Any] = Field(..., description="Pathway overlap analysis")
    gene_overlap: Dict[str, Any] = Field(..., description="Gene overlap analysis")
    mechanistic_differences: List[Dict[str, Any]] = Field(
        ..., description="Mechanistic differences"
    )
    expression_context: Dict[str, Any] = Field(
        ..., description="Expression context comparison"
    )
    consensus_pathways: List[str] = Field(..., description="Consensus pathway IDs")
    database_specific_insights: Dict[str, List[str]] = Field(
        ..., description="Database-specific unique insights"
    )
    validation_score: float = Field(..., description="Cross-validation score")
    data_sources: Optional[List[DataSourceStatus]] = Field(
        None, description="Data source status and success rates"
    )
    completeness_metrics: Optional[CompletenessMetrics] = Field(
        None, description="Data completeness metrics"
    )


class DrugRepurposingResult(BaseModel):
    """Result from Scenario 6: Drug Repurposing."""
    disease_query: str = Field(..., description="Original disease query")
    disease_pathways: List[Pathway] = Field(..., description="Disease-associated pathways")
    candidate_drugs: List[DrugInfo] = Field(..., description="Candidate drugs")
    repurposing_scores: Dict[str, float] = Field(
        ..., description="Drug ID -> repurposing score"
    )
    network_validation: Dict[str, Any] = Field(
        ..., description="Network-based validation"
    )
    off_target_analysis: Dict[str, Any] = Field(
        ..., description="Off-target effect analysis"
    )
    expression_validation: Dict[str, Any] = Field(
        ..., description="Expression-based validation"
    )
    combination_opportunities: List[Dict[str, Any]] = Field(
        ..., description="Drug combination opportunities"
    )
    safety_profiles: Dict[str, Dict[str, Any]] = Field(
        ..., description="Drug safety profiles"
    )
    clinical_evidence: Dict[str, List[str]] = Field(
        ..., description="Clinical evidence for repurposing"
    )
    validation_score: float = Field(..., description="Cross-validation score")
    data_sources: Optional[List[DataSourceStatus]] = Field(
        None, description="Data source status and success rates"
    )
    completeness_metrics: Optional[CompletenessMetrics] = Field(
        None, description="Data completeness metrics"
    )
