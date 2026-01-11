"""
Pydantic Data Models

Unified data models for all MCP outputs and pipeline data structures.
"""

from typing import Dict, List, Optional, Literal, Any
from pydantic import BaseModel, Field


class DataSourceStatus(BaseModel):
    """Track data source completeness and success rates."""
    source_name: str = Field(..., description="Database name (hpa, kegg, string, chembl, etc.)")
    requested: int = Field(..., ge=0, description="Number of queries attempted")
    successful: int = Field(..., ge=0, description="Number of successful queries")
    failed: int = Field(..., ge=0, description="Number of failed queries")
    success_rate: float = Field(..., ge=0.0, le=1.0, description="Success rate (0-1)")
    error_types: List[str] = Field(default_factory=list, description="Types of errors encountered")
    avg_response_time: Optional[float] = Field(None, description="Average response time in seconds")


class CompletenessMetrics(BaseModel):
    """Overall data completeness metrics."""
    expression_data: Optional[float] = Field(None, ge=0.0, le=1.0, description="Expression data completeness")
    pathology_data: Optional[float] = Field(None, ge=0.0, le=1.0, description="Pathology data completeness")
    network_data: Optional[float] = Field(None, ge=0.0, le=1.0, description="Network data completeness")
    pathway_data: Optional[float] = Field(None, ge=0.0, le=1.0, description="Pathway data completeness")
    drug_data: Optional[float] = Field(None, ge=0.0, le=1.0, description="Drug data completeness")
    overall_completeness: float = Field(..., ge=0.0, le=1.0, description="Overall completeness score")


class Disease(BaseModel):
    """Disease information from KEGG or Reactome."""
    id: str = Field(..., description="Disease identifier")
    name: str = Field(..., description="Disease name")
    source_db: Literal['kegg', 'reactome'] = Field(..., description="Source database")
    pathways: List[str] = Field(default_factory=list, description="Associated pathway IDs")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence score (0-1)")
    description: Optional[str] = Field(None, description="Disease description")
    category: Optional[str] = Field(None, description="Disease category")


class Pathway(BaseModel):
    """Pathway information from KEGG or Reactome."""
    id: str = Field(..., description="Pathway identifier")
    name: str = Field(..., description="Pathway name")
    source_db: Literal['kegg', 'reactome'] = Field(..., description="Source database")
    genes: List[str] = Field(default_factory=list, description="Gene symbols in pathway")
    hierarchy_level: Optional[int] = Field(None, description="Hierarchy level in Reactome")
    parent_pathway: Optional[str] = Field(None, description="Parent pathway ID")
    description: Optional[str] = Field(None, description="Pathway description")
    confidence: float = Field(ge=0.0, le=1.0, default=1.0, description="Pathway confidence")


class Protein(BaseModel):
    """Protein information with cross-database identifiers."""
    gene_symbol: str = Field(..., description="Gene symbol")
    uniprot_id: Optional[str] = Field(None, description="UniProt identifier")
    string_id: Optional[str] = Field(None, description="STRING identifier")
    kegg_id: Optional[str] = Field(None, description="KEGG gene identifier")
    ensembl_id: Optional[str] = Field(None, description="Ensembl identifier")
    hpa_id: Optional[str] = Field(None, description="HPA identifier")
    description: Optional[str] = Field(None, description="Protein description")
    molecular_weight: Optional[float] = Field(None, description="Molecular weight in Da")
    protein_class: Optional[str] = Field(None, description="Protein class/type")
    confidence: float = Field(0.8, description="Confidence score for protein data")


class Interaction(BaseModel):
    """Protein-protein interaction with evidence."""
    protein_a: str = Field(..., description="First protein identifier")
    protein_b: str = Field(..., description="Second protein identifier")
    combined_score: float = Field(ge=0.0, le=1.0, description="Combined confidence score")
    evidence_types: Dict[str, float] = Field(
        default_factory=dict, 
        description="Evidence type scores (experimental, database, etc.)"
    )
    pathway_context: Optional[str] = Field(None, description="Shared pathway context")
    interaction_type: Optional[str] = Field(None, description="Type of interaction")
    source_database: Literal['string', 'reactome'] = Field(..., description="Source database")


class ExpressionProfile(BaseModel):
    """Tissue-specific expression profile."""
    gene: str = Field(..., description="Gene symbol")
    tissue: str = Field(..., description="Tissue name")
    expression_level: Literal['Not detected', 'Low', 'Medium', 'High'] = Field(
        ..., description="Expression level"
    )
    reliability: Literal['Approved', 'Supported', 'Uncertain'] = Field(
        default='Uncertain', description="Expression reliability"
    )
    cell_type_specific: bool = Field(False, description="Cell type specificity")
    subcellular_location: List[str] = Field(
        default_factory=list, 
        description="Subcellular localization"
    )



class CancerMarker(BaseModel):
    """Cancer prognostic marker information."""
    gene: str = Field(..., description="Gene symbol")
    cancer_type: str = Field(..., description="Cancer type")
    prognostic_value: Literal['favorable', 'unfavorable', 'variable', 'unknown'] = Field(
        ..., description="Prognostic value (favorable/unfavorable for clear associations, variable/unknown for complex cases)"
    )
    survival_association: str = Field(..., description="Survival association")
    expression_pattern: Dict[str, Any] = Field(
        default_factory=dict, 
        description="Expression pattern in cancer vs normal"
    )
    clinical_relevance: Optional[str] = Field(None, description="Clinical relevance")
    confidence: float = Field(ge=0.0, le=1.0, description="Marker confidence score")


class DrugInfo(BaseModel):
    """Drug information from KEGG."""
    drug_id: str = Field(..., description="Drug identifier")
    name: str = Field(..., description="Drug name")
    indication: Optional[str] = Field(None, description="Therapeutic indication")
    mechanism: Optional[str] = Field(None, description="Mechanism of action")
    targets: List[str] = Field(default_factory=list, description="Target protein IDs")
    development_status: Optional[str] = Field(None, description="Development status")
    drug_class: Optional[str] = Field(None, description="Drug class")
    approval_status: Optional[str] = Field(None, description="FDA approval status")
    source: Optional[str] = Field(None, description="Source database (kegg, chembl, etc.)")
    bioactivity_nm: Optional[float] = Field(None, description="Median bioactivity value (nM)")
    drug_likeness_score: Optional[float] = Field(
        None, ge=0.0, le=1.0, description="Drug-likeness score (0-1)")
    similarity_score: Optional[float] = Field(
        None, ge=0.0, le=1.0, description="Structural/functional similarity to known drugs")


class NetworkNode(BaseModel):
    """Network node with metadata."""
    id: str = Field(..., description="Node identifier")
    node_type: Literal['protein', 'complex', 'small_molecule'] = Field(
        ..., description="Node type"
    )
    gene_symbol: Optional[str] = Field(None, description="Gene symbol")
    uniprot_id: Optional[str] = Field(None, description="UniProt ID")
    pathways: List[str] = Field(default_factory=list, description="Associated pathways")
    expression_level: Optional[str] = Field(None, description="Expression level")
    centrality_measures: Dict[str, float] = Field(
        default_factory=dict, 
        description="Network centrality measures"
    )
    # Phase 1 (P1) enhancements
    function: Optional[str] = Field(None, description="Protein function/description from UniProt")
    subcellular_location: List[str] = Field(
        default_factory=list, 
        description="Subcellular localization (e.g., membrane, cytoplasm, nucleus)"
    )
    # Phase 2 (P2) preparation
    domains: Optional[List[Dict[str, Any]]] = Field(
        None, 
        description="Protein domains and binding sites (P2 enhancement)"
    )


class NetworkEdge(BaseModel):
    """Network edge with interaction data."""
    source: str = Field(..., description="Source node ID")
    target: str = Field(..., description="Target node ID")
    weight: float = Field(ge=0.0, le=1.0, description="Edge weight")
    interaction_type: Optional[str] = Field(None, description="Interaction type")
    evidence_score: float = Field(ge=0.0, le=1.0, description="Evidence score")
    pathway_context: Optional[str] = Field(None, description="Pathway context")
    # Phase 2 (P2) enhancement
    regulatory_action: Optional[str] = Field(
        None, 
        description="Regulatory action (activation, inhibition, binding, expression, catalysis)"
    )


class PrioritizedTarget(BaseModel):
    """Prioritized target for cancer analysis."""
    target_id: str = Field(..., description="Target identifier")
    target_name: str = Field(..., description="Target name")
    priority_score: float = Field(..., ge=0.0, le=1.0, description="Overall priority score (0-1)")
    druggability_score: float = Field(..., ge=0.0, le=1.0, description="Druggability score (0-1)")
    cancer_specificity: float = Field(..., ge=0.0, le=1.0, description="Cancer specificity score (0-1)")
    network_centrality: float = Field(..., ge=0.0, le=1.0, description="Network centrality score (0-1)")
    prognostic_value: float = Field(..., ge=0.0, le=1.0, description="Prognostic value score (0-1)")
    pathway_impact: float = Field(..., ge=0.0, le=1.0, description="Pathway impact score (0-1)")
    validation_status: str = Field(..., description="Validation status")
    external_ids: Dict[str, str] = Field({}, description="External database IDs")


class SimulationResult(BaseModel):
    """Result from simulation engine."""
    target_node: str = Field(..., description="Target node identifier")
    mode: Literal['inhibit', 'activate'] = Field(..., description="Simulation mode")
    affected_nodes: Dict[str, float] = Field(..., description="Node effects")
    direct_targets: List[str] = Field(..., description="Direct interaction targets")
    downstream: List[str] = Field(..., description="Downstream affected nodes")
    upstream: List[str] = Field(default_factory=list, description="Upstream regulators")
    feedback_loops: List[str] = Field(default_factory=list, description="Feedback loop nodes")
    network_impact: Dict[str, Any] = Field(..., description="Network-level metrics")
    confidence_scores: Dict[str, float] = Field(..., description="Confidence scores")
    execution_time: float = Field(ge=0.0, description="Execution time in seconds")
    biological_context: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Biological annotations (functions, pathways)"
    )
    drug_info: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Known drug annotations for the target"
    )


class NetworkExpansionConfig(BaseModel):
    """Configuration for network expansion in MRA simulation."""
    initial_neighbors: int = Field(
        2,
        ge=1,
        le=10,
        description="Number of neighbors to expand per target in Step 3 (default: 2 for >5 targets, 5 for <=5 targets)"
    )
    expansion_neighbors: int = Field(
        1,
        ge=0,
        le=5,
        description="Number of neighbors to expand per target in Step 6 (default: 1)"
    )
    max_network_size: int = Field(
        100,
        ge=20,
        le=500,
        description="Maximum network size limit (default: 100 nodes)"
    )
    step_timeouts: Optional[Dict[int, int]] = Field(
        None,
        description="Override step timeouts in seconds (e.g., {3: 300, 6: 400})"
    )


class MRASimulationResult(BaseModel):
    """Result from MRA simulation engine."""
    target_node: str = Field(..., description="Target node identifier")
    steady_state: List[float] = Field(..., description="Steady-state response values")
    convergence_info: Dict[str, Any] = Field(..., description="Convergence information")
    upstream_effects: List[str] = Field(..., description="Upstream affected nodes")
    downstream_effects: List[str] = Field(..., description="Downstream affected nodes")
    feedback_loops: List[str] = Field(default_factory=list, description="Detected feedback loops")
    execution_time: float = Field(ge=0.0, description="Execution time in seconds")


class GeneProvenance(BaseModel):
    """Gene provenance tracking for disease-gene associations."""
    gene_symbol: str = Field(..., description="Gene symbol")
    source: Literal['pathway', 'association', 'both'] = Field(
        ..., description="Discovery source (pathway-based, disease-association, or both)"
    )
    confidence: float = Field(
        ge=0.0, le=1.0,
        description="Overall confidence score (0-1)"
    )
    pathway_count: int = Field(
        ge=0,
        description="Number of pathways containing this gene"
    )
    association_confidence: float = Field(
        ge=0.0, le=1.0,
        description="Disease association confidence from UniProt annotations (0-1)"
    )


class DiseaseNetworkResult(BaseModel):
    """Result from Scenario 1: Disease Network Construction."""
    disease: Disease = Field(..., description="Disease information")
    pathways: List[Pathway] = Field(..., description="Associated pathways")
    network_nodes: List[NetworkNode] = Field(..., description="Network nodes")
    network_edges: List[NetworkEdge] = Field(..., description="Network edges")
    expression_profiles: List[ExpressionProfile] = Field(
        ..., description="Tissue expression data"
    )
    cancer_markers: List[CancerMarker] = Field(
        default_factory=list,
        description="Cancer prognostic markers"
    )
    enrichment_results: Dict[str, Any] = Field(
        ..., description="Functional enrichment results"
    )
    validation_score: float = Field(ge=0.0, le=1.0, description="Overall validation score")
    data_sources: Optional[List[DataSourceStatus]] = Field(
        None, description="Data source status and success rates"
    )
    completeness_metrics: Optional[CompletenessMetrics] = Field(
        None, description="Data completeness metrics"
    )
    network_summary: Optional[Dict[str, Any]] = Field(
        None, description="Network topology metrics (node/edge counts, hubs)"
    )
    expression_summary: Optional[Dict[str, Any]] = Field(
        None, description="Expression coverage and scoring metrics"
    )
    pathway_summary: Optional[Dict[str, Any]] = Field(
        None, description="Pathway coverage and gene distribution metrics"
    )
    # Phase 1.5 enhancements: Gene provenance tracking
    gene_provenance: Optional[Dict[str, GeneProvenance]] = Field(
        None,
        description="Gene provenance tracking (source, confidence, pathway membership)"
    )
    association_summary: Optional[Dict[str, Any]] = Field(
        None,
        description="Phase 1.5 disease-gene association enrichment summary"
    )


class TargetAnalysisResult(BaseModel):
    """Result from Scenario 2: Target-Centric Analysis."""
    target: Protein = Field(..., description="Target protein information")
    pathways: List[Pathway] = Field(..., description="Associated pathways")
    interactions: List[Interaction] = Field(..., description="Protein interactions")
    expression_profiles: List[ExpressionProfile] = Field(
        ..., description="Tissue expression data"
    )
    subcellular_location: List[str] = Field(
        ..., description="Subcellular localization"
    )
    druggability_score: float = Field(ge=0.0, le=1.0, description="Druggability assessment")
    known_drugs: List[DrugInfo] = Field(..., description="Known drugs targeting protein")
    safety_profile: Dict[str, Any] = Field(..., description="Safety assessment")
    validation_score: float = Field(ge=0.0, le=1.0, description="Overall validation score")
    data_sources: Optional[List[DataSourceStatus]] = Field(
        None, description="Data source status and success rates"
    )
    completeness_metrics: Optional[CompletenessMetrics] = Field(
        None, description="Data completeness metrics"
    )
    network_summary: Optional[Dict[str, Any]] = Field(
        None, description="Target-centric network topology metrics"
    )
    expression_summary: Optional[Dict[str, Any]] = Field(
        None, description="Target expression coverage metrics"
    )
    pathway_summary: Optional[Dict[str, Any]] = Field(
        None, description="Target pathway coverage metrics"
    )
    prioritization_summary: Optional[Dict[str, Any]] = Field(
        None, description="Composite scoring breakdown for prioritization"
    )


class CancerAnalysisResult(BaseModel):
    """Result from Scenario 3: Cancer-Specific Analysis."""
    cancer_type: str = Field(..., description="Cancer type")
    prognostic_markers: List[CancerMarker] = Field(..., description="Prognostic markers")
    cancer_pathways: List[Pathway] = Field(..., description="Cancer-associated pathways")
    network_nodes: List[NetworkNode] = Field(..., description="Cancer network nodes")
    network_edges: List[NetworkEdge] = Field(..., description="Cancer network edges")
    expression_dysregulation: Dict[str, Any] = Field(
        ..., description="Expression dysregulation patterns"
    )
    prioritized_targets: List[Dict[str, Any]] = Field(
        ..., description="Prioritized therapeutic targets"
    )
    combination_opportunities: List[Dict[str, Any]] = Field(
        ..., description="Combination therapy opportunities"
    )
    validation_score: float = Field(ge=0.0, le=1.0, description="Overall validation score")
    data_sources: Optional[List[DataSourceStatus]] = Field(
        None, description="Data source status and success rates"
    )
    completeness_metrics: Optional[CompletenessMetrics] = Field(
        None, description="Data completeness metrics"
    )
    network_summary: Optional[Dict[str, Any]] = Field(
        None, description="Cancer network topology metrics"
    )
    expression_summary: Optional[Dict[str, Any]] = Field(
        None, description="Expression dysregulation coverage metrics"
    )
    marker_summary: Optional[Dict[str, Any]] = Field(
        None, description="Prognostic marker distribution summary"
    )


class MultiTargetSimulationResult(BaseModel):
    """Result from Scenario 4: Multi-Target Simulation."""
    validated_targets: List[str] = Field(..., description="Validated target genes")
    network: Dict[str, Any] = Field(..., description="Network structure")
    simulation_result: SimulationResult = Field(..., description="Simulation results")
    impact_assessment: Dict[str, Any] = Field(..., description="Impact assessment")
    validation_score: float = Field(ge=0.0, le=1.0, description="Overall validation score")
    data_sources: Optional[List[DataSourceStatus]] = Field(
        None, description="Data source query status"
    )
    completeness_metrics: Optional[CompletenessMetrics] = Field(
        None, description="Scenario data completeness metrics"
    )
    # NEW: Track warnings and skipped items for transparency
    warnings: List[str] = Field(
        default_factory=list,
        description="Warnings generated during execution (timeouts, data issues)"
    )
    skipped_items: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Items skipped during execution (targets, pathways, etc.) with reason"
    )


class PathwayComparisonResult(BaseModel):
    """Result from Scenario 5: Pathway Comparison."""
    pathway_a: Dict[str, Any] = Field(..., description="First pathway")
    pathway_b: Dict[str, Any] = Field(..., description="Second pathway")
    jaccard_similarity: float = Field(..., description="Jaccard similarity score")
    pathway_concordance: float = Field(..., description="Pathway concordance score")
    common_genes: List[str] = Field(..., description="Common genes")
    pathway_overlap: Dict[str, Any] = Field(..., description="Pathway overlap analysis")
    validation_score: float = Field(ge=0.0, le=1.0, description="Overall validation score")
    data_sources: Optional[List[DataSourceStatus]] = Field(
        None, description="Data source query status"
    )
    completeness_metrics: Optional[CompletenessMetrics] = Field(
        None, description="Scenario data completeness metrics"
    )


class DrugTarget(BaseModel):
    """Drug-target interaction information."""
    drug_id: str = Field(..., description="Drug identifier")
    target_id: str = Field(..., description="Target protein identifier")
    target_protein: Optional[str] = Field(
        None,
        description="Canonical gene symbol for the protein target"
    )
    interaction_type: str = Field(..., description="Type of interaction")
    affinity: Optional[float] = Field(None, description="Binding affinity")
    mechanism: Optional[str] = Field(None, description="Mechanism of action")
    confidence: float = Field(ge=0.0, le=1.0, description="Interaction confidence")


class RepurposingCandidate(BaseModel):
    """Drug repurposing candidate."""
    drug_id: str = Field(..., description="Drug identifier")
    drug_name: str = Field(..., description="Drug name")
    target_protein: str = Field(..., description="Target protein")
    repurposing_score: float = Field(..., description="Repurposing score (0-1)")
    network_impact: float = Field(..., description="Network impact score")
    expression_specificity: float = Field(..., description="Expression specificity score")
    safety_profile: Dict[str, Any] = Field(..., description="Safety profile")
    efficacy_prediction: float = Field(..., description="Efficacy prediction score")
    bioactivity_nm: Optional[float] = Field(
        None, description="Median bioactivity value (nM) from ChEMBL/assays"
    )
    drug_likeness_score: Optional[float] = Field(
        None, ge=0.0, le=1.0, description="Drug-likeness score (0-1)"
    )


class DrugRepurposingResult(BaseModel):
    """Result from Scenario 6: Drug Repurposing."""
    disease_pathways: List[Dict[str, Any]] = Field(..., description="Disease-associated pathways")
    known_drugs: List[DrugInfo] = Field(..., description="Known drugs for disease")
    repurposing_scores: Dict[str, float] = Field(..., description="Repurposing scores")
    network_overlap: float = Field(..., description="Network overlap score")
    safety_profiles: Dict[str, Any] = Field(..., description="Safety profiles")
    validation_score: float = Field(ge=0.0, le=1.0, description="Overall validation score")
    data_sources: Optional[List[DataSourceStatus]] = Field(
        None, description="Data source query status"
    )
    completeness_metrics: Optional[CompletenessMetrics] = Field(
        None, description="Scenario data completeness metrics"
    )


# ============================================================================
# ChEMBL-Specific Data Models (Phase 2 Integration)
# ============================================================================


class Compound(BaseModel):
    """Chemical compound from ChEMBL database."""
    chembl_id: str = Field(..., description="ChEMBL compound ID (e.g., CHEMBL25)")
    name: Optional[str] = Field(None, description="Preferred compound name")

    # Chemical structure
    smiles: Optional[str] = Field(None, description="Canonical SMILES string")
    inchi: Optional[str] = Field(None, description="InChI string")
    inchi_key: Optional[str] = Field(None, description="InChI key")
    molecular_formula: Optional[str] = Field(None, description="Molecular formula")

    # Physicochemical properties
    molecular_weight: Optional[float] = Field(None, description="Molecular weight (Da)")
    alogp: Optional[float] = Field(None, description="Calculated logP (lipophilicity)")
    hba: Optional[int] = Field(None, description="Hydrogen bond acceptors")
    hbd: Optional[int] = Field(None, description="Hydrogen bond donors")
    psa: Optional[float] = Field(None, description="Polar surface area (Ų)")
    rtb: Optional[int] = Field(None, description="Rotatable bonds")
    ro5_violations: int = Field(0, description="Lipinski Rule of Five violations")

    # Metadata
    molecule_type: Optional[str] = Field(None, description="Molecule type (Small molecule, Protein, etc.)")
    source_db: Literal['chembl'] = Field('chembl', description="Source database")
    confidence: float = Field(ge=0.0, le=1.0, default=0.8, description="Data confidence score")
    synonyms: List[str] = Field(default_factory=list, description="Alternative names")

    # Cross-references
    external_refs: Dict[str, str] = Field(
        default_factory=dict,
        description="External database references (PubChem, DrugBank, etc.)"
    )


class Bioactivity(BaseModel):
    """Bioactivity measurement from ChEMBL."""
    activity_id: str = Field(..., description="Unique activity ID")

    # References
    assay_chembl_id: str = Field(..., description="ChEMBL assay ID")
    target_chembl_id: Optional[str] = Field(None, description="ChEMBL target ID")
    molecule_chembl_id: str = Field(..., description="ChEMBL compound ID")

    # Activity measurement
    activity_type: str = Field(..., description="Activity type (IC50, Ki, EC50, Kd, etc.)")
    activity_value: Optional[float] = Field(None, description="Numeric activity value")
    activity_units: Optional[str] = Field(None, description="Activity units (nM, uM, M, etc.)")
    activity_relation: Optional[str] = Field(None, description="Relation (=, <, >, ~, <=, >=)")

    # Assay information
    assay_type: Optional[str] = Field(None, description="Assay type (B=Binding, F=Functional)")
    assay_organism: Optional[str] = Field(None, description="Test organism")

    # Quality metrics
    confidence: float = Field(
        ge=0.0, le=1.0, default=0.7,
        description="Measurement confidence (based on assay type, reproducibility)"
    )
    pchembl_value: Optional[float] = Field(
        None,
        description="pChEMBL value (-log10 of molar IC50/Ki/EC50)"
    )

    # Metadata
    activity_comment: Optional[str] = Field(None, description="Activity comments")
    data_validity_comment: Optional[str] = Field(None, description="Data validity notes")


class TargetBioactivity(BaseModel):
    """Aggregated bioactivity data for a target-compound pair."""
    # Target information
    target_id: str = Field(..., description="Target identifier (gene symbol or ChEMBL ID)")
    target_name: str = Field(..., description="Target name")
    target_chembl_id: Optional[str] = Field(None, description="ChEMBL target ID")
    uniprot_id: Optional[str] = Field(None, description="UniProt accession")

    # Compound information
    compound_id: str = Field(..., description="Compound identifier (ChEMBL ID)")
    compound_name: str = Field(..., description="Compound name")
    compound_smiles: Optional[str] = Field(None, description="Compound SMILES")

    # Bioactivity measurements
    bioactivities: List[Bioactivity] = Field(
        default_factory=list,
        description="All bioactivity measurements"
    )

    # Aggregated activity values (in nM, standardized)
    median_ic50: Optional[float] = Field(None, description="Median IC50 value (nM)")
    median_ki: Optional[float] = Field(None, description="Median Ki value (nM)")
    median_ec50: Optional[float] = Field(None, description="Median EC50 value (nM)")
    median_kd: Optional[float] = Field(None, description="Median Kd value (nM)")

    # Activity counts
    activity_count: int = Field(0, description="Total number of activities")
    ic50_count: int = Field(0, description="Number of IC50 measurements")
    ki_count: int = Field(0, description="Number of Ki measurements")

    # Quality assessment
    data_quality: Literal['high', 'medium', 'low'] = Field(
        'medium',
        description="Overall data quality (high: >5 measurements, medium: 2-5, low: 1)"
    )

    # Druggability metrics
    druggability_score: float = Field(
        ge=0.0, le=1.0,
        description="Druggability score (0-1, based on potency and drug-likeness)"
    )
    potency_category: Literal['very_high', 'high', 'moderate', 'low', 'inactive'] = Field(
        'moderate',
        description="Potency category (very_high: <10nM, high: 10-100nM, moderate: 100-1000nM, low: >1000nM)"
    )

    # Metadata
    confidence: float = Field(
        ge=0.0, le=1.0, default=0.7,
        description="Overall confidence in target-compound relationship"
    )


class DrugLikenessAssessment(BaseModel):
    """Drug-likeness assessment using multiple rules."""
    compound_id: str = Field(..., description="Compound identifier")

    # Lipinski Rule of Five
    lipinski_compliant: bool = Field(..., description="Passes Lipinski Rule of Five")
    ro5_violations: int = Field(..., description="Number of Rule of Five violations")
    detailed_violations: Dict[str, bool] = Field(
        default_factory=dict,
        description="Detailed Lipinski violations (mw, logp, hba, hbd)"
    )

    # Veber's rules (oral bioavailability)
    veber_compliant: bool = Field(
        default=True,
        description="Passes Veber's rules (PSA ≤140 Ų, RTB ≤10)"
    )
    veber_violations: Dict[str, Any] = Field(
        default_factory=dict,
        description="Veber rule violations"
    )

    # Pfizer 3/75 rule (for CNS drugs)
    pfizer_compliant: bool = Field(
        default=True,
        description="Passes Pfizer 3/75 rule (logP <3 and TPSA <75)"
    )

    # Physicochemical properties
    molecular_weight: Optional[float] = Field(None, description="Molecular weight (≤500 for RO5)")
    alogp: Optional[float] = Field(None, description="LogP (≤5 for RO5)")
    hbd: Optional[int] = Field(None, description="H-bond donors (≤5 for RO5)")
    hba: Optional[int] = Field(None, description="H-bond acceptors (≤10 for RO5)")
    psa: Optional[float] = Field(None, description="Polar surface area (≤140 for Veber)")
    rtb: Optional[int] = Field(None, description="Rotatable bonds (≤10 for Veber)")

    # Overall assessment
    overall_assessment: Literal['drug-like', 'lead-like', 'non-drug-like'] = Field(
        ...,
        description="Overall drug-likeness assessment"
    )
    drug_likeness_score: float = Field(
        ge=0.0, le=1.0,
        description="Composite drug-likeness score (0-1)"
    )

    # Recommendations
    issues: List[str] = Field(
        default_factory=list,
        description="List of drug-likeness issues"
    )
    recommendations: List[str] = Field(
        default_factory=list,
        description="Recommendations for improvement"
    )


class MolecularDescriptors(BaseModel):
    """Comprehensive molecular descriptors and properties."""
    compound_id: str = Field(..., description="Compound identifier")

    # Basic properties
    molecular_weight: Optional[float] = Field(None, description="Molecular weight (Da)")
    molecular_formula: Optional[str] = Field(None, description="Molecular formula")

    # Lipophilicity descriptors
    logp: Optional[float] = Field(None, description="Calculated logP")
    logd: Optional[float] = Field(None, description="Calculated logD at pH 7.4")
    logs: Optional[float] = Field(None, description="Aqueous solubility (logS)")

    # Hydrogen bonding
    hba: Optional[int] = Field(None, description="Hydrogen bond acceptors")
    hbd: Optional[int] = Field(None, description="Hydrogen bond donors")

    # Surface properties
    tpsa: Optional[float] = Field(None, description="Topological polar surface area (Ų)")
    psa: Optional[float] = Field(None, description="Polar surface area (Ų)")

    # Flexibility
    rotatable_bonds: Optional[int] = Field(None, description="Number of rotatable bonds")

    # Ring properties
    aromatic_rings: Optional[int] = Field(None, description="Number of aromatic rings")
    aliphatic_rings: Optional[int] = Field(None, description="Number of aliphatic rings")

    # Complexity
    num_atoms: Optional[int] = Field(None, description="Total number of atoms")
    num_heavy_atoms: Optional[int] = Field(None, description="Number of heavy atoms")
    num_stereocenters: Optional[int] = Field(None, description="Number of stereocenters")

    # ADMET properties (if available)
    caco2_permeability: Optional[float] = Field(None, description="Caco-2 permeability")
    blood_brain_barrier: Optional[float] = Field(None, description="Blood-brain barrier penetration")
    plasma_protein_binding: Optional[float] = Field(None, description="Plasma protein binding (%)")

    # Fingerprints (for similarity calculations)
    morgan_fingerprint: Optional[str] = Field(None, description="Morgan fingerprint")
    maccs_keys: Optional[str] = Field(None, description="MACCS keys fingerprint")

    # Metadata
    descriptor_source: str = Field('chembl', description="Source of descriptors")
    calculation_method: Optional[str] = Field(None, description="Calculation method used")
    confidence: float = Field(ge=0.0, le=1.0, default=0.8, description="Descriptor confidence")
