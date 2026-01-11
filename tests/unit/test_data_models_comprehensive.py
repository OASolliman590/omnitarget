"""
Comprehensive Unit Tests for Data Models

Fast unit tests (<1s each) for all Pydantic data models.
"""

import pytest
from pydantic import ValidationError

pytestmark = pytest.mark.unit
from src.models.data_models import (
    Protein, Pathway, Disease, Interaction, ExpressionProfile,
    CancerMarker, DrugInfo, DrugTarget, NetworkNode, NetworkEdge,
    SimulationResult, MRASimulationResult, PrioritizedTarget,
    DiseaseNetworkResult, TargetAnalysisResult, CancerAnalysisResult,
    MultiTargetSimulationResult, PathwayComparisonResult, DrugRepurposingResult
)


class TestProteinModel:
    """Test Protein model validation and functionality."""
    
    def test_protein_creation_valid(self):
        """Test valid protein creation"""
        protein = Protein(
            gene_symbol="TP53",
            uniprot_id="P04637",
            string_id="9606.ENSP00000269305",
            description="Tumor protein p53",
            molecular_weight=43653.0,
            protein_class="Transcription factor"
        )
        assert protein.gene_symbol == "TP53"
        assert protein.uniprot_id == "P04637"
        assert protein.string_id == "9606.ENSP00000269305"
        assert protein.molecular_weight == 43653.0
    
    def test_protein_creation_minimal(self):
        """Test protein creation with minimal required fields"""
        protein = Protein(gene_symbol="TP53")
        assert protein.gene_symbol == "TP53"
        assert protein.uniprot_id is None
        assert protein.string_id is None


class TestPathwayModel:
    """Test Pathway model validation and functionality."""
    
    def test_pathway_creation_valid(self):
        """Test valid pathway creation"""
        pathway = Pathway(
            id="hsa05224",
            name="Breast cancer",
            source_db="kegg",
            genes=["TP53", "BRCA1", "BRCA2"],
            hierarchy_level=1,
            confidence=0.9
        )
        assert pathway.id == "hsa05224"
        assert pathway.name == "Breast cancer"
        assert pathway.source_db == "kegg"
        assert len(pathway.genes) == 3
        assert pathway.confidence == 0.9
    
    def test_pathway_validation_source_db(self):
        """Test pathway source_db validation"""
        with pytest.raises(ValidationError):
            Pathway(
                id="hsa05224",
                name="Breast cancer",
                source_db="invalid_db"  # Must be 'kegg' or 'reactome'
            )


class TestDiseaseModel:
    """Test Disease model validation and functionality."""
    
    def test_disease_creation_valid(self):
        """Test valid disease creation"""
        disease = Disease(
            id="hsa05224",
            name="Breast cancer",
            source_db="kegg",
            pathways=["hsa05224", "hsa05225"],
            confidence=0.85,
            description="Malignant neoplasm of breast",
            category="Cancer"
        )
        assert disease.id == "hsa05224"
        assert disease.name == "Breast cancer"
        assert disease.source_db == "kegg"
        assert len(disease.pathways) == 2
        assert disease.confidence == 0.85
    
    def test_disease_confidence_validation(self):
        """Test disease confidence validation"""
        # Valid confidence
        disease = Disease(id="hsa05224", name="Breast cancer", source_db="kegg", confidence=0.5)
        assert disease.confidence == 0.5
        
        # Invalid confidence (too high)
        with pytest.raises(ValidationError):
            Disease(id="hsa05224", name="Breast cancer", source_db="kegg", confidence=1.5)
        
        # Invalid confidence (too low)
        with pytest.raises(ValidationError):
            Disease(id="hsa05224", name="Breast cancer", source_db="kegg", confidence=-0.1)


class TestInteractionModel:
    """Test Interaction model validation and functionality."""
    
    def test_interaction_creation_valid(self):
        """Test valid interaction creation"""
        interaction = Interaction(
            protein_a="TP53",
            protein_b="MDM2",
            combined_score=0.9,
            evidence_types={"experimental": 0.8, "database": 0.7},
            pathway_context="cell_cycle",
            interaction_type="binding",
            source_database="string"
        )
        assert interaction.protein_a == "TP53"
        assert interaction.protein_b == "MDM2"
        assert interaction.combined_score == 0.9
        assert interaction.evidence_types["experimental"] == 0.8
        assert interaction.source_database == "string"
    
    def test_interaction_score_validation(self):
        """Test interaction score validation"""
        # Valid score
        interaction = Interaction(
            protein_a="TP53", protein_b="MDM2", combined_score=0.5, source_database="string"
        )
        assert interaction.combined_score == 0.5
        
        # Invalid score (too high)
        with pytest.raises(ValidationError):
            Interaction(
                protein_a="TP53", protein_b="MDM2", combined_score=1.5, source_database="string"
            )
        
        # Invalid score (too low)
        with pytest.raises(ValidationError):
            Interaction(
                protein_a="TP53", protein_b="MDM2", combined_score=-0.1, source_database="string"
            )


class TestExpressionProfileModel:
    """Test ExpressionProfile model validation and functionality."""
    
    def test_expression_profile_creation_valid(self):
        """Test valid expression profile creation"""
        profile = ExpressionProfile(
            gene="TP53",
            tissue="breast",
            expression_level="High",
            reliability="Approved",
            cell_type_specific=True,
            subcellular_location=["nucleus", "cytoplasm"]
        )
        assert profile.gene == "TP53"
        assert profile.tissue == "breast"
        assert profile.expression_level == "High"
        assert profile.reliability == "Approved"
        assert profile.cell_type_specific == True
        assert len(profile.subcellular_location) == 2
    
    def test_expression_level_validation(self):
        """Test expression level validation"""
        valid_levels = ["Not detected", "Low", "Medium", "High"]
        for level in valid_levels:
            profile = ExpressionProfile(
                gene="TP53", tissue="breast", expression_level=level, reliability="Approved"
            )
            assert profile.expression_level == level
        
        # Invalid level
        with pytest.raises(ValidationError):
            ExpressionProfile(
                gene="TP53", tissue="breast", expression_level="Invalid", reliability="Approved"
            )


class TestCancerMarkerModel:
    """Test CancerMarker model validation and functionality."""
    
    def test_cancer_marker_creation_valid(self):
        """Test valid cancer marker creation"""
        marker = CancerMarker(
            gene="TP53",
            cancer_type="breast cancer",
            prognostic_value="favorable",
            survival_association="improved survival",
            expression_pattern={"cancer": "high", "normal": "low"},
            clinical_relevance="prognostic biomarker",
            confidence=0.8
        )
        assert marker.gene == "TP53"
        assert marker.cancer_type == "breast cancer"
        assert marker.prognostic_value == "favorable"
        assert marker.confidence == 0.8
    
    def test_prognostic_value_validation(self):
        """Test prognostic value validation"""
        valid_values = ["favorable", "unfavorable"]
        for value in valid_values:
            marker = CancerMarker(
                gene="TP53", cancer_type="breast cancer", prognostic_value=value,
                survival_association="test", confidence=0.8
            )
            assert marker.prognostic_value == value
        
        # Invalid value
        with pytest.raises(ValidationError):
            CancerMarker(
                gene="TP53", cancer_type="breast cancer", prognostic_value="invalid",
                survival_association="test", confidence=0.8
            )


class TestSimulationResultModel:
    """Test SimulationResult model validation and functionality."""
    
    def test_simulation_result_creation_valid(self):
        """Test valid simulation result creation"""
        result = SimulationResult(
            target_node="TP53",
            mode="inhibit",
            affected_nodes={"TP53": 0.9, "MDM2": -0.7, "BRCA1": -0.5},
            direct_targets=["MDM2"],
            downstream=["MDM2", "BRCA1"],
            upstream=["ATM"],
            feedback_loops=["MDM2"],
            network_impact={"total_affected": 3, "mean_effect": 0.7},
            confidence_scores={"TP53": 1.0, "MDM2": 0.8, "BRCA1": 0.6},
            execution_time=0.15
        )
        assert result.target_node == "TP53"
        assert result.mode == "inhibit"
        assert len(result.affected_nodes) == 3
        assert result.affected_nodes["TP53"] == 0.9
        assert result.execution_time == 0.15
    
    def test_simulation_result_mode_validation(self):
        """Test simulation result mode validation"""
        valid_modes = ["inhibit", "activate"]
        for mode in valid_modes:
            result = SimulationResult(
                target_node="TP53", mode=mode, affected_nodes={}, direct_targets=[],
                downstream=[], network_impact={}, confidence_scores={}, execution_time=0.0
            )
            assert result.mode == mode
        
        # Invalid mode
        with pytest.raises(ValidationError):
            SimulationResult(
                target_node="TP53", mode="invalid", affected_nodes={}, direct_targets=[],
                downstream=[], network_impact={}, confidence_scores={}, execution_time=0.0
            )


class TestPrioritizedTargetModel:
    """Test PrioritizedTarget model validation and functionality."""
    
    def test_prioritized_target_creation_valid(self):
        """Test valid prioritized target creation"""
        target = PrioritizedTarget(
            target_id="TP53",
            target_name="Tumor protein p53",
            priority_score=0.9,
            druggability_score=0.8,
            cancer_specificity=0.85,
            network_centrality=0.7,
            prognostic_value=0.75,
            pathway_impact=0.8,
            validation_status="validated",
            external_ids={"entrez": "7157", "uniprot": "P04637"}
        )
        assert target.target_id == "TP53"
        assert target.target_name == "Tumor protein p53"
        assert target.priority_score == 0.9
        assert target.druggability_score == 0.8
        assert target.validation_status == "validated"
    
    def test_prioritized_target_score_validation(self):
        """Test prioritized target score validation"""
        # Valid scores
        target = PrioritizedTarget(
            target_id="TP53", target_name="TP53", priority_score=0.5,
            druggability_score=0.5, cancer_specificity=0.5, network_centrality=0.5,
            prognostic_value=0.5, pathway_impact=0.5, validation_status="test"
        )
        assert target.priority_score == 0.5
        
        # Invalid score (too high)
        with pytest.raises(ValidationError):
            PrioritizedTarget(
                target_id="TP53", target_name="TP53", priority_score=1.5,
                druggability_score=0.5, cancer_specificity=0.5, network_centrality=0.5,
                prognostic_value=0.5, pathway_impact=0.5, validation_status="test"
            )


class TestResultModels:
    """Test result model validation and functionality."""
    
    def test_disease_network_result_creation(self):
        """Test DiseaseNetworkResult creation"""
        disease = Disease(id="hsa05224", name="Breast cancer", source_db="kegg", confidence=0.8)
        pathway = Pathway(id="hsa05224", name="Breast cancer", source_db="kegg", genes=["TP53"])
        
        result = DiseaseNetworkResult(
            disease=disease,
            pathways=[pathway],
            network_nodes=[],
            network_edges=[],
            expression_profiles=[],
            cancer_markers=[],
            enrichment_results={"GO:0006915": 0.001},
            validation_score=0.85
        )
        assert result.disease.name == "Breast cancer"
        assert len(result.pathways) == 1
        assert result.validation_score == 0.85
    
    def test_target_analysis_result_creation(self):
        """Test TargetAnalysisResult creation"""
        protein = Protein(gene_symbol="TP53", uniprot_id="P04637")
        pathway = Pathway(id="hsa05224", name="Breast cancer", source_db="kegg", genes=["TP53"])
        interaction = Interaction(
            protein_a="TP53", protein_b="MDM2", combined_score=0.9, source_database="string"
        )
        
        result = TargetAnalysisResult(
            target=protein,
            pathways=[pathway],
            interactions=[interaction],
            expression_profiles=[],
            subcellular_location=["nucleus"],
            druggability_score=0.8,
            known_drugs=[],
            safety_profile={"toxicity": "low"}
        )
        assert result.target.gene_symbol == "TP53"
        assert result.druggability_score == 0.8
        assert len(result.interactions) == 1


class TestModelSerialization:
    """Test model serialization and deserialization."""
    
    def test_protein_serialization(self):
        """Test protein model serialization"""
        protein = Protein(
            gene_symbol="TP53",
            uniprot_id="P04637",
            string_id="9606.ENSP00000269305"
        )
        
        # Test to dict
        protein_dict = protein.model_dump()
        assert protein_dict["gene_symbol"] == "TP53"
        assert protein_dict["uniprot_id"] == "P04637"
        assert protein_dict["string_id"] == "9606.ENSP00000269305"
        
        # Test from dict
        protein_from_dict = Protein(**protein_dict)
        assert protein_from_dict.gene_symbol == "TP53"
        assert protein_from_dict.uniprot_id == "P04637"


class TestModelValidation:
    """Test comprehensive model validation."""
    
    def test_all_models_instantiation(self):
        """Test that all models can be instantiated with valid data"""
        # Test all model types
        models = [
            Protein(gene_symbol="TP53"),
            Pathway(id="hsa05224", name="Breast cancer", source_db="kegg"),
            Disease(id="hsa05224", name="Breast cancer", source_db="kegg", confidence=0.8),
            Interaction(protein_a="TP53", protein_b="MDM2", combined_score=0.9, source_database="string"),
            ExpressionProfile(gene="TP53", tissue="breast", expression_level="High", reliability="Approved"),
            CancerMarker(gene="TP53", cancer_type="breast cancer", prognostic_value="favorable", 
                       survival_association="test", confidence=0.8),
            DrugInfo(drug_id="CHEMBL123", name="Aspirin"),
            DrugTarget(drug_id="CHEMBL123", target_id="TP53", interaction_type="binding", confidence=0.8),
            NetworkNode(id="TP53", node_type="protein", gene_symbol="TP53"),
            NetworkEdge(source="TP53", target="MDM2", weight=0.9, evidence_score=0.9),
            SimulationResult(target_node="TP53", mode="inhibit", affected_nodes={}, direct_targets=[],
                           downstream=[], network_impact={}, confidence_scores={}, execution_time=0.0),
            PrioritizedTarget(target_id="TP53", target_name="TP53", priority_score=0.8,
                             druggability_score=0.7, cancer_specificity=0.8, network_centrality=0.6,
                             prognostic_value=0.7, pathway_impact=0.8, validation_status="test")
        ]
        
        # All models should be created successfully
        assert len(models) == 12
        for model in models:
            assert model is not None
    
    def test_model_field_validation(self):
        """Test field validation across all models"""
        # Test required fields
        with pytest.raises(ValidationError):
            Protein()  # Missing required gene_symbol
        
        with pytest.raises(ValidationError):
            Pathway()  # Missing required id, name, source_db
        
        # Test field constraints
        with pytest.raises(ValidationError):
            Disease(id="hsa05224", name="Breast cancer", source_db="kegg", confidence=1.5)  # Invalid confidence
        
        with pytest.raises(ValidationError):
            Interaction(protein_a="TP53", protein_b="MDM2", combined_score=1.5, source_database="string")  # Invalid score
