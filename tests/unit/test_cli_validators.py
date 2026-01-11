"""
Unit Tests for CLI Validators

Tests for parameter validation in interactive and YAML modes.
"""

import pytest
from src.cli.validators import CLIValidator


class TestCLIValidator:
    """Test CLI parameter validation."""
    
    @pytest.fixture
    def validator(self):
        """Create validator instance."""
        return CLIValidator()
    
    # Disease Query Validation Tests
    def test_validate_disease_query_valid(self, validator):
        """Test valid disease query."""
        is_valid, error = validator.validate_disease_query("breast cancer")
        assert is_valid is True
        assert error is None
    
    def test_validate_disease_query_empty(self, validator):
        """Test empty disease query."""
        is_valid, error = validator.validate_disease_query("")
        assert is_valid is False
        assert "cannot be empty" in error.lower()
    
    def test_validate_disease_query_too_short(self, validator):
        """Test disease query too short."""
        is_valid, error = validator.validate_disease_query("ab")
        assert is_valid is False
        assert "too short" in error.lower()
    
    def test_validate_disease_query_too_long(self, validator):
        """Test disease query too long."""
        long_query = "a" * 101
        is_valid, error = validator.validate_disease_query(long_query)
        assert is_valid is False
        assert "too long" in error.lower()
    
    # Tissue Context Validation Tests
    def test_validate_tissue_context_valid(self, validator):
        """Test valid tissue context."""
        is_valid, error, _ = validator.validate_tissue_context("breast")
        assert is_valid is True
        assert error is None
    
    def test_validate_tissue_context_empty(self, validator):
        """Test empty tissue context (should be valid as optional)."""
        is_valid, error, _ = validator.validate_tissue_context("")
        assert is_valid is True
        assert error is None
    
    def test_validate_tissue_context_invalid_with_suggestions(self, validator):
        """Test invalid tissue with suggestions."""
        is_valid, error, suggestions = validator.validate_tissue_context("brest")
        assert is_valid is False
        assert suggestions is not None
        assert "breast" in suggestions
    
    def test_validate_tissue_context_custom(self, validator):
        """Test custom tissue context (should warn but allow)."""
        is_valid, error, _ = validator.validate_tissue_context("custom_tissue")
        assert is_valid is True
    
    # Target Gene Validation Tests
    def test_validate_target_gene_valid(self, validator):
        """Test valid gene symbol."""
        is_valid, error = validator.validate_target_gene("TP53")
        assert is_valid is True
        assert error is None
    
    def test_validate_target_gene_lowercase(self, validator):
        """Test lowercase gene symbol (should be valid)."""
        is_valid, error = validator.validate_target_gene("brca1")
        assert is_valid is True
    
    def test_validate_target_gene_empty(self, validator):
        """Test empty gene symbol."""
        is_valid, error = validator.validate_target_gene("")
        assert is_valid is False
        assert "cannot be empty" in error.lower()
    
    def test_validate_target_gene_invalid_format(self, validator):
        """Test invalid gene symbol format."""
        is_valid, error = validator.validate_target_gene("TP-53")
        assert is_valid is False
        assert "invalid" in error.lower()
    
    def test_validate_target_gene_too_short(self, validator):
        """Test gene symbol too short."""
        is_valid, error = validator.validate_target_gene("A")
        assert is_valid is False
        assert "too short" in error.lower()
    
    def test_validate_target_gene_too_long(self, validator):
        """Test gene symbol too long."""
        is_valid, error = validator.validate_target_gene("VERYLONGGENENAMEOVER15CHARS")
        assert is_valid is False
        assert "too long" in error.lower()
    
    # Targets List Validation Tests
    def test_validate_targets_valid(self, validator):
        """Test valid targets list."""
        is_valid, error, valid_targets = validator.validate_targets(["TP53", "BRCA1", "BRCA2"])
        assert is_valid is True
        assert error is None
        assert len(valid_targets) == 3
        assert all(t.isupper() for t in valid_targets)
    
    def test_validate_targets_empty(self, validator):
        """Test empty targets list."""
        is_valid, error, _ = validator.validate_targets([])
        assert is_valid is False
        assert "cannot be empty" in error.lower()
    
    def test_validate_targets_with_invalid(self, validator):
        """Test targets list with invalid entries."""
        is_valid, error, valid_targets = validator.validate_targets(["TP53", "INVALID-GENE", "BRCA1"])
        assert is_valid is False
        assert "INVALID-GENE" in error
    
    def test_validate_targets_duplicates(self, validator):
        """Test targets list with duplicates."""
        is_valid, error, _ = validator.validate_targets(["TP53", "BRCA1", "TP53"])
        assert is_valid is False
        assert "duplicate" in error.lower()
    
    # Simulation Mode Validation Tests
    def test_validate_simulation_mode_simple(self, validator):
        """Test simple simulation mode."""
        is_valid, error = validator.validate_simulation_mode("simple")
        assert is_valid is True
        assert error is None
    
    def test_validate_simulation_mode_mra(self, validator):
        """Test MRA simulation mode."""
        is_valid, error = validator.validate_simulation_mode("mra")
        assert is_valid is True
        assert error is None
    
    def test_validate_simulation_mode_invalid(self, validator):
        """Test invalid simulation mode."""
        is_valid, error = validator.validate_simulation_mode("advanced")
        assert is_valid is False
        assert "invalid" in error.lower()
    
    def test_validate_simulation_mode_empty(self, validator):
        """Test empty simulation mode."""
        is_valid, error = validator.validate_simulation_mode("")
        assert is_valid is False
    
    # Perturbation Type Validation Tests
    def test_validate_perturbation_type_valid(self, validator):
        """Test valid perturbation types."""
        for perturb_type in ['inhibit', 'activate', 'knockdown', 'overexpress']:
            is_valid, error = validator.validate_perturbation_type(perturb_type)
            assert is_valid is True
            assert error is None
    
    def test_validate_perturbation_type_invalid(self, validator):
        """Test invalid perturbation type."""
        is_valid, error = validator.validate_perturbation_type("stimulate")
        assert is_valid is False
        assert "invalid" in error.lower()
    
    def test_validate_perturbation_type_empty(self, validator):
        """Test empty perturbation type (should be valid as optional)."""
        is_valid, error = validator.validate_perturbation_type("")
        assert is_valid is True
    
    # Perturbation Strength Validation Tests
    def test_validate_perturbation_strength_valid(self, validator):
        """Test valid perturbation strength."""
        is_valid, error = validator.validate_perturbation_strength(0.9)
        assert is_valid is True
        assert error is None
    
    def test_validate_perturbation_strength_min(self, validator):
        """Test minimum perturbation strength."""
        is_valid, error = validator.validate_perturbation_strength(0.0)
        assert is_valid is True
    
    def test_validate_perturbation_strength_max(self, validator):
        """Test maximum perturbation strength."""
        is_valid, error = validator.validate_perturbation_strength(1.0)
        assert is_valid is True
    
    def test_validate_perturbation_strength_too_low(self, validator):
        """Test perturbation strength below minimum."""
        is_valid, error = validator.validate_perturbation_strength(-0.1)
        assert is_valid is False
        assert "between 0.0 and 1.0" in error
    
    def test_validate_perturbation_strength_too_high(self, validator):
        """Test perturbation strength above maximum."""
        is_valid, error = validator.validate_perturbation_strength(1.5)
        assert is_valid is False
        assert "between 0.0 and 1.0" in error
    
    def test_validate_perturbation_strength_invalid_type(self, validator):
        """Test perturbation strength with invalid type."""
        is_valid, error = validator.validate_perturbation_strength("high")
        assert is_valid is False
        assert "must be a number" in error
    
    def test_validate_perturbation_strength_none(self, validator):
        """Test None perturbation strength (should be valid as optional)."""
        is_valid, error = validator.validate_perturbation_strength(None)
        assert is_valid is True
    
    # Scenario ID Validation Tests
    def test_validate_scenario_id_valid(self, validator):
        """Test valid scenario IDs."""
        for sid in range(1, 7):
            is_valid, error = validator.validate_scenario_id(sid)
            assert is_valid is True
            assert error is None
    
    def test_validate_scenario_id_too_low(self, validator):
        """Test scenario ID below minimum."""
        is_valid, error = validator.validate_scenario_id(0)
        assert is_valid is False
        assert "between 1 and 6" in error
    
    def test_validate_scenario_id_too_high(self, validator):
        """Test scenario ID above maximum."""
        is_valid, error = validator.validate_scenario_id(7)
        assert is_valid is False
        assert "between 1 and 6" in error
    
    def test_validate_scenario_id_invalid_type(self, validator):
        """Test scenario ID with invalid type."""
        is_valid, error = validator.validate_scenario_id("one")
        assert is_valid is False
        assert "must be an integer" in error
    
    # Scenario Parameters Validation Tests
    def test_validate_scenario_1_parameters(self, validator):
        """Test Scenario 1 parameter validation."""
        params = {
            'disease_query': 'breast cancer',
            'tissue_context': 'breast'
        }
        is_valid, errors = validator.validate_scenario_parameters(1, params)
        assert is_valid is True
        assert len(errors) == 0
    
    def test_validate_scenario_1_missing_required(self, validator):
        """Test Scenario 1 with missing required parameter."""
        params = {'tissue_context': 'breast'}
        is_valid, errors = validator.validate_scenario_parameters(1, params)
        assert is_valid is False
        assert any('disease_query' in error for error in errors)
    
    def test_validate_scenario_2_parameters(self, validator):
        """Test Scenario 2 parameter validation."""
        params = {'target_query': 'TP53'}
        is_valid, errors = validator.validate_scenario_parameters(2, params)
        assert is_valid is True
        assert len(errors) == 0
    
    def test_validate_scenario_3_parameters(self, validator):
        """Test Scenario 3 parameter validation."""
        params = {
            'cancer_type': 'breast cancer',
            'tissue_context': 'breast'
        }
        is_valid, errors = validator.validate_scenario_parameters(3, params)
        assert is_valid is True
        assert len(errors) == 0
    
    def test_validate_scenario_4_parameters(self, validator):
        """Test Scenario 4 parameter validation."""
        params = {
            'targets': ['TP53', 'BRCA1', 'BRCA2'],
            'disease_context': 'breast cancer',
            'simulation_mode': 'mra',
            'tissue_context': 'breast'
        }
        is_valid, errors = validator.validate_scenario_parameters(4, params)
        assert is_valid is True
        assert len(errors) == 0
    
    def test_validate_scenario_4_invalid_simulation_mode(self, validator):
        """Test Scenario 4 with invalid simulation mode."""
        params = {
            'targets': ['TP53', 'BRCA1'],
            'disease_context': 'breast cancer',
            'simulation_mode': 'invalid'
        }
        is_valid, errors = validator.validate_scenario_parameters(4, params)
        assert is_valid is False
        assert any('simulation_mode' in error.lower() for error in errors)
    
    def test_validate_scenario_5_parameters(self, validator):
        """Test Scenario 5 parameter validation."""
        params = {'pathway_query': 'apoptosis'}
        is_valid, errors = validator.validate_scenario_parameters(5, params)
        assert is_valid is True
        assert len(errors) == 0
    
    def test_validate_scenario_6_parameters(self, validator):
        """Test Scenario 6 parameter validation."""
        params = {
            'disease_query': 'breast cancer',
            'tissue_context': 'breast',
            'simulation_mode': 'simple'
        }
        is_valid, errors = validator.validate_scenario_parameters(6, params)
        assert is_valid is True
        assert len(errors) == 0
    
    def test_validate_scenario_multiple_errors(self, validator):
        """Test scenario with multiple validation errors."""
        params = {
            'targets': [],  # Invalid: empty
            'disease_context': '',  # Invalid: empty
            'simulation_mode': 'invalid'  # Invalid: not simple/mra
        }
        is_valid, errors = validator.validate_scenario_parameters(4, params)
        assert is_valid is False
        assert len(errors) >= 2  # Multiple errors should be caught

