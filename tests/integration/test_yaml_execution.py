"""
Integration Tests for YAML Runner

Tests for YAML configuration loading, validation, and execution.
Uses mock MCP responses to avoid dependency on real servers.
"""

import pytest
import asyncio
from pathlib import Path
import tempfile
import yaml

from src.cli.yaml_runner import YAMLRunner
from src.cli.validators import CLIValidator


@pytest.fixture
def temp_yaml_file():
    """Create temporary YAML file for testing."""
    def _create_yaml(content):
        temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False)
        yaml.dump(content, temp_file)
        temp_file.close()
        return temp_file.name
    
    return _create_yaml


@pytest.fixture
def validator():
    """Create validator instance."""
    return CLIValidator()


class TestYAMLConfigLoading:
    """Test YAML configuration loading and validation."""
    
    def test_load_valid_yaml(self, temp_yaml_file):
        """Test loading valid YAML configuration."""
        config = {
            'hypothesis': 'Test hypothesis',
            'scenarios': [
                {'id': 1, 'disease_query': 'breast cancer'}
            ]
        }
        
        yaml_path = temp_yaml_file(config)
        runner = YAMLRunner()
        
        # Should load without errors
        loaded_config = asyncio.run(runner._load_config(yaml_path))
        assert loaded_config is not None
        assert loaded_config['hypothesis'] == 'Test hypothesis'
        assert len(loaded_config['scenarios']) == 1
        
        # Cleanup
        Path(yaml_path).unlink()
    
    def test_load_yaml_missing_scenarios(self, temp_yaml_file):
        """Test loading YAML without scenarios section."""
        config = {
            'hypothesis': 'Test hypothesis'
            # Missing 'scenarios'
        }
        
        yaml_path = temp_yaml_file(config)
        runner = YAMLRunner()
        
        # Should fail validation
        loaded_config = asyncio.run(runner._load_config(yaml_path))
        assert loaded_config is None
        
        # Cleanup
        Path(yaml_path).unlink()
    
    def test_load_yaml_empty_scenarios(self, temp_yaml_file):
        """Test loading YAML with empty scenarios."""
        config = {
            'hypothesis': 'Test hypothesis',
            'scenarios': []
        }
        
        yaml_path = temp_yaml_file(config)
        runner = YAMLRunner()
        
        # Should fail validation
        loaded_config = asyncio.run(runner._load_config(yaml_path))
        assert loaded_config is None
        
        # Cleanup
        Path(yaml_path).unlink()
    
    def test_load_yaml_scenario_missing_id(self, temp_yaml_file):
        """Test loading YAML with scenario missing ID."""
        config = {
            'scenarios': [
                {'disease_query': 'breast cancer'}  # Missing 'id'
            ]
        }
        
        yaml_path = temp_yaml_file(config)
        runner = YAMLRunner()
        
        # Should fail validation
        loaded_config = asyncio.run(runner._load_config(yaml_path))
        assert loaded_config is None
        
        # Cleanup
        Path(yaml_path).unlink()
    
    def test_load_yaml_invalid_scenario_id(self, temp_yaml_file):
        """Test loading YAML with invalid scenario ID."""
        config = {
            'scenarios': [
                {'id': 10, 'disease_query': 'breast cancer'}  # Invalid ID
            ]
        }
        
        yaml_path = temp_yaml_file(config)
        runner = YAMLRunner()
        
        # Should fail validation
        loaded_config = asyncio.run(runner._load_config(yaml_path))
        assert loaded_config is None
        
        # Cleanup
        Path(yaml_path).unlink()


class TestYAMLParameterValidation:
    """Test scenario parameter validation."""
    
    def test_validate_scenario_1_params(self, temp_yaml_file):
        """Test Scenario 1 parameter validation."""
        config = {
            'scenarios': [
                {
                    'id': 1,
                    'disease_query': 'breast cancer',
                    'tissue_context': 'breast'
                }
            ]
        }
        
        yaml_path = temp_yaml_file(config)
        runner = YAMLRunner()
        
        # Should pass validation
        loaded_config = asyncio.run(runner._load_config(yaml_path))
        assert loaded_config is not None
        
        # Cleanup
        Path(yaml_path).unlink()
    
    def test_validate_scenario_1_missing_required(self, temp_yaml_file):
        """Test Scenario 1 with missing required parameter."""
        config = {
            'scenarios': [
                {
                    'id': 1,
                    'tissue_context': 'breast'  # Missing disease_query
                }
            ]
        }
        
        yaml_path = temp_yaml_file(config)
        runner = YAMLRunner()
        
        # Should fail validation
        loaded_config = asyncio.run(runner._load_config(yaml_path))
        assert loaded_config is None
        
        # Cleanup
        Path(yaml_path).unlink()
    
    def test_validate_scenario_4_params(self, temp_yaml_file):
        """Test Scenario 4 parameter validation."""
        config = {
            'scenarios': [
                {
                    'id': 4,
                    'targets': ['TP53', 'BRCA1', 'BRCA2'],
                    'disease_context': 'breast cancer',
                    'simulation_mode': 'mra'
                }
            ]
        }
        
        yaml_path = temp_yaml_file(config)
        runner = YAMLRunner()
        
        # Should pass validation
        loaded_config = asyncio.run(runner._load_config(yaml_path))
        assert loaded_config is not None
        
        # Cleanup
        Path(yaml_path).unlink()
    
    def test_validate_scenario_4_invalid_simulation_mode(self, temp_yaml_file):
        """Test Scenario 4 with invalid simulation mode."""
        config = {
            'scenarios': [
                {
                    'id': 4,
                    'targets': ['TP53', 'BRCA1'],
                    'disease_context': 'breast cancer',
                    'simulation_mode': 'invalid'
                }
            ]
        }
        
        yaml_path = temp_yaml_file(config)
        runner = YAMLRunner()
        
        # Should fail validation
        loaded_config = asyncio.run(runner._load_config(yaml_path))
        assert loaded_config is None
        
        # Cleanup
        Path(yaml_path).unlink()


class TestYAMLGlobalParameters:
    """Test global parameter inheritance."""
    
    def test_global_params_inheritance(self, temp_yaml_file):
        """Test that global parameters are inherited by scenarios."""
        config = {
            'global_params': {
                'tissue_context': 'breast',
                'simulation_mode': 'simple'
            },
            'scenarios': [
                {
                    'id': 1,
                    'disease_query': 'breast cancer'
                    # Should inherit tissue_context: breast
                }
            ]
        }
        
        yaml_path = temp_yaml_file(config)
        runner = YAMLRunner()
        
        # Should pass validation with inherited params
        loaded_config = asyncio.run(runner._load_config(yaml_path))
        assert loaded_config is not None
        assert loaded_config['global_params']['tissue_context'] == 'breast'
        
        # Cleanup
        Path(yaml_path).unlink()
    
    def test_scenario_params_override_global(self, temp_yaml_file):
        """Test that scenario parameters override global ones."""
        config = {
            'global_params': {
                'simulation_mode': 'simple'
            },
            'scenarios': [
                {
                    'id': 4,
                    'targets': ['TP53', 'BRCA1'],
                    'disease_context': 'breast cancer',
                    'simulation_mode': 'mra'  # Override global
                }
            ]
        }
        
        yaml_path = temp_yaml_file(config)
        runner = YAMLRunner()
        
        # Should pass validation
        loaded_config = asyncio.run(runner._load_config(yaml_path))
        assert loaded_config is not None
        
        # Cleanup
        Path(yaml_path).unlink()


class TestYAMLMultipleScenarios:
    """Test configurations with multiple scenarios."""
    
    def test_multiple_scenarios_valid(self, temp_yaml_file):
        """Test configuration with multiple valid scenarios."""
        config = {
            'hypothesis': 'Multi-scenario test',
            'scenarios': [
                {'id': 1, 'disease_query': 'breast cancer'},
                {'id': 2, 'target_query': 'TP53'},
                {'id': 3, 'cancer_type': 'breast cancer', 'tissue_context': 'breast'}
            ]
        }
        
        yaml_path = temp_yaml_file(config)
        runner = YAMLRunner()
        
        # Should pass validation
        loaded_config = asyncio.run(runner._load_config(yaml_path))
        assert loaded_config is not None
        assert len(loaded_config['scenarios']) == 3
        
        # Cleanup
        Path(yaml_path).unlink()
    
    def test_multiple_scenarios_one_invalid(self, temp_yaml_file):
        """Test configuration with one invalid scenario."""
        config = {
            'scenarios': [
                {'id': 1, 'disease_query': 'breast cancer'},  # Valid
                {'id': 2},  # Invalid - missing target_query
                {'id': 3, 'cancer_type': 'breast cancer', 'tissue_context': 'breast'}  # Valid
            ]
        }
        
        yaml_path = temp_yaml_file(config)
        runner = YAMLRunner()
        
        # Should fail validation due to invalid scenario
        loaded_config = asyncio.run(runner._load_config(yaml_path))
        assert loaded_config is None
        
        # Cleanup
        Path(yaml_path).unlink()


class TestYAMLExamples:
    """Test that example YAML files are valid."""
    
    def test_axl_breast_cancer_yaml(self):
        """Test axl_breast_cancer.yaml example."""
        yaml_path = "examples/yaml_configs/axl_breast_cancer.yaml"
        
        if not Path(yaml_path).exists():
            pytest.skip(f"Example file not found: {yaml_path}")
        
        runner = YAMLRunner()
        loaded_config = asyncio.run(runner._load_config(yaml_path))
        
        assert loaded_config is not None
        assert 'hypothesis' in loaded_config
        assert len(loaded_config['scenarios']) > 0
    
    def test_disease_analysis_yaml(self):
        """Test disease_analysis.yaml example."""
        yaml_path = "examples/yaml_configs/disease_analysis.yaml"
        
        if not Path(yaml_path).exists():
            pytest.skip(f"Example file not found: {yaml_path}")
        
        runner = YAMLRunner()
        loaded_config = asyncio.run(runner._load_config(yaml_path))
        
        assert loaded_config is not None
        assert len(loaded_config['scenarios']) > 0
    
    def test_target_analysis_yaml(self):
        """Test target_analysis.yaml example."""
        yaml_path = "examples/yaml_configs/target_analysis.yaml"
        
        if not Path(yaml_path).exists():
            pytest.skip(f"Example file not found: {yaml_path}")
        
        runner = YAMLRunner()
        loaded_config = asyncio.run(runner._load_config(yaml_path))
        
        assert loaded_config is not None
        assert loaded_config['scenarios'][0]['id'] == 2
    
    def test_batch_analysis_yaml(self):
        """Test batch_analysis.yaml example."""
        yaml_path = "examples/yaml_configs/batch_analysis.yaml"
        
        if not Path(yaml_path).exists():
            pytest.skip(f"Example file not found: {yaml_path}")
        
        runner = YAMLRunner()
        loaded_config = asyncio.run(runner._load_config(yaml_path))
        
        assert loaded_config is not None
        assert len(loaded_config['scenarios']) > 1  # Multiple scenarios
    
    def test_drug_repurposing_yaml(self):
        """Test drug_repurposing.yaml example."""
        yaml_path = "examples/yaml_configs/drug_repurposing.yaml"
        
        if not Path(yaml_path).exists():
            pytest.skip(f"Example file not found: {yaml_path}")
        
        runner = YAMLRunner()
        loaded_config = asyncio.run(runner._load_config(yaml_path))
        
        assert loaded_config is not None
        assert any(s['id'] == 6 for s in loaded_config['scenarios'])

