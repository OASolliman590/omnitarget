"""
CLI Parameter Validators

Validation functions for interactive and YAML modes.
Integrates with existing DataValidator for consistency.
"""

import logging
from typing import Dict, List, Any, Optional, Tuple
import re

logger = logging.getLogger(__name__)


class CLIValidator:
    """Parameter validation for CLI modes."""
    
    # Common tissue contexts
    VALID_TISSUES = [
        'breast', 'lung', 'liver', 'kidney', 'brain', 'heart', 'pancreas',
        'prostate', 'ovary', 'testis', 'colon', 'stomach', 'skin', 'blood',
        'muscle', 'bone', 'thyroid', 'adrenal', 'spleen', 'lymph node'
    ]
    
    # Common simulation modes
    VALID_SIMULATION_MODES = ['simple', 'mra']
    
    # Perturbation types
    VALID_PERTURBATION_TYPES = ['inhibit', 'activate', 'knockdown', 'overexpress']
    
    def __init__(self):
        """Initialize CLI validator."""
        self.gene_pattern = re.compile(r'^[A-Z0-9]+$')
    
    def validate_disease_query(self, query: str) -> Tuple[bool, Optional[str]]:
        """
        Validate disease query.
        
        Args:
            query: Disease name to validate
            
        Returns:
            (is_valid, error_message)
        """
        if not query or len(query.strip()) == 0:
            return False, "Disease query cannot be empty"
        
        if len(query) < 3:
            return False, "Disease query too short (minimum 3 characters)"
        
        if len(query) > 100:
            return False, "Disease query too long (maximum 100 characters)"
        
        return True, None
    
    def validate_tissue_context(self, tissue: str) -> Tuple[bool, Optional[str], Optional[List[str]]]:
        """
        Validate tissue context and suggest alternatives.
        
        Args:
            tissue: Tissue context to validate
            
        Returns:
            (is_valid, error_message, suggestions)
        """
        if not tissue:
            return True, None, None  # Optional parameter
        
        tissue_lower = tissue.lower().strip()
        
        # Check exact match
        if tissue_lower in self.VALID_TISSUES:
            return True, None, None
        
        # Find similar tissues (fuzzy match)
        suggestions = [t for t in self.VALID_TISSUES if tissue_lower in t or t in tissue_lower]
        
        if suggestions:
            return False, f"Tissue '{tissue}' not recognized", suggestions
        
        # Still allow custom tissues but warn
        logger.warning(f"Using custom tissue context: {tissue}")
        return True, None, None
    
    def validate_target_gene(self, gene: str) -> Tuple[bool, Optional[str]]:
        """
        Validate a single gene/target symbol.
        
        Args:
            gene: Gene symbol to validate
            
        Returns:
            (is_valid, error_message)
        """
        if not gene or len(gene.strip()) == 0:
            return False, "Gene symbol cannot be empty"
        
        gene = gene.strip().upper()
        
        # Basic format validation
        if not self.gene_pattern.match(gene):
            return False, f"Invalid gene symbol format: {gene} (should be alphanumeric, e.g., TP53, BRCA1)"
        
        if len(gene) < 2:
            return False, "Gene symbol too short (minimum 2 characters)"
        
        if len(gene) > 15:
            return False, "Gene symbol too long (maximum 15 characters)"
        
        return True, None
    
    def validate_targets(self, targets: List[str]) -> Tuple[bool, Optional[str], List[str]]:
        """
        Validate list of target genes.
        
        Args:
            targets: List of gene symbols
            
        Returns:
            (is_valid, error_message, valid_targets)
        """
        if not targets or len(targets) == 0:
            return False, "Target list cannot be empty", []
        
        valid_targets = []
        invalid_targets = []
        
        for target in targets:
            is_valid, error = self.validate_target_gene(target)
            if is_valid:
                valid_targets.append(target.strip().upper())
            else:
                invalid_targets.append(target)
        
        if invalid_targets:
            return False, f"Invalid targets: {', '.join(invalid_targets)}", valid_targets
        
        # Check for duplicates
        if len(valid_targets) != len(set(valid_targets)):
            return False, "Duplicate targets found", valid_targets
        
        return True, None, valid_targets
    
    def validate_simulation_mode(self, mode: str) -> Tuple[bool, Optional[str]]:
        """
        Validate simulation mode.
        
        Args:
            mode: Simulation mode ('simple' or 'mra')
            
        Returns:
            (is_valid, error_message)
        """
        if not mode:
            return False, "Simulation mode cannot be empty"
        
        mode_lower = mode.lower().strip()
        
        if mode_lower not in self.VALID_SIMULATION_MODES:
            return False, f"Invalid simulation mode: {mode}. Must be one of: {', '.join(self.VALID_SIMULATION_MODES)}"
        
        return True, None
    
    def validate_perturbation_type(self, perturbation_type: str) -> Tuple[bool, Optional[str]]:
        """
        Validate perturbation type.
        
        Args:
            perturbation_type: Type of perturbation
            
        Returns:
            (is_valid, error_message)
        """
        if not perturbation_type:
            return True, None  # Optional parameter
        
        perturb_lower = perturbation_type.lower().strip()
        
        if perturb_lower not in self.VALID_PERTURBATION_TYPES:
            return False, f"Invalid perturbation type: {perturbation_type}. Must be one of: {', '.join(self.VALID_PERTURBATION_TYPES)}"
        
        return True, None
    
    def validate_perturbation_strength(self, strength: float) -> Tuple[bool, Optional[str]]:
        """
        Validate perturbation strength.
        
        Args:
            strength: Perturbation strength (0.0-1.0)
            
        Returns:
            (is_valid, error_message)
        """
        if strength is None:
            return True, None  # Optional parameter
        
        try:
            strength_float = float(strength)
        except (ValueError, TypeError):
            return False, f"Perturbation strength must be a number, got: {strength}"
        
        if strength_float < 0.0 or strength_float > 1.0:
            return False, f"Perturbation strength must be between 0.0 and 1.0, got: {strength_float}"
        
        return True, None
    
    def validate_scenario_id(self, scenario_id: int) -> Tuple[bool, Optional[str]]:
        """
        Validate scenario ID.
        
        Args:
            scenario_id: Scenario ID (1-6)
            
        Returns:
            (is_valid, error_message)
        """
        try:
            sid = int(scenario_id)
        except (ValueError, TypeError):
            return False, f"Scenario ID must be an integer, got: {scenario_id}"
        
        if sid < 1 or sid > 6:
            return False, f"Scenario ID must be between 1 and 6, got: {sid}"
        
        return True, None
    
    def validate_scenario_parameters(self, scenario_id: int, params: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Validate parameters for a specific scenario.
        
        Args:
            scenario_id: Scenario ID (1-6)
            params: Parameters dictionary
            
        Returns:
            (is_valid, list_of_errors)
        """
        errors = []
        
        # Scenario-specific required parameters
        required_params = {
            1: ['disease_query'],
            2: ['target_query'],
            3: ['cancer_type', 'tissue_context'],
            4: ['targets', 'disease_context'],
            5: ['pathway_query'],
            6: ['disease_query', 'tissue_context']
        }
        
        # Check required parameters
        if scenario_id in required_params:
            for param in required_params[scenario_id]:
                if param not in params or params[param] is None:
                    errors.append(f"Missing required parameter: {param}")
        
        # Validate individual parameters
        if 'disease_query' in params:
            is_valid, error = self.validate_disease_query(params['disease_query'])
            if not is_valid:
                errors.append(error)
        
        if 'tissue_context' in params and params['tissue_context']:
            is_valid, error, _ = self.validate_tissue_context(params['tissue_context'])
            if not is_valid:
                errors.append(error)
        
        if 'target_query' in params:
            is_valid, error = self.validate_target_gene(params['target_query'])
            if not is_valid:
                errors.append(error)
        
        if 'targets' in params:
            is_valid, error, _ = self.validate_targets(params['targets'])
            if not is_valid:
                errors.append(error)
        
        if 'simulation_mode' in params and params['simulation_mode']:
            is_valid, error = self.validate_simulation_mode(params['simulation_mode'])
            if not is_valid:
                errors.append(error)
        
        if 'perturbation_strength' in params and params['perturbation_strength'] is not None:
            is_valid, error = self.validate_perturbation_strength(params['perturbation_strength'])
            if not is_valid:
                errors.append(error)

        if 'network_expansion' in params and params['network_expansion'] is not None:
            is_valid, validation_errors = self.validate_network_expansion(params['network_expansion'])
            if not is_valid:
                errors.extend(validation_errors)

        return len(errors) == 0, errors

    def validate_network_expansion(self, config: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Validate network expansion configuration.

        Args:
            config: Network expansion configuration dictionary

        Returns:
            (is_valid, list_of_errors)
        """
        errors = []

        # Validate initial_neighbors
        if 'initial_neighbors' in config:
            try:
                value = int(config['initial_neighbors'])
                if value < 1 or value > 10:
                    errors.append("initial_neighbors must be between 1 and 10")
            except (ValueError, TypeError):
                errors.append(f"initial_neighbors must be an integer, got: {config['initial_neighbors']}")

        # Validate expansion_neighbors
        if 'expansion_neighbors' in config:
            try:
                value = int(config['expansion_neighbors'])
                if value < 0 or value > 5:
                    errors.append("expansion_neighbors must be between 0 and 5")
            except (ValueError, TypeError):
                errors.append(f"expansion_neighbors must be an integer, got: {config['expansion_neighbors']}")

        # Validate max_network_size
        if 'max_network_size' in config:
            try:
                value = int(config['max_network_size'])
                if value < 20 or value > 500:
                    errors.append("max_network_size must be between 20 and 500")
            except (ValueError, TypeError):
                errors.append(f"max_network_size must be an integer, got: {config['max_network_size']}")

        # Validate step_timeouts
        if 'step_timeouts' in config:
            if not isinstance(config['step_timeouts'], dict):
                errors.append("step_timeouts must be a dictionary")
            else:
                for step, timeout in config['step_timeouts'].items():
                    try:
                        step_int = int(step)
                        if step_int < 1 or step_int > 8:
                            errors.append(f"Invalid step number in step_timeouts: {step} (must be 1-8)")
                        timeout_int = int(timeout)
                        if timeout_int < 10 or timeout_int > 600:
                            errors.append(f"Step {step} timeout must be between 10 and 600 seconds, got: {timeout}")
                    except (ValueError, TypeError):
                        errors.append(f"Invalid step_timeouts entry: step={step}, timeout={timeout}")

        return len(errors) == 0, errors


async def check_mcp_health(config_path: str = "config/mcp_servers.json") -> Tuple[bool, Dict[str, Any]]:
    """
    Check MCP server health.
    
    Args:
        config_path: Path to MCP server configuration
        
    Returns:
        (all_healthy, health_status_dict)
    """
    from ..core.mcp_client_manager import MCPClientManager
    
    try:
        manager = MCPClientManager(config_path)
        async with manager:
            health_status = {}
            
            # Test each server
            for server_name in ['kegg', 'reactome', 'string', 'hpa']:
                try:
                    client = await manager.get_client(server_name)
                    health_status[server_name] = {
                        'status': 'healthy',
                        'message': 'Server responding'
                    }
                except Exception as e:
                    health_status[server_name] = {
                        'status': 'unhealthy',
                        'message': str(e)
                    }
            
            all_healthy = all(s['status'] == 'healthy' for s in health_status.values())
            return all_healthy, health_status
            
    except Exception as e:
        logger.error(f"MCP health check failed: {e}")
        return False, {'error': str(e)}

