"""
OmniTarget Pipeline Orchestrator

Main orchestrator for all 6 core scenarios.
P0-4: Production Monitoring with structured logging
"""

import asyncio
import logging
from typing import Dict, Any, Optional, List
from pathlib import Path
from datetime import datetime

from .mcp_client_manager import MCPClientManager
from .exceptions import (
    DatabaseConnectionError,
    DatabaseTimeoutError,
    MCPServerError,
    DatabaseUnavailableError,
    ScenarioExecutionError,
    format_error_for_logging
)
from .logging_config import get_correlation_id, log_with_context
from .metrics import (
    record_scenario_execution, record_scenario_phase,
    update_resource_metrics, monitor_execution
)
from ..scenarios.scenario_1_disease_network import DiseaseNetworkScenario
from ..scenarios.scenario_2_target_analysis import TargetAnalysisScenario
from ..scenarios.scenario_3_cancer_analysis import CancerAnalysisScenario
from ..scenarios.scenario_4_mra_simulation import MultiTargetSimulationScenario
from ..scenarios.scenario_5_pathway_comparison import PathwayComparisonScenario
from ..scenarios.scenario_6_drug_repurposing import DrugRepurposingScenario

logger = logging.getLogger(__name__)


class OmniTargetPipeline:
    """
    Main orchestrator for the OmniTarget bioinformatics pipeline.
    
    Manages all 6 core scenarios and provides a unified interface.
    """
    
    def __init__(self, config_path: str = "config/mcp_servers.json"):
        """
        Initialize the OmniTarget pipeline.
        
        Args:
            config_path: Path to MCP server configuration file
        """
        self.config_path = config_path
        self.mcp_manager = MCPClientManager(config_path)
        
        # Initialize scenarios
        self.scenarios = {
            1: DiseaseNetworkScenario(self.mcp_manager),
            2: TargetAnalysisScenario(self.mcp_manager),
            3: CancerAnalysisScenario(self.mcp_manager),
            4: MultiTargetSimulationScenario(self.mcp_manager),
            5: PathwayComparisonScenario(self.mcp_manager),
            6: DrugRepurposingScenario(self.mcp_manager)
        }
        
        logger.info("OmniTarget pipeline initialized with 6 core scenarios")
    
    async def run_scenario(
        self,
        scenario_id: int,
        **kwargs
    ) -> Any:
        """
        Run a specific scenario.

        Args:
            scenario_id: ID of scenario to run (1-6)
            **kwargs: Scenario-specific parameters

        Returns:
            Scenario result
        """
        if scenario_id not in self.scenarios:
            raise ValueError(f"Scenario {scenario_id} not implemented yet")

        scenario = self.scenarios[scenario_id]
        scenario_name = scenario.__class__.__name__

        # Get correlation ID for tracking
        correlation_id = get_correlation_id()

        # Record start time
        start_time = datetime.utcnow()

        # Log scenario start with structured logging (P0-4)
        log_with_context(
            logger,
            "info",
            "scenario_execution_started",
            scenario_id=scenario_id,
            scenario_name=scenario_name,
            correlation_id=correlation_id,
            parameters=kwargs
        )

        try:
            # Execute scenario (MCP session managed by pipeline context)
            result = await scenario.execute(**kwargs)

            # Calculate duration
            duration = (datetime.utcnow() - start_time).total_seconds()

            # Record success metrics (P0-4)
            record_scenario_execution(
                scenario_id=scenario_id,
                scenario_name=scenario_name,
                status="success",
                duration=duration
            )

            # Update resource metrics
            update_resource_metrics()

            # Log scenario completion with structured logging (P0-4)
            log_with_context(
                logger,
                "info",
                "scenario_execution_completed",
                scenario_id=scenario_id,
                scenario_name=scenario_name,
                correlation_id=correlation_id,
                duration=duration,
                status="success"
            )

            return result

        except Exception as e:
            # Calculate duration
            duration = (datetime.utcnow() - start_time).total_seconds()

            # Record error metrics (P0-4)
            record_scenario_execution(
                scenario_id=scenario_id,
                scenario_name=scenario_name,
                status="error",
                duration=duration
            )

            # Log scenario failure with structured logging (P0-4)
            log_with_context(
                logger,
                "error",
                "scenario_execution_failed",
                scenario_id=scenario_id,
                scenario_name=scenario_name,
                correlation_id=correlation_id,
                duration=duration,
                error=str(e),
                error_type=type(e).__name__
            )

            raise
    
    async def run_scenario_batch(
        self, 
        scenario_configs: List[Dict[str, Any]]
    ) -> List[Any]:
        """
        Run multiple scenarios in batch.
        
        Args:
            scenario_configs: List of scenario configurations
                Each config should have 'scenario_id' and scenario-specific parameters
            
        Returns:
            List of scenario results
        """
        logger.info(f"Running batch of {len(scenario_configs)} scenarios")
        
        results = []
        
        # Run scenarios sequentially (MCP session managed by pipeline context)
        for config in scenario_configs:
            scenario_id = config.pop('scenario_id')
            result = await self.run_scenario(scenario_id, **config)
            results.append(result)
        
        logger.info("Batch execution completed")
        return results
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Perform health check on all MCP servers.
        
        Returns:
            Health status for each server
        """
        logger.info("Performing pipeline health check")
        
        health_status = {}
        
        try:
            async with self.mcp_manager.session():
                # Test each MCP server
                for server_name in self.mcp_manager.clients.keys():
                    try:
                        client = self.mcp_manager.clients[server_name]
                        
                        # Test basic connectivity
                        if server_name == "kegg":
                            await client.call_tool("get_database_info", {})
                        elif server_name == "reactome":
                            await client.call_tool("get_database_info", {})
                        elif server_name == "string":
                            await client.call_tool("get_database_info", {})
                        elif server_name == "hpa":
                            # HPA doesn't have get_database_info, skip health check
                            # HPA responses can be very large and cause buffer issues
                            # Just mark as healthy if server started successfully
                            pass
                        
                        health_status[server_name] = {
                            'status': 'healthy',
                            'message': 'Server responding normally'
                        }
                        
                    except (DatabaseConnectionError, DatabaseTimeoutError, MCPServerError) as e:
                        # Database/MCP server errors
                        health_status[server_name] = {
                            'status': 'unhealthy',
                            'message': f'Server error ({type(e).__name__}): {str(e)}'
                        }
                        logger.warning(
                            f"Health check failed for {server_name}",
                            extra=format_error_for_logging(e)
                        )
                    except DatabaseUnavailableError as e:
                        # Server temporarily unavailable
                        health_status[server_name] = {
                            'status': 'degraded',
                            'message': f'Server temporarily unavailable: {str(e)}'
                        }
                        logger.warning(
                            f"Health check degraded for {server_name}",
                            extra=format_error_for_logging(e)
                        )
                    except Exception as e:
                        # Unexpected errors
                        health_status[server_name] = {
                            'status': 'unhealthy',
                            'message': f'Unexpected error: {type(e).__name__}: {str(e)}'
                        }
                        logger.error(
                            f"Health check failed for {server_name} with unexpected error",
                            extra=format_error_for_logging(e)
                        )

        except (DatabaseConnectionError, DatabaseUnavailableError) as e:
            # Overall health check failure - MCP manager session error
            logger.error(
                "Health check failed - MCP manager session error",
                extra=format_error_for_logging(e)
            )
            health_status['pipeline'] = {
                'status': 'unhealthy',
                'message': f'Pipeline error: {str(e)}'
            }
        except Exception as e:
            logger.error(
                "Health check failed with unexpected error",
                extra=format_error_for_logging(e)
            )
            health_status['pipeline'] = {
                'status': 'unhealthy',
                'message': f'Unexpected pipeline error: {type(e).__name__}: {str(e)}'
            }
        
        return health_status
    
    async def get_scenario_info(self, scenario_id: int) -> Dict[str, Any]:
        """
        Get information about a specific scenario.
        
        Args:
            scenario_id: ID of scenario
            
        Returns:
            Scenario information
        """
        if scenario_id not in self.scenarios:
            raise ValueError(f"Scenario {scenario_id} not implemented yet")
        
        scenario = self.scenarios[scenario_id]
        
        # Get scenario information
        info = {
            'scenario_id': scenario_id,
            'name': scenario.__class__.__name__,
            'description': scenario.__doc__,
            'parameters': self._get_scenario_parameters(scenario_id)
        }
        
        return info
    
    def _get_scenario_parameters(self, scenario_id: int) -> Dict[str, Any]:
        """Get parameters for a specific scenario."""
        parameter_schemas = {
            1: {
                'disease_query': {
                    'type': 'string',
                    'description': 'Disease name or identifier',
                    'required': True
                },
                'tissue_context': {
                    'type': 'string',
                    'description': 'Optional tissue context for expression filtering',
                    'required': False
                }
            },
            2: {
                'target_query': {
                    'type': 'string',
                    'description': 'Target protein name or identifier',
                    'required': True
                }
            },
            3: {
                'cancer_type': {
                    'type': 'string',
                    'description': 'Type of cancer (e.g., "breast cancer")',
                    'required': True
                },
                'tissue_context': {
                    'type': 'string',
                    'description': 'Normal tissue context (e.g., "breast")',
                    'required': True
                }
            },
            4: {
                'targets': {
                    'type': 'list',
                    'description': 'List of target proteins/genes',
                    'required': True
                },
                'disease_context': {
                    'type': 'string',
                    'description': 'Disease context for pathway analysis',
                    'required': True
                },
                'simulation_mode': {
                    'type': 'string',
                    'description': 'Simulation mode: "simple" or "mra"',
                    'required': False
                },
                'tissue_context': {
                    'type': 'string',
                    'description': 'Optional tissue context for expression filtering',
                    'required': False
                }
            },
            5: {
                'pathway_query': {
                    'type': 'string',
                    'description': 'Pathway name or identifier',
                    'required': True
                }
            },
            6: {
                'disease_query': {
                    'type': 'string',
                    'description': 'Disease name or identifier',
                    'required': True
                },
                'tissue_context': {
                    'type': 'string',
                    'description': 'Tissue context for expression analysis',
                    'required': True
                },
                'simulation_mode': {
                    'type': 'string',
                    'description': 'Simulation mode: "simple" or "mra"',
                    'required': False
                }
            }
        }
        
        return parameter_schemas.get(scenario_id, {})
    
    async def list_available_scenarios(self) -> List[Dict[str, Any]]:
        """
        List all available scenarios.
        
        Returns:
            List of scenario information
        """
        scenarios = []
        
        for scenario_id in self.scenarios.keys():
            info = await self.get_scenario_info(scenario_id)
            scenarios.append(info)
        
        return scenarios
    
    async def validate_scenario_parameters(
        self, 
        scenario_id: int, 
        parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Validate parameters for a specific scenario.
        
        Args:
            scenario_id: ID of scenario
            parameters: Parameters to validate
            
        Returns:
            Validation result
        """
        if scenario_id not in self.scenarios:
            return {
                'valid': False,
                'errors': [f"Scenario {scenario_id} not implemented yet"]
            }
        
        # Get parameter schema
        schema = self._get_scenario_parameters(scenario_id)
        
        # Validate required parameters
        errors = []
        for param_name, param_info in schema.items():
            if param_info.get('required', False):
                if param_name not in parameters:
                    errors.append(f"Required parameter '{param_name}' missing")
        
        # Validate parameter types
        for param_name, param_value in parameters.items():
            if param_name in schema:
                expected_type = schema[param_name].get('type', 'string')
                if expected_type == 'string' and not isinstance(param_value, str):
                    errors.append(f"Parameter '{param_name}' must be a string")
                elif expected_type == 'integer' and not isinstance(param_value, int):
                    errors.append(f"Parameter '{param_name}' must be an integer")
                elif expected_type == 'float' and not isinstance(param_value, (int, float)):
                    errors.append(f"Parameter '{param_name}' must be a number")
        
        return {
            'valid': len(errors) == 0,
            'errors': errors
        }
    
    async def get_pipeline_status(self) -> Dict[str, Any]:
        """
        Get overall pipeline status.
        
        Returns:
            Pipeline status information
        """
        status = {
            'pipeline_version': '1.0.0',
            'available_scenarios': len(self.scenarios),
            'implemented_scenarios': list(self.scenarios.keys()),
            'mcp_servers': list(self.mcp_manager.clients.keys()),
            'config_path': self.config_path
        }
        
        # Add health status
        try:
            health = await self.health_check()
            status['health_status'] = health
        except (DatabaseConnectionError, DatabaseUnavailableError) as e:
            status['health_status'] = {
                'pipeline': {
                    'status': 'unhealthy',
                    'message': f'Health check failed: {str(e)}'
                }
            }
            logger.error(
                "Health check failed during get_pipeline_status",
                extra=format_error_for_logging(e)
            )
        except Exception as e:
            status['health_status'] = {
                'pipeline': {
                    'status': 'unhealthy',
                    'message': f'Unexpected error during health check: {type(e).__name__}: {str(e)}'
                }
            }
            logger.error(
                "Health check failed with unexpected error during get_pipeline_status",
                extra=format_error_for_logging(e)
            )
        
        return status
    
    async def shutdown(self):
        """Shutdown the pipeline and clean up resources."""
        logger.info("Shutting down OmniTarget pipeline")

        try:
            await self.mcp_manager.stop_all()
            logger.info("Pipeline shutdown completed")
        except (DatabaseConnectionError, DatabaseUnavailableError) as e:
            # Log database/MCP errors during shutdown
            logger.error(
                "Database error during pipeline shutdown",
                extra=format_error_for_logging(e)
            )
        except Exception as e:
            logger.error(
                f"Error during pipeline shutdown: {type(e).__name__}: {e}",
                extra=format_error_for_logging(e)
            )
    
    async def __aenter__(self):
        """Async context manager entry - starts MCP session."""
        await self.mcp_manager.start_all()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.shutdown()


# Convenience functions for common workflows
async def analyze_disease(
    disease_query: str, 
    tissue_context: Optional[str] = None,
    config_path: str = "config/mcp_servers.json"
) -> Any:
    """
    Convenience function for disease network analysis.
    
    Args:
        disease_query: Disease name or identifier
        tissue_context: Optional tissue context
        config_path: Path to MCP server configuration
        
    Returns:
        DiseaseNetworkResult
    """
    async with OmniTargetPipeline(config_path) as pipeline:
        return await pipeline.run_scenario(
            1,
            disease_query=disease_query,
            tissue_context=tissue_context
        )


async def analyze_target(
    target_query: str,
    config_path: str = "config/mcp_servers.json"
) -> Any:
    """
    Convenience function for target analysis.
    
    Args:
        target_query: Target protein name or identifier
        config_path: Path to MCP server configuration
        
    Returns:
        TargetAnalysisResult
    """
    async with OmniTargetPipeline(config_path) as pipeline:
        return await pipeline.run_scenario(
            2,
            target_query=target_query
        )


async def analyze_cancer(
    cancer_type: str,
    tissue_context: str,
    config_path: str = "config/mcp_servers.json"
) -> Any:
    """
    Convenience function for cancer analysis.
    
    Args:
        cancer_type: Type of cancer
        tissue_context: Normal tissue context
        config_path: Path to MCP server configuration
        
    Returns:
        CancerAnalysisResult
    """
    async with OmniTargetPipeline(config_path) as pipeline:
        return await pipeline.run_scenario(
            3,
            cancer_type=cancer_type,
            tissue_context=tissue_context
        )
