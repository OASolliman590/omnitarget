"""
YAML Runner

YAML-based batch execution for OmniTarget pipeline.
Supports multi-scenario workflows with parameter inheritance.
"""

import asyncio
import logging
import json
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime

try:
    import yaml
except ImportError:
    raise ImportError("PyYAML is required for YAML mode. Install with: pip install pyyaml>=6.0")

try:
    import aiofiles
except ImportError:
    raise ImportError("aiofiles is required for async file operations. Install with: pip install aiofiles")

from ..core.pipeline_orchestrator import OmniTargetPipeline
from ..visualization.orchestrator import VisualizationOrchestrator
from .validators import CLIValidator

logger = logging.getLogger(__name__)


class YAMLRunner:
    """YAML-based batch executor for OmniTarget pipeline."""
    
    def __init__(self, config_path: str = "config/mcp_servers.json"):
        """Initialize YAML runner."""
        self.config_path = config_path
        self.validator = CLIValidator()
        self.pipeline = None
        self.results_dir = Path("results")
        self.results_dir.mkdir(exist_ok=True)
        self.last_results_path: Optional[Path] = None
    
    async def run(
        self,
        yaml_path: str,
        visualize: bool = False,
        visualize_options: Optional[Dict[str, Any]] = None,
    ) -> int:
        """
        Run scenarios from YAML configuration.
        
        Args:
            yaml_path: Path to YAML configuration file
            
        Returns:
            Exit code (0 for success, 1 for failure)
        """
        print("\n" + "=" * 80)
        print("🧬 OmniTarget Pipeline - YAML Batch Mode")
        print("=" * 80)
        print()
        
        try:
            # Load and validate configuration
            config = await self._load_config(yaml_path)
            if config is None:
                return 1
            
            # Display configuration summary
            self._display_config_summary(config)
            
            # Execute scenarios
            results = await self._execute_scenarios(config)
            
            # Save results
            results_path = await self._save_results(config, results, yaml_path)
            self.last_results_path = results_path
            
            # Display summary
            self._display_execution_summary(results)

            # Optional visualization run
            if visualize and results_path:
                self._generate_visualizations(
                    results_path,
                    visualize_options or {},
                )
            
            print("\n✅ YAML batch execution completed successfully!")
            return 0
            
        except Exception as e:
            print(f"\n❌ YAML execution failed: {e}")
            logger.exception("YAML runner error")
            return 1
        finally:
            if self.pipeline:
                await self.pipeline.shutdown()
    
    async def _load_config(self, yaml_path: str) -> Optional[Dict[str, Any]]:
        """Load and validate YAML configuration."""
        yaml_file = Path(yaml_path)
        
        if not yaml_file.exists():
            print(f"❌ YAML file not found: {yaml_path}")
            return None
        
        print(f"📄 Loading configuration from: {yaml_path}")
        logger.info(f"📄 Loading YAML configuration from: {yaml_path}")

        try:
            with open(yaml_file, 'r') as f:
                config = yaml.safe_load(f)
            
            # Validate configuration structure
            if not self._validate_config_structure(config):
                return None
            
            # Validate scenario parameters
            if not await self._validate_scenario_configs(config):
                return None
            
            print("✅ Configuration validated successfully")
            print()
            
            return config
            
        except yaml.YAMLError as e:
            print(f"❌ YAML parsing error: {e}")
            return None
        except Exception as e:
            print(f"❌ Configuration loading error: {e}")
            return None
    
    def _validate_config_structure(self, config: Dict[str, Any]) -> bool:
        """Validate YAML configuration structure."""
        if 'scenarios' not in config:
            print("❌ Configuration missing 'scenarios' section")
            return False
        
        if not isinstance(config['scenarios'], list):
            print("❌ 'scenarios' must be a list")
            return False
        
        if len(config['scenarios']) == 0:
            print("❌ At least one scenario must be defined")
            return False
        
        # Validate each scenario has an ID
        for i, scenario in enumerate(config['scenarios']):
            if 'id' not in scenario:
                print(f"❌ Scenario #{i+1} missing 'id' field")
                return False
            
            is_valid, error = self.validator.validate_scenario_id(scenario['id'])
            if not is_valid:
                print(f"❌ Scenario #{i+1}: {error}")
                return False
        
        return True
    
    async def _validate_scenario_configs(self, config: Dict[str, Any]) -> bool:
        """Validate scenario-specific parameters."""
        global_params = config.get('global_params', {})
        
        for i, scenario_config in enumerate(config['scenarios']):
            scenario_id = scenario_config['id']
            
            # Merge global and scenario-specific params
            params = {**global_params, **scenario_config}
            params.pop('id')  # Remove id from params
            params.pop('name', None)  # Remove optional name field
            
            # Validate parameters
            is_valid, errors = self.validator.validate_scenario_parameters(scenario_id, params)
            
            if not is_valid:
                print(f"❌ Scenario #{i+1} (ID: {scenario_id}) validation errors:")
                for error in errors:
                    print(f"  - {error}")
                return False
        
        return True
    
    def _display_config_summary(self, config: Dict[str, Any]):
        """Display configuration summary."""
        print("📋 Configuration Summary")
        print("-" * 80)
        
        if 'hypothesis' in config:
            print(f"Hypothesis: {config['hypothesis']}")
        
        if 'description' in config:
            print(f"Description: {config['description']}")
        
        if 'global_params' in config:
            print(f"\nGlobal Parameters:")
            for key, value in config['global_params'].items():
                print(f"  {key}: {value}")
        
        print(f"\nScenarios to Execute: {len(config['scenarios'])}")
        for i, scenario in enumerate(config['scenarios'], 1):
            name = scenario.get('name', f"Scenario {scenario['id']}")
            print(f"  {i}. {name} (ID: {scenario['id']})")
        
        print("-" * 80)
        print()
    
    async def _execute_scenarios(self, config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Execute all scenarios sequentially."""
        global_params = config.get('global_params', {})
        results = []
        
        print("⚡ Executing Scenarios")
        print("-" * 80)
        print()

        logger.info(f"⚡ Starting execution of {len(config['scenarios'])} scenario(s)")

        # Initialize pipeline with context manager for MCP session
        async with OmniTargetPipeline(self.config_path) as pipeline:
            for i, scenario_config in enumerate(config['scenarios'], 1):
                scenario_id = scenario_config['id']
                scenario_name = scenario_config.get('name', f"Scenario {scenario_id}")
                
                print(f"[{i}/{len(config['scenarios'])}] Executing: {scenario_name}")
                
                # Merge parameters
                params = {**global_params, **scenario_config}
                params.pop('id')
                params.pop('name', None)
                
                try:
                    # Execute scenario
                    result = await pipeline.run_scenario(scenario_id, **params)
                    
                    # Store result with metadata
                    result_data = {
                        'scenario_id': scenario_id,
                        'scenario_name': scenario_name,
                        'status': 'success',
                        'result': result,
                        'execution_time': datetime.now().isoformat()
                    }
                    
                    results.append(result_data)
                    
                    print(f"  ✅ Completed successfully")
                    
                    # Display quick summary
                    if hasattr(result, 'validation_score'):
                        print(f"     Validation Score: {result.validation_score:.3f}")
                    if hasattr(result, 'network_nodes'):
                        print(f"     Network Nodes: {len(result.network_nodes)}")
                    
                    print()
                    
                except Exception as e:
                    print(f"  ❌ Failed: {e}")
                    
                    result_data = {
                        'scenario_id': scenario_id,
                        'scenario_name': scenario_name,
                        'status': 'failed',
                        'error': str(e),
                        'execution_time': datetime.now().isoformat()
                    }
                    
                    results.append(result_data)
                    
                    logger.exception(f"Scenario {scenario_id} execution error")
                    
                    # Ask if we should continue
                    if i < len(config['scenarios']):
                        print("⚠️  Scenario failed. Continuing with next scenario...")
                        print()
        
        print("-" * 80)
        return results
    
    async def _save_results(self, config: Dict[str, Any], results: List[Dict[str, Any]], yaml_path: str) -> Path:
        """
        Save results to file with proper error handling.

        Args:
            config: Configuration dictionary
            results: List of result dictionaries
            yaml_path: Path to the YAML configuration file

        Returns:
            Path to the saved file

        Raises:
            IOError: If file write fails or verification fails
        """
        # Determine output path
        if 'output' in config and 'path' in config['output']:
            output_path = Path(config['output']['path'])
        else:
            # Generate default filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            yaml_name = Path(yaml_path).stem
            output_path = self.results_dir / f"{yaml_name}_{timestamp}.json"

        # Ensure directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Prepare output data
        output_data = {
            'hypothesis': config.get('hypothesis', ''),
            'description': config.get('description', ''),
            'execution_metadata': {
                'yaml_config': yaml_path,
                'timestamp': datetime.now().isoformat(),
                'total_scenarios': len(results),
                'successful_scenarios': sum(1 for r in results if r['status'] == 'success'),
                'failed_scenarios': sum(1 for r in results if r['status'] == 'failed')
            },
            'results': []
        }

        # Process results
        for result_data in results:
            processed_result = {
                'scenario_id': result_data['scenario_id'],
                'scenario_name': result_data['scenario_name'],
                'status': result_data['status'],
                'execution_time': result_data['execution_time']
            }

            if result_data['status'] == 'success':
                # Extract key information from result
                result_obj = result_data['result']

                # DEBUG: Log what we're trying to serialize
                logger.info(f"[DEBUG] Serializing result for scenario {result_data['scenario_id']}: {type(result_obj).__name__}")
                logger.info(f"[DEBUG] Result type: {type(result_obj)}")
                if hasattr(result_obj, '__dict__'):
                    logger.info(f"[DEBUG] Result fields: {list(result_obj.__dict__.keys())}")
                if hasattr(result_obj, 'model_dump'):
                    try:
                        model_data = result_obj.model_dump()
                        logger.info(f"[DEBUG] Result model_dump keys: {list(model_data.keys())}")
                        if 'validation_score' in model_data:
                            logger.info(f"[DEBUG] validation_score present: {model_data['validation_score']}")
                        else:
                            logger.warning(f"[DEBUG] validation_score NOT in model_dump!")
                    except Exception as e:
                        logger.error(f"[DEBUG] model_dump() failed: {e}", exc_info=True)

                try:
                    if hasattr(result_obj, 'model_dump'):
                        # Pydantic v2 model (preferred) - use 'json' mode to avoid re-validation
                        processed_result['data'] = result_obj.model_dump(mode='json')
                        logger.info(f"[DEBUG] Successfully serialized with model_dump(mode='json')")
                    elif hasattr(result_obj, 'dict'):
                        # Pydantic v1 model
                        processed_result['data'] = result_obj.dict()
                        logger.info(f"[DEBUG] Successfully serialized with dict()")
                    elif hasattr(result_obj, '__dict__'):
                        processed_result['data'] = result_obj.__dict__
                        logger.info(f"[DEBUG] Successfully serialized with __dict__")
                    else:
                        processed_result['data'] = str(result_obj)
                        logger.info(f"[DEBUG] Successfully serialized with str()")
                except Exception as e:
                    # Enhanced error logging
                    logger.error(f"Failed to serialize result for scenario {result_data['scenario_id']}: {e}", exc_info=True)
                    if hasattr(result_obj, '__dict__'):
                        processed_result['data'] = result_obj.__dict__
                    else:
                        processed_result['data'] = str(result_obj)
            else:
                processed_result['error'] = result_data.get('error', 'Unknown error')

            output_data['results'].append(processed_result)

        try:
            # DEBUG: Log what we're about to write
            logger.info(f"[DEBUG] About to write results to {output_path}")
            logger.info(f"[DEBUG] Total scenarios in output_data: {len(output_data.get('results', []))}")
            for i, r in enumerate(output_data.get('results', [])):
                logger.info(f"[DEBUG] Result {i}: scenario_id={r.get('scenario_id')}, status={r.get('status')}")
                if 'data' in r:
                    if isinstance(r['data'], dict):
                        logger.info(f"[DEBUG] Result {i} has 'data' dict with keys: {list(r['data'].keys())}")
                        if 'validation_score' in r['data']:
                            logger.info(f"[DEBUG] Result {i} validation_score: {r['data']['validation_score']}")
                    else:
                        logger.info(f"[DEBUG] Result {i} data is not a dict: {type(r['data'])}")

            # Write file asynchronously
            async with aiofiles.open(output_path, 'w') as f:
                json_content = json.dumps(output_data, indent=2, default=str)
                logger.info(f"[DEBUG] JSON content length: {len(json_content)} bytes")
                await f.write(json_content)

            # CRITICAL: Verify file was written
            if not output_path.exists():
                raise IOError(f"File write verification failed: File does not exist at {output_path}")

            if output_path.stat().st_size == 0:
                raise IOError(f"File write verification failed: File is empty at {output_path}")

            file_size = output_path.stat().st_size
            logger.info(f"✅ Results saved to {output_path} (size: {file_size} bytes)")

            # Read back and verify content
            async with aiofiles.open(output_path, 'r') as f:
                content = await f.read(1000)  # Read first 1000 chars
                logger.info(f"[DEBUG] File content preview (first 1000 chars): {content[:1000]}")

            print(f"\n💾 Results saved to: {output_path}")
            return output_path

        except (IOError, OSError) as e:
            logger.error(f"❌ File I/O error: {e}")
            print(f"\n❌ Failed to save results: {e}")
            raise

        except (TypeError, ValueError) as e:
            # Handle JSON serialization errors (e.g., from Pydantic model_dump issues)
            logger.error(f"❌ Serialization error: {e}")
            print(f"\n❌ Failed to serialize results: {e}")
            # Create a minimal result file with just metadata
            try:
                minimal_output = {
                    'hypothesis': output_data.get('hypothesis', ''),
                    'description': output_data.get('description', ''),
                    'execution_metadata': output_data.get('execution_metadata', {}),
                    'error': f'Serialization failed: {str(e)[:200]}',
                    'results': []
                }
                async with aiofiles.open(output_path, 'w') as f:
                    await f.write(json.dumps(minimal_output, indent=2, default=str))
                print(f"\n⚠️  Saved minimal results file due to serialization error: {output_path}")
                return output_path
            except Exception as e2:
                logger.error(f"❌ Failed to save minimal results: {e2}")
                raise

        except Exception as e:
            logger.error(f"❌ Unexpected error during save: {e}")
            print(f"\n❌ Failed to save results: {e}")
            raise

        return output_path
    
    def _display_execution_summary(self, results: List[Dict[str, Any]]):
        """Display execution summary."""
        print("\n📊 Execution Summary")
        print("-" * 80)
        
        total = len(results)
        successful = sum(1 for r in results if r['status'] == 'success')
        failed = sum(1 for r in results if r['status'] == 'failed')
        
        print(f"Total Scenarios: {total}")
        print(f"Successful: {successful} ✅")
        print(f"Failed: {failed} ❌")
        print(f"Success Rate: {successful/total*100:.1f}%")
        
        print("\nScenario Status:")
        for result in results:
            status_icon = "✅" if result['status'] == 'success' else "❌"
            print(f"  {status_icon} {result['scenario_name']} (ID: {result['scenario_id']})")
        
        print("-" * 80)

    def _generate_visualizations(self, results_path: Path, options: Dict[str, Any]):
        """Generate visualizations for the most recent run."""
        output_dir = options.get('output_dir', 'results/figures')
        style = options.get('style', 'publication')
        scenario = options.get('scenario')
        interactive = options.get('interactive', False)

        formats_opt = options.get('formats', ['png'])
        if isinstance(formats_opt, str):
            if formats_opt == 'all':
                formats = ['png', 'pdf', 'svg']
            else:
                formats = [fmt.strip() for fmt in formats_opt.split(',') if fmt.strip()]
        elif isinstance(formats_opt, list):
            formats = formats_opt
        else:
            formats = ['png']

        try:
            orchestrator = VisualizationOrchestrator(style=style)
            print("\n🎨 Auto-visualization enabled")
            print(f"   Results file: {results_path}")
            print(f"   Output dir: {output_dir}")

            if scenario:
                orchestrator.visualize_scenario(
                    scenario_id=scenario,
                    json_path=str(results_path),
                    output_dir=output_dir,
                    interactive=interactive,
                    formats=formats,
                )
            else:
                orchestrator.visualize_all_scenarios(
                    json_path=str(results_path),
                    output_dir=output_dir,
                    interactive=interactive,
                    formats=formats,
                )

            print("✅ Visualization generation complete\n")

        except Exception as exc:
            logger.error(f"Visualization failed: {exc}", exc_info=True)
            print(f"\n⚠️  Visualization generation failed: {exc}\n")


async def yaml_mode(yaml_path: str, config_path: str = "config/mcp_servers.json") -> int:
    """
    Run YAML batch mode.

    Args:
        yaml_path: Path to YAML configuration file
        config_path: Path to MCP server configuration

    Returns:
        Exit code (0 for success, 1 for failure)
    """
    runner = YAMLRunner(config_path)
    return await runner.run(yaml_path)
