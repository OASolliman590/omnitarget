"""
Interactive CLI Wizard

Guided terminal wizard for OmniTarget pipeline execution.
Uses questionary for rich terminal UI with validation.
"""

import asyncio
import logging
import json
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime

try:
    import questionary
    from questionary import Style
except ImportError:
    raise ImportError("questionary is required for interactive mode. Install with: pip install questionary>=1.10.0")

from ..core.pipeline_orchestrator import OmniTargetPipeline
from .validators import CLIValidator, check_mcp_health

logger = logging.getLogger(__name__)

# Custom style for questionary
custom_style = Style([
    ('qmark', 'fg:#673ab7 bold'),
    ('question', 'bold'),
    ('answer', 'fg:#f44336 bold'),
    ('pointer', 'fg:#673ab7 bold'),
    ('highlighted', 'fg:#673ab7 bold'),
    ('selected', 'fg:#cc5454'),
    ('separator', 'fg:#cc5454'),
    ('instruction', ''),
    ('text', ''),
    ('disabled', 'fg:#858585 italic')
])


class InteractiveWizard:
    """Interactive wizard for OmniTarget pipeline."""
    
    def __init__(self, config_path: str = "config/mcp_servers.json"):
        """Initialize interactive wizard."""
        self.config_path = config_path
        self.validator = CLIValidator()
        self.pipeline = None
        self.results_dir = Path("results")
        self.results_dir.mkdir(exist_ok=True)
    
    async def run(self):
        """Run the interactive wizard."""
        print("\n" + "=" * 80)
        print("🧬 OmniTarget Pipeline - Interactive Mode")
        print("=" * 80)
        print()
        
        try:
            # Step 1: Health check
            if not await self._health_check_step():
                return 1
            
            # Step 2: Scenario selection
            scenario_id = await self._select_scenario_step()
            if scenario_id is None:
                return 0
            
            # Step 3: Parameter collection
            params = await self._collect_parameters_step(scenario_id)
            if params is None:
                return 0
            
            # Step 4: Confirmation
            if not await self._confirmation_step(scenario_id, params):
                return 0
            
            # Step 5: Execute
            result = await self._execute_step(scenario_id, params)
            if result is None:
                return 1
            
            # Step 6: Display results
            await self._display_results_step(scenario_id, result)
            
            # Step 7: Next actions
            await self._next_actions_step()
            
            return 0
            
        except KeyboardInterrupt:
            print("\n\n⚠️  Operation cancelled by user")
            return 1
        except Exception as e:
            print(f"\n\n❌ Error: {e}")
            logger.exception("Interactive wizard error")
            return 1
        finally:
            if self.pipeline:
                await self.pipeline.shutdown()
    
    async def _health_check_step(self) -> bool:
        """Step 1: Check MCP server health."""
        print("🏥 Step 1: Checking MCP Server Health...")
        print()
        
        all_healthy, health_status = await check_mcp_health(self.config_path)
        
        for server, status in health_status.items():
            icon = "✅" if status['status'] == 'healthy' else "❌"
            print(f"  {icon} {server.upper()}: {status['message']}")
        
        print()
        
        if not all_healthy:
            print("⚠️  Some MCP servers are not responding.")
            
            continue_anyway = questionary.confirm(
                "Continue anyway?",
                default=False,
                style=custom_style
            ).ask()
            
            if not continue_anyway:
                print("Exiting...")
                return False
        
        return True
    
    async def _select_scenario_step(self) -> Optional[int]:
        """Step 2: Select scenario."""
        print("📋 Step 2: Select Analysis Scenario")
        print()
        
        scenarios = {
            1: "Disease Network Construction - Build comprehensive disease networks",
            2: "Target-Centric Analysis - Analyze specific protein targets",
            3: "Cancer-Specific Analysis - Discover cancer markers & therapeutic targets",
            4: "Multi-Target Simulation (MRA) - Simulate perturbation effects",
            5: "Pathway Comparison - Cross-validate KEGG vs Reactome pathways",
            6: "Drug Repurposing - Identify drug repurposing opportunities"
        }
        
        choices = [f"{k}. {v}" for k, v in scenarios.items()]
        choices.append("Exit")
        
        selected = questionary.select(
            "Select a scenario:",
            choices=choices,
            style=custom_style
        ).ask()
        
        if selected == "Exit":
            return None
        
        scenario_id = int(selected.split('.')[0])
        print()
        return scenario_id
    
    async def _collect_parameters_step(self, scenario_id: int) -> Optional[Dict[str, Any]]:
        """Step 3: Collect parameters for selected scenario."""
        print(f"⚙️  Step 3: Configure Parameters for Scenario {scenario_id}")
        print()
        
        if scenario_id == 1:
            return await self._collect_scenario_1_params()
        elif scenario_id == 2:
            return await self._collect_scenario_2_params()
        elif scenario_id == 3:
            return await self._collect_scenario_3_params()
        elif scenario_id == 4:
            return await self._collect_scenario_4_params()
        elif scenario_id == 5:
            return await self._collect_scenario_5_params()
        elif scenario_id == 6:
            return await self._collect_scenario_6_params()
        
        return None
    
    async def _collect_scenario_1_params(self) -> Optional[Dict[str, Any]]:
        """Collect parameters for Scenario 1: Disease Network Construction."""
        params = {}
        
        # Disease query
        params['disease_query'] = questionary.text(
            "Enter disease name:",
            validate=lambda x: self.validator.validate_disease_query(x)[0] or self.validator.validate_disease_query(x)[1],
            style=custom_style
        ).ask()
        
        # Tissue context (optional)
        add_tissue = questionary.confirm(
            "Add tissue context? (optional, improves expression filtering)",
            default=True,
            style=custom_style
        ).ask()
        
        if add_tissue:
            params['tissue_context'] = questionary.text(
                "Enter tissue context:",
                default=params['disease_query'].split()[0] if ' ' in params['disease_query'] else "",
                style=custom_style
            ).ask()
        
        return params
    
    async def _collect_scenario_2_params(self) -> Optional[Dict[str, Any]]:
        """Collect parameters for Scenario 2: Target-Centric Analysis."""
        params = {}
        
        # Target query
        params['target_query'] = questionary.text(
            "Enter target gene/protein (e.g., TP53, BRCA1):",
            validate=lambda x: self.validator.validate_target_gene(x)[0] or self.validator.validate_target_gene(x)[1],
            style=custom_style
        ).ask()
        
        return params
    
    async def _collect_scenario_3_params(self) -> Optional[Dict[str, Any]]:
        """Collect parameters for Scenario 3: Cancer-Specific Analysis."""
        params = {}
        
        # Cancer type
        params['cancer_type'] = questionary.text(
            "Enter cancer type (e.g., breast cancer, lung adenocarcinoma):",
            validate=lambda x: len(x) >= 3 or "Cancer type too short",
            style=custom_style
        ).ask()
        
        # Tissue context
        params['tissue_context'] = questionary.text(
            "Enter tissue context:",
            default=params['cancer_type'].split()[0] if ' ' in params['cancer_type'] else "",
            style=custom_style
        ).ask()
        
        return params
    
    async def _collect_scenario_4_params(self) -> Optional[Dict[str, Any]]:
        """Collect parameters for Scenario 4: Multi-Target Simulation."""
        params = {}
        
        # Targets
        targets_str = questionary.text(
            "Enter target genes (comma-separated, e.g., TP53,BRCA1,BRCA2):",
            validate=lambda x: len(x.strip()) > 0 or "Targets cannot be empty",
            style=custom_style
        ).ask()
        
        targets = [t.strip().upper() for t in targets_str.split(',')]
        is_valid, error, valid_targets = self.validator.validate_targets(targets)
        
        if not is_valid:
            print(f"❌ {error}")
            return None
        
        params['targets'] = valid_targets
        
        # Disease context
        params['disease_context'] = questionary.text(
            "Enter disease context:",
            style=custom_style
        ).ask()
        
        # Simulation mode
        params['simulation_mode'] = questionary.select(
            "Select simulation mode:",
            choices=[
                'simple - Fast, confidence-weighted propagation',
                'mra - Advanced Modular Response Analysis (slower, more accurate)'
            ],
            style=custom_style
        ).ask().split(' - ')[0]
        
        # Tissue context (optional)
        add_tissue = questionary.confirm(
            "Add tissue context?",
            default=True,
            style=custom_style
        ).ask()
        
        if add_tissue:
            params['tissue_context'] = questionary.text(
                "Enter tissue context:",
                style=custom_style
            ).ask()
        
        return params
    
    async def _collect_scenario_5_params(self) -> Optional[Dict[str, Any]]:
        """Collect parameters for Scenario 5: Pathway Comparison."""
        params = {}
        
        # Pathway query
        params['pathway_query'] = questionary.text(
            "Enter pathway name or identifier:",
            validate=lambda x: len(x) >= 3 or "Pathway query too short",
            style=custom_style
        ).ask()
        
        return params
    
    async def _collect_scenario_6_params(self) -> Optional[Dict[str, Any]]:
        """Collect parameters for Scenario 6: Drug Repurposing."""
        params = {}
        
        # Disease query
        params['disease_query'] = questionary.text(
            "Enter disease name:",
            validate=lambda x: self.validator.validate_disease_query(x)[0] or self.validator.validate_disease_query(x)[1],
            style=custom_style
        ).ask()
        
        # Tissue context
        params['tissue_context'] = questionary.text(
            "Enter tissue context:",
            default=params['disease_query'].split()[0] if ' ' in params['disease_query'] else "",
            style=custom_style
        ).ask()
        
        # Simulation mode (optional)
        params['simulation_mode'] = questionary.select(
            "Select simulation mode:",
            choices=['simple', 'mra'],
            default='simple',
            style=custom_style
        ).ask()
        
        return params
    
    async def _confirmation_step(self, scenario_id: int, params: Dict[str, Any]) -> bool:
        """Step 4: Confirm parameters."""
        print()
        print("✅ Step 4: Confirmation")
        print()
        print(f"Scenario: {scenario_id}")
        print("Parameters:")
        for key, value in params.items():
            print(f"  {key}: {value}")
        print()
        
        confirmed = questionary.confirm(
            "Execute with these parameters?",
            default=True,
            style=custom_style
        ).ask()
        
        return confirmed
    
    async def _execute_step(self, scenario_id: int, params: Dict[str, Any]) -> Optional[Any]:
        """Step 5: Execute scenario."""
        print()
        print(f"⚡ Step 5: Executing Scenario {scenario_id}...")
        print()
        print("Please wait, this may take a few minutes...")
        print()
        
        try:
            # Use pipeline as context manager for MCP session management
            async with OmniTargetPipeline(self.config_path) as pipeline:
                result = await pipeline.run_scenario(scenario_id, **params)
            
            print()
            print("✅ Execution completed successfully!")
            print()
            
            return result
            
        except Exception as e:
            print(f"\n❌ Execution failed: {e}")
            logger.exception("Scenario execution error")
            return None
    
    async def _display_results_step(self, scenario_id: int, result: Any):
        """Step 6: Display results."""
        print("📊 Step 6: Results Summary")
        print()
        
        # Display summary based on scenario
        if hasattr(result, 'validation_score'):
            print(f"Validation Score: {result.validation_score:.3f}")
        
        if hasattr(result, 'pathways'):
            print(f"Pathways Found: {len(result.pathways)}")
        
        if hasattr(result, 'network_nodes'):
            print(f"Network Nodes: {len(result.network_nodes)}")
        
        if hasattr(result, 'network_edges'):
            print(f"Network Edges: {len(result.network_edges)}")
        
        if hasattr(result, 'prioritized_targets'):
            print(f"Prioritized Targets: {len(result.prioritized_targets)}")
            if result.prioritized_targets:
                print("\nTop 5 Targets:")
                for i, target in enumerate(result.prioritized_targets[:5], 1):
                    print(f"  {i}. {target.gene_symbol}: {target.prioritization_score:.3f}")
        
        print()
        
        # Save results
        save_results = questionary.confirm(
            "Save results to file?",
            default=True,
            style=custom_style
        ).ask()
        
        if save_results:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"scenario_{scenario_id}_{timestamp}.json"
            filepath = self.results_dir / filename
            
            with open(filepath, 'w') as f:
                json.dump(result.__dict__ if hasattr(result, '__dict__') else str(result), 
                         f, indent=2, default=str)
            
            print(f"💾 Results saved to: {filepath}")
            print()
    
    async def _next_actions_step(self):
        """Step 7: Next actions."""
        action = questionary.select(
            "What would you like to do next?",
            choices=[
                "View detailed results",
                "Run another scenario",
                "Export configuration to YAML",
                "Exit"
            ],
            style=custom_style
        ).ask()
        
        if action == "Run another scenario":
            # Restart wizard
            await self.run()
        elif action == "View detailed results":
            print("\nDetailed results are saved in the results directory.")
        elif action == "Export configuration to YAML":
            print("\nYAML export feature coming soon...")
        
        print("\nThank you for using OmniTarget Pipeline!")


async def interactive_mode(config_path: str = "config/mcp_servers.json") -> int:
    """
    Run interactive wizard mode.
    
    Args:
        config_path: Path to MCP server configuration
        
    Returns:
        Exit code (0 for success, 1 for failure)
    """
    wizard = InteractiveWizard(config_path)
    return await wizard.run()

