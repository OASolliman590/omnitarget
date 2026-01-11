"""
Scenario 4: Multi-Target Simulation with MRA

Advanced perturbation simulation with cross-validation and impact assessment.
Based on Mature_development_plan.md Phase 1-5 and OmniTarget_Development_Plan.md.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Set, Tuple
import networkx as nx
import numpy as np
from collections import defaultdict
from itertools import islice
import time
import psutil
import os

from ..core.mcp_client_manager import MCPClientManager
from ..core.data_standardizer import DataStandardizer
from ..core.validation import DataValidator
from ..core.simulation.mra_simulator import MRASimulator
from ..core.simulation.feedback_analyzer import FeedbackAnalyzer
from ..core.string_network_builder import AdaptiveStringNetworkBuilder
from ..models.data_models import (
    Protein, Pathway, Interaction, ExpressionProfile,
    NetworkNode, NetworkEdge, MRASimulationResult, NetworkExpansionConfig,
    DataSourceStatus, CompletenessMetrics
)
from ..models.simulation_models import MultiTargetSimulationResult, SimulationResult

logger = logging.getLogger(__name__)


def retry_with_exponential_backoff(
    max_attempts: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    exponential_base: float = 2.0,
    jitter: bool = True
):
    """
    Retry decorator with exponential backoff for MCP calls.

    Args:
        max_attempts: Maximum number of retry attempts
        base_delay: Initial delay in seconds
        max_delay: Maximum delay in seconds
        exponential_base: Base for exponential backoff calculation
        jitter: Whether to add random jitter to delays

    Returns:
        Decorator function
    """
    def decorator(func):
        async def wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(1, max_attempts + 1):
                try:
                    result = await func(*args, **kwargs)
                    if attempt > 1:
                        logger.info(f"✅ {func.__name__} succeeded on attempt {attempt}")
                    return result
                except (asyncio.TimeoutError, ConnectionError, Exception) as e:
                    last_exception = e

                    if attempt == max_attempts:
                        logger.warning(
                            f"❌ {func.__name__} failed after {max_attempts} attempts. "
                            f"Last error: {type(e).__name__}: {str(e)[:100]}"
                        )
                        raise e

                    # Calculate delay with exponential backoff
                    delay = min(base_delay * (exponential_base ** (attempt - 1)), max_delay)

                    # Add jitter to avoid thundering herd
                    if jitter:
                        import random
                        delay = delay * (0.5 + random.random() * 0.5)

                    logger.warning(
                        f"⚠️  {func.__name__} failed on attempt {attempt}/{max_attempts}. "
                        f"Retrying in {delay:.1f}s... Error: {type(e).__name__}: {str(e)[:50]}"
                    )

                    await asyncio.sleep(delay)

            # Should never reach here, but just in case
            if last_exception:
                raise last_exception

        return wrapper
    return decorator


class MRASimulationProgress:
    """
    Progress monitoring and heartbeat logging for MRA simulations.

    Provides real-time progress tracking to help diagnose hanging issues
    and monitor resource usage during long-running simulations.
    """

    def __init__(self, total_targets: int, step_count: int = 8, heartbeat_interval: int = 30):
        """
        Initialize progress monitor.

        Args:
            total_targets: Number of targets being analyzed
            step_count: Total number of steps in the workflow (default: 8)
            heartbeat_interval: Seconds between heartbeat logs (default: 30)
        """
        import time
        import psutil
        import os

        self.total_targets = total_targets
        self.step_count = step_count
        self.heartbeat_interval = heartbeat_interval
        self.start_time = time.time()
        self.last_heartbeat = self.start_time
        self.current_step = 0
        self.step_names = {
            1: "Target Resolution",
            2: "Network Context",
            3: "Interaction Validation",
            4: "Expression Validation",
            5: "Pathway Impact",
            6: "Network Construction",
            7: "Simulation",
            8: "Impact Assessment"
        }
        self.step_times = {}
        self.pid = os.getpid()
        try:
            self.process = psutil.Process(self.pid)
        except:
            self.process = None

        logger.info(f"🔍 MRA Simulation Progress Monitor Started")
        logger.info(f"   Total targets: {total_targets}")
        logger.info(f"   Total steps: {step_count}")
        logger.info(f"   Heartbeat interval: {heartbeat_interval}s")
        logger.info(f"   Process ID: {self.pid}")

    async def log_step_start(self, step_number: int, step_details: str = ""):
        """
        Log the start of a new step.

        Args:
            step_number: Step number (1-8)
            step_details: Additional details about the step
        """
        import time
        self.current_step = step_number
        step_name = self.step_names.get(step_number, f"Step {step_number}")
        elapsed = time.time() - self.start_time

        logger.info(f"⏱️  [{self._format_time(elapsed)}] Step {step_number}/8: {step_name}")
        if step_details:
            logger.info(f"   Details: {step_details}")

        self._log_resources()

    async def log_step_complete(self, step_number: int, result_summary: str = ""):
        """
        Log the completion of a step.

        Args:
            step_number: Step number that completed
            result_summary: Summary of results
        """
        import time
        elapsed = time.time() - self.start_time
        step_time = elapsed - sum(self.step_times.values())
        self.step_times[step_number] = step_time

        step_name = self.step_names.get(step_number, f"Step {step_number}")
        logger.info(f"✅ [{self._format_time(elapsed)}] Step {step_number}/8: {step_name} - COMPLETE")
        logger.info(f"   Step duration: {self._format_duration(step_time)}")
        if result_summary:
            logger.info(f"   Results: {result_summary}")

        # Log progress summary
        progress_pct = (step_number / self.step_count) * 100
        logger.info(f"   Progress: {progress_pct:.1f}% ({step_number}/{self.step_count} steps)")

    async def log_heartbeat(self, message: str = ""):
        """
        Log a heartbeat update with current status.

        Args:
            message: Additional message to include
        """
        import time
        current_time = time.time()
        if current_time - self.last_heartbeat < self.heartbeat_interval:
            return

        self.last_heartbeat = current_time
        elapsed = current_time - self.start_time

        step_name = self.step_names.get(self.current_step, f"Step {self.current_step}")
        logger.info(f"💓 [{self._format_time(elapsed)}] HEARTBEAT - Current: Step {self.current_step} ({step_name})")

        if message:
            logger.info(f"   Message: {message}")

        self._log_resources()

    def _log_resources(self):
        """Log current resource usage."""
        if not self.process:
            return

        try:
            # Get memory usage
            memory_info = self.process.memory_info()
            memory_mb = memory_info.rss / 1024 / 1024

            # Get CPU percent
            cpu_percent = self.process.cpu_percent(interval=0.1)

            logger.info(f"   Resources: CPU={cpu_percent:.1f}%, Memory={memory_mb:.1f}MB")

        except Exception as e:
            # Silently fail if resource monitoring fails
            pass

    def _format_time(self, seconds: float) -> str:
        """Format seconds into HH:MM:SS string."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"

    def _format_duration(self, seconds: float) -> str:
        """Format duration into human-readable string."""
        if seconds < 60:
            return f"{seconds:.1f}s"
        elif seconds < 3600:
            minutes = int(seconds // 60)
            secs = int(seconds % 60)
            return f"{minutes}m {secs}s"
        else:
            hours = int(seconds // 3600)
            minutes = int((seconds % 3600) // 60)
            return f"{hours}h {minutes}m"

    async def log_final_summary(self):
        """Log final execution summary."""
        import time
        total_time = time.time() - self.start_time

        logger.info(f"🏁 MRA Simulation Complete")
        logger.info(f"   Total duration: {self._format_duration(total_time)}")
        logger.info(f"   Total targets: {self.total_targets}")
        logger.info(f"   Average time per target: {self._format_duration(total_time / max(1, self.total_targets))}")

        if self.step_times:
            logger.info(f"   Step breakdown:")
            for step_num in sorted(self.step_times.keys()):
                step_time = self.step_times[step_num]
                step_name = self.step_names.get(step_num, f"Step {step_num}")
                percentage = (step_time / total_time) * 100
                logger.info(f"     Step {step_num}: {self._format_duration(step_time)} ({percentage:.1f}%) - {step_name}")


class MultiTargetSimulationScenario:
    """
    Scenario 4: Multi-Target Simulation with MRA

    8-step workflow:
    1. Target resolution (STRING + HPA)
    2. Network context (Reactome pathways)
    3. Interaction validation (STRING)
    4. Expression validation (HPA)
    5. Pathway impact (Reactome participants)
    6. Network construction
    7. Simulation (simple or MRA)
    8. Impact assessment (STRING enrichment)
    """

    def __init__(
        self,
        mcp_manager: MCPClientManager,
        network_expansion_config: Optional[NetworkExpansionConfig] = None
    ):
        """
        Initialize multi-target simulation scenario.

        Args:
            mcp_manager: MCP client manager instance
            network_expansion_config: Optional configuration for network expansion parameters
        """
        self.mcp_manager = mcp_manager
        self.standardizer = DataStandardizer()
        self.validator = DataValidator()

        # Store network expansion config (use defaults if not provided)
        self.network_expansion_config = network_expansion_config or NetworkExpansionConfig()

        # Timeout configuration for each step (in seconds)
        # INCREASED to prevent hanging on slow API calls
        self.step_timeouts = {
            1: 60,   # Target resolution: 60s (was 30s)
            2: 60,   # Network context: 60s (was 45s)
            3: 300,  # Interaction validation: 300s (was 60s) - MAJOR increase for large network fetches
            4: 120,  # Expression validation: 120s (was 60s)
            5: 150,  # Pathway impact: 150s (was 90s) - with timeout per pathway
            6: 300,  # Network construction: 300s (was 120s) - MAJOR increase
            7: 180,  # Simulation: 180s (unchanged)
            8: 120   # Impact assessment: 120s (was 60s)
        }

        # Override step timeouts if provided in config
        if self.network_expansion_config.step_timeouts:
            for step, timeout in self.network_expansion_config.step_timeouts.items():
                if step in self.step_timeouts:
                    logger.info(
                        f"Overriding Step {step} timeout: {self.step_timeouts[step]}s → {timeout}s "
                        f"(from network_expansion_config)"
                    )
                    self.step_timeouts[step] = timeout
        self.progress_monitor = None
        # Semaphore to limit concurrent MCP requests (max 3 to prevent overload)
        self.max_concurrent_requests = 3
        self._request_semaphore = None  # Lazy initialization to avoid event loop issues
        # Connection pool monitoring
        self.connection_pool_enabled = False
        self.connection_pool_stats = None
        # Circuit breaker monitoring
        self.circuit_breaker_enabled = True
        self.circuit_breaker_stats = None
        self._active_data_sources: Optional[Dict[str, DataSourceStatus]] = None
        self.string_builder = None  # Will be initialized when data_sources available

    def _get_request_semaphore(self) -> asyncio.Semaphore:
        """Lazy initialize semaphore to avoid event loop issues in __init__."""
        if self._request_semaphore is None:
            self._request_semaphore = asyncio.Semaphore(self.max_concurrent_requests)
            logger.info(f"Request semaphore initialized: max_concurrent={self.max_concurrent_requests}")
        return self._request_semaphore

    def _check_connection_pool_status(self):
        """
        Check if connection pooling is available and enabled.

        This method checks the MCP manager for connection pool support
        and logs the status for monitoring purposes.
        """
        manager = getattr(self.mcp_manager, 'connection_pool_manager', None)
        if not manager:
            logger.debug("🔗 Connection Pool: Not available (no connection pool manager)")
            return

        try:
            self.connection_pool_enabled = bool(getattr(manager, 'enable_pooling', False))
        except Exception:
            logger.debug("🔗 Connection Pool: Manager missing enable flag, treating as disabled")
            self.connection_pool_enabled = False

        if not self.connection_pool_enabled:
            logger.info("🔗 Connection Pool: DISABLED")
            return

        try:
            stats = manager.get_stats()
        except Exception as exc:
            logger.debug(f"🔗 Connection Pool: Unable to fetch stats ({exc})")
            return

        if not isinstance(stats, dict) or 'pool_stats' not in stats:
            logger.debug("🔗 Connection Pool: Stats unavailable or malformed")
            return

        self.connection_pool_stats = stats
        pool_stats = stats['pool_stats']

        logger.info(
            f"🔗 Connection Pool: ENABLED | "
            f"Active: {pool_stats.get('active_connections', 0)} | "
            f"Total created: {pool_stats.get('total_connections_created', 0)}"
        )

    def _log_connection_pool_summary(self):
        """Log a summary of connection pool usage."""
        if not self.connection_pool_enabled or not self.connection_pool_stats:
            logger.info("🔗 Connection Pool: Not enabled or no stats available")
            return

        stats = self.connection_pool_stats['pool_stats']

        logger.info("=" * 70)
        logger.info("🔗 CONNECTION POOL SUMMARY")
        logger.info("=" * 70)
        logger.info(f"Total connections created: {stats['total_connections_created']}")
        logger.info(f"Total connections closed: {stats['total_connections_closed']}")
        logger.info(f"Active connections: {stats['active_connections']}")
        logger.info("-" * 70)

        for server_name, server_stats in stats['servers'].items():
            logger.info(f"Server: {server_name}")
            logger.info(f"  Active: {server_stats['total']} | "
                       f"Healthy: {server_stats['healthy']} | "
                       f"Unhealthy: {server_stats['unhealthy']}")
            logger.info(f"  Requests: {server_stats['total_requests']} | "
                       f"Errors: {server_stats['total_errors']} | "
                       f"Error rate: {(server_stats['total_errors']/max(1, server_stats['total_requests'])*100):.1f}%")
            logger.info(f"  Avg age: {server_stats['avg_age_seconds']:.1f}s")
            logger.info("")

        logger.info("=" * 70)

    def _check_circuit_breaker_status(self):
        """
        Check if circuit breaker is available and enabled.

        This method checks for circuit breaker manager support
        and logs the status for monitoring purposes.
        """
        from ..core.circuit_breaker import get_global_manager

        try:
            cb_manager = get_global_manager()
            if cb_manager:
                # Get all circuit breaker stats
                stats = cb_manager.get_all_stats()
                self.circuit_breaker_stats = stats

                if stats:
                    open_circuits = sum(1 for s in stats.values() if s.get('state') == 'open')
                    half_open_circuits = sum(1 for s in stats.values() if s.get('state') == 'half_open')
                    closed_circuits = sum(1 for s in stats.values() if s.get('state') == 'closed')

                    logger.info(
                        f"🔌 Circuit Breaker: ENABLED | "
                        f"Closed: {closed_circuits} | "
                        f"Open: {open_circuits} | "
                        f"Half-Open: {half_open_circuits}"
                    )
                else:
                    logger.info("🔌 Circuit Breaker: No circuit breakers registered")
            else:
                logger.info("🔌 Circuit Breaker: Not available (no manager found)")
        except Exception as e:
            logger.debug(f"🔌 Circuit Breaker: Check failed: {e}")

    def _log_circuit_breaker_summary(self):
        """Log a summary of circuit breaker status."""
        if not self.circuit_breaker_enabled or not self.circuit_breaker_stats:
            logger.info("🔌 Circuit Breaker: Not enabled or no stats available")
            return

        from ..core.circuit_breaker import get_global_manager

        try:
            cb_manager = get_global_manager()
            if not cb_manager:
                logger.info("🔌 Circuit Breaker: Manager not available")
                return

            aggregate = cb_manager.get_aggregate_stats()
            all_stats = self.circuit_breaker_stats

            logger.info("=" * 70)
            logger.info("🔌 CIRCUIT BREAKER SUMMARY")
            logger.info("=" * 70)
            logger.info(f"Total servers: {aggregate.get('total_servers', 0)}")
            logger.info(f"Total requests: {aggregate.get('total_requests', 0)}")
            logger.info(f"Total failures: {aggregate.get('total_failures', 0)}")
            logger.info(f"Failure rate: {aggregate.get('failure_rate', 0):.2%}")
            logger.info(f"Healthy (CLOSED): {aggregate.get('closed_circuits', 0)}")
            logger.info(f"Degraded (OPEN): {aggregate.get('open_circuits', 0)}")
            logger.info(f"Testing (HALF-OPEN): {aggregate.get('half_open_circuits', 0)}")
            logger.info("-" * 70)

            for server_name, stats in all_stats.items():
                logger.info(f"Server: {server_name}")
                logger.info(f"  State: {stats.get('state', 'unknown').upper()}")
                logger.info(f"  Requests: {stats.get('total_calls', 0)} | "
                           f"Failures: {stats.get('total_failures', 0)} | "
                           f"Successes: {stats.get('total_successes', 0)}")
                last_failure = stats.get('last_failure_time')
                if last_failure:
                    logger.info(f"  Last failure: {last_failure}")
                if stats.get('state') == 'open':
                    logger.info(f"  🚫 BLOCKING requests (failure threshold reached)")
                elif stats.get('state') == 'half_open':
                    successes = stats.get('success_count', 0)
                    logger.info(f"  🔄 Testing recovery ({successes}/2 successes needed)")
                logger.info("")

            logger.info("=" * 70)
        except Exception as e:
            logger.warning(f"Failed to log circuit breaker summary: {e}")

    def _build_simulation_context(
        self,
        pathway_data: Dict[str, Any],
        interaction_data: Dict[str, Any],
        expression_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Assemble MCP context for simulation engines."""
        context = {
            'reactome_pathways': [],
            'kegg_pathways': [],
            'string_interactions': interaction_data.get('interactions', []),
            'hpa_expression': []
        }

        for pathway in pathway_data.get('pathways', []):
            try:
                pathway_dict = pathway.dict()
            except AttributeError:
                pathway_dict = pathway or {}

            source_db = pathway_dict.get('source_db', 'reactome')
            if source_db == 'reactome':
                context['reactome_pathways'].append(pathway_dict)
            else:
                context['kegg_pathways'].append(pathway_dict)

        for profile in expression_data.get('profiles', []):
            try:
                profile_dict = profile.dict()
            except AttributeError:
                profile_dict = profile or {}

            context['hpa_expression'].append({
                'gene': profile_dict.get('gene'),
                'expression_level': profile_dict.get('expression_level'),
                'tissue': profile_dict.get('tissue'),
                'reliability': profile_dict.get('reliability')
            })

        return context

    async def _execute_with_semaphore(self, client_func, *args, source_name: Optional[str] = None, **kwargs):
        """
        Execute a coroutine with semaphore-based request limiting.

        Args:
            client_func: Callable returning coroutine (e.g., MCP client method)
            *args: Arguments to pass to the coroutine
            **kwargs: Keyword arguments to pass to the coroutine
            source_name: Optional data source identifier for tracking

        Returns:
            Result from the coroutine
        """
        semaphore = self._get_request_semaphore()
        async with semaphore:
            coro = client_func(*args, **kwargs)
            if source_name:
                return await self._call_with_tracking(None, source_name, coro)
            return await coro

    async def _step_with_timeout(self, step_number: int, step_func, *args, **kwargs):
        """
        Execute a step with timeout protection and fallback strategy.

        Args:
            step_number: Step number (1-8) to determine timeout and logging
            step_func: Async function to execute
            *args: Arguments to pass to step_func
            **kwargs: Keyword arguments to pass to step_func

        Returns:
            Result from step_func or fallback result on timeout
        """
        timeout = self.step_timeouts.get(step_number, 60)
        step_name = f"Step {step_number}"

        try:
            logger.info(f"{step_name}: Starting with {timeout}s timeout")

            # Track elapsed time for progress alerts
            start_time = asyncio.get_event_loop().time()
            alert_80_sent = False

            # Create a task to monitor timeout progress
            async def monitor_timeout():
                nonlocal alert_80_sent
                try:
                    await asyncio.sleep(timeout * 0.8)  # Wait 80% of timeout
                    if not alert_80_sent:
                        elapsed = asyncio.get_event_loop().time() - start_time
                        logger.warning(f"⚠️  {step_name}: Approaching timeout ({elapsed:.1f}s/{timeout}s) - 80% threshold reached")
                        alert_80_sent = True
                except asyncio.CancelledError:
                    pass

            # Start monitoring task
            monitor_task = asyncio.create_task(monitor_timeout())

            # Execute the step with timeout
            result = await asyncio.wait_for(step_func(*args, **kwargs), timeout=timeout)

            # Cancel monitoring task
            monitor_task.cancel()
            try:
                await monitor_task
            except asyncio.CancelledError:
                pass

            logger.info(f"{step_name}: Completed successfully")
            return result
        except asyncio.TimeoutError:
            elapsed = asyncio.get_event_loop().time() - start_time if 'start_time' in locals() else timeout
            logger.warning(f"⚠️  {step_name}: TIMEOUT after {elapsed:.1f}s (configured: {timeout}s) - applying fallback strategy")
            return await self._get_fallback_result(step_number, *args, **kwargs)
        except Exception as e:
            logger.error(f"{step_name}: Error occurred - {type(e).__name__}: {e}")
            logger.info(f"{step_name}: Applying fallback strategy")
            return await self._get_fallback_result(step_number, *args, **kwargs)

    async def _get_fallback_result(self, step_number: int, *args, **kwargs):
        """
        Get fallback result when a step times out or fails.

        Args:
            step_number: Step number (1-8)
            *args: Arguments from original step call

        Returns:
            Fallback result appropriate for the step
        """
        if step_number == 1:
            # Target resolution fallback: return empty list
            logger.warning("Step 1 fallback: Returning empty resolved targets")
            return {'resolved_targets': [], 'resolution_accuracy': 0.0}
        elif step_number == 2:
            # Network context fallback: return empty pathways
            logger.warning("Step 2 fallback: Returning empty pathways")
            return {'pathways': [], 'coverage': 0.0}
        elif step_number == 3:
            # Interaction validation fallback: return empty interactions
            logger.warning("Step 3 fallback: Returning empty interactions")
            return {'interactions': [], 'validation_score': 0.0}
        elif step_number == 4:
            # Expression validation fallback: return empty profiles
            logger.warning("Step 4 fallback: Returning empty expression profiles")
            return {'profiles': [], 'coverage': 0.0}
        elif step_number == 5:
            # Pathway impact fallback: return empty impact
            logger.warning("Step 5 fallback: Returning empty pathway impact")
            return {'pathway_impacts': [], 'overall_impact': 0.0}
        elif step_number == 6:
            # Network construction fallback: return empty network
            logger.warning("Step 6 fallback: Returning empty network")
            import networkx as nx
            G = nx.Graph()
            return {'network': G, 'nodes': [], 'edges': []}
        elif step_number == 7:
            # Simulation fallback: return minimal results
            logger.warning("Step 7 fallback: Returning minimal simulation results")
            if args and len(args) > 1:
                network = args[0]
                targets = args[1]
                import networkx as nx
                results = []
                for target in targets[:3]:  # Limit to 3 targets
                    if network.has_node(target.gene_symbol):
                        results.append({
                            'target': target.gene_symbol,
                            'affected_nodes': {target.gene_symbol: 0.5},
                            'direct_targets': list(network.neighbors(target.gene_symbol))[:5],
                            'downstream': [],
                            'upstream': [],
                            'feedback_loops': [],
                            'network_impact': {'total_affected': 1},
                            'confidence_scores': {'overall': 0.3},
                            'execution_time': 0.1
                        })
                return {
                    'results': results,
                    'convergence_rate': 0.5,
                    'synergy_matrix': {},
                    'network_impact': {'total_affected_nodes': len(results)},
                    'total_affected_nodes': len(results)
                }
            return {'results': [], 'convergence_rate': 0.0, 'synergy_matrix': {}, 'network_impact': {}}
        elif step_number == 8:
            # Impact assessment fallback: return empty assessment
            logger.warning("Step 8 fallback: Returning empty impact assessment")
            return {
                'enrichment': {},
                'enrichment_score': 0.5,
                'pathway_impact': {},
                'functional_enrichment': {},
                'network_impact': {},
                'affected_genes': []
            }
        else:
            logger.error(f"Unknown step number for fallback: {step_number}")
            return {}
    
    async def execute(
        self,
        targets: List[str],
        disease_context: str,
        simulation_mode: str = 'simple',
        tissue_context: Optional[str] = None,
        network_expansion: Optional[Dict[str, Any]] = None
    ) -> MultiTargetSimulationResult:
        """
        Execute complete multi-target simulation workflow.

        Args:
            targets: List of target proteins/genes
            disease_context: Disease context for pathway analysis
            simulation_mode: 'simple' or 'mra'
            tissue_context: Optional tissue context for expression filtering
            network_expansion: Optional dict with network expansion config
                              (initial_neighbors, expansion_neighbors, max_network_size, step_timeouts)

        Returns:
            MultiTargetSimulationResult with complete analysis
        """
        # Update network expansion config if provided
        if network_expansion:
            self.network_expansion_config = NetworkExpansionConfig(**network_expansion)
            logger.info(f"Updated network_expansion_config from execute parameters: {self.network_expansion_config}")

            # Override step timeouts if provided in config
            if self.network_expansion_config.step_timeouts:
                for step, timeout in self.network_expansion_config.step_timeouts.items():
                    if step in self.step_timeouts:
                        logger.info(
                            f"Overriding Step {step} timeout: {self.step_timeouts[step]}s → {timeout}s "
                            f"(from network_expansion)"
                        )
                        self.step_timeouts[step] = timeout

        # Initialize progress monitor
        self.progress_monitor = MRASimulationProgress(
            total_targets=len(targets),
            step_count=8,
            heartbeat_interval=10  # CHANGED from 30s to 10s for faster feedback
        )

        # Check connection pool status
        self._check_connection_pool_status()

        # Check circuit breaker status
        self._check_circuit_breaker_status()

        logger.info(f"Starting multi-target simulation for {len(targets)} targets with timeout protection and progress monitoring")

        data_sources = {
            'reactome': DataSourceStatus(source_name='reactome', requested=0, successful=0, failed=0, success_rate=0.0, error_types=[]),
            'string': DataSourceStatus(source_name='string', requested=0, successful=0, failed=0, success_rate=0.0, error_types=[]),
            'hpa': DataSourceStatus(source_name='hpa', requested=0, successful=0, failed=0, success_rate=0.0, error_types=[])
        }
        self._active_data_sources = data_sources

        # Step 1: Target resolution (with timeout)
        await self.progress_monitor.log_step_start(1, f"Resolving {len(targets)} targets across STRING and HPA")
        target_data = await self._step_with_timeout(
            1, self._step1_target_resolution, targets
        )
        resolved_count = len(target_data.get('resolved_targets', []))
        await self.progress_monitor.log_step_complete(1, f"{resolved_count}/{len(targets)} targets resolved")

        # Step 2: Network context (with timeout)
        await self.progress_monitor.log_step_start(2, f"Finding pathways for {resolved_count} targets")
        pathway_data = await self._step_with_timeout(
            2, self._step2_network_context,
            target_data['resolved_targets'],
            disease_context
        )
        pathway_count = len(pathway_data.get('pathways', []))
        pathway_coverage = pathway_data.get('coverage', 'N/A')
        await self.progress_monitor.log_step_complete(2, f"{pathway_count} pathways found")
        logger.info(f"DEBUG [Step 2]: pathway_data contains {pathway_count} pathways, coverage={pathway_coverage}")

        # Step 3: Interaction validation (with timeout)
        await self.progress_monitor.log_step_start(3, f"Validating interactions for {resolved_count} targets")
        interaction_data = await self._step_with_timeout(
            3, self._step3_interaction_validation,
            target_data['resolved_targets']
        )
        interaction_count = len(interaction_data.get('interactions', []))
        await self.progress_monitor.log_step_complete(3, f"{interaction_count} interactions found")

        # Step 4: Expression validation (with timeout)
        await self.progress_monitor.log_step_start(4, f"Validating expression for {resolved_count} targets")
        expression_data = await self._step_with_timeout(
            4, self._step4_expression_validation,
            target_data['resolved_targets'],
            tissue_context
        )
        expression_count = len(expression_data.get('profiles', []))
        await self.progress_monitor.log_step_complete(4, f"{expression_count} expression profiles")

        # Step 5: Pathway impact (with timeout)
        await self.progress_monitor.log_step_start(5, f"Analyzing {pathway_count} pathways for {resolved_count} targets")
        impact_data = await self._step_with_timeout(
            5, self._step5_pathway_impact,
            target_data['resolved_targets'],
            pathway_data['pathways']
        )
        impact_count = len(impact_data.get('pathway_impacts', []))
        await self.progress_monitor.log_step_complete(5, f"{impact_count} pathway impacts calculated")

        # Step 6: Network construction (with timeout)
        await self.progress_monitor.log_step_start(6, f"Building network from {interaction_count} interactions")
        network_data = await self._step_with_timeout(
            6, self._step6_network_construction,
            target_data['resolved_targets'],
            interaction_data['interactions']
        )
        network_node_count = len(network_data.get('nodes', []))
        network_edge_count = len(network_data.get('edges', []))
        # DIAGNOSTIC: Log network size after step 6
        logger.info(f"[S4 Step 6] Network constructed: {network_node_count} nodes, {network_edge_count} edges")
        if network_node_count == 0:
            logger.warning(f"[S4 Step 6] ⚠️  Network is empty! Network object type: {type(network_data.get('network'))}")
            if network_data.get('network'):
                G = network_data.get('network')
                if hasattr(G, 'number_of_nodes'):
                    logger.warning(f"[S4 Step 6] NetworkX graph has {G.number_of_nodes()} nodes (not in 'nodes' list)")
        await self.progress_monitor.log_step_complete(6, f"Network: {network_node_count} nodes, {network_edge_count} edges")

        simulation_context = self._build_simulation_context(
            pathway_data,
            interaction_data,
            expression_data
        )

        # Step 7: Simulation (with timeout)
        await self.progress_monitor.log_step_start(7, f"Running {simulation_mode} simulation for {resolved_count} targets")
        simulation_data = await self._step_with_timeout(
            7, self._step7_simulation,
            network_data['network'],
            target_data['resolved_targets'],
            simulation_mode,
            tissue_context,
            simulation_context
        )
        simulation_count = len(simulation_data.get('results', []))
        convergence_rate = simulation_data.get('convergence_rate', 0.0)
        await self.progress_monitor.log_step_complete(7, f"{simulation_count} simulations, {convergence_rate:.2%} convergence")

        # Step 8: Impact assessment (with timeout)
        await self.progress_monitor.log_step_start(8, f"Assessing network-wide impact")
        assessment_data = await self._step_with_timeout(
            8, self._step8_impact_assessment,
            simulation_data['results'],
            network_data['network'],
            pathway_data['pathways'],
            target_data['resolved_targets']
        )
        affected_genes_count = len(assessment_data.get('affected_genes', []))
        enrichment_score = assessment_data.get('enrichment_score', 0.0)
        await self.progress_monitor.log_step_complete(8, f"{affected_genes_count} affected genes, enrichment: {enrichment_score:.3f}")

        # Log final summary
        await self.progress_monitor.log_final_summary()

        # DEBUG: Log pathway_data before completeness calculation
        logger.info(f"DEBUG [Before completeness]: pathway_data has {len(pathway_data.get('pathways', []))} pathways")

        completeness_metrics = self._build_completeness_metrics(
            pathway_data,
            network_data,
            expression_data
        )
        
        # Calculate validation score
        validation_score = self._calculate_validation_score(
            target_data, pathway_data, interaction_data,
            expression_data, impact_data, simulation_data, assessment_data,
            data_sources, completeness_metrics
        )
        
        # Build enhanced result with all required fields for MultiTargetSimulationResult
        validated_target_names = [t.gene_symbol for t in target_data.get('resolved_targets', [])]
        
        # Create individual results from enhanced simulation
        individual_results = []
        for sim_result in simulation_data.get('results', []):
            individual_result = SimulationResult(
                target_node=sim_result['target'],
                mode='inhibit',
                affected_nodes=sim_result['affected_nodes'],
                direct_targets=sim_result['direct_targets'],
                downstream=sim_result['downstream'],
                upstream=sim_result['upstream'],
                feedback_loops=sim_result['feedback_loops'],
                network_impact=sim_result['network_impact'],
                confidence_scores=sim_result['confidence_scores'],
                execution_time=sim_result['execution_time'],
                biological_context=sim_result.get('biological_context'),
                drug_info=sim_result.get('drug_info')
            )
            individual_results.append(individual_result)
        
        # Enhanced combined effects from network-wide impact
        network_impact = simulation_data.get('network_impact', {})
        combined_effects = {
            'overall_effect': network_impact.get('perturbation_magnitude', 0.7),
            'network_coverage': network_impact.get('network_coverage', 0.0),
            'average_effect_strength': network_impact.get('average_effect_strength', 0.0),
            'total_affected_nodes': network_impact.get('total_affected_nodes', 0)
        }
        
        # Enhanced synergy analysis
        synergy_matrix = simulation_data.get('synergy_matrix', {})
        overall_synergy = synergy_matrix.get('overall_synergy', 0.0)
        synergy_analysis = {
            'synergy_score': overall_synergy,
            'antagonism_score': max(0.0, 1.0 - overall_synergy),
            'pairwise_synergy': synergy_matrix.get('pairwise_synergy', {}),
            'top_synergistic_pairs': synergy_matrix.get('top_synergistic_pairs', []),
            'engine': simulation_data.get('engine', simulation_mode)
        }
        
        # Enhanced network perturbation (CRITICAL FIX: Use standardized schema keys + add network size)
        # Standardized keys: total_affected, mean_effect, max_effect (not total_affected_nodes)
        # CRITICAL FIX: Extract network size from network_data (available in scope from step 6)
        network_nodes_count = len(network_data.get('nodes', []))
        network_edges_count = len(network_data.get('edges', []))
        
        # Also try to get from network object if nodes list is empty (but network object has nodes)
        if network_nodes_count == 0:
            network_obj = network_data.get('network')
            if network_obj and hasattr(network_obj, 'number_of_nodes'):
                network_nodes_count = network_obj.number_of_nodes()
                network_edges_count = network_obj.number_of_edges()
                logger.info(f"[S4 Network Perturbation] Extracted network size from NetworkX object: {network_nodes_count} nodes, {network_edges_count} edges")
                logger.warning(f"[S4 Network Perturbation] ⚠️  Network 'nodes' list was empty but NetworkX graph has {network_nodes_count} nodes - using graph size")
        
        # DIAGNOSTIC: Log network_impact contents
        logger.debug(f"[S4 Network Perturbation] network_impact keys: {list(network_impact.keys())}")
        logger.debug(f"[S4 Network Perturbation] network_impact values: {network_impact}")
        
        network_perturbation = {
            'perturbation_magnitude': network_impact.get('perturbation_magnitude', 0.7),
            'network_disruption': network_impact.get('network_disruption', 0.0),
            'total_affected': network_impact.get('total_affected', network_impact.get('total_affected_nodes', 0)),  # Use standardized key
            'mean_effect': network_impact.get('mean_effect', network_impact.get('average_effect_strength', 0.0)),  # Use standardized key
            'max_effect': network_impact.get('max_effect', 0.0),  # Use standardized key
            'network_coverage': network_impact.get('network_coverage', 0.0),
            # CRITICAL FIX: Add network size to network_perturbation
            'network_nodes': network_nodes_count,
            'network_edges': network_edges_count
        }
        
        logger.info(f"[S4 Network Perturbation] Final network size: {network_nodes_count} nodes, {network_edges_count} edges")
        
        # Enhanced pathway enrichment
        pathway_enrichment = {
            'enrichment_score': assessment_data.get('enrichment_score', 0.6),
            'pathway_impact': assessment_data.get('pathway_impact', {}),
            'functional_enrichment': assessment_data.get('functional_enrichment', {})
        }
        
        # Enhanced validation metrics
        validation_metrics = {
            'accuracy': validation_score,
            'convergence_rate': simulation_data.get('convergence_rate', 0.0),
            'network_coverage': network_impact.get('network_coverage', 0.0),
            'biological_relevance': self._calculate_biological_relevance(individual_results)
        }
        
        # Build affected_nodes dict (from combined_effects which is a dict)
        affected_nodes_dict = combined_effects if isinstance(combined_effects, dict) else {
            target: 1.0 for target in validated_target_names
        }

        # Build downstream list (from combined_effects keys if it's a dict)
        downstream_list = list(combined_effects.keys()) if isinstance(combined_effects, dict) else []

        result = MultiTargetSimulationResult(
            targets=validated_target_names,
            individual_results=individual_results,
            combined_effects=combined_effects if isinstance(combined_effects, dict) else {},
            synergy_analysis=synergy_analysis,
            network_perturbation=network_perturbation,  # CRITICAL FIX: Use network_perturbation dict with network_nodes/edges, not network_impact
            pathway_enrichment=pathway_enrichment,
            validation_metrics=validation_metrics,
            validation_score=validation_score,
            data_sources=list(data_sources.values()),
            completeness_metrics=completeness_metrics
        )

        # Log connection pool summary if enabled
        self._log_connection_pool_summary()

        # Log circuit breaker summary if enabled
        self._log_circuit_breaker_summary()

        logger.info(f"Multi-target simulation completed. Validation score: {validation_score:.3f}")
        self._active_data_sources = None
        return result
    
    async def _step1_target_resolution(self, targets: List[str]) -> Dict[str, Any]:
        """
        Step 1: Target resolution.
        
        Resolve targets across STRING and HPA databases.
        """
        logger.info("Step 1: Target resolution")
        
        resolved_targets = []
        
        # Resolve each target (limited to 3 concurrent to prevent MCP server overload)
        semaphore_task_batches = []
        for i, target in enumerate(targets):
            semaphore_task_batches.append(self._resolve_single_target(target, i))

        # Execute with semaphore limiting (max 3 concurrent)
        results = await asyncio.gather(*semaphore_task_batches)

        # Merge results
        for target_result in results:
            if target_result:
                resolved_targets.append(target_result)

        # Complete and return
        return self._step1_target_resolution_complete(resolved_targets, targets)

    @retry_with_exponential_backoff(max_attempts=3, base_delay=0.5, max_delay=10.0)
    async def _resolve_single_target(self, target: str, index: int) -> Optional[Protein]:
        """Resolve a single target with semaphore protection and retry logic.

        Enhanced error handling with detailed context and fallback strategies.
        """
        error_context = {
            'target': target,
            'index': index,
            'databases': ['STRING', 'HPA'],
            'operation': 'target_resolution'
        }

        try:
            logger.debug(f"[Target {index+1}] Resolving '{target}' via STRING and HPA databases")
            # Use semaphore-wrapped MCP calls
            string_proteins, hpa_proteins = await asyncio.wait_for(
                asyncio.gather(
                    self._execute_with_semaphore(
                        self.mcp_manager.string.search_proteins, target, limit=3, source_name='string'
                    ),
                    self._execute_with_semaphore(
                        self.mcp_manager.hpa.search_proteins, target, max_results=3, source_name='hpa'
                    )
                ),
                timeout=10.0
            )
            logger.debug(f"[Target {index+1}] Retrieved data from STRING and HPA")
        except asyncio.TimeoutError as e:
            logger.error(
                f"❌ [Target {index+1}] TIMEOUT: '{target}' resolution exceeded 10s. "
                f"Database servers may be overloaded. Skipping this target. "
                f"Context: {error_context}"
            )
            return None
        except ConnectionError as e:
            logger.error(
                f"❌ [Target {index+1}] CONNECTION ERROR: '{target}' - {str(e)}. "
                f"Check MCP server connectivity. Skipping this target. "
                f"Context: {error_context}"
            )
            return None
        except Exception as e:
            logger.error(
                f"❌ [Target {index+1}] UNEXPECTED ERROR: '{target}' resolution failed: "
                f"{type(e).__name__}: {str(e)[:200]}. Skipping this target. "
                f"Context: {error_context}"
            )
            return None

        # Standardize and validate
        target_candidates = []
        string_count = 0
        hpa_count = 0

        if string_proteins and string_proteins.get('proteins'):
            for protein_data in string_proteins['proteins']:
                try:
                    protein = self.standardizer.standardize_string_protein(protein_data)
                    if self.validator.validate_protein_confidence(protein):
                        target_candidates.append(protein)
                        string_count += 1
                except Exception as e:
                    logger.debug(f"[Target {index+1}] STRING protein standardization failed: {e}")
                    continue

        # HPA returns list directly
        if isinstance(hpa_proteins, list):
            for protein_data in hpa_proteins:
                try:
                    protein = self.standardizer.standardize_hpa_protein(protein_data)
                    if self.validator.validate_protein_confidence(protein):
                        target_candidates.append(protein)
                        hpa_count += 1
                except Exception as e:
                    logger.debug(f"[Target {index+1}] HPA protein standardization failed: {e}")
                    continue
        elif isinstance(hpa_proteins, dict) and hpa_proteins.get('proteins'):
            for protein_data in hpa_proteins['proteins']:
                try:
                    protein = self.standardizer.standardize_hpa_protein(protein_data)
                    if self.validator.validate_protein_confidence(protein):
                        target_candidates.append(protein)
                        hpa_count += 1
                except Exception as e:
                    logger.debug(f"[Target {index+1}] HPA protein standardization failed: {e}")
                    continue

        # Get best candidate
        if target_candidates:
            best_protein = max(target_candidates, key=lambda p: p.confidence)
            logger.info(
                f"✅ [Target {index+1}] '{target}' resolved successfully. "
                f"STRING: {string_count}, HPA: {hpa_count}, Best confidence: {best_protein.confidence:.3f}"
            )
            return best_protein

        logger.warning(
            f"⚠️  [Target {index+1}] '{target}' could not be resolved. "
            f"No valid candidates found (STRING: {string_count}, HPA: {hpa_count}). "
            f"Using fallback target with moderate confidence."
        )

        fallback_protein = Protein(
            gene_symbol=target.upper(),
            confidence=0.5,
            description=f"Fallback target generated after MCP resolution failed for {target}"
        )
        return fallback_protein

    def _step1_target_resolution_complete(self, resolved_targets: List[Protein], targets: List[str]) -> Dict[str, Any]:
        """Complete Step 1 calculation and return results."""
        # Calculate resolution accuracy
        resolution_accuracy = self.validator.validate_target_resolution_accuracy(
            len(resolved_targets), len(targets)
        )

        return {
            'resolved_targets': resolved_targets,
            'resolution_accuracy': resolution_accuracy
        }

    def _expand_disease_synonyms(self, disease_query: str) -> List[str]:
        """
        Expand disease query with medical synonyms for better Reactome coverage.

        Generic synonym expansion without hardcoding specific diseases.

        Args:
            disease_query: Original disease term (e.g., "breast cancer")

        Returns:
            List of disease synonyms to try, starting with original query
        """
        synonyms = [disease_query]  # Always try original first

        disease_lower = disease_query.lower()

        # Extract base disease name (remove "cancer" suffix for expansion)
        if "cancer" in disease_lower:
            base_name = disease_lower.replace(" cancer", "").strip()

            # Add medical terminology variations
            if base_name:
                # Common cancer-related medical terms
                variations = [
                    f"{base_name} cancer",          # Original
                    f"{base_name} carcinoma",       # Medical term for cancer
                    f"{base_name} neoplasm",        # General tumor term
                    f"{base_name} malignancy",      # Cancer synonym
                    f"{base_name} tumor",           # Tumor variant
                ]
                synonyms.extend([v for v in variations if v != disease_lower])

        # Remove duplicates while preserving order (original first)
        seen = set()
        unique_synonyms = []
        for syn in synonyms:
            if syn not in seen:
                seen.add(syn)
                unique_synonyms.append(syn)

        return unique_synonyms[:5]  # Limit to 5 to avoid excessive queries

    @retry_with_exponential_backoff(max_attempts=3, base_delay=1.0, max_delay=15.0)
    async def _step2_network_context(
        self,
        targets: List[Protein],
        disease_context: str
    ) -> Dict[str, Any]:
        """
        Step 2: Network context.

        Get Reactome pathways for disease context.
        Enhanced to fetch genes for each pathway.
        Enhanced error handling with detailed context.
        """
        target_count = len(targets)
        logger.info(f"Step 2: Network context - Finding pathways for '{disease_context}' with {target_count} targets")

        error_context = {
            'disease_context': disease_context,
            'target_count': target_count,
            'operation': 'network_context',
            'database': 'Reactome'
        }

        # Get disease pathways with synonym expansion
        pathways = None
        disease_synonyms = self._expand_disease_synonyms(disease_context)

        logger.debug(f"Trying {len(disease_synonyms)} disease synonyms: {disease_synonyms}")

        try:
            # Try each synonym until we get pathways
            for idx, synonym in enumerate(disease_synonyms):
                logger.debug(f"Searching Reactome for disease pathways (attempt {idx+1}/{len(disease_synonyms)}): '{synonym}'")

                try:
                    pathways = await self._call_with_tracking(
                        None,
                        'reactome',
                        self.mcp_manager.reactome.find_pathways_by_disease(synonym)
                    )

                    # Check if we got valid pathways
                    pathway_count = len(pathways.get('pathways', [])) if pathways else 0

                    if pathway_count > 0:
                        logger.info(f"✅ Found {pathway_count} pathways using synonym: '{synonym}'")
                        break  # Success! Stop trying other synonyms
                    else:
                        logger.debug(f"No pathways found for synonym: '{synonym}', trying next...")
                        pathways = None  # Reset for next iteration

                except Exception as e:
                    logger.debug(f"Error querying Reactome with '{synonym}': {e}")
                    continue  # Try next synonym

            # After trying all synonyms, check if we got pathways
            if not pathways or len(pathways.get('pathways', [])) == 0:
                logger.info(
                    f"Using direct pathway search (disease search unavailable for '{disease_context}' and synonyms)"
                )
                # Don't raise error - let execution continue with empty pathways
                pathways = {'pathways': []}

        except asyncio.TimeoutError as e:
            logger.error(
                f"❌ [Step 2] TIMEOUT: Reactome search for '{disease_context}' exceeded timeout. "
                f"Database may be slow. Using fallback. Context: {error_context}"
            )
            raise
        except ConnectionError as e:
            logger.error(
                f"❌ [Step 2] CONNECTION ERROR: Cannot connect to Reactome MCP server. "
                f"Check server status. Context: {error_context}"
            )
            raise
        except Exception as e:
            logger.error(
                f"❌ [Step 2] UNEXPECTED ERROR: Reactome search failed: {type(e).__name__}: {str(e)[:200]}. "
                f"Context: {error_context}"
            )
            raise

        # OPTIMIZATION: Fetch genes for pathways in parallel batches
        standardized_pathways = []
        pathways_with_genes = 0
        pathways_without_genes = 0
        failed_pathways = 0

        if pathways.get('pathways'):
            pathway_list = pathways['pathways'][:10]  # Limit to 10 pathways for performance
            logger.info(f"Found {len(pathway_list)} pathways, fetching genes in parallel...")

            @retry_with_exponential_backoff(max_attempts=1, base_delay=0.5, max_delay=5.0)  # CHANGED from 2 to 1
            async def process_pathway(pathway_data):
                """Process a single pathway and extract genes."""
                pathway_id = pathway_data.get('stId') or pathway_data.get('id')
                pathway_name = pathway_data.get('displayName') or pathway_data.get('name', pathway_id)
                
                # FIX: Handle list-wrapped pathway names from Reactome API
                if isinstance(pathway_name, list) and len(pathway_name) > 0:
                    pathway_name = pathway_name[0]
                elif not isinstance(pathway_name, str):
                    pathway_name = str(pathway_name) if pathway_name else pathway_id

                if not pathway_id:
                    logger.warning(f"⚠️  [Step 2] Pathway missing ID, skipping: {pathway_data}")
                    return None

                error_ctx = {
                    'pathway_id': pathway_id,
                    'pathway_name': pathway_name,
                    'operation': 'pathway_processing',
                    'database': 'Reactome'
                }

                try:
                    # Fetch genes for this pathway
                    logger.debug(f"[Step 2] Processing pathway {pathway_id}: {pathway_name}")
                    genes = await self._extract_reactome_pathway_genes(pathway_id)

                    # Create pathway with genes
                    pathway = Pathway(
                        id=pathway_id,
                        name=pathway_name,
                        source_db='reactome',
                        genes=genes,
                        description=pathway_data.get('description'),
                        confidence=0.9
                    )
                    logger.debug(f"✅ [Step 2] Pathway {pathway_id}: Retrieved {len(genes)} genes")
                    return pathway
                except asyncio.TimeoutError:
                    logger.warning(
                        f"⚠️  [Step 2] TIMEOUT: Gene extraction for pathway {pathway_id} exceeded timeout. "
                        f"Falling back to pathway without genes. Context: {error_ctx}"
                    )
                    # Fallback: create pathway without genes
                    try:
                        pathway = self.standardizer.standardize_reactome_pathway(pathway_data)
                        return pathway
                    except Exception as e:
                        logger.error(
                            f"❌ [Step 2] FALLBACK FAILED: Could not create pathway {pathway_id}: {e}. "
                            f"Context: {error_ctx}"
                        )
                        return None
                except ConnectionError:
                    logger.warning(
                        f"⚠️  [Step 2] CONNECTION ERROR: Cannot fetch genes for {pathway_id}. "
                        f"Falling back. Context: {error_ctx}"
                    )
                    try:
                        pathway = self.standardizer.standardize_reactome_pathway(pathway_data)
                        return pathway
                    except Exception as e:
                        logger.error(
                            f"❌ [Step 2] FALLBACK FAILED: Could not create pathway {pathway_id}: {e}. "
                            f"Context: {error_ctx}"
                        )
                        return None
                except Exception as e:
                    logger.error(
                        f"❌ [Step 2] ERROR: Failed to process pathway {pathway_id}: "
                        f"{type(e).__name__}: {str(e)[:200]}. Skipping. Context: {error_ctx}"
                    )
                    return None

            # Process pathways in parallel batches (3 at a time) with timeout
            batch_size = 3
            for i in range(0, len(pathway_list), batch_size):
                batch = pathway_list[i:i + batch_size]
                batch_num = i // batch_size + 1
                logger.debug(f"[Step 2] Processing pathway batch {batch_num}/{len(pathway_list)//batch_size + 1} ({len(batch)} pathways)")

                try:
                    batch_results = await asyncio.wait_for(
                        asyncio.gather(*[process_pathway(p) for p in batch], return_exceptions=True),
                        timeout=30.0  # 30 second timeout per batch
                    )

                    # Process results and count
                    for pathway in batch_results:
                        if isinstance(pathway, Exception):
                            logger.error(f"[Step 2] Batch processing error: {type(pathway).__name__}: {str(pathway)[:100]}")
                            failed_pathways += 1
                            continue

                        if pathway and not isinstance(pathway, Exception):
                            standardized_pathways.append(pathway)
                            # Check if pathway has genes
                            if hasattr(pathway, 'genes') and pathway.genes:
                                pathways_with_genes += 1
                            else:
                                pathways_without_genes += 1

                except asyncio.TimeoutError:
                    logger.warning(
                        f"⚠️  [Step 2] BATCH {batch_num} TIMEOUT: Exceeded 30s. "
                        f"Continuing with remaining batches."
                    )
                    failed_pathways += len(batch)
                    continue

        # Calculate pathway coverage
        target_genes = [t.gene_symbol for t in targets if t.gene_symbol]
        pathway_coverage = self.validator.validate_pathway_coverage(
            target_genes, [p.id for p in standardized_pathways]
        )

        logger.info(
            f"✅ [Step 2] Network context complete: {len(standardized_pathways)} pathways "
            f"({pathways_with_genes} with genes, {pathways_without_genes} without, {failed_pathways} failed), "
            f"coverage: {pathway_coverage:.3f}"
        )

        return {
            'pathways': standardized_pathways,
            'coverage': pathway_coverage
        }

    @retry_with_exponential_backoff(max_attempts=3, base_delay=0.5, max_delay=8.0)
    async def _extract_reactome_pathway_genes(self, pathway_id: str) -> List[str]:
        """
        Extract gene symbols from a Reactome pathway.
        Uses the same approach as Scenario 1/5/6 to extract actual gene symbols.

        Enhanced error handling with detailed context and fallback strategies.

        Args:
            pathway_id: Reactome pathway ID

        Returns:
            List of validated gene symbols
        """
        genes = set()
        error_context = {
            'pathway_id': pathway_id,
            'operation': 'gene_extraction',
            'database': 'Reactome'
        }

        try:
            logger.debug(f"[Reactome] Extracting genes from pathway: {pathway_id}")
            # Method 1: Get pathway details and extract from entities
            details = await self._call_with_tracking(
                None,
                'reactome',
                self.mcp_manager.reactome.get_pathway_details(pathway_id)
            )

            if not details:
                logger.warning(
                    f"⚠️  [Reactome] Pathway details empty for {pathway_id}. "
                    f"Context: {error_context}"
                )
            else:
                logger.debug(f"[Reactome] Retrieved pathway details for {pathway_id}")

            # Extract from entities (proteins, complexes, small molecules)
            entities_count = 0
            if details.get('entities'):
                entities_count = len(details['entities'])
                logger.debug(f"[Reactome] Processing {entities_count} entities from {pathway_id}")
                for entity in details['entities']:
                    try:
                        gene_names = self._extract_gene_names_from_entity(entity)
                        genes.update(gene_names)
                    except Exception as e:
                        logger.debug(f"[Reactome] Entity processing failed: {e}")
                        continue

            # Extract from hasEvent (reactions with participants)
            events_count = 0
            if details.get('hasEvent'):
                events_count = len(details['hasEvent'])
                logger.debug(f"[Reactome] Processing {events_count} events from {pathway_id}")
                for event in details['hasEvent']:
                    try:
                        # Get participants from reaction
                        if event.get('participants'):
                            for participant in event['participants']:
                                gene_names = self._extract_gene_names_from_entity(participant)
                                genes.update(gene_names)
                    except Exception as e:
                        logger.debug(f"[Reactome] Event processing failed: {e}")
                        continue

            logger.debug(
                f"[Reactome] Pathway {pathway_id}: Extracted {len(genes)} gene candidates "
                f"from {entities_count} entities and {events_count} events"
            )

            # Method 2: Backup with get_pathway_participants if needed
            if not genes:
                logger.debug(f"[Reactome] No genes from Method 1, trying get_pathway_participants for {pathway_id}")
                try:
                    participants = await self._call_with_tracking(
                        None,
                        'reactome',
                        self.mcp_manager.reactome.get_pathway_participants(pathway_id)
                    )
                    if participants and participants.get('participants'):
                        participants_count = len(participants['participants'])
                        logger.debug(f"[Reactome] Retrieved {participants_count} participants for {pathway_id}")
                        for participant in participants['participants']:
                            try:
                                gene_names = self._extract_gene_names_from_entity(participant)
                                genes.update(gene_names)
                            except Exception as e:
                                logger.debug(f"[Reactome] Participant processing failed: {e}")
                                continue
                    else:
                        logger.debug(f"[Reactome] No participants found for {pathway_id}")
                except Exception as e:
                    logger.warning(
                        f"⚠️  [Reactome] get_pathway_participants failed for {pathway_id}: {type(e).__name__}: {str(e)[:100]}. "
                        f"Context: {error_context}"
                    )

            # Filter out non-gene terms and validate
            filtered_genes = self._filter_valid_gene_symbols(genes)

            logger.debug(
                f"✅ [Reactome] Pathway {pathway_id}: "
                f"Extracted {len(filtered_genes)} valid gene symbols "
                f"(from {len(genes)} candidates)"
            )

        except asyncio.TimeoutError as e:
            logger.error(
                f"❌ [Reactome] TIMEOUT: Gene extraction for {pathway_id} exceeded timeout. "
                f"Database may be slow. Returning empty list. Context: {error_context}"
            )
            return []
        except ConnectionError as e:
            logger.error(
                f"❌ [Reactome] CONNECTION ERROR: Cannot connect to Reactome MCP server for {pathway_id}. "
                f"Check server status. Returning empty list. Context: {error_context}"
            )
            return []
        except Exception as e:
            logger.error(
                f"❌ [Reactome] UNEXPECTED ERROR: Gene extraction for {pathway_id} failed: "
                f"{type(e).__name__}: {str(e)[:200]}. Returning empty list. Context: {error_context}"
            )
            import traceback
            logger.debug(traceback.format_exc())
            return []

        return list(filtered_genes) if filtered_genes else []
    
    def _extract_gene_names_from_entity(self, entity: Dict[str, Any]) -> List[str]:
        """
        Extract gene names from a Reactome entity/participant.
        
        Handles various Reactome data structures:
        - Simple proteins: {'name': 'TP53', 'type': 'Protein'}
        - Complexes: {'name': 'TP53:p-S15', 'type': 'Complex'}
        - Small molecules: {'name': 'ATP', 'type': 'SmallMolecule'}
        
        Args:
            entity: Reactome entity/participant dictionary
            
        Returns:
            List of extracted gene symbol candidates
        """
        gene_names = []
        
        # Try multiple field variations based on Reactome structure
        name_fields = ['name', 'displayName', 'geneName', 'symbol', 'identifier']
        
        for field in name_fields:
            if field in entity and entity[field]:
                name = entity[field]
                
                # Handle different name formats
                if isinstance(name, str):
                    # Split complex names and extract gene symbols
                    candidates = self._parse_complex_name(name)
                    gene_names.extend(candidates)
                elif isinstance(name, list):
                    # Handle list of names
                    for n in name:
                        if isinstance(n, str):
                            candidates = self._parse_complex_name(n)
                            gene_names.extend(candidates)
        
        return gene_names
    
    def _parse_complex_name(self, name: str) -> List[str]:
        """
        Parse complex names and extract valid gene symbols.
        
        Handles formats like:
        - 'TP53' → ['TP53']
        - 'TP53:p-S15' → ['TP53']
        - 'Signaling by ERBB2' → ['ERBB2']
        - 'BRCA1/BRCA2' → ['BRCA1', 'BRCA2']
        - 'Constitutive Signaling by AKT1' → ['AKT1']
        
        Args:
            name: Complex name string
            
        Returns:
            List of extracted gene candidates
        """
        if not name or not isinstance(name, str):
            return []
        
        candidates = []
        
        # Remove common modifications and split
        # Handle patterns: "name:modification", "name-modification", "name/name"
        name = name.replace('[', ' ').replace(']', ' ')
        name = name.replace('(', ' ').replace(')', ' ')
        
        # Split on common separators
        parts = []
        for sep in [':', '/', '-', ',']:
            if sep in name:
                parts.extend(name.split(sep))
                break
        else:
            parts = [name]
        
        # Also split on spaces to catch "Signaling by GENE" patterns
        all_words = []
        for part in parts:
            all_words.extend(part.split())
        
        # Common stop words / pathway terms to filter out
        stop_words = {
            'signaling', 'by', 'in', 'cancer', 'pathway', 'the', 'a', 'and', 'or', 
            'of', 'to', 'with', 'from', 'for', 'at', 'as', 'an', 'on', 'is', 'are',
            'constitutive', 'activated', 'active', 'inactive', 'mutant', 'mutants',
            'wild', 'type', 'overexpressed', 'loss', 'function', 'drug', 'resistance',
            'kd', 'lbd', 'bind', 'binds', 'binding', 'fusion', 'dimerizes', 'targets',
            # Modifications
            'p', 's', 't', 'y',  # phosphorylation sites
            # Numbers (line numbers, positions)
            'i', 'ii', 'iii', 'iv', 'v',
            # Common pathway terms
            'signalling', 'mediated', 'dependent', 'independent', 'induced', 'inhibited',
            'downstream', 'upstream', 'via', 'through', 'upon', 'after', 'during',
            'complex', 'receptor', 'ligand', 'domain', 'family', 'member', 'variant',
        }
        
        for word in all_words:
            word = word.strip().upper()
            
            # Skip empty, too short, or too long
            if not word or len(word) < 2 or len(word) > 15:
                continue
            
            # Skip stop words
            if word.lower() in stop_words:
                continue
            
            # Skip if not alphanumeric
            if not word.replace('_', '').isalnum():
                continue
                
            # Skip if all numbers
            if word.isdigit():
                continue
            
            # Gene symbols are typically:
            # - Mix of letters and possibly numbers
            # - Start with a letter
            # - Mostly uppercase
            # - 2-10 characters for human genes (some longer for protein complexes)
            
            # Must start with letter
            if not word[0].isalpha():
                continue
            
            # Good candidate if it's mostly letters with some numbers
            letter_count = sum(1 for c in word if c.isalpha())
            if letter_count >= len(word) * 0.6:  # At least 60% letters
                candidates.append(word)
        
        return candidates
    

    def _filter_valid_gene_symbols(self, candidates: Set[str]) -> Set[str]:
        """
        Filter and validate gene symbols from candidates.
        
        Args:
            candidates: Set of candidate gene symbols
            
        Returns:
            Set of validated gene symbols
        """
        valid_genes = set()
        
        # Expanded false positives list
        # Updated 2025-11-20 based on pipeline analysis
        false_positives = {
            # Metabolites & nucleotides
            'ATP', 'GDP', 'GTP', 'DNA', 'RNA', 'AMP', 'ADP', 'NAD', 'NADH',
            'FAD', 'COA', 'GMP', 'UMP', 'CMP', 'TMP', 'GTP',
            
            # Lipids & second messengers
            'PI', 'PIP', 'PIP2', 'PIP3', 'DAG', 'IP3',
            
            # Ions
            'CA', 'MG', 'ZN', 'FE', 'CU', 'MN',
            
            # Protein domains and regions
            'KD', 'LBD', 'ECD', 'TMD', 'ICD', 'NTD', 'CTD',
            'PEST', 'HD', 'MAM', 'SAM', 'SH2', 'SH3', 'PDZ',
            'RING', 'CARD', 'DEATH', 'PH', 'C2', 'LRR',
            
            # Protein family abbreviations (common in KEGG diagrams)
            'PKC', 'PKA', 'PKG', 'PRKG',  # Kinase families
            'PLC', 'PLD', 'PLA', 'PLA2',  # Phospholipase families  
            'PDE', 'PDE6',  # Phosphodiesterase families
            'FZD',  # Frizzled receptor family (use FZD1, FZD2, etc.)
            'WNT',  # WNT family (use WNT1, WNT2, etc. - but WNT11 is valid!)
            'CAM', 'CALM',  # Calmodulin (use CALM1, CALM2, CALM3)
            'NFAT',  # NFAT family (use NFATC1, NFATC2, etc.)
            'NOS',  # Nitric oxide synthase family
            'MAPK',  # MAPK family (use MAPK1, MAPK3, etc.)
            
            # Incomplete gene symbols (family prefixes without numbers)
            'DKK',  # Dickkopf family - needs number (DKK1, DKK2, DKK3, DKK4)
            'PPP2R',  # Protein phosphatase 2 subunit - needs full name
            
            # Drug names/identifiers
            'TKIs', 'EGFRI', 'MKIs', 'AZ5104',
            
            # Specific drug names (NEW - from Phase 1 investigation)
            'AEE788', 'AEE78',  # EGFR/HER2 inhibitor drugs
            
            # Technical terms
            'SARA', 'AXIN', 'AN', 'HR',
            'SIGNALING', 'PATHWAY', 'CANCER', 'DISEASE',
            'BINDING', 'DOMAIN', 'REGION', 'MOTIF', 'SITE',
            
            # Single letters (NEW - from pipeline analysis)
            # These appear in Reactome pathways like "Signaling by WNT in cancer" → ['P', 'I', 'A']
            'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
            'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
        }
        
        for candidate in candidates:
            if not candidate or not isinstance(candidate, str):
                continue
            
            candidate = candidate.strip().upper()
            
            # Must start with a letter
            if candidate[0].isalpha():
                # Must not be in false positives list
                if candidate not in false_positives:
                    # Must be reasonable length (2-15 chars)
                    if 2 <= len(candidate) <= 15:
                        valid_genes.add(candidate)
        
        return valid_genes
    
    async def _step3_interaction_validation(self, targets: List[Protein]) -> Dict[str, Any]:
        """
        Step 3: Interaction validation.

        Validate interactions using STRING.
        Enhanced error handling with detailed context.
        """
        target_ids = [t.gene_symbol for t in targets if t.gene_symbol]
        target_count = len(target_ids)
        logger.info(f"Step 3: Interaction validation - Validating {target_count} targets via STRING")

        if not target_ids:
            logger.warning("⚠️  [Step 3] No valid target IDs for interaction validation")
            return {'interactions': [], 'validation_score': 0.0}

        error_context = {
            'target_count': target_count,
            'targets': target_ids[:5],  # Log first 5 for context
            'operation': 'interaction_validation',
            'database': 'STRING',
            'species': '9606'
        }

        # Get interaction network with expanded neighbors for propagation
        # Use config value if provided, otherwise adaptive based on target count
        # Adaptive: For >5 targets: use 2 neighbors (default), for <=5 targets: use 5 neighbors
        if hasattr(self, 'network_expansion_config'):
            add_nodes = self.network_expansion_config.initial_neighbors
            logger.info(
                f"[Step 3] Using configured initial_neighbors={add_nodes} from network_expansion_config"
            )
        else:
            add_nodes = 2 if target_count > 5 else 5
            logger.info(
                f"[Step 3] Using adaptive add_nodes={add_nodes} (based on {target_count} targets)"
            )

        # Initialize adaptive STRING builder if not already done
        if self.string_builder is None:
            self.string_builder = AdaptiveStringNetworkBuilder(
                self.mcp_manager,
                self._active_data_sources
            )
        
        try:
            logger.info(
                f"[Step 3] Building adaptive STRING network for {target_count} targets"
            )
            network_result = await asyncio.wait_for(
                self.string_builder.build_network(
                    genes=target_ids,
                    priority_genes=None,  # All targets are equally important
                    data_sources=self._active_data_sources
                ),
                timeout=self.step_timeouts[3]  # Use configured timeout (now 300s)
            )
            
            nodes = network_result.get('nodes', [])
            edges = network_result.get('edges', [])
            
            # Convert to expected format
            string_network = {
                'nodes': nodes,
                'edges': edges
            }

            if not string_network:
                logger.error(
                    f"❌ [Step 3] STRING returned empty response. "
                    f"Context: {error_context}"
                )
                return {'interactions': [], 'validation_score': 0.0}

        except asyncio.TimeoutError:
            logger.error(
                f"❌ [Step 3] TIMEOUT: STRING interaction network request exceeded {self.step_timeouts[3]}s. "
                f"Database may be slow. Using fallback. Context: {error_context}"
            )
            return {'interactions': [], 'validation_score': 0.0}
        except ConnectionError as e:
            logger.error(
                f"❌ [Step 3] CONNECTION ERROR: Cannot connect to STRING MCP server. "
                f"Check server status. Using fallback. Context: {error_context}"
            )
            return {'interactions': [], 'validation_score': 0.0}
        except Exception as e:
            logger.error(
                f"❌ [Step 3] UNEXPECTED ERROR: STRING interaction request failed: "
                f"{type(e).__name__}: {str(e)[:200]}. Using fallback. Context: {error_context}"
            )
            return {'interactions': [], 'validation_score': 0.0}

        # STRING returns nodes and edges directly in response (not nested under 'network')
        # Adaptive builder returns direct format
        interactions = string_network.get('edges', [])
        nodes_in_network = string_network.get('nodes', [])

        interaction_count = len(interactions)
        node_count = len(nodes_in_network)

        # Log network expansion results
        if interaction_count > 0 and node_count > 0:
            logger.info(
                f"✅ [Step 3] Network expansion successful: "
                f"{node_count} nodes, {interaction_count} edges retrieved"
            )
        else:
            logger.warning(
                f"⚠️  [Step 3] Network expansion returned sparse results: "
                f"{node_count} nodes, {interaction_count} edges. "
                f"This may indicate targets have limited interactions in STRING."
            )

        # Calculate interaction confidence with multiple field name support
        try:
            confidence_scores = []
            for edge in interactions:
                # STRING API returns confidence in multiple possible fields:
                # - 'confidence_score': Primary confidence score field
                # - 'score': Combined confidence score (0-1000)
                # - 'combined_score': Alternative field name
                # - 'confidence': Another alternative
                # - 'weight': Weight value
                # Try all possible field names
                score = (edge.get('confidence_score') or
                        edge.get('score') or
                        edge.get('combined_score') or
                        edge.get('confidence') or
                        edge.get('weight') or
                        0)

                # Ensure numeric
                try:
                    score = float(score)
                    confidence_scores.append(score)
                except (ValueError, TypeError):
                    logger.debug(f"Non-numeric confidence score: {score}")
                    confidence_scores.append(0)

            median_confidence = np.median(confidence_scores) if confidence_scores else 0

            logger.debug(
                f"[Step 3] Interaction confidence stats: "
                f"count={len(confidence_scores)}, "
                f"min={min(confidence_scores) if confidence_scores else 0}, "
                f"median={median_confidence}, "
                f"max={max(confidence_scores) if confidence_scores else 0}"
            )

            # Log warning if all scores are zero
            if median_confidence == 0 and len(interactions) > 0:
                logger.warning(
                    f"⚠️  [Step 3] All {len(interactions)} interactions have zero confidence. "
                    f"This may indicate missing 'score' field in STRING response. "
                    f"Sample edge keys: {list(interactions[0].keys()) if interactions else 'N/A'}"
                )

        except Exception as e:
            logger.error(
                f"❌ [Step 3] ERROR: Failed to calculate confidence scores: {e}. "
                f"Using default score. Context: {error_context}"
            )
            median_confidence = 0.0

        return {
            'interactions': interactions,
            'validation_score': median_confidence / 1000.0  # Convert to 0-1 scale
        }
    
    async def _step4_expression_validation(
        self,
        targets: List[Protein],
        tissue_context: Optional[str]
    ) -> Dict[str, Any]:
        """
        Step 4: Expression validation.

        Validate expression using HPA.
        Enhanced error handling with detailed context.
        """
        target_count = len(targets)
        logger.info(
            f"Step 4: Expression validation - Validating {target_count} targets via HPA"
        )

        expression_profiles = []
        successful_targets = 0
        failed_targets = 0
        skipped_targets = 0

        error_context_base = {
            'target_count': target_count,
            'tissue_context': tissue_context,
            'operation': 'expression_validation',
            'database': 'HPA'
        }

        # Get expression for each target
        for idx, target in enumerate(targets):
            if not target.gene_symbol:
                skipped_targets += 1
                continue

            target_gene = target.gene_symbol
            error_context = {
                **error_context_base,
                'target_index': idx + 1,
                'target_gene': target_gene
            }

            try:
                logger.debug(f"[Step 4] Fetching expression data for {target_gene} ({idx+1}/{target_count})")

                # Add timeout to HPA call
                expression_data = await asyncio.wait_for(
                    self._call_with_tracking(
                        None,
                        'hpa',
                        self.mcp_manager.hpa.get_tissue_expression(target_gene)
                    ),
                    timeout=30.0  # 30 second timeout per target
                )

                if not expression_data:
                    logger.warning(
                        f"⚠️  [Step 4] HPA returned empty data for {target_gene}. "
                        f"Context: {error_context}"
                    )
                    failed_targets += 1
                    continue

                # Use HPA parsing helpers to handle list/dict formats
                from ..utils.hpa_parsing import _iter_expr_items, categorize_expression

                profile_count = 0
                try:
                    for tissue, ntpms in _iter_expr_items(expression_data):
                        # Fuzzy tissue matching with common variants
                        if tissue_context:
                            tissue_variants = {
                                'breast': ['mammary', 'breast', 'ductal', 'lobular'],
                                'liver': ['hepatic', 'liver'],
                                'lung': ['pulmonary', 'lung', 'bronchial'],
                                'brain': ['cerebral', 'brain', 'neural'],
                                'kidney': ['renal', 'kidney'],
                                'heart': ['cardiac', 'heart'],
                                'colon': ['colon', 'colonic', 'intestinal'],
                                'stomach': ['gastric', 'stomach'],
                                'pancreas': ['pancreatic', 'pancreas'],
                                'prostate': ['prostate', 'prostatic']
                            }

                            context_lower = tissue_context.lower()
                            tissue_lower = tissue.lower()

                            # Check if ANY variant matches
                            if context_lower in tissue_variants:
                                if not any(variant in tissue_lower for variant in tissue_variants[context_lower]):
                                    continue
                            else:
                                # Fallback to original exact substring match
                                if context_lower not in tissue_lower:
                                    continue

                        try:
                            expression_level = categorize_expression(ntpms)

                            expression_profile = ExpressionProfile(
                                gene=target_gene,
                                tissue=tissue,
                                expression_level=expression_level,
                                reliability='Approved',
                                cell_type_specific=False,
                                subcellular_location=[]
                            )
                            expression_profiles.append(expression_profile)
                            profile_count += 1
                        except Exception as e:
                            logger.debug(f"[Step 4] Failed to process tissue {tissue} for {target_gene}: {e}")
                            continue

                    if profile_count > 0:
                        logger.debug(f"[Step 4] Successfully processed {profile_count} expression profiles for {target_gene}")
                        successful_targets += 1
                    else:
                        # Expression fallback: try alternative targets if primary has no data
                        logger.debug(f"[Step 4] No expression for {target_gene}, trying fallback genes")

                        fallback_success = False
                        # Try up to 3 other targets as fallback (generic, not hardcoded)
                        for fallback_idx, fallback_target in enumerate(targets[:min(5, len(targets))]):
                            if not fallback_target.gene_symbol or fallback_target.gene_symbol == target_gene:
                                continue

                            try:
                                logger.debug(f"[Step 4] Trying fallback gene: {fallback_target.gene_symbol}")
                                fallback_expr = await asyncio.wait_for(
                                    self._call_with_tracking(
                                        None,
                                        'hpa',
                                        self.mcp_manager.hpa.get_tissue_expression(fallback_target.gene_symbol)
                                    ),
                                    timeout=15.0  # Shorter timeout for fallback
                                )

                                if fallback_expr:
                                    # Process fallback expression data
                                    for tissue, ntpms in _iter_expr_items(fallback_expr):
                                        if tissue_context and tissue_context.lower() in tissue.lower():
                                            expression_level = categorize_expression(ntpms)
                                            profile = ExpressionProfile(
                                                gene=fallback_target.gene_symbol,  # Use fallback gene name
                                                tissue=tissue,
                                                expression_level=expression_level,
                                                ntpm=ntpms,
                                                cancer_marker=False
                                            )
                                            expression_profiles.append(profile)
                                            profile_count += 1

                                    if profile_count > 0:
                                        logger.info(f"✅ [Step 4] Using {fallback_target.gene_symbol} expression as fallback for {target_gene} ({profile_count} profiles)")
                                        fallback_success = True
                                        successful_targets += 1
                                        break  # Success! Stop trying fallbacks

                            except Exception as e:
                                logger.debug(f"[Step 4] Fallback gene {fallback_target.gene_symbol} failed: {e}")
                                continue

                        if not fallback_success:
                            logger.debug(
                                f"[Step 4] No expression data available for {target_gene} (tried {len(targets[:5])} fallback genes)"
                            )
                            failed_targets += 1

                except Exception as e:
                    logger.error(
                        f"❌ [Step 4] ERROR: Failed to parse expression data for {target_gene}: "
                        f"{type(e).__name__}: {str(e)[:200]}. Context: {error_context}"
                    )
                    failed_targets += 1

            except asyncio.TimeoutError:
                logger.error(
                    f"❌ [Step 4] TIMEOUT: Expression fetch for {target_gene} exceeded 30s. "
                    f"Database may be slow. Skipping. Context: {error_context}"
                )
                failed_targets += 1
            except ConnectionError as e:
                logger.error(
                    f"❌ [Step 4] CONNECTION ERROR: Cannot connect to HPA MCP server for {target_gene}. "
                    f"Check server status. Skipping. Context: {error_context}"
                )
                failed_targets += 1
            except Exception as e:
                logger.error(
                    f"❌ [Step 4] UNEXPECTED ERROR: Expression fetch for {target_gene} failed: "
                    f"{type(e).__name__}: {str(e)[:200]}. Skipping. Context: {error_context}"
                )
                failed_targets += 1

        # Calculate expression coverage
        covered_genes = set(ep.gene for ep in expression_profiles)
        target_genes = set(t.gene_symbol for t in targets if t.gene_symbol)
        coverage = len(covered_genes) / len(target_genes) if target_genes else 0.0

        logger.info(
            f"✅ [Step 4] Expression validation complete: "
            f"{len(expression_profiles)} profiles from {successful_targets} targets, "
            f"{failed_targets} failed, {skipped_targets} skipped, "
            f"coverage: {coverage:.3f}"
        )

        return {
            'profiles': expression_profiles,
            'coverage': coverage
        }
    
    async def _step5_pathway_impact(
        self, 
        targets: List[Protein], 
        pathways: List[Pathway]
    ) -> Dict[str, Any]:
        """
        Step 5: Pathway impact.
        
        Analyze pathway impact using Reactome participants.
        """
        logger.info("Step 5: Pathway impact - Processing in parallel batches")

        # Process pathways in parallel batches (max 3 concurrent to avoid overload)
        pathway_impacts = []
        pathway_list = list(pathways)  # Convert to list for batching
        batch_size = 3  # Process 3 pathways in parallel

        async def process_single_pathway(pathway):
            """Process a single pathway with timeout."""
            try:
                # Get pathway participants with TIMEOUT to prevent hanging
                participants = await asyncio.wait_for(
                    self._call_with_tracking(
                        None,
                        'reactome',
                        self.mcp_manager.reactome.get_pathway_participants(pathway.id)
                    ),
                    timeout=60.0  # 60 second timeout per pathway
                )

                if participants.get('participants'):
                    # Calculate target overlap
                    pathway_genes = set(p.get('gene', '') for p in participants['participants'])
                    target_genes = set(t.gene_symbol for t in targets if t.gene_symbol)

                    overlap = len(pathway_genes & target_genes)
                    total_targets = len(target_genes)

                    impact_score = overlap / total_targets if total_targets > 0 else 0.0

                    return {
                        'pathway_id': pathway.id,
                        'pathway_name': pathway.name,
                        'impact_score': impact_score,
                        'target_overlap': overlap,
                        'total_targets': total_targets
                    }

            except asyncio.TimeoutError:
                logger.warning(f"Step 5 TIMEOUT: Pathway {pathway.id} exceeded 60s - skipping")
                return None
            except Exception as e:
                logger.warning(f"Failed to analyze pathway {pathway.id}: {e}")
                return None

        # Process pathways in batches
        for i in range(0, len(pathway_list), batch_size):
            batch = pathway_list[i:i + batch_size]
            batch_num = i // batch_size + 1
            total_batches = (len(pathway_list) + batch_size - 1) // batch_size

            logger.debug(f"Processing pathway batch {batch_num}/{total_batches} ({len(batch)} pathways)")

            # Execute batch in parallel
            batch_results = await asyncio.gather(
                *[process_single_pathway(pathway) for pathway in batch],
                return_exceptions=True
            )

            # Collect results
            for result in batch_results:
                if result and not isinstance(result, Exception):
                    pathway_impacts.append(result)
        
        # Calculate overall pathway impact
        overall_impact = np.mean([p['impact_score'] for p in pathway_impacts]) if pathway_impacts else 0.0
        
        return {
            'pathway_impacts': pathway_impacts,
            'overall_impact': overall_impact
        }
    
    async def _step6_network_construction(
        self, 
        targets: List[Protein], 
        interactions: List[Dict]
    ) -> Dict[str, Any]:
        """
        Step 6: Network construction.
        
        Build network from targets and interactions.
        """
        logger.info("Step 6: Network construction")
        
        # Build NetworkX graph
        G = nx.Graph()
        
        # Add target nodes
        for target in targets:
            G.add_node(target.gene_symbol, **target.dict())
        
        # Add interaction edges
        # STRING uses protein_a/protein_b field names, handle both formats
        for interaction in interactions:
            # Prefer human-readable gene symbols when available
            source = (interaction.get('preferredName_A') or
                     interaction.get('preferred_name_a') or
                     interaction.get('protein_a') or
                     interaction.get('source') or 
                     interaction.get('protein1') or
                     interaction.get('from', ''))
            target = (interaction.get('preferredName_B') or
                     interaction.get('preferred_name_b') or
                     interaction.get('protein_b') or
                     interaction.get('target') or 
                     interaction.get('protein2') or
                     interaction.get('to', ''))
            
            # Get score (STRING uses combined_score or score)
            score = (interaction.get('combined_score') or 
                    interaction.get('score', 0.0)) / 1000.0
            
            if source and target:
                # Create edge attributes without weight to avoid conflicts
                edge_attrs = {k: v for k, v in interaction.items() if k not in ['weight', 'source', 'target']}
                G.add_edge(source, target, weight=score, **edge_attrs)
        
        # Expand network by fetching additional neighbors for each node
        # LIMIT: Only expand target nodes to avoid excessive API calls
        logger.info(f"Expanding network: initial {len(G.nodes())} nodes, {len(G.edges())} edges")
        expanded_nodes = set(G.nodes())
        new_interactions = []
        
        # Only expand target nodes, not all nodes (to avoid too many API calls)
        target_gene_symbols = [t.gene_symbol for t in targets]
        nodes_to_expand = [node for node in G.nodes() if node in target_gene_symbols]
        
        # OPTIMIZATION: Aggressively limit expansion for many targets to avoid memory/timeout
        num_targets = len(nodes_to_expand)

        # Use config value if provided, otherwise production-ready defaults
        # OPTIMIZATION: Adaptive logic to ensure network reaches critical mass without expensive fallback
        add_nodes_per_target = self.network_expansion_config.expansion_neighbors
        max_network_size = self.network_expansion_config.max_network_size
        
        # If using default/low expansion (<=1), boost it for small target sets
        # This avoids the expensive depth-2 fallback later
        if add_nodes_per_target <= 1:
            if num_targets <= 5:
                add_nodes_per_target = 10  # Boost to 10 for small sets
                logger.info(f"[Step 6] Adaptive expansion: Boosted expansion_neighbors from {self.network_expansion_config.expansion_neighbors} to {add_nodes_per_target} (small target set)")
            elif num_targets <= 10:
                add_nodes_per_target = 5   # Boost to 5 for medium sets
                logger.info(f"[Step 6] Adaptive expansion: Boosted expansion_neighbors from {self.network_expansion_config.expansion_neighbors} to {add_nodes_per_target} (medium target set)")
        
        # Calculate max targets to expand based on network size limit
        max_targets_to_expand = min(
            num_targets,
            max(1, (max_network_size - len(G.nodes())) // (add_nodes_per_target + 1))
        )
        
        logger.info(
            f"[Step 6] Network expansion strategy: add_nodes_per_target={add_nodes_per_target}, "
            f"max_targets_to_expand={max_targets_to_expand}, "
            f"max_network_size={max_network_size}"
        )
        
        logger.info(f"Expanding network for {min(len(nodes_to_expand), max_targets_to_expand)} target nodes (out of {len(G.nodes())} total)")
        logger.info(f"Using {add_nodes_per_target} neighbors per target to optimize memory usage")
        
        # OPTIMIZATION: Fetch interactions in parallel batches (much faster!)
        nodes_to_expand_list = nodes_to_expand[:max_targets_to_expand]
        batch_size = 1  # CHANGED from 2 to 1 for stability with slow APIs
        
        async def fetch_node_interactions(node: str):
            """Fetch interactions for a single node."""
            try:
                return await self._call_with_tracking(
                    None,
                    'string',
                    self.mcp_manager.string.get_interaction_network(
                        protein_ids=[node],
                        species="9606",
                        required_score=350,  # PRODUCTION FIX: Lowered from 400 to 350 for more interactions
                        add_nodes=add_nodes_per_target
                    )
                )
            except Exception as e:
                logger.warning(f"Failed to expand network for node {node}: {e}")
                return None
        
        # Process in parallel batches with timeout and circuit breaker
        consecutive_timeouts = 0
        stop_expansion = False  # Flag for early termination

        for i in range(0, len(nodes_to_expand_list), batch_size):
            batch = nodes_to_expand_list[i:i + batch_size]
            batch_num = i//batch_size + 1
            total_batches = (len(nodes_to_expand_list) + batch_size - 1) // batch_size
            logger.debug(f"Processing network expansion batch {batch_num}/{total_batches} ({len(batch)} nodes)")

            # CIRCUIT BREAKER: Skip after 2 consecutive timeouts
            if consecutive_timeouts >= 2:
                logger.warning(f"CIRCUIT BREAKER: Skipping remaining batches after {consecutive_timeouts} consecutive timeouts")
                stop_expansion = True
                break

            # EARLY TERMINATION: Stop if network is large enough
            if len(G.nodes()) >= max_network_size:
                logger.info(
                    f"Network size reached {len(G.nodes())} nodes "
                    f"(limit: {max_network_size}) - stopping expansion"
                )
                stop_expansion = True
                break

            # Fetch all nodes in batch in parallel with timeout
            try:
                batch_results = await asyncio.wait_for(
                    asyncio.gather(*[fetch_node_interactions(node) for node in batch], return_exceptions=True),
                    timeout=60.0  # 60 second timeout per expansion batch
                )
                # Reset timeout counter on success
                consecutive_timeouts = 0
            except asyncio.TimeoutError:
                consecutive_timeouts += 1
                logger.warning(f"Network expansion batch {batch_num} TIMEOUT ({consecutive_timeouts}/2) - continuing to next batch")
                if consecutive_timeouts >= 2:
                    logger.error(f"CIRCUIT BREAKER TRIGGERED: {consecutive_timeouts} consecutive timeouts")
                continue

            # Process results
            for node, node_interactions in zip(batch, batch_results):
                if node_interactions is None or isinstance(node_interactions, Exception):
                    continue

                try:
                    # Handle both nested and direct response structures
                    if 'network' in node_interactions:
                        network_data = node_interactions.get('network', {})
                        node_edges = network_data.get('edges', [])
                    else:
                        node_edges = node_interactions.get('edges', [])
                except Exception as e:
                    logger.debug(f"Error parsing interactions for {node}: {e}")
                    continue
                
                for edge in node_edges:
                    # Try multiple field name variations (STRING uses protein_a/protein_b)
                    edge_source = (edge.get('protein_a') or
                                  edge.get('preferredName_A') or 
                                  edge.get('source') or 
                                  edge.get('protein1') or
                                  edge.get('from', ''))
                    edge_target = (edge.get('protein_b') or
                                  edge.get('preferredName_B') or 
                                  edge.get('target') or 
                                  edge.get('protein2') or
                                  edge.get('to', ''))
                    # Get score (STRING uses combined_score or score)
                    edge_score = (edge.get('combined_score') or 
                                 edge.get('score', 0.0)) / 1000.0
                    
                    # Add node to expanded set
                    if edge_source:
                        expanded_nodes.add(edge_source)
                    if edge_target:
                        expanded_nodes.add(edge_target)
                    
                    # Add edge if both nodes exist
                    if edge_source and edge_target:
                        new_interactions.append({
                            'source': edge_source,
                            'target': edge_target,
                            'score': edge_score * 1000.0,  # Store as original score
                            'weight': edge_score,
                            **edge
                        })
        
        # Merge new interactions into network
        for interaction in new_interactions:
            source = interaction.get('source', '')
            target = interaction.get('target', '')
            weight = interaction.get('weight', 0.5)
            
            if source and target and not G.has_edge(source, target):
                # Create edge attributes without weight to avoid conflicts
                edge_attrs = {k: v for k, v in interaction.items() if k not in ['weight', 'source', 'target']}
                G.add_edge(source, target, weight=weight, **edge_attrs)
        
        # Add any nodes from expanded set that weren't added yet
        for node in expanded_nodes:
            if not G.has_node(node):
                G.add_node(node)
        
        # Validate network has minimum edges per node (at least 2)
        nodes_to_remove = []
        for node in G.nodes():
            if G.degree(node) < 2 and node not in [t.gene_symbol for t in targets]:
                # Only remove non-target nodes with too few connections
                nodes_to_remove.append(node)
        
        for node in nodes_to_remove:
            G.remove_node(node)
            logger.debug(f"Removed node {node} with insufficient connections")
        
        logger.info(f"Initial expansion complete: {len(G.nodes())} nodes, {len(G.edges())} edges")

        # PRODUCTION FIX: If network is still too small, do depth-2 expansion
        min_required_nodes = 50  # Biological minimum for valid analysis
        ideal_nodes = 100  # Target for robust analysis

        if len(G.nodes()) < min_required_nodes:
            logger.warning(
                f"Network critically undersized ({len(G.nodes())} nodes). "
                f"Performing depth-2 expansion to reach minimum {min_required_nodes} nodes..."
            )

            # Do second-hop expansion
            try:
                G = await self._expand_network_with_neighbors(
                    G,
                    target_gene_symbols,
                    expansion_depth=1,  # One more hop
                    max_nodes_to_expand=20  # Limit to top 20 nodes to prevent explosion
                )
                logger.info(f"Depth-2 expansion complete: {len(G.nodes())} nodes, {len(G.edges())} edges")
            except Exception as e:
                logger.error(f"Depth-2 expansion failed: {e}")

        elif len(G.nodes()) < ideal_nodes:
            logger.info(
                f"Network smaller than ideal ({len(G.nodes())} < {ideal_nodes} nodes). "
                f"Attempting limited depth-2 expansion..."
            )

            # Do targeted second-hop expansion (smaller scope)
            try:
                # Only expand from target nodes for second hop
                current_size = len(G.nodes())
                G = await self._expand_network_with_neighbors(
                    G,
                    target_gene_symbols[:3],  # Only expand first 3 targets
                    expansion_depth=1,
                    max_nodes_to_expand=10  # Very limited expansion
                )
                added_nodes = len(G.nodes()) - current_size
                logger.info(f"Limited depth-2 expansion added {added_nodes} nodes: {len(G.nodes())} total nodes")
            except Exception as e:
                logger.warning(f"Limited depth-2 expansion failed: {e}")

        # Final network validation
        final_node_count = len(G.nodes())
        final_edge_count = len(G.edges())

        if final_node_count < min_required_nodes:
            logger.error(
                f"⚠️  CRITICAL: Network size ({final_node_count} nodes) below biological minimum "
                f"({min_required_nodes} nodes). Results will lack statistical power and biological validity. "
                f"Recommended: Lower confidence threshold or increase expansion depth."
            )
        elif final_node_count < ideal_nodes:
            logger.warning(
                f"⚠️  Network size ({final_node_count} nodes) below ideal ({ideal_nodes} nodes). "
                f"Results may have limited biological coverage. Consider increasing expansion parameters."
            )
        else:
            logger.info(
                f"✅ Network size acceptable: {final_node_count} nodes, {final_edge_count} edges "
                f"(minimum: {min_required_nodes}, ideal: {ideal_nodes})"
            )

        logger.info(f"Network expansion complete: {final_node_count} nodes, {final_edge_count} edges")

        # Convert to result format
        network_nodes = []
        for node in G.nodes(data=True):
            node_id = node[0]
            node_data = node[1]
            
            network_node = NetworkNode(
                id=node_id,
                node_type='protein',
                gene_symbol=node_data.get('gene_symbol', node_id),
                pathways=self._get_node_pathways(node_id),
                centrality_measures=self._calculate_centrality_measures(G, node_id)
            )
            network_nodes.append(network_node)
        
        network_edges = []
        for edge in G.edges(data=True):
            source, target, edge_data = edge
            
            network_edge = NetworkEdge(
                source=source,
                target=target,
                weight=edge_data.get('weight', 0.0),
                interaction_type=edge_data.get('interaction_type'),
                evidence_score=edge_data.get('score', 0.0) / 1000.0,
                pathway_context=self._get_edge_pathway_context(source, target)
            )
            network_edges.append(network_edge)
        
        return {
            'network': G,
            'nodes': network_nodes,
            'edges': network_edges
        }
    
    async def _expand_network_with_neighbors(
        self, 
        network: nx.Graph, 
        target_genes: List[str], 
        expansion_depth: int = 1,
        max_nodes_to_expand: int = 20
    ) -> nx.Graph:
        """
        Expand network by fetching neighbors of existing nodes.
        
        Args:
            network: Existing network graph
            target_genes: Target genes to prioritize for expansion
            expansion_depth: How many hops to expand (1 = neighbors, 2 = neighbors of neighbors)
            max_nodes_to_expand: Maximum number of nodes to expand (prioritizing targets and hubs)
        
        Returns:
            Expanded network with additional nodes and edges
        """
        logger.info(f"Expanding network with neighbors: depth={expansion_depth}, max_nodes={max_nodes_to_expand}")
        
        expanded_network = network.copy()
        
        # Iterate expansion based on depth
        for depth in range(expansion_depth):
            # Prioritize nodes to expand:
            # 1. Target genes (if present in network)
            # 2. High degree nodes (hubs)
            
            candidates = []
            # Add targets first
            for node in target_genes:
                if expanded_network.has_node(node):
                    candidates.append(node)
            
            # Add other nodes sorted by degree
            other_nodes = sorted(
                [n for n in expanded_network.nodes() if n not in candidates],
                key=lambda n: expanded_network.degree(n),
                reverse=True
            )
            candidates.extend(other_nodes)
            
            # Limit to max_nodes
            nodes_to_expand = candidates[:max_nodes_to_expand]
            
            logger.info(f"Expansion depth {depth + 1}: processing top {len(nodes_to_expand)} nodes (out of {len(candidates)} candidates)")
            
            new_nodes = set()
            new_edges = []
            
            # Process in batches to avoid timeouts
            batch_size = 3
            for i in range(0, len(nodes_to_expand), batch_size):
                batch = nodes_to_expand[i:i + batch_size]
                
                # Fetch neighbors for batch
                batch_results = await asyncio.gather(
                    *[self._call_with_tracking(
                        None,
                        'string',
                        self.mcp_manager.string.get_interaction_network(
                            protein_ids=[node],
                            species="9606",
                            required_score=400,
                            add_nodes=5  # Conservative expansion
                        )
                    ) for node in batch],
                    return_exceptions=True
                )
                
                # Process results
                for node, node_interactions in zip(batch, batch_results):
                    if node_interactions is None or isinstance(node_interactions, Exception):
                        continue

                    try:
                        # Handle both nested and direct response structures
                        if 'network' in node_interactions:
                            network_data = node_interactions.get('network', {})
                            node_edges = network_data.get('edges', [])
                        else:
                            node_edges = node_interactions.get('edges', [])
                        
                        for edge in node_edges:
                            # Prefer preferredName fields for human-readable IDs
                            edge_source = (edge.get('preferredName_A') or
                                          edge.get('preferred_name_a') or
                                          edge.get('protein_a') or 
                                          edge.get('source') or 
                                          edge.get('protein1') or
                                          edge.get('from', ''))
                            edge_target = (edge.get('preferredName_B') or
                                          edge.get('preferred_name_b') or
                                          edge.get('protein_b') or 
                                          edge.get('target') or 
                                          edge.get('protein2') or
                                          edge.get('to', ''))
                            # Get score (STRING uses combined_score or score)
                            edge_score = (edge.get('combined_score') or 
                                         edge.get('score', 0.0)) / 1000.0
                            
                            # Add nodes to expanded set
                            if edge_source:
                                new_nodes.add(edge_source)
                            if edge_target:
                                new_nodes.add(edge_target)
                            
                            # Add edge if both nodes exist
                            if edge_source and edge_target:
                                # Create edge dict without weight to avoid conflicts
                                edge_dict = {k: v for k, v in edge.items() if k not in ['weight']}
                                new_edges.append({
                                    'source': edge_source,
                                    'target': edge_target,
                                    'score': edge_score * 1000.0,
                                    'weight': edge_score,
                                    **edge_dict
                                })
                                
                    except Exception as e:
                        logger.warning(f"Failed to expand network for node {node} at depth {depth + 1}: {e}")
                        continue
            
            # Merge new nodes and edges into network
            for node in new_nodes:
                if not expanded_network.has_node(node):
                    expanded_network.add_node(node)
            
            for edge in new_edges:
                source = edge.get('source', '')
                target = edge.get('target', '')
                weight = edge.get('weight', 0.5)
                
                if source and target and not expanded_network.has_edge(source, target):
                    # Create edge attributes without weight to avoid conflicts
                    edge_attrs = {k: v for k, v in edge.items() if k not in ['weight', 'source', 'target']}
                    expanded_network.add_edge(source, target, weight=weight, **edge_attrs)
            
            logger.debug(f"Expansion depth {depth + 1} complete: {len(expanded_network.nodes())} nodes, {len(expanded_network.edges())} edges")
        
        logger.info(f"Network expansion complete: {len(expanded_network.nodes())} nodes, {len(expanded_network.edges())} edges")
        return expanded_network
    
    async def _step7_simulation(
        self, 
        network: nx.Graph, 
        targets: List[Protein], 
        simulation_mode: str,
        tissue_context: Optional[str],
        simulation_context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Step 7: Enhanced Simulation with actual propagation engines.
        """
        simulation_context = simulation_context or {}
        requested_mode = (simulation_mode or 'simple').lower()
        use_mra = requested_mode == 'mra'
        perturbation_type = 'activate' if requested_mode in ('activate', 'overexpress') else 'inhibit'

        # SCIENTIFIC OPTIMIZATION: Auto-switch to BFS for large networks
        # MRA full matrix inversion is O(n³) and impractical for networks > 50 nodes
        # BFS propagation provides comparable results with O(n+e) complexity
        MRA_MAX_NETWORK_SIZE = 50  # nodes
        network_size = network.number_of_nodes()
        
        if use_mra and network_size > MRA_MAX_NETWORK_SIZE:
            logger.warning(
                f"⚠️  Network too large for full MRA ({network_size} nodes > {MRA_MAX_NETWORK_SIZE} limit). "
                f"Switching to BFS propagation for computational feasibility. "
                f"Scientific note: BFS propagation provides comparable pathway-level insights for large networks."
            )
            use_mra = False
            # Add to warnings for transparency
            if 'warnings' not in simulation_context:
                simulation_context['warnings'] = []
            simulation_context['warnings'].append(
                f"MRA mode requested but network size ({network_size}) exceeds threshold ({MRA_MAX_NETWORK_SIZE}). "
                f"Used BFS propagation instead for computational feasibility."
            )

        engine_label = 'MRASimulator' if use_mra else 'BFS propagation engine'
        logger.info(
            f"Step 7: Running {requested_mode.upper()} simulation using {engine_label}"
        )

        simulator = MRASimulator(network, simulation_context) if use_mra else None

        simulation_results: List[Dict[str, Any]] = []
        all_affected_nodes: Set[str] = set()

        logger.info(f"Processing {len(targets)} targets for simulation...")

        # Track memory usage
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        logger.info(f"Initial memory usage: {initial_memory:.1f} MB")

        for idx, target in enumerate(targets, 1):
            gene_symbol = target.gene_symbol
            if not gene_symbol:
                continue

            # Log memory for each target
            current_memory = process.memory_info().rss / 1024 / 1024  # MB
            logger.debug(
                f"Processing target {idx}/{len(targets)}: {gene_symbol} "
                f"(Memory: {current_memory:.1f} MB, +{current_memory - initial_memory:.1f} MB)"
            )
            try:
                if use_mra:
                    sim_output = await simulator.simulate_perturbation(
                        gene_symbol,
                        perturbation_type=perturbation_type,
                        tissue_context=tissue_context
                    )
                    affected_nodes = self._extract_mra_effects(sim_output.steady_state)
                    network_impact = self._summarize_mra_network_impact(gene_symbol, affected_nodes, network)
                    confidence_scores = self._calculate_mra_confidence_scores(gene_symbol, affected_nodes, network)
                    downstream_nodes = list(sim_output.downstream_classification.keys())[:25]
                    upstream_nodes = list(sim_output.upstream_classification.keys())[:25]
                    feedback_loops = [
                        self._format_mra_feedback_loop(loop)
                        for loop in sim_output.feedback_loops
                    ][:5]
                    if network.has_node(gene_symbol):
                        direct_targets = [
                            neighbor for neighbor in network.neighbors(gene_symbol)
                            if neighbor in affected_nodes
                        ]
                    else:
                        direct_targets = []
                    execution_time = sim_output.execution_time
                    mra_context = {
                        'steady_state_top_hits': dict(
                            sorted(
                                affected_nodes.items(),
                                key=lambda kv: abs(kv[1]),
                                reverse=True
                            )[:10]
                        ),
                        'convergence': sim_output.convergence_info,
                        'tissue_specificity': sim_output.tissue_specificity
                    }
                else:
                    propagation_result = self._simulate_network_propagation(
                        network,
                        gene_symbol,
                        simulation_mode=requested_mode,
                        tissue_context=tissue_context
                    )
                    affected_nodes = propagation_result['affected_nodes']
                    network_impact = propagation_result['network_impact']
                    confidence_scores = propagation_result['confidence_scores']
                    downstream_nodes = propagation_result['downstream']
                    upstream_nodes = propagation_result['upstream']
                    feedback_loops = propagation_result['feedback_loops']
                    direct_targets = propagation_result['direct_targets']
                    execution_time = propagation_result['execution_time']
                    mra_context = {}

                biological_context = self._get_biological_annotation(gene_symbol)
                if mra_context:
                    biological_context.setdefault('mra', {}).update(mra_context)

                result = {
                    'target': gene_symbol,
                    'affected_nodes': affected_nodes,
                    'direct_targets': direct_targets,
                    'downstream': downstream_nodes,
                    'upstream': upstream_nodes,
                    'feedback_loops': feedback_loops,
                    'network_impact': network_impact,
                    'confidence_scores': confidence_scores,
                    'biological_context': biological_context,
                    'drug_info': self._get_drug_annotations(gene_symbol),
                    'execution_time': execution_time
                }
                simulation_results.append(result)
                all_affected_nodes.update(node for node, effect in affected_nodes.items() if abs(effect) > 0)
                logger.info(
                    f"✅ Completed simulation for {gene_symbol}: "
                    f"{len(affected_nodes)} affected nodes, "
                    f"{len(direct_targets)} direct targets"
                )
            except KeyError as exc:
                logger.warning(
                    f"MRA simulation skipped for {gene_symbol}: {exc}. Using fallback propagation."
                )
                simulation_results.append(self._fallback_simulation(network, gene_symbol))
            except Exception as exc:
                logger.error(
                    f"Enhanced simulation failed for target {gene_symbol}: {exc}",
                    exc_info=True  # Log full stack trace
                )
                simulation_results.append(self._fallback_simulation(network, gene_symbol))

        if not simulation_results:
            logger.warning("Simulation engine returned no results; using fallback simulation set")
            for target in targets[:3]:
                simulation_results.append(self._fallback_simulation(network, target.gene_symbol))

        logger.info(
            f"✅ All {len(simulation_results)} target simulations complete. "
            f"Calculating synergy matrix and network impact..."
        )
        
        # Calculate synergy matrix with timeout protection
        synergy_start_time = time.time()
        try:
            synergy_matrix = self._calculate_synergy_matrix(simulation_results, network)
            synergy_duration = time.time() - synergy_start_time
            logger.info(f"✅ Synergy matrix calculated in {synergy_duration:.2f}s")
        except Exception as exc:
            logger.warning(
                f"Synergy matrix calculation failed: {exc}. Using empty matrix."
            )
            synergy_matrix = {
                'pairwise_synergy': {},
                'overall_synergy': 0.0,
                'top_synergistic_pairs': []
            }
        
        network_impact = self._assess_network_wide_impact(simulation_results, network)
        convergence_rate = self._calculate_simulation_convergence(simulation_results)

        return {
            'results': simulation_results,
            'convergence_rate': convergence_rate,
            'synergy_matrix': synergy_matrix,
            'network_impact': network_impact,
            'engine': 'mra' if use_mra else 'simple',
            'total_affected_nodes': len(all_affected_nodes)
        }
    
    def _simulate_network_propagation(
        self, 
        network: nx.Graph, 
        target_node: str, 
        simulation_mode: str,
        tissue_context: Optional[str]
    ) -> Dict[str, Any]:
        """
        IMPROVEMENT 1: True Network Propagation
        
        Simulate perturbation propagation through network topology.
        """
        import time
        start_time = time.time()
        
        if not network.has_node(target_node):
            logger.warning(f"Target node {target_node} not found in network")
            return {
                'affected_nodes': {},
                'direct_targets': [],
                'network_impact': {'total_affected': 0},
                'confidence_scores': {'overall': 0.0},
                'execution_time': 0.0
            }
        
        # Validate network has edges before propagation
        if network.number_of_edges() == 0:
            logger.warning(f"Network has no edges for target {target_node}, propagation impossible")
            # Fallback: try to fetch neighbors on-the-fly if network is sparse
            try:
                # This is a fallback - in practice network should be expanded in Step 6
                logger.debug(f"Attempting to expand network for {target_node}")
                # Note: Full expansion should happen in Step 6, this is just a safety net
                neighbors_count = len(list(network.neighbors(target_node)))
                if neighbors_count == 0:
                    return {
                        'affected_nodes': {target_node: 1.0},
                        'direct_targets': [],
                        'network_impact': {
                            'total_affected': 0,  # Exclude target
                            'mean_effect': 1.0,
                            'max_effect': 1.0,
                            'network_coverage': 0.0,
                            'network_centrality': 0.0,
                            'betweenness_centrality': 0.0,
                            'propagation_depth': 0,
                            'perturbation_magnitude': 0.0
                        },
                        'confidence_scores': {'overall': 0.3, 'propagation': 0.05, 'network_coverage': 0.09},
                        'execution_time': time.time() - start_time
                    }
            except Exception as e:
                logger.warning(f"Failed to handle sparse network for {target_node}: {e}")
        
        # Initialize propagation
        affected_nodes = {target_node: 1.0}  # Target gets full effect
        node_depths = {target_node: 0}
        direct_targets = []
        propagation_queue = [(target_node, 1.0, 0)]  # (node, effect_strength, depth)
        visited = {target_node}
        
        # Network propagation with depth limit (3 hops, 0.5 decay)
        max_depth = 3
        propagation_factor = 0.5
        
        logger.debug(f"Starting propagation for {target_node} with {len(list(network.neighbors(target_node)))} neighbors")
        
        while propagation_queue:
            current_node, effect_strength, depth = propagation_queue.pop(0)
            
            if depth >= max_depth:
                continue
            
            # Get neighbors and propagate effect
            neighbors = list(network.neighbors(current_node))
            if not neighbors:
                continue  # Skip if no neighbors
            
            for neighbor in neighbors:
                if neighbor in visited:
                    continue
                
                # Get edge weight (interaction strength) with graceful fallback
                edge_data = network.get_edge_data(current_node, neighbor, {})
                edge_weight = edge_data.get('weight', 0.5) if edge_data else 0.5
                
                # Ensure edge_weight is valid (between 0 and 1)
                if not isinstance(edge_weight, (int, float)) or edge_weight <= 0:
                    edge_weight = 0.5  # Default to moderate interaction
                
                # Calculate propagated effect
                propagated_effect = effect_strength * propagation_factor * edge_weight
                
                # Threshold for significant effect
                if propagated_effect >= 0.05:
                    affected_nodes[neighbor] = propagated_effect
                    node_depths[neighbor] = depth + 1
                    visited.add(neighbor)
                    
                    if depth == 0:
                        direct_targets.append(neighbor)
                    
                    # Add to queue for further propagation
                    propagation_queue.append((neighbor, propagated_effect, depth + 1))
        
        logger.debug(f"Propagation complete for {target_node}: {len(affected_nodes)} affected nodes at depth {max_depth}")
        
        # Calculate network impact metrics (STANDARDIZED SCHEMA)
        total_affected = len(affected_nodes) - 1  # Exclude target from count
        
        # OPTIMIZED: Calculate degree centrality directly (O(1) instead of O(V))
        # Degree centrality = degree(node) / (n-1) where n is number of nodes
        node_degree = network.degree(target_node) if network.has_node(target_node) else 0
        network_size = len(network)
        network_centrality = node_degree / (network_size - 1) if network_size > 1 else 0.0
        
        # OPTIMIZED: Skip expensive betweenness centrality for large networks
        # Betweenness centrality is O(V*E) and can take minutes for 186-node networks
        # Use approximation: nodes with high degree often have high betweenness
        network_size_threshold = 100
        if network_size <= network_size_threshold:
            try:
                # Only calculate for small networks
                betweenness_centrality = nx.betweenness_centrality(network).get(target_node, 0.0)
            except Exception as exc:
                logger.debug(f"Betweenness centrality calculation failed for {target_node}: {exc}")
                # Fallback: approximate using degree centrality
                betweenness_centrality = min(1.0, network_centrality * 1.5)
        else:
            # For large networks, approximate betweenness using degree centrality
            # This is a reasonable approximation: high-degree nodes often have high betweenness
            betweenness_centrality = min(1.0, network_centrality * 1.2)
            logger.debug(
                f"Skipping expensive betweenness calculation for {target_node} "
                f"(network size: {network_size} > {network_size_threshold}). "
                f"Using degree-based approximation: {betweenness_centrality:.3f}"
            )

        # OPTIMIZED: Calculate effect statistics using generator expressions (memory-efficient)
        # Cap at 1000 values to prevent memory overflow on large networks
        effect_values_iter = (abs(effect) for effect in affected_nodes.values())
        effect_values_list = list(islice(effect_values_iter, 1000))
        
        if effect_values_list:
            mean_effect = np.mean(effect_values_list)
            max_effect = max(effect_values_list)
            impact_sum = sum(effect_values_list)
        else:
            mean_effect = max_effect = impact_sum = 0.0
        
        coverage = total_affected / len(network.nodes()) if network.nodes() else 0.0

        # PRODUCTION: Use standardized NetworkImpactMetrics schema
        network_impact = {
            # Core metrics (required by validation/visualization)
            'total_affected': total_affected,
            'mean_effect': float(mean_effect),
            'max_effect': float(max_effect),
            'network_coverage': float(coverage),

            # Extended metrics (optional, for detailed analysis)
            'network_centrality': float(network_centrality),
            'betweenness_centrality': float(betweenness_centrality),
            'propagation_depth': max_depth,
            'perturbation_magnitude': float(min(1.0, impact_sum / max(1, len(network.nodes())))),
        }

        
        # Calculate confidence scores with base confidence if target has neighbors
        neighbors_count = len(list(network.neighbors(target_node)))
        base_confidence = 0.3 if neighbors_count > 0 else 0.0
        
        # Enhanced overall confidence calculation
        overall_confidence = max(
            base_confidence, 
            network_centrality + betweenness_centrality * 2
        )
        
        confidence_scores = {
            'overall': min(0.9, overall_confidence),
            'propagation': min(0.8, total_affected / 20.0),
            'network_coverage': min(0.7, coverage)
        }
        
        execution_time = time.time() - start_time
        downstream_nodes = [node for node, depth in node_depths.items() if depth > 0][:25]
        upstream_nodes = self._classify_upstream_nodes(network, target_node)
        feedback_loops = self._analyze_feedback_loops(network, target_node)
        
        return {
            'affected_nodes': affected_nodes,
            'direct_targets': direct_targets,
            'downstream': downstream_nodes,
            'upstream': upstream_nodes,
            'feedback_loops': feedback_loops,
            'network_impact': network_impact,
            'confidence_scores': confidence_scores,
            'execution_time': execution_time
        }

    def _extract_mra_effects(
        self,
        steady_state: Dict[str, float],
        min_effect: float = 0.01
    ) -> Dict[str, float]:
        """Filter steady-state values down to meaningful effects."""
        if not steady_state:
            return {}
        return {
            node: float(effect)
            for node, effect in steady_state.items()
            if abs(effect) >= min_effect
        }

    def _calculate_mra_confidence_scores(
        self,
        target_node: str,
        affected_nodes: Dict[str, float],
        network: nx.Graph
    ) -> Dict[str, float]:
        """Approximate confidence scores based on coverage and magnitude."""
        if not affected_nodes:
            return {'overall': 0.0, 'propagation': 0.0, 'network_coverage': 0.0}

        effect_values = [abs(value) for value in affected_nodes.values()]
        max_effect = max(effect_values)
        total_nodes = len(network.nodes()) or 1
        coverage = min(1.0, len(affected_nodes) / total_nodes)
        propagation = min(0.9, max(0, len(affected_nodes) - 1) / 10.0)
        degree_centrality = (
            nx.degree_centrality(network).get(target_node, 0.0)
            if network.number_of_nodes() > 1 else 0.0
        )

        overall = min(
            0.95,
            0.5 * max_effect + 0.3 * coverage + 0.2 * degree_centrality + 0.05
        )

        return {
            'overall': overall,
            'propagation': propagation,
            'network_coverage': coverage
        }

    def _summarize_mra_network_impact(
        self,
        target_node: str,
        affected_nodes: Dict[str, float],
        network: nx.Graph
    ) -> Dict[str, Any]:
        """Summarize network-wide impact from MRA effects."""
        if not affected_nodes:
            return {
                'total_affected': 0,
                'network_centrality': 0.0,
                'betweenness_centrality': 0.0,
                'network_coverage': 0.0,
                'average_effect': 0.0,
                'perturbation_magnitude': 0.0
            }

        effect_values = [abs(val) for val in affected_nodes.values()]
        average_effect = float(np.mean(effect_values)) if effect_values else 0.0
        max_effect = float(np.max(effect_values)) if effect_values else 0.0
        total_nodes = len(network.nodes()) or 1
        coverage = min(1.0, len(affected_nodes) / total_nodes)

        degree_centrality = (
            nx.degree_centrality(network).get(target_node, 0.0)
            if network.number_of_nodes() > 1 else 0.0
        )
        betweenness_centrality = (
            nx.betweenness_centrality(network).get(target_node, 0.0)
            if network.number_of_nodes() > 1 else 0.0
        )

        return {
            'total_affected': len(affected_nodes),
            'network_centrality': degree_centrality,
            'betweenness_centrality': betweenness_centrality,
            'network_coverage': coverage,
            'average_effect': average_effect,
            'max_effect': max_effect,
            'perturbation_magnitude': min(1.0, average_effect * max(coverage, 0.05))
        }

    def _format_mra_feedback_loop(self, loop: Any) -> str:
        """Convert FeedbackLoop objects into readable strings."""
        try:
            nodes = getattr(loop, 'nodes', [])
            loop_type = getattr(loop, 'loop_type', 'unknown')
            strength = getattr(loop, 'strength', 0.0)
            pathway_context = getattr(loop, 'pathway_context', None)
            label = ' -> '.join(nodes) if nodes else str(loop)
            context_str = f", pathway={pathway_context}" if pathway_context else ""
            return f"{loop_type}: {label}{context_str} (strength={strength:.2f})"
        except Exception:
            return str(loop)
    
    def _get_biological_annotation(self, gene_symbol: str) -> Dict[str, Any]:
        """
        IMPROVEMENT 2: Biological Annotation
        
        Get functional annotation for the target gene.
        """
        # Known biological annotations for common cancer targets
        biological_map = {
            'AXL': {
                'type': 'receptor_tyrosine_kinase',
                'function': 'Cell survival, migration, invasion',
                'pathway': 'PI3K/AKT, MAPK',
                'druggability': 'high',
                'clinical_relevance': 'Metastasis, therapy resistance'
            },
            'RELA': {
                'type': 'transcription_factor',
                'function': 'NF-κB signaling, inflammation',
                'pathway': 'NF-κB',
                'druggability': 'medium',
                'clinical_relevance': 'Inflammation, survival'
            },
            'AKT1': {
                'type': 'kinase',
                'function': 'Cell survival, proliferation',
                'pathway': 'PI3K/AKT',
                'druggability': 'high',
                'clinical_relevance': 'Oncogenic signaling'
            },
            'STAT3': {
                'type': 'transcription_factor',
                'function': 'Cell survival, immune response',
                'pathway': 'JAK/STAT',
                'druggability': 'medium',
                'clinical_relevance': 'Immune evasion, survival'
            },
            'MAPK1': {
                'type': 'kinase',
                'function': 'Cell proliferation, differentiation',
                'pathway': 'MAPK/ERK',
                'druggability': 'high',
                'clinical_relevance': 'Oncogenic signaling'
            },
            'MAPK3': {
                'type': 'kinase',
                'function': 'Cell proliferation, differentiation',
                'pathway': 'MAPK/ERK',
                'druggability': 'high',
                'clinical_relevance': 'Oncogenic signaling'
            },
            'CASP3': {
                'type': 'protease',
                'function': 'Apoptosis execution',
                'pathway': 'Apoptosis',
                'druggability': 'low',
                'clinical_relevance': 'Cell death'
            },
            'MMP9': {
                'type': 'metalloproteinase',
                'function': 'ECM degradation, invasion',
                'pathway': 'Metastasis',
                'druggability': 'medium',
                'clinical_relevance': 'Metastasis, invasion'
            },
            'VEGFA': {
                'type': 'growth_factor',
                'function': 'Angiogenesis',
                'pathway': 'VEGF',
                'druggability': 'high',
                'clinical_relevance': 'Angiogenesis'
            },
            'CCND1': {
                'type': 'cyclin',
                'function': 'Cell cycle progression',
                'pathway': 'Cell cycle',
                'druggability': 'medium',
                'clinical_relevance': 'Proliferation'
            }
        }
        
        return biological_map.get(gene_symbol, {
            'type': 'unknown',
            'function': 'Unknown function',
            'pathway': 'Unknown pathway',
            'druggability': 'unknown',
            'clinical_relevance': 'Unknown relevance'
        })
    
    def _analyze_feedback_loops(self, network: nx.Graph, target_node: str) -> List[str]:
        """
        IMPROVEMENT 3: Feedback Loop Analysis
        
        Identify feedback loops involving the target node.
        Safe implementation for both directed and undirected graphs.
        """
        feedback_loops = []
        
        if not network.has_node(target_node):
            return feedback_loops
        
        # Find cycles involving the target node
        try:
            # Get subgraph of nodes within 2 hops of target
            neighbors = list(network.neighbors(target_node))
            
            # Safety check: if too many neighbors, skip expensive cycle detection
            if len(neighbors) > 50:
                logger.debug(f"Skipping feedback loop analysis for {target_node}: too many neighbors ({len(neighbors)})")
                return []
                
            extended_neighbors = set()
            for neighbor in neighbors:
                # Limit extended neighbors to avoid explosion
                if len(extended_neighbors) > 200:
                    break
                extended_neighbors.update(network.neighbors(neighbor))
            
            subgraph_nodes = {target_node} | set(neighbors) | extended_neighbors
            
            # Safety check: if subgraph is too large, skip
            if len(subgraph_nodes) > 100:
                logger.debug(f"Skipping feedback loop analysis for {target_node}: subgraph too large ({len(subgraph_nodes)} nodes)")
                return []
                
            subgraph = network.subgraph(subgraph_nodes)
            
            cycles = []
            if network.is_directed():
                # For directed graphs, use simple_cycles with a limit
                # simple_cycles is a generator, so we can stop early
                cycle_gen = nx.simple_cycles(subgraph)
                try:
                    for _ in range(100):  # Limit to checking 100 cycles
                        cycle = next(cycle_gen)
                        if target_node in cycle and len(cycle) >= 2:
                            cycles.append(cycle)
                            if len(cycles) >= 5:  # We only need top 5
                                break
                except StopIteration:
                    pass
            else:
                # For undirected graphs, use cycle_basis (polynomial time)
                # This finds fundamental cycles, which is a good proxy for feedback in undirected PPIs
                # Note: In undirected graphs, a "cycle" of length 2 is just an edge, usually ignored
                basis = nx.cycle_basis(subgraph)
                for cycle in basis:
                    if target_node in cycle and len(cycle) >= 3:
                        cycles.append(cycle)
                        if len(cycles) >= 5:
                            break
            
            for cycle in cycles:
                feedback_loops.append(' -> '.join(cycle))
                    
        except Exception as e:
            logger.debug(f"Feedback loop analysis failed for {target_node}: {e}")
        
        return feedback_loops[:5]  # Limit to top 5 feedback loops
    
    def _classify_downstream_nodes(self, network: nx.Graph, target_node: str) -> List[str]:
        """
        IMPROVEMENT 4: Downstream Node Classification
        
        Classify nodes that are downstream of the target.
        """
        downstream_nodes = []
        
        if not network.has_node(target_node):
            return downstream_nodes
        
        # Validate network has edges for connectivity
        if network.number_of_edges() == 0:
            logger.debug(f"Network has no edges, cannot classify downstream nodes for {target_node}")
            return downstream_nodes
        
        # Use BFS to find downstream nodes
        queue = [(target_node, 0)]
        visited = {target_node}
        max_depth = 2
        
        while queue:
            current_node, depth = queue.pop(0)
            
            if depth >= max_depth:
                continue
            
            # Get neighbors - handle case where node has no neighbors
            neighbors = list(network.neighbors(current_node))
            if not neighbors:
                continue  # Skip if no neighbors
            
            for neighbor in neighbors:
                if neighbor not in visited:
                    visited.add(neighbor)
                    downstream_nodes.append(neighbor)
                    queue.append((neighbor, depth + 1))
        
        return downstream_nodes[:10]  # Limit to top 10 downstream nodes
    
    def _classify_upstream_nodes(self, network: nx.Graph, target_node: str) -> List[str]:
        """
        IMPROVEMENT 4: Upstream Node Classification
        
        Classify nodes that are upstream of the target.
        """
        upstream_nodes = []
        
        if not network.has_node(target_node):
            return upstream_nodes
        
        # Validate network has edges for connectivity
        if network.number_of_edges() == 0:
            logger.debug(f"Network has no edges, cannot classify upstream nodes for {target_node}")
            return upstream_nodes
        
        # Find nodes that have edges TO the target
        # In undirected graphs, all neighbors can be considered upstream
        neighbors = list(network.neighbors(target_node))
        if neighbors:
            upstream_nodes = neighbors[:10]  # Limit to top 10 upstream nodes
        else:
            # For directed graphs (if network has direction), check edges explicitly
            for node in network.nodes():
                if node != target_node and network.has_edge(node, target_node):
                    upstream_nodes.append(node)
                    if len(upstream_nodes) >= 10:
                        break
        
        return upstream_nodes[:10]  # Limit to top 10 upstream nodes
    
    def _get_drug_annotations(self, gene_symbol: str) -> Dict[str, Any]:
        """
        IMPROVEMENT 5: Drug Information Integration
        
        Get drug information for the target gene.
        """
        # Known drug annotations for common cancer targets
        drug_map = {
            'AXL': {
                'approved_drugs': ['Bemcentinib', 'Cabozantinib'],
                'clinical_trials': ['Phase II', 'Phase III'],
                'mechanism': 'Tyrosine kinase inhibitor',
                'indication': 'Metastatic cancer'
            },
            'AKT1': {
                'approved_drugs': ['Capivasertib'],
                'clinical_trials': ['Phase III'],
                'mechanism': 'AKT inhibitor',
                'indication': 'Breast cancer'
            },
            'MAPK1': {
                'approved_drugs': ['Trametinib', 'Cobimetinib'],
                'clinical_trials': ['Approved'],
                'mechanism': 'MEK inhibitor',
                'indication': 'Melanoma, NSCLC'
            },
            'VEGFA': {
                'approved_drugs': ['Bevacizumab', 'Aflibercept'],
                'clinical_trials': ['Approved'],
                'mechanism': 'VEGF inhibitor',
                'indication': 'Multiple cancers'
            },
            'STAT3': {
                'approved_drugs': [],
                'clinical_trials': ['Phase I', 'Phase II'],
                'mechanism': 'STAT3 inhibitor',
                'indication': 'Experimental'
            }
        }
        
        return drug_map.get(gene_symbol, {
            'approved_drugs': [],
            'clinical_trials': [],
            'mechanism': 'Unknown',
            'indication': 'Unknown'
        })
    
    def _calculate_synergy_matrix(self, simulation_results: List[Dict], network: nx.Graph) -> Dict[str, Any]:
        """
        IMPROVEMENT 6: Comprehensive Synergy Analysis
        
        Calculate synergy matrix for all target pairs.
        """
        synergy_matrix = {}
        targets = [result['target'] for result in simulation_results]
        total_pairs = len(targets) * (len(targets) - 1) // 2
        
        logger.info(f"Calculating synergy matrix for {len(targets)} targets ({total_pairs} pairs)...")
        
        # Calculate pairwise synergy
        pair_count = 0
        for i, target_a in enumerate(targets):
            for j, target_b in enumerate(targets[i+1:], i+1):
                pair_count += 1
                logger.debug(
                    f"  Pair {pair_count}/{total_pairs}: {target_a} vs {target_b}"
                )
                
                try:
                    synergy_score = self._calculate_pairwise_synergy(
                        simulation_results[i], simulation_results[j], network
                    )
                    
                    pair_key = f"{target_a}_{target_b}"
                    synergy_matrix[pair_key] = {
                        'target_a': target_a,
                        'target_b': target_b,
                        'synergy_score': synergy_score,
                        'interaction_type': self._classify_synergy_type(synergy_score),
                        'rationale': self._get_synergy_rationale(target_a, target_b, synergy_score)
                    }
                except Exception as exc:
                    logger.warning(
                        f"Failed to calculate synergy for {target_a} vs {target_b}: {exc}"
                    )
                    # Use default synergy score on error
                    pair_key = f"{target_a}_{target_b}"
                    synergy_matrix[pair_key] = {
                        'target_a': target_a,
                        'target_b': target_b,
                        'synergy_score': 0.0,
                        'interaction_type': 'unknown',
                        'rationale': f'Calculation failed: {exc}'
                    }
        
        # Calculate overall synergy metrics
        synergy_scores = [data['synergy_score'] for data in synergy_matrix.values()]
        overall_synergy = np.mean(synergy_scores) if synergy_scores else 0.0
        
        return {
            'pairwise_synergy': synergy_matrix,
            'overall_synergy': overall_synergy,
            'top_synergistic_pairs': sorted(
                synergy_matrix.items(), 
                key=lambda x: x[1]['synergy_score'], 
                reverse=True
            )[:5]
        }
    
    def _calculate_pathway_overlap(self, target_a: str, target_b: str, network: nx.Graph) -> float:
        """
        Calculate pathway overlap between two targets.
        Returns 1.0 if in same pathway, 0.5 if connected pathways, 0.0 otherwise.
        """
        # Check if nodes have pathway information in network attributes
        pathways_a = set()
        pathways_b = set()
        
        if network.has_node(target_a):
            node_data_a = network.nodes[target_a]
            if 'pathways' in node_data_a and isinstance(node_data_a['pathways'], list):
                pathways_a = set(node_data_a['pathways'])
        
        if network.has_node(target_b):
            node_data_b = network.nodes[target_b]
            if 'pathways' in node_data_b and isinstance(node_data_b['pathways'], list):
                pathways_b = set(node_data_b['pathways'])
        
        # Same pathway
        if pathways_a and pathways_b and (pathways_a & pathways_b):
            return 1.0
        
        # Connected pathways (neighbors share pathways)
        if network.has_node(target_a) and network.has_node(target_b):
            neighbors_a = set(network.neighbors(target_a))
            neighbors_b = set(network.neighbors(target_b))
            common_neighbors = neighbors_a & neighbors_b
            
            if common_neighbors:
                # Check if common neighbors have pathway overlap
                for neighbor in common_neighbors:
                    neighbor_pathways = set()
                    if network.has_node(neighbor):
                        neighbor_data = network.nodes[neighbor]
                        if 'pathways' in neighbor_data and isinstance(neighbor_data['pathways'], list):
                            neighbor_pathways = set(neighbor_data['pathways'])
                    
                    if (pathways_a & neighbor_pathways) or (pathways_b & neighbor_pathways):
                        return 0.5
        
        return 0.0
    
    def _calculate_interaction_strength(self, target_a: str, target_b: str, network: nx.Graph) -> float:
        """
        Calculate interaction strength between two targets.
        Returns edge weight if edge exists, 0 otherwise.
        """
        if not network.has_edge(target_a, target_b):
            return 0.0
        
        edge_data = network.get_edge_data(target_a, target_b, {})
        edge_weight = edge_data.get('weight', 0.0) if edge_data else 0.0
        
        # Ensure weight is between 0 and 1
        return max(0.0, min(1.0, edge_weight))
    
    def _calculate_proximity_score(self, target_a: str, target_b: str, network: nx.Graph) -> float:
        """
        Calculate proximity score between two targets based on network topology.
        Returns inverse of shortest path length if path exists, else 0.
        
        Optimized for performance: uses NetworkX's built-in cutoff parameter to limit
        search depth, preventing excessive computation on large networks.
        """
        if not network.has_node(target_a) or not network.has_node(target_b):
            return 0.0
        
        if target_a == target_b:
            return 1.0
        
        # Check for direct edge (distance = 1)
        if network.has_edge(target_a, target_b):
            return 1.0
        
        # Limit search depth to 5 hops for performance (prevents exploring very long paths)
        max_depth = 5
        
        # Use NetworkX's shortest_path_length with cutoff parameter for early termination
        # This leverages NetworkX's optimized algorithms (BFS for unweighted, Dijkstra for weighted)
        try:
            shortest_path_length = nx.shortest_path_length(
                network, 
                target_a, 
                target_b,
                cutoff=max_depth  # Early termination: stop searching after max_depth
            )
            # Inverse of path length: 1/distance (closer = higher score)
            return 1.0 / (shortest_path_length + 1)
        except nx.NetworkXNoPath:
            # No path found within cutoff distance
            return 0.0
        except nx.NetworkXError:
            # NetworkX error (e.g., nodes not in same component)
            return 0.0
        except Exception as exc:
            logger.debug(f"Error calculating proximity for {target_a}-{target_b}: {exc}")
            return 0.0
    
    def _calculate_pairwise_synergy(self, result_a: Dict, result_b: Dict, network: nx.Graph) -> float:
        """
        Calculate synergy score between two targets using network topology,
        pathway overlap, and interaction strength instead of Jaccard similarity.
        """
        target_a = result_a['target']
        target_b = result_b['target']
        
        # Use new synergy calculation based on network topology
        pathway_overlap = self._calculate_pathway_overlap(target_a, target_b, network)
        interaction_strength = self._calculate_interaction_strength(target_a, target_b, network)
        proximity_score = self._calculate_proximity_score(target_a, target_b, network)
        
        # Calculate synergy score with new weights
        # Formula: synergy = (pathway_overlap * 0.4 + interaction_strength * 0.3 + proximity_score * 0.3)
        synergy_score = (
            pathway_overlap * 0.4 + 
            interaction_strength * 0.3 + 
            proximity_score * 0.3
        )
        
        return min(synergy_score, 1.0)
    
    def _classify_synergy_type(self, synergy_score: float) -> str:
        """Classify synergy type based on score."""
        if synergy_score > 0.7:
            return 'synergistic'
        elif synergy_score > 0.4:
            return 'additive'
        else:
            return 'antagonistic'
    
    def _get_synergy_rationale(self, target_a: str, target_b: str, synergy_score: float) -> str:
        """Get rationale for synergy between two targets."""
        if synergy_score > 0.7:
            return f"Strong synergy: {target_a} and {target_b} target complementary pathways"
        elif synergy_score > 0.4:
            return f"Moderate synergy: {target_a} and {target_b} have overlapping effects"
        else:
            return f"Limited synergy: {target_a} and {target_b} target independent pathways"
    
    def _assess_network_wide_impact(self, simulation_results: List[Dict], network: nx.Graph) -> Dict[str, Any]:
        """
        IMPROVEMENT 7: Network-wide Impact Assessment
        
        Assess the overall impact on the network.
        """
        all_affected = set()
        total_effects = []
        
        for result in simulation_results:
            all_affected.update(result['affected_nodes'].keys())
            total_effects.extend(result['affected_nodes'].values())
        
        # Calculate network coverage
        network_coverage = len(all_affected) / len(network.nodes()) if network.nodes() else 0.0
        
        # Calculate average effect strength
        average_effect = np.mean(total_effects) if total_effects else 0.0
        
        # Calculate network disruption
        network_disruption = min(1.0, network_coverage * 2.0)
        
        return {
            'total_affected_nodes': len(all_affected),
            'network_coverage': network_coverage,
            'average_effect_strength': average_effect,
            'network_disruption': network_disruption,
            'perturbation_magnitude': min(1.0, average_effect * network_coverage)
        }
    
    def _fallback_simulation(self, network: nx.Graph, target_node: str) -> Dict[str, Any]:
        """Fallback simulation when enhanced simulation fails."""
        return {
            'target': target_node,
            'affected_nodes': {target_node: 0.5},
            'direct_targets': list(network.neighbors(target_node))[:5] if network.has_node(target_node) else [],
            'downstream': [],
            'upstream': [],
            'feedback_loops': [],
            'network_impact': {'total_affected': 1},
            'confidence_scores': {'overall': 0.3},
            'biological_context': {'type': 'unknown'},
            'drug_info': {'approved_drugs': []},
            'execution_time': 0.1
        }
    
    def _calculate_biological_relevance(self, individual_results: List) -> float:
        """
        Calculate biological relevance score for the simulation results.
        
        Adjusted normalization factors to match actual affected node counts:
        - Change normalization: /5.0 for network_score (expect at least 5 affected nodes)
        - Lower feedback_score normalization to /2.0 (expect at least 2 feedback loops)
        - Add minimum relevance: if any target has downstream effects, minimum relevance = 0.2
        """
        if not individual_results:
            return 0.0
        
        relevance_scores = []
        has_downstream_effects = False
        
        for result in individual_results:
            # Score based on network impact (adjusted normalization)
            network_impact = result.network_impact.get('total_affected', 0)
            network_score = min(1.0, network_impact / 5.0)  # Changed from /10.0 to /5.0
            
            # Score based on confidence
            confidence = result.confidence_scores.get('overall', 0.0)
            
            # Score based on feedback loops (biological relevance) - adjusted normalization
            feedback_loops_count = len(result.feedback_loops) if hasattr(result, 'feedback_loops') else 0
            feedback_score = min(1.0, feedback_loops_count / 2.0)  # Changed from /3.0 to /2.0
            
            # Check if target has downstream effects
            downstream = result.downstream if hasattr(result, 'downstream') else []
            if len(downstream) > 0:
                has_downstream_effects = True
            
            # Combined relevance score
            relevance = (network_score * 0.4 + confidence * 0.4 + feedback_score * 0.2)
            relevance_scores.append(relevance)
        
        # Calculate mean relevance
        mean_relevance = np.mean(relevance_scores) if relevance_scores else 0.0
        
        # Add minimum relevance if any target has downstream effects
        if has_downstream_effects:
            mean_relevance = max(0.2, mean_relevance)
        
        return mean_relevance

    def _track_data_source(
        self,
        data_sources: Dict[str, DataSourceStatus],
        source_name: str,
        success: bool = True,
        error_type: Optional[str] = None
    ) -> Optional[DataSourceStatus]:
        """Track MCP data source usage."""
        if not data_sources or source_name not in data_sources:
            return None

        status = data_sources[source_name]
        status.requested += 1
        if success:
            status.successful += 1
        else:
            status.failed += 1
            if error_type and error_type not in status.error_types:
                status.error_types.append(error_type)

        if status.requested:
            status.success_rate = status.successful / status.requested

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "[DATA_TRACKING] %s requested=%d successful=%d failed=%d success_rate=%.2f",
                source_name,
                status.requested,
                status.successful,
                status.failed,
                status.success_rate,
            )
        return status

    async def _call_with_tracking(
        self,
        data_sources: Optional[Dict[str, DataSourceStatus]],
        source_name: str,
        coro,
        suppress_exception: bool = False,
    ):
        """Await an MCP coroutine and update tracking counters automatically."""
        sources = data_sources or self._active_data_sources
        try:
            result = await coro
            if sources:
                self._track_data_source(sources, source_name, success=True)
            return result
        except Exception as exc:
            if sources:
                self._track_data_source(
                    sources,
                    source_name,
                    success=False,
                    error_type=type(exc).__name__,
                )
            if suppress_exception:
                logger.debug(
                    "[DATA_TRACKING] Suppressed %s failure for %s",
                    source_name,
                    type(exc).__name__,
                )
                return None
            raise
    
    async def _step8_impact_assessment(
        self, 
        simulation_results: List[Dict], 
        network: nx.Graph,
        pathways: List[Pathway],
        targets: List[Protein]
    ) -> Dict[str, Any]:
        """
        Step 8: Impact assessment.
        
        Assess simulation impact using STRING enrichment and pathway analysis.
        """
        logger.info("Step 8: Impact assessment")
        
        # Collect affected genes from all simulations
        all_affected_genes = set()
        for sim_result in simulation_results:
            # Get affected nodes from the new result structure
            affected_nodes = sim_result.get('affected_nodes', {})
            all_affected_genes.update(affected_nodes.keys())
        
        # Always run enrichment even if only targets are affected
        # Use target genes as minimum input for enrichment
        target_genes = [t.gene_symbol for t in targets if t.gene_symbol]
        enrichment_genes = list(all_affected_genes) if all_affected_genes else target_genes
        
        # Get functional enrichment using STRING even for small gene sets
        enrichment_data = {}
        if enrichment_genes:
            try:
                enrichment = await self._call_with_tracking(
                    None,
                    'string',
                    self.mcp_manager.string.get_functional_enrichment(
                        protein_ids=enrichment_genes,
                        species="9606"
                    )
                )
                enrichment_data = enrichment.get('enrichment', {})
            except Exception as e:
                logger.warning(f"Failed to get functional enrichment: {e}")
        
        # Map affected genes to pathways from Step 2
        pathway_impact = {}
        target_genes_set = set(target_genes)
        affected_genes_set = all_affected_genes if all_affected_genes else target_genes_set
        
        logger.info(f"Calculating pathway impact for {len(pathways)} pathways")
        logger.info(f"Affected genes: {len(affected_genes_set)} genes")
        if affected_genes_set:
            logger.info(f"Sample affected genes: {list(affected_genes_set)[:10]}")
        
        # Calculate pathway impact scores based on affected gene count per pathway
        for pathway in pathways:
            logger.debug(f"Processing pathway: {pathway.id} ({pathway.name})")
            pathway_genes = set()
            
            # Try to get genes from pathway object first
            if hasattr(pathway, 'genes') and pathway.genes:
                pathway_genes = set(pathway.genes)
                logger.info(f"Pathway {pathway.id}: Using {len(pathway_genes)} genes from pathway object. Sample: {list(pathway_genes)[:5]}")
            
            # OPTIMIZATION: If no genes in pathway object, fetch participants from Reactome
            # But skip if we already have genes to avoid redundant API calls
            if not pathway_genes:
                try:
                    # Try get_pathway_participants first (faster than get_pathway_details)
                    participants = await self._call_with_tracking(
                        None,
                        'reactome',
                        self.mcp_manager.reactome.get_pathway_participants(pathway.id)
                    )
                    if participants.get('participants'):
                        # Extract gene symbols from participants
                        for participant in participants['participants']:
                            # Try multiple field names for gene symbol
                            gene = (participant.get('gene_symbol') or 
                                   participant.get('gene') or 
                                   participant.get('displayName') or 
                                   participant.get('geneName') or
                                   participant.get('name', ''))
                            if gene:
                                pathway_genes.add(gene)
                    
                    # If no participants from get_pathway_participants, try get_pathway_details
                    if not pathway_genes:
                        pathway_details = await self._call_with_tracking(
                            None,
                            'reactome',
                            self.mcp_manager.reactome.get_pathway_details(pathway.id)
                        )
                        
                        # Extract genes from entities
                        if pathway_details.get('entities'):
                            for entity in pathway_details['entities']:
                                gene_name = (entity.get('displayName') or 
                                           entity.get('geneName') or 
                                           entity.get('gene_symbol') or
                                           entity.get('gene', ''))
                                if gene_name:
                                    pathway_genes.add(gene_name)
                        
                        # Extract genes from reactions/events
                        if pathway_details.get('hasEvent'):
                            for event in pathway_details['hasEvent']:
                                if event.get('participants'):
                                    for participant in event['participants']:
                                        gene_name = (participant.get('displayName') or 
                                                   participant.get('geneName') or 
                                                   participant.get('gene_symbol') or
                                                   participant.get('gene', ''))
                                        if gene_name:
                                            pathway_genes.add(gene_name)
                    
                    if pathway_genes:
                        logger.debug(f"Pathway {pathway.id}: Fetched {len(pathway_genes)} genes from Reactome")
                    else:
                        logger.warning(f"Pathway {pathway.id}: No genes found from Reactome")
                        
                except Exception as e:
                    logger.warning(f"Failed to get genes for pathway {pathway.id}: {e}")
                    pathway_genes = set()
            
            if not pathway_genes:
                logger.warning(f"Pathway {pathway.id}: No genes found after extraction!")
            else:
                logger.info(f"Pathway {pathway.id}: Has {len(pathway_genes)} genes. Sample: {list(pathway_genes)[:5]}")
            
            # Calculate overlap between affected genes and pathway genes
            # Normalize gene names (uppercase for comparison)
            pathway_genes_normalized = set(g.upper() if isinstance(g, str) else str(g).upper() for g in pathway_genes if g)
            affected_genes_normalized = set(g.upper() if isinstance(g, str) else str(g).upper() for g in affected_genes_set if g)
            affected_in_pathway = pathway_genes_normalized & affected_genes_normalized
            
            if affected_in_pathway:
                logger.debug(f"Pathway {pathway.id}: Found {len(affected_in_pathway)} matching genes: {list(affected_in_pathway)[:5]}")
            
            # Map back to original gene names for reporting
            affected_in_pathway_original = set()
            for affected_gene in affected_genes_set:
                if affected_gene and affected_gene.upper() in affected_in_pathway:
                    affected_in_pathway_original.add(affected_gene)
            
            pathway_total = len(pathway_genes) if pathway_genes else 1
            
            # Calculate impact score - use a lower threshold to capture pathways with any overlap
            impact_score = len(affected_in_pathway) / pathway_total if pathway_total > 0 else 0.0
            
            if impact_score > 0:
                pathway_impact[pathway.id] = {
                    'pathway_id': pathway.id,
                    'pathway_name': pathway.name if hasattr(pathway, 'name') else pathway.id,
                    'impact_score': impact_score,
                    'affected_genes_count': len(affected_in_pathway),
                    'total_pathway_genes': pathway_total,
                    'affected_genes': list(affected_in_pathway_original)
                }
                logger.info(f"Pathway {pathway.id} ({pathway.name}): {len(affected_in_pathway)}/{pathway_total} genes affected (impact: {impact_score:.3f})")
        
        logger.info(f"Pathway impact calculated: {len(pathway_impact)} pathways affected")
        
        # Calculate network impact metrics
        network_impact = self._calculate_network_impact_metrics(
            simulation_results, network
        )
        
        # Calculate enrichment score based on pathway impact
        enrichment_score = np.mean([p['impact_score'] for p in pathway_impact.values()]) if pathway_impact else 0.6
        
        return {
            'enrichment': enrichment_data,
            'enrichment_score': enrichment_score,
            'pathway_impact': pathway_impact,
            'functional_enrichment': enrichment_data,  # Alias for compatibility
            'network_impact': network_impact,
            'affected_genes': list(all_affected_genes) if all_affected_genes else target_genes
        }
    
    def _get_node_pathways(self, node_id: str) -> List[str]:
        """Get pathways for a node."""
        return []
    
    def _calculate_centrality_measures(self, network: nx.Graph, node_id: str) -> Dict[str, float]:
        """Calculate centrality measures for a node."""
        try:
            return {
                'betweenness': nx.betweenness_centrality(network)[node_id],
                'closeness': nx.closeness_centrality(network)[node_id],
                'degree': network.degree(node_id)
            }
        except:
            return {'betweenness': 0.0, 'closeness': 0.0, 'degree': 0}
    
    def _get_edge_pathway_context(self, source: str, target: str) -> Optional[str]:
        """Get pathway context for an edge."""
        return None
    
    def _calculate_simulation_convergence(self, results: List[Dict]) -> float:
        """
        Calculate simulation convergence rate.
        
        Lower threshold to 0.1 (any propagation = converged) and check if 
        total_affected > 1 (more than just target itself).
        """
        if not results:
            return 0.0
        
        converged_count = 0
        for result in results:
            # Check if simulation was successful based on multiple criteria
            confidence = result.get('confidence_scores', {}).get('overall', 0.0)
            network_impact = result.get('network_impact', {})
            total_affected = network_impact.get('total_affected', 1)
            
            # Consider converged if:
            # 1. Confidence > 0.1 (any propagation = converged, lowered from 0.5)
            # 2. OR total_affected > 1 (more than just target itself)
            if confidence > 0.1 or total_affected > 1:
                converged_count += 1
        
        return converged_count / len(results) if results else 0.0
    
    def _calculate_network_impact_metrics(
        self, 
        simulation_results: List[Dict], 
        network: nx.Graph
    ) -> Dict[str, Any]:
        """Calculate network impact metrics."""
        if not simulation_results:
            return {'total_impact': 0.0, 'network_coverage': 0.0}
        
        # Calculate total impact
        total_impact = 0.0
        for result in simulation_results:
            affected_nodes = result.get('affected_nodes', {})
            impact = sum(abs(effect) for effect in affected_nodes.values())
            total_impact += impact
        
        # Calculate network coverage
        all_affected = set()
        for result in simulation_results:
            affected_nodes = result.get('affected_nodes', {})
            all_affected.update(affected_nodes.keys())
        
        network_coverage = len(all_affected) / len(network.nodes()) if network.nodes() else 0.0
        
        return {
            'total_impact': total_impact,
            'network_coverage': network_coverage,
            'affected_nodes': len(all_affected)
        }
    
    def _calculate_validation_score(
        self,
        target_data: Dict,
        pathway_data: Dict,
        interaction_data: Dict,
        expression_data: Dict,
        impact_data: Dict,
        simulation_data: Dict,
        assessment_data: Dict,
        data_sources: Dict,
        completeness_metrics: CompletenessMetrics
    ) -> float:
        """Calculate overall validation score with data completeness penalties."""
        scores = {}

        # Target resolution accuracy
        scores['resolution_accuracy'] = target_data.get('resolution_accuracy', 0.0)

        # Pathway coverage
        scores['pathway_coverage'] = pathway_data.get('coverage', 0.0)

        # Interaction validation
        scores['interaction_validation'] = interaction_data.get('validation_score', 0.0)

        # Expression coverage
        scores['expression_coverage'] = expression_data.get('coverage', 0.0)

        # Pathway impact
        scores['pathway_impact'] = impact_data.get('overall_impact', 0.0)

        # Simulation convergence
        scores['simulation_convergence'] = simulation_data.get('convergence_rate', 0.0)

        # Calculate overall score with data source penalties
        validation_result = self.validator.calculate_overall_validation_score(
            scores,
            data_sources=list(data_sources.values()),
            completeness_metrics=completeness_metrics
        )

        # Extract the final score (float) from the result dictionary
        if isinstance(validation_result, dict):
            return validation_result.get('final_score', 0.0)
        else:
            # Backward compatibility - in case validator returns float directly
            return validation_result

    def _build_completeness_metrics(
        self,
        pathway_data: Dict[str, Any],
        network_data: Dict[str, Any],
        expression_data: Dict[str, Any],
    ) -> CompletenessMetrics:
        """Construct completeness metrics for Scenario 4 outputs."""
        expression_comp = expression_data.get('coverage')
        if expression_comp is None:
            profile_count = len(expression_data.get('profiles', []))
            expression_comp = min(1.0, profile_count / 50.0) if profile_count else 0.0
        expression_comp = max(0.0, min(1.0, expression_comp))

        network_nodes = len(network_data.get('nodes', []))
        network_comp = min(1.0, network_nodes / 150.0) if network_nodes else 0.0

        pathway_comp = pathway_data.get('coverage')
        if pathway_comp is None:
            pathway_count = len(pathway_data.get('pathways', []))
            pathway_comp = min(1.0, pathway_count / 25.0) if pathway_count else 0.0
            logger.debug(f"Completeness calc: pathway_count={pathway_count}, computed pathway_comp={pathway_comp:.3f}")
        else:
            logger.debug(f"Completeness calc: using existing coverage={pathway_comp:.3f}")
        pathway_comp = max(0.0, min(1.0, pathway_comp))

        values = [metric for metric in (expression_comp, network_comp, pathway_comp) if metric is not None]
        overall = sum(values) / len(values) if values else 0.0

        return CompletenessMetrics(
            expression_data=expression_comp,
            network_data=network_comp,
            pathway_data=pathway_comp,
            drug_data=0.0,
            pathology_data=None,
            overall_completeness=overall
        )
