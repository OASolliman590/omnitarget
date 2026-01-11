"""
Prometheus Metrics

Exposes metrics for monitoring pipeline performance and health.
P0-4: Production Monitoring (Week 8)

Updated: 2025-11-07
"""

import logging
import psutil
import time
from typing import Dict, Any, Optional
from functools import wraps

logger = logging.getLogger(__name__)

try:
    from prometheus_client import (
        Counter, Histogram, Gauge, start_http_server,
        Info, CollectorRegistry
    )
    PROMETHEUS_AVAILABLE = True
except ImportError:
    # prometheus_client not installed, create no-op implementations
    PROMETHEUS_AVAILABLE = False

    class Counter:
        def __init__(self, *args, **kwargs): pass
        def inc(self, *args, **kwargs): pass
        def labels(self, *args, **kwargs): return self

    class Histogram:
        def __init__(self, *args, **kwargs): pass
        def observe(self, *args, **kwargs): pass
        def time(self): return NullTimer()
        def labels(self, *args, **kwargs): return self

    class Gauge:
        def __init__(self, *args, **kwargs): pass
        def set(self, *args, **kwargs): pass
        def inc(self, *args, **kwargs): pass
        def dec(self, *args, **kwargs): pass
        def labels(self, *args, **kwargs): return self

    class Info:
        def __init__(self, *args, **kwargs): pass
        def info(self, *args, **kwargs): pass

    class NullTimer:
        def __enter__(self): return self
        def __exit__(self, *args): pass

    def start_http_server(*args, **kwargs): pass


# ============================================================================
# Scenario Execution Metrics
# ============================================================================

scenario_executions_total = Counter(
    'omnitarget_scenario_executions_total',
    'Total number of scenario executions',
    ['scenario_id', 'scenario_name', 'status'],
    registry=CollectorRegistry() if PROMETHEUS_AVAILABLE else None
)

scenario_duration_seconds = Histogram(
    'omnitarget_scenario_duration_seconds',
    'Scenario execution duration in seconds',
    ['scenario_id', 'scenario_name'],
    registry=CollectorRegistry() if PROMETHEUS_AVAILABLE else None
)

scenario_phases_total = Counter(
    'omnitarget_scenario_phases_total',
    'Total number of scenario phase executions',
    ['scenario_id', 'scenario_name', 'phase'],
    registry=CollectorRegistry() if PROMETHEUS_AVAILABLE else None
)


# ============================================================================
# MCP Request Metrics
# ============================================================================

mcp_requests_total = Counter(
    'omnitarget_mcp_requests_total',
    'Total number of MCP requests',
    ['server_name', 'tool_name', 'status'],
    registry=CollectorRegistry() if PROMETHEUS_AVAILABLE else None
)

mcp_request_duration_seconds = Histogram(
    'omnitarget_mcp_request_duration_seconds',
    'MCP request duration in seconds',
    ['server_name', 'tool_name'],
    registry=CollectorRegistry() if PROMETHEUS_AVAILABLE else None
)

mcp_requests_in_flight = Gauge(
    'omnitarget_mcp_requests_in_flight',
    'Number of MCP requests currently in flight',
    ['server_name'],
    registry=CollectorRegistry() if PROMETHEUS_AVAILABLE else None
)


# ============================================================================
# Batch Query Metrics
# ============================================================================

batch_queries_total = Counter(
    'omnitarget_batch_queries_total',
    'Total number of batch queries',
    ['operation', 'batch_size', 'status'],
    registry=CollectorRegistry() if PROMETHEUS_AVAILABLE else None
)

batch_query_duration_seconds = Histogram(
    'omnitarget_batch_query_duration_seconds',
    'Batch query duration in seconds',
    ['operation', 'batch_size'],
    registry=CollectorRegistry() if PROMETHEUS_AVAILABLE else None
)

parallel_queries_total = Counter(
    'omnitarget_parallel_queries_total',
    'Total number of parallel queries',
    ['operation', 'query_count'],
    registry=CollectorRegistry() if PROMETHEUS_AVAILABLE else None
)


# ============================================================================
# Error Metrics
# ============================================================================

errors_total = Counter(
    'omnitarget_errors_total',
    'Total number of errors',
    ['error_type', 'component', 'severity'],
    registry=CollectorRegistry() if PROMETHEUS_AVAILABLE else None
)

retries_total = Counter(
    'omnitarget_retries_total',
    'Total number of retry attempts',
    ['component', 'operation'],
    registry=CollectorRegistry() if PROMETHEUS_AVAILABLE else None
)


# ============================================================================
# Resource Metrics
# ============================================================================

active_mcp_connections = Gauge(
    'omnitarget_active_mcp_connections',
    'Number of active MCP connections',
    ['server_name'],
    registry=CollectorRegistry() if PROMETHEUS_AVAILABLE else None
)

memory_usage_bytes = Gauge(
    'omnitarget_memory_usage_bytes',
    'Current memory usage in bytes',
    registry=CollectorRegistry() if PROMETHEUS_AVAILABLE else None
)

cpu_usage_percent = Gauge(
    'omnitarget_cpu_usage_percent',
    'Current CPU usage percentage',
    registry=CollectorRegistry() if PROMETHEUS_AVAILABLE else None
)


# ============================================================================
# Data Quality Metrics
# ============================================================================

validation_scores = Histogram(
    'omnitarget_validation_scores',
    'Validation scores from quality checks',
    ['validation_type'],
    registry=CollectorRegistry() if PROMETHEUS_AVAILABLE else None
)

pathway_coverage = Histogram(
    'omnitarget_pathway_coverage',
    'Pathway coverage percentage',
    ['database'],
    registry=CollectorRegistry() if PROMETHEUS_AVAILABLE else None
)

cross_database_concordance = Histogram(
    'omnitarget_cross_database_concordance',
    'Cross-database concordance score',
    ['db1', 'db2'],
    registry=CollectorRegistry() if PROMETHEUS_AVAILABLE else None
)


# ============================================================================
# System Information
# ============================================================================

system_info = Info(
    'omnitarget_system_info',
    'System information',
    registry=CollectorRegistry() if PROMETHEUS_AVAILABLE else None
)


def start_metrics_server(port: int = 8000) -> None:
    """
    Start Prometheus metrics HTTP server.

    Args:
        port: Port to expose metrics on (default: 8000)
    """
    if not PROMETHEUS_AVAILABLE:
        logger.warning("Prometheus client not installed, metrics server not started")
        return

    start_http_server(port, registry=CollectorRegistry())
    logger.info(f"Prometheus metrics server started on port {port}")
    logger.info(f"Metrics available at: http://localhost:{port}/metrics")


def record_scenario_execution(
    scenario_id: int,
    scenario_name: str,
    status: str,
    duration: float
) -> None:
    """
    Record scenario execution metrics.

    Args:
        scenario_id: Scenario ID
        scenario_name: Scenario name
        status: Execution status (success, error, timeout)
        duration: Execution duration in seconds
    """
    scenario_executions_total.labels(
        scenario_id=scenario_id,
        scenario_name=scenario_name,
        status=status
    ).inc()

    scenario_duration_seconds.labels(
        scenario_id=scenario_id,
        scenario_name=scenario_name
    ).observe(duration)

    logger.debug(
        f"Recorded metrics for scenario {scenario_id}",
        extra={
            "extra_fields": {
                "scenario_id": scenario_id,
                "scenario_name": scenario_name,
                "status": status,
                "duration": duration
            }
        }
    )


def record_scenario_phase(
    scenario_id: int,
    scenario_name: str,
    phase: str
) -> None:
    """
    Record scenario phase execution.

    Args:
        scenario_id: Scenario ID
        scenario_name: Scenario name
        phase: Phase name
    """
    scenario_phases_total.labels(
        scenario_id=scenario_id,
        scenario_name=scenario_name,
        phase=phase
    ).inc()


def record_mcp_request(
    server_name: str,
    tool_name: str,
    status: str,
    duration: float
) -> None:
    """
    Record MCP request metrics.

    Args:
        server_name: MCP server name (kegg, reactome, etc.)
        tool_name: Tool being called
        status: Request status (success, error, timeout)
        duration: Request duration in seconds
    """
    mcp_requests_total.labels(
        server_name=server_name,
        tool_name=tool_name,
        status=status
    ).inc()

    mcp_request_duration_seconds.labels(
        server_name=server_name,
        tool_name=tool_name
    ).observe(duration)


def record_batch_query(
    operation: str,
    batch_size: int,
    status: str,
    duration: float
) -> None:
    """
    Record batch query metrics.

    Args:
        operation: Operation name (e.g., pathway_extraction)
        batch_size: Number of items in batch
        status: Query status (success, partial, error)
        duration: Query duration in seconds
    """
    batch_queries_total.labels(
        operation=operation,
        batch_size=batch_size,
        status=status
    ).inc()

    batch_query_duration_seconds.labels(
        operation=operation,
        batch_size=batch_size
    ).observe(duration)


def record_error(
    error_type: str,
    component: str,
    severity: str = "error"
) -> None:
    """
    Record error metrics.

    Args:
        error_type: Type of error (e.g., DatabaseConnectionError)
        component: Component where error occurred
        severity: Error severity (error, warning, critical)
    """
    errors_total.labels(
        error_type=error_type,
        component=component,
        severity=severity
    ).inc()


def increment_retries(component: str, operation: str) -> None:
    """
    Record retry attempt.

    Args:
        component: Component performing retry
        operation: Operation being retried
    """
    retries_total.labels(
        component=component,
        operation=operation
    ).inc()


def update_resource_metrics() -> None:
    """
    Update system resource metrics (memory, CPU, connections).
    """
    try:
        # Memory usage
        process = psutil.Process()
        memory_bytes = process.memory_info().rss
        memory_usage_bytes.set(memory_bytes)

        # CPU usage
        cpu_percent = process.cpu_percent()
        cpu_usage_percent.set(cpu_percent)

    except Exception as e:
        logger.warning(f"Failed to update resource metrics: {e}")


def set_mcp_connections(server_name: str, count: int) -> None:
    """
    Set active MCP connection count.

    Args:
        server_name: MCP server name
        count: Number of active connections
    """
    active_mcp_connections.labels(server_name=server_name).set(count)


def record_validation_score(validation_type: str, score: float) -> None:
    """
    Record validation score.

    Args:
        validation_type: Type of validation
        score: Validation score (0.0 to 1.0)
    """
    validation_scores.labels(validation_type=validation_type).observe(score)


def record_pathway_coverage(database: str, coverage: float) -> None:
    """
    Record pathway coverage.

    Args:
        database: Database name (kegg, reactome, etc.)
        coverage: Coverage percentage (0.0 to 1.0)
    """
    pathway_coverage.labels(database=database).observe(coverage)


def record_cross_database_concordance(db1: str, db2: str, concordance: float) -> None:
    """
    Record cross-database concordance score.

    Args:
        db1: First database name
        db2: Second database name
        concordance: Concordance score (0.0 to 1.0)
    """
    cross_database_concordance.labels(db1=db1, db2=db2).observe(concordance)


# Decorator for automatic metrics collection
def monitor_execution(metric_name: str, labels: Optional[Dict[str, str]] = None):
    """
    Decorator to automatically collect execution metrics.

    Args:
        metric_name: Name of the metric
        labels: Labels to attach to the metric

    Example:
        >>> @monitor_execution("database_query", {"db": "kegg"})
        ... async def query_database():
        ...     pass
    """
    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            start_time = time.time()
            status = "success"

            try:
                result = await func(*args, **kwargs)
                return result
            except Exception as e:
                status = "error"
                raise
            finally:
                duration = time.time() - start_time
                # Record metrics here if needed
                logger.debug(
                    f"Executed {func.__name__}",
                    extra={
                        "extra_fields": {
                            "function": func.__name__,
                            "duration": duration,
                            "status": status
                        }
                    }
                )

        return async_wrapper
    return decorator


# Initialize system info
if PROMETHEUS_AVAILABLE:
    system_info.info({
        "version": "1.0.0",
        "python_version": "3.9+",
        "implementation": "OmniTarget Production Pipeline"
    })
