"""
ChEMBL Monitoring and Data Lineage

Provides monitoring, quality checks, and data lineage tracking for ChEMBL integration.
Features:
- Query performance monitoring
- Data quality metrics
- Integration health checks
- Error tracking and alerting
- Data lineage/provenance tracking
"""

import asyncio
import logging
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple, Callable
from enum import Enum

logger = logging.getLogger(__name__)


class HealthStatus(Enum):
    """Health status for monitoring."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


class DataQualityLevel(Enum):
    """Data quality levels."""
    HIGH = "high"        # >90% quality score
    MEDIUM = "medium"    # 70-90% quality score
    LOW = "low"          # 50-70% quality score
    POOR = "poor"        # <50% quality score


@dataclass
class QueryMetrics:
    """Metrics for a single query."""
    query_type: str
    duration: float
    success: bool
    error_message: Optional[str] = None
    timestamp: float = field(default_factory=time.time)
    cache_hit: bool = False
    result_count: int = 0


@dataclass
class DataQualityMetrics:
    """Data quality metrics."""
    total_records: int
    valid_records: int
    invalid_records: int
    missing_fields: Dict[str, int] = field(default_factory=dict)
    validation_errors: List[str] = field(default_factory=list)
    quality_score: float = 0.0
    quality_level: DataQualityLevel = DataQualityLevel.MEDIUM

    def calculate_quality_score(self):
        """Calculate overall quality score."""
        if self.total_records == 0:
            self.quality_score = 0.0
            self.quality_level = DataQualityLevel.POOR
            return

        # Base score from valid records
        validity_score = self.valid_records / self.total_records

        # Penalty for missing fields
        missing_penalty = min(0.2, len(self.missing_fields) * 0.02)

        # Final score
        self.quality_score = max(0.0, validity_score - missing_penalty)

        # Determine quality level
        if self.quality_score > 0.9:
            self.quality_level = DataQualityLevel.HIGH
        elif self.quality_score > 0.7:
            self.quality_level = DataQualityLevel.MEDIUM
        elif self.quality_score > 0.5:
            self.quality_level = DataQualityLevel.LOW
        else:
            self.quality_level = DataQualityLevel.POOR


@dataclass
class IntegrationHealthCheck:
    """Health check result."""
    component: str
    status: HealthStatus
    message: str
    timestamp: float = field(default_factory=time.time)
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DataLineage:
    """Data lineage/provenance tracking."""
    record_id: str
    record_type: str
    source: str  # "chembl", "kegg", "merged"
    created_at: float
    transformations: List[str] = field(default_factory=list)
    validation_results: Dict[str, bool] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def add_transformation(self, transformation: str):
        """Record a transformation step."""
        self.transformations.append(f"{time.time():.0f}: {transformation}")

    def add_validation(self, validator_name: str, passed: bool):
        """Record validation result."""
        self.validation_results[validator_name] = passed

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            'record_id': self.record_id,
            'record_type': self.record_type,
            'source': self.source,
            'created_at': self.created_at,
            'age_seconds': time.time() - self.created_at,
            'transformations': self.transformations,
            'validation_results': self.validation_results,
            'validation_pass_rate': (
                sum(self.validation_results.values()) / len(self.validation_results)
                if self.validation_results else 1.0
            ),
            'metadata': self.metadata
        }


class ChEMBLMonitor:
    """
    Monitoring system for ChEMBL integration.

    Tracks query performance, data quality, errors, and system health.

    Example:
        >>> monitor = ChEMBLMonitor(window_size=1000)
        >>> monitor.record_query("search_targets", 0.5, True, result_count=10)
        >>> health = await monitor.check_health()
        >>> print(f"Status: {health.status}")
    """

    def __init__(self, window_size: int = 1000, alert_threshold: int = 10):
        """
        Initialize monitor.

        Args:
            window_size: Number of recent queries to track
            alert_threshold: Number of errors before alerting
        """
        self.window_size = window_size
        self.alert_threshold = alert_threshold

        # Query tracking
        self._query_history: deque = deque(maxlen=window_size)
        self._query_counts = defaultdict(int)
        self._query_errors = defaultdict(int)

        # Performance tracking
        self._query_durations = defaultdict(list)

        # Data quality tracking
        self._quality_metrics: List[DataQualityMetrics] = []

        # Error tracking
        self._recent_errors: deque = deque(maxlen=100)

        # Health status
        self._last_health_check: Optional[IntegrationHealthCheck] = None

        # Alerts
        self._alert_callbacks: List[Callable] = []

    def record_query(
        self,
        query_type: str,
        duration: float,
        success: bool,
        error_message: Optional[str] = None,
        cache_hit: bool = False,
        result_count: int = 0
    ):
        """
        Record query metrics.

        Args:
            query_type: Type of query (e.g., "search_targets")
            duration: Query duration in seconds
            success: Whether query succeeded
            error_message: Error message if failed
            cache_hit: Whether result was from cache
            result_count: Number of results returned
        """
        metrics = QueryMetrics(
            query_type=query_type,
            duration=duration,
            success=success,
            error_message=error_message,
            cache_hit=cache_hit,
            result_count=result_count
        )

        self._query_history.append(metrics)
        self._query_counts[query_type] += 1

        if success:
            self._query_durations[query_type].append(duration)
        else:
            self._query_errors[query_type] += 1
            self._recent_errors.append(metrics)

            # Check if we should alert
            if self._query_errors[query_type] >= self.alert_threshold:
                self._trigger_alert(
                    f"High error rate for {query_type}: {self._query_errors[query_type]} errors"
                )

        logger.debug(
            f"Query recorded: {query_type} "
            f"duration={duration:.3f}s success={success} cache_hit={cache_hit}"
        )

    def record_data_quality(self, metrics: DataQualityMetrics):
        """
        Record data quality metrics.

        Args:
            metrics: Data quality metrics
        """
        metrics.calculate_quality_score()
        self._quality_metrics.append(metrics)

        # Alert on poor quality
        if metrics.quality_level == DataQualityLevel.POOR:
            self._trigger_alert(
                f"Poor data quality detected: {metrics.quality_score:.1%} "
                f"({metrics.valid_records}/{metrics.total_records} valid)"
            )

        logger.info(
            f"Data quality: {metrics.quality_level.value} "
            f"({metrics.quality_score:.1%})"
        )

    def get_query_stats(self, query_type: Optional[str] = None) -> Dict[str, Any]:
        """
        Get query statistics.

        Args:
            query_type: Optional specific query type

        Returns:
            Dictionary with query statistics
        """
        if query_type:
            queries = [q for q in self._query_history if q.query_type == query_type]
        else:
            queries = list(self._query_history)

        if not queries:
            return {
                'total_queries': 0,
                'success_rate': 0.0,
                'cache_hit_rate': 0.0,
                'avg_duration': 0.0,
                'p95_duration': 0.0,
                'p99_duration': 0.0
            }

        successful = [q for q in queries if q.success]
        cache_hits = [q for q in queries if q.cache_hit]
        durations = [q.duration for q in queries]
        durations.sort()

        return {
            'total_queries': len(queries),
            'successful_queries': len(successful),
            'failed_queries': len(queries) - len(successful),
            'success_rate': len(successful) / len(queries),
            'cache_hit_rate': len(cache_hits) / len(queries) if queries else 0.0,
            'avg_duration': sum(durations) / len(durations) if durations else 0.0,
            'p50_duration': durations[len(durations) // 2] if durations else 0.0,
            'p95_duration': durations[int(len(durations) * 0.95)] if durations else 0.0,
            'p99_duration': durations[int(len(durations) * 0.99)] if durations else 0.0,
            'min_duration': min(durations) if durations else 0.0,
            'max_duration': max(durations) if durations else 0.0,
        }

    def get_quality_summary(self) -> Dict[str, Any]:
        """
        Get data quality summary.

        Returns:
            Dictionary with quality metrics summary
        """
        if not self._quality_metrics:
            return {
                'num_checks': 0,
                'avg_quality_score': 0.0,
                'quality_distribution': {}
            }

        avg_score = sum(m.quality_score for m in self._quality_metrics) / len(self._quality_metrics)

        distribution = defaultdict(int)
        for metrics in self._quality_metrics:
            distribution[metrics.quality_level.value] += 1

        return {
            'num_checks': len(self._quality_metrics),
            'avg_quality_score': avg_score,
            'quality_distribution': dict(distribution),
            'recent_quality': self._quality_metrics[-1].quality_score if self._quality_metrics else 0.0
        }

    def get_error_summary(self) -> Dict[str, Any]:
        """
        Get error summary.

        Returns:
            Dictionary with error statistics
        """
        if not self._recent_errors:
            return {
                'total_errors': 0,
                'errors_by_type': {},
                'recent_errors': []
            }

        errors_by_type = defaultdict(int)
        for error in self._recent_errors:
            errors_by_type[error.query_type] += 1

        recent_errors = [
            {
                'query_type': e.query_type,
                'error_message': e.error_message,
                'timestamp': e.timestamp,
                'age_seconds': time.time() - e.timestamp
            }
            for e in list(self._recent_errors)[-10:]  # Last 10 errors
        ]

        return {
            'total_errors': len(self._recent_errors),
            'errors_by_type': dict(errors_by_type),
            'recent_errors': recent_errors
        }

    async def check_health(self) -> IntegrationHealthCheck:
        """
        Perform health check.

        Returns:
            Health check result
        """
        # Get recent query stats (last 100 queries or last 5 minutes)
        recent_window = 100
        recent_queries = list(self._query_history)[-recent_window:]

        if not recent_queries:
            health = IntegrationHealthCheck(
                component="chembl_integration",
                status=HealthStatus.UNKNOWN,
                message="No recent queries",
                details={'query_count': 0}
            )
            self._last_health_check = health
            return health

        # Calculate metrics
        success_rate = sum(1 for q in recent_queries if q.success) / len(recent_queries)
        avg_duration = sum(q.duration for q in recent_queries) / len(recent_queries)
        error_rate = 1.0 - success_rate

        # Determine health status
        if success_rate >= 0.95 and avg_duration < 5.0:
            status = HealthStatus.HEALTHY
            message = "All systems operational"
        elif success_rate >= 0.80 and avg_duration < 10.0:
            status = HealthStatus.DEGRADED
            message = f"Performance degraded (success_rate={success_rate:.1%}, avg_duration={avg_duration:.2f}s)"
        else:
            status = HealthStatus.UNHEALTHY
            message = f"System unhealthy (success_rate={success_rate:.1%}, avg_duration={avg_duration:.2f}s)"

        health = IntegrationHealthCheck(
            component="chembl_integration",
            status=status,
            message=message,
            details={
                'query_count': len(recent_queries),
                'success_rate': success_rate,
                'error_rate': error_rate,
                'avg_duration': avg_duration,
                'cache_hit_rate': sum(1 for q in recent_queries if q.cache_hit) / len(recent_queries)
            }
        )

        self._last_health_check = health
        logger.info(f"Health check: {status.value} - {message}")

        return health

    def add_alert_callback(self, callback: Callable[[str], None]):
        """
        Add alert callback.

        Args:
            callback: Function to call when alert is triggered
        """
        self._alert_callbacks.append(callback)

    def _trigger_alert(self, message: str):
        """
        Trigger alert.

        Args:
            message: Alert message
        """
        logger.warning(f"ALERT: {message}")

        for callback in self._alert_callbacks:
            try:
                callback(message)
            except Exception as e:
                logger.error(f"Alert callback failed: {e}")

    def get_monitoring_report(self) -> Dict[str, Any]:
        """
        Get comprehensive monitoring report.

        Returns:
            Dictionary with all monitoring data
        """
        return {
            'timestamp': time.time(),
            'query_stats': self.get_query_stats(),
            'quality_summary': self.get_quality_summary(),
            'error_summary': self.get_error_summary(),
            'health_check': {
                'status': self._last_health_check.status.value if self._last_health_check else 'unknown',
                'message': self._last_health_check.message if self._last_health_check else '',
                'details': self._last_health_check.details if self._last_health_check else {}
            } if self._last_health_check else None
        }

    def reset_stats(self):
        """Reset all statistics (for testing or new session)."""
        self._query_history.clear()
        self._query_counts.clear()
        self._query_errors.clear()
        self._query_durations.clear()
        self._quality_metrics.clear()
        self._recent_errors.clear()
        self._last_health_check = None
        logger.info("Monitor statistics reset")


class DataLineageTracker:
    """
    Data lineage and provenance tracking.

    Tracks data origins, transformations, and quality throughout the pipeline.

    Example:
        >>> tracker = DataLineageTracker()
        >>> lineage = tracker.create_lineage("CHEMBL25", "compound", "chembl")
        >>> lineage.add_transformation("standardized_to_pydantic")
        >>> lineage.add_validation("drug_likeness", True)
        >>> tracker.store_lineage(lineage)
    """

    def __init__(self, max_records: int = 10000):
        """
        Initialize lineage tracker.

        Args:
            max_records: Maximum number of lineage records to keep
        """
        self.max_records = max_records
        self._lineage: Dict[str, DataLineage] = {}
        self._access_times: Dict[str, float] = {}

    def create_lineage(
        self,
        record_id: str,
        record_type: str,
        source: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> DataLineage:
        """
        Create new lineage record.

        Args:
            record_id: Unique record identifier
            record_type: Type of record (compound, target, bioactivity, etc.)
            source: Data source (chembl, kegg, merged)
            metadata: Optional metadata

        Returns:
            DataLineage object
        """
        lineage = DataLineage(
            record_id=record_id,
            record_type=record_type,
            source=source,
            created_at=time.time(),
            metadata=metadata or {}
        )

        self.store_lineage(lineage)
        return lineage

    def store_lineage(self, lineage: DataLineage):
        """
        Store lineage record.

        Args:
            lineage: DataLineage object
        """
        # Evict oldest if at capacity
        if len(self._lineage) >= self.max_records:
            oldest_id = min(self._access_times, key=self._access_times.get)
            del self._lineage[oldest_id]
            del self._access_times[oldest_id]

        self._lineage[lineage.record_id] = lineage
        self._access_times[lineage.record_id] = time.time()

    def get_lineage(self, record_id: str) -> Optional[DataLineage]:
        """
        Get lineage for record.

        Args:
            record_id: Record identifier

        Returns:
            DataLineage object or None
        """
        if record_id in self._lineage:
            self._access_times[record_id] = time.time()
            return self._lineage[record_id]
        return None

    def get_lineage_summary(self, record_id: str) -> Optional[Dict[str, Any]]:
        """
        Get lineage summary for record.

        Args:
            record_id: Record identifier

        Returns:
            Dictionary with lineage summary
        """
        lineage = self.get_lineage(record_id)
        return lineage.to_dict() if lineage else None

    def query_lineage(
        self,
        record_type: Optional[str] = None,
        source: Optional[str] = None,
        min_age: Optional[float] = None,
        max_age: Optional[float] = None
    ) -> List[DataLineage]:
        """
        Query lineage records.

        Args:
            record_type: Filter by record type
            source: Filter by source
            min_age: Minimum age in seconds
            max_age: Maximum age in seconds

        Returns:
            List of matching DataLineage objects
        """
        results = []
        current_time = time.time()

        for lineage in self._lineage.values():
            # Apply filters
            if record_type and lineage.record_type != record_type:
                continue
            if source and lineage.source != source:
                continue

            age = current_time - lineage.created_at
            if min_age is not None and age < min_age:
                continue
            if max_age is not None and age > max_age:
                continue

            results.append(lineage)

        return results

    def get_lineage_stats(self) -> Dict[str, Any]:
        """
        Get lineage statistics.

        Returns:
            Dictionary with lineage stats
        """
        if not self._lineage:
            return {
                'total_records': 0,
                'by_type': {},
                'by_source': {},
                'avg_transformations': 0.0,
                'avg_validations': 0.0
            }

        by_type = defaultdict(int)
        by_source = defaultdict(int)
        total_transformations = 0
        total_validations = 0

        for lineage in self._lineage.values():
            by_type[lineage.record_type] += 1
            by_source[lineage.source] += 1
            total_transformations += len(lineage.transformations)
            total_validations += len(lineage.validation_results)

        num_records = len(self._lineage)

        return {
            'total_records': num_records,
            'by_type': dict(by_type),
            'by_source': dict(by_source),
            'avg_transformations': total_transformations / num_records,
            'avg_validations': total_validations / num_records,
            'oldest_record_age': max(time.time() - l.created_at for l in self._lineage.values()),
            'newest_record_age': min(time.time() - l.created_at for l in self._lineage.values())
        }


# Global instances
_global_monitor: Optional[ChEMBLMonitor] = None
_global_lineage_tracker: Optional[DataLineageTracker] = None


def get_monitor() -> ChEMBLMonitor:
    """Get or create global monitor instance."""
    global _global_monitor
    if _global_monitor is None:
        _global_monitor = ChEMBLMonitor()
    return _global_monitor


def get_lineage_tracker() -> DataLineageTracker:
    """Get or create global lineage tracker instance."""
    global _global_lineage_tracker
    if _global_lineage_tracker is None:
        _global_lineage_tracker = DataLineageTracker()
    return _global_lineage_tracker
