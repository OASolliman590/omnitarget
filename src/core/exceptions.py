"""
OmniTarget Custom Exception Hierarchy

Provides specific exception types for better error handling and debugging.
Part of P0-2: Error Handling & Retry Logic critical fix.

Author: OmniTarget Team
Date: 2025-01-06
"""

from typing import Optional, Dict, Any


class OmniTargetException(Exception):
    """
    Base exception for all OmniTarget errors.

    All custom exceptions should inherit from this class.
    This allows catching all OmniTarget-specific errors with a single except clause.
    """

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        """
        Initialize exception with message and optional details.

        Args:
            message: Human-readable error message
            details: Additional context (server name, parameters, etc.)
        """
        super().__init__(message)
        self.message = message
        self.details = details or {}

    def __str__(self):
        """String representation with details."""
        if self.details:
            details_str = ", ".join(f"{k}={v}" for k, v in self.details.items())
            return f"{self.message} ({details_str})"
        return self.message


# =============================================================================
# Database & MCP Server Errors
# =============================================================================

class DatabaseError(OmniTargetException):
    """
    Base class for database-related errors.

    Use for errors related to external database access (KEGG, Reactome, etc.)
    """
    pass


class DatabaseConnectionError(DatabaseError):
    """
    Database connection failed.

    Raised when unable to establish connection to database or MCP server.
    This is typically a transient error - retry may succeed.
    """

    def __init__(self, server_name: str, message: str, details: Optional[Dict[str, Any]] = None):
        """
        Initialize with server information.

        Args:
            server_name: Name of the database/server (e.g., "KEGG", "Reactome")
            message: Error description
            details: Additional context
        """
        details = details or {}
        details['server'] = server_name
        super().__init__(message, details)
        self.server_name = server_name


class DatabaseTimeoutError(DatabaseError):
    """
    Database query timed out.

    Raised when a database query exceeds the timeout threshold.
    This is typically a transient error - retry may succeed.
    """

    def __init__(self, server_name: str, timeout: float, query: str, details: Optional[Dict[str, Any]] = None):
        """
        Initialize with timeout information.

        Args:
            server_name: Name of the database/server
            timeout: Timeout threshold in seconds
            query: Query that timed out
            details: Additional context
        """
        details = details or {}
        details.update({
            'server': server_name,
            'timeout_seconds': timeout,
            'query': query
        })
        message = f"{server_name} query timed out after {timeout}s: {query}"
        super().__init__(message, details)
        self.server_name = server_name
        self.timeout = timeout
        self.query = query


class DatabaseUnavailableError(DatabaseError):
    """
    Database service is unavailable.

    Raised when database/MCP server is down or circuit breaker is open.
    This typically requires waiting before retry.
    """

    def __init__(self, server_name: str, reason: str, retry_after: Optional[int] = None):
        """
        Initialize with unavailability information.

        Args:
            server_name: Name of the database/server
            reason: Why service is unavailable
            retry_after: Suggested retry delay in seconds
        """
        details = {
            'server': server_name,
            'reason': reason
        }
        if retry_after:
            details['retry_after_seconds'] = retry_after

        message = f"{server_name} is unavailable: {reason}"
        super().__init__(message, details)
        self.server_name = server_name
        self.retry_after = retry_after


class MCPServerError(DatabaseError):
    """
    MCP server returned an error.

    Raised when MCP server returns an error response (not a connection issue).
    This may or may not be retryable depending on the error code.
    """

    def __init__(self, server_name: str, error_code: Optional[int], error_message: str,
                 tool_name: Optional[str] = None):
        """
        Initialize with MCP error information.

        Args:
            server_name: Name of the MCP server
            error_code: MCP error code (e.g., -32602 for invalid params)
            error_message: Error message from server
            tool_name: Name of the tool that failed
        """
        details = {
            'server': server_name,
            'error_code': error_code,
            'error_message': error_message
        }
        if tool_name:
            details['tool'] = tool_name

        message = f"{server_name} MCP error"
        if error_code:
            message += f" [{error_code}]"
        if tool_name:
            message += f" in {tool_name}"
        message += f": {error_message}"

        super().__init__(message, details)
        self.server_name = server_name
        self.error_code = error_code
        self.error_message = error_message
        self.tool_name = tool_name

    def is_retryable(self) -> bool:
        """
        Determine if this error is retryable.

        Returns:
            True if error might succeed on retry, False otherwise
        """
        # Error codes that are NOT retryable (client errors)
        non_retryable_codes = {
            -32600,  # Invalid Request
            -32601,  # Method not found
            -32602,  # Invalid params
            -32603,  # Internal error (usually not retryable)
        }

        # CRITICAL FIX: ECONNRESET errors are retryable even with -32603 code
        # Connection resets are transient network issues that should be retried
        if 'ECONNRESET' in self.error_message.upper() or 'connection reset' in self.error_message.lower():
            return True

        if self.error_code in non_retryable_codes:
            return False

        # Server errors (5xx equivalent) might be retryable
        return True


# =============================================================================
# Data Validation Errors
# =============================================================================

class DataValidationError(OmniTargetException):
    """
    Data validation failed.

    Raised when data does not meet quality or format requirements.
    This is typically NOT retryable - the data itself is the problem.
    """

    def __init__(self, message: str, field: Optional[str] = None,
                 value: Optional[Any] = None, expected: Optional[str] = None,
                 fallback_available: bool = False):
        """
        Initialize with validation details.

        Args:
            message: Validation error description
            field: Field that failed validation
            value: Invalid value
            expected: Expected format/value
            fallback_available: Whether a fallback method exists
        """
        details = {}
        if field:
            details['field'] = field
        if value is not None:
            details['value'] = str(value)
        if expected:
            details['expected'] = expected
        if fallback_available:
            details['fallback_available'] = True

        super().__init__(message, details)
        self.field = field
        self.value = value
        self.expected = expected
        self.fallback_available = fallback_available


class EmptyResultError(DataValidationError):
    """
    Query returned no results.

    Raised when a database query returns empty results unexpectedly.
    This may be expected (no genes in pathway) or unexpected (network issue).
    """

    def __init__(self, query_type: str, query: str, expected_min: int = 1,
                 fallback_available: bool = False):
        """
        Initialize with query information.

        Args:
            query_type: Type of query (e.g., "pathway_search", "gene_lookup")
            query: The query that returned no results
            expected_min: Minimum expected results
            fallback_available: Whether alternative query methods exist
        """
        message = f"{query_type} returned no results for: {query}"
        if expected_min > 0:
            message += f" (expected at least {expected_min})"

        details = {
            'query_type': query_type,
            'query': query,
            'expected_min': expected_min
        }

        super().__init__(message, fallback_available=fallback_available)
        self.details.update(details)
        self.query_type = query_type
        self.query = query


class InvalidGeneSymbolError(DataValidationError):
    """
    Gene symbol is invalid or not recognized.

    Raised when a gene symbol fails validation checks.
    """

    def __init__(self, gene_symbol: str, reason: str):
        """
        Initialize with gene symbol information.

        Args:
            gene_symbol: The invalid gene symbol
            reason: Why it's invalid
        """
        message = f"Invalid gene symbol: {gene_symbol} - {reason}"
        super().__init__(message, field='gene_symbol', value=gene_symbol,
                        expected='Valid gene symbol (2-15 alphanumeric characters)')
        self.gene_symbol = gene_symbol
        self.reason = reason


# =============================================================================
# Configuration Errors
# =============================================================================

class ConfigurationError(OmniTargetException):
    """
    Configuration error.

    Raised when configuration is invalid or missing.
    This is NOT retryable - requires fixing configuration.
    """

    def __init__(self, config_key: str, message: str, config_file: Optional[str] = None):
        """
        Initialize with configuration details.

        Args:
            config_key: Configuration key that's problematic
            message: Error description
            config_file: Path to configuration file
        """
        details = {'config_key': config_key}
        if config_file:
            details['config_file'] = config_file

        super().__init__(message, details)
        self.config_key = config_key
        self.config_file = config_file


class MissingConfigurationError(ConfigurationError):
    """
    Required configuration is missing.

    Raised when required configuration key is not found.
    """

    def __init__(self, config_key: str, config_file: Optional[str] = None):
        """
        Initialize with missing configuration information.

        Args:
            config_key: Missing configuration key
            config_file: Path to configuration file
        """
        message = f"Missing required configuration: {config_key}"
        if config_file:
            message += f" in {config_file}"

        super().__init__(config_key, message, config_file)


# =============================================================================
# Scenario Execution Errors
# =============================================================================

class ScenarioExecutionError(OmniTargetException):
    """
    Scenario execution failed.

    Raised when a scenario fails to complete successfully.
    """

    def __init__(self, scenario_id: int, scenario_name: str, phase: str,
                 original_error: Optional[Exception] = None):
        """
        Initialize with scenario execution details.

        Args:
            scenario_id: Scenario ID (1-6)
            scenario_name: Human-readable scenario name
            phase: Phase that failed (e.g., "marker_discovery", "network_construction")
            original_error: The underlying exception that caused failure
        """
        message = f"Scenario {scenario_id} ({scenario_name}) failed in phase: {phase}"

        details = {
            'scenario_id': scenario_id,
            'scenario_name': scenario_name,
            'phase': phase
        }

        if original_error:
            details['original_error'] = str(original_error)
            details['original_error_type'] = type(original_error).__name__

        super().__init__(message, details)
        self.scenario_id = scenario_id
        self.scenario_name = scenario_name
        self.phase = phase
        self.original_error = original_error


# =============================================================================
# Helper Functions
# =============================================================================

def is_transient_error(error: Exception) -> bool:
    """
    Determine if an error is transient (retryable).

    Args:
        error: The exception to check

    Returns:
        True if error is likely transient and should be retried

    Example:
        >>> try:
        ...     await mcp_client.call_tool(...)
        ... except Exception as e:
        ...     if is_transient_error(e):
        ...         # Retry
        ...     else:
        ...         # Don't retry
    """
    # Transient error types
    transient_types = (
        DatabaseConnectionError,
        DatabaseTimeoutError,
        ConnectionError,
        ConnectionResetError,  # CRITICAL FIX: Add ConnectionResetError
        TimeoutError,
    )

    if isinstance(error, transient_types):
        return True

    # MCPServerError: check if retryable (includes ECONNRESET check)
    if isinstance(error, MCPServerError):
        return error.is_retryable()

    # DatabaseUnavailableError: retryable after delay
    if isinstance(error, DatabaseUnavailableError):
        return True

    # CRITICAL FIX: Check error message for ECONNRESET (handles cases where it's wrapped)
    error_msg = str(error).upper()
    if 'ECONNRESET' in error_msg or 'CONNECTION RESET' in error_msg:
        return True

    # Not transient
    return False


def format_error_for_logging(error: Exception) -> Dict[str, Any]:
    """
    Format exception for structured logging.

    Args:
        error: The exception to format

    Returns:
        Dictionary with error details for logging

    Example:
        >>> logger.error("Operation failed", extra=format_error_for_logging(e))
    """
    base_info = {
        'error_type': type(error).__name__,
        'error_message': str(error),
        'is_transient': is_transient_error(error)
    }

    # Add OmniTarget-specific details
    if isinstance(error, OmniTargetException):
        base_info.update(error.details)

    return base_info
