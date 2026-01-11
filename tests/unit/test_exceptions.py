"""
Unit tests for custom exception hierarchy (P0-2).

Tests all custom exception classes and helper functions.
"""

import pytest
from src.core.exceptions import (
    OmniTargetException,
    DatabaseError,
    DatabaseConnectionError,
    DatabaseTimeoutError,
    DatabaseUnavailableError,
    MCPServerError,
    DataValidationError,
    EmptyResultError,
    InvalidGeneSymbolError,
    ConfigurationError,
    MissingConfigurationError,
    ScenarioExecutionError,
    is_transient_error,
    format_error_for_logging,
)


class TestBaseExceptions:
    """Test base exception classes."""

    def test_omnitarget_exception_basic(self):
        """Test basic OmniTargetException."""
        error = OmniTargetException("Test error")
        assert str(error) == "Test error"
        assert error.message == "Test error"
        assert error.details == {}

    def test_omnitarget_exception_with_details(self):
        """Test OmniTargetException with details."""
        details = {"server": "KEGG", "query": "hsa:123"}
        error = OmniTargetException("Test error", details=details)
        assert "Test error" in str(error)
        assert "server=KEGG" in str(error)
        assert "query=hsa:123" in str(error)
        assert error.details == details

    def test_database_error_inheritance(self):
        """Test DatabaseError inherits from OmniTargetException."""
        error = DatabaseError("DB error")
        assert isinstance(error, OmniTargetException)
        assert isinstance(error, DatabaseError)


class TestDatabaseErrors:
    """Test database-related exception classes."""

    def test_database_connection_error(self):
        """Test DatabaseConnectionError."""
        error = DatabaseConnectionError(
            server_name="KEGG",
            message="Connection refused",
            details={"host": "localhost", "port": 8080}
        )
        assert error.server_name == "KEGG"
        assert "Connection refused" in str(error)
        assert "server=KEGG" in str(error)
        assert error.details["host"] == "localhost"

    def test_database_timeout_error(self):
        """Test DatabaseTimeoutError."""
        error = DatabaseTimeoutError(
            server_name="Reactome",
            timeout=30.0,
            query="get_pathway(12345)"
        )
        assert error.server_name == "Reactome"
        assert error.timeout == 30.0
        assert error.query == "get_pathway(12345)"
        assert "30" in str(error)
        assert "Reactome" in str(error)
        assert error.details["timeout_seconds"] == 30.0

    def test_database_unavailable_error_without_retry_after(self):
        """Test DatabaseUnavailableError without retry_after."""
        error = DatabaseUnavailableError(
            server_name="STRING",
            reason="Circuit breaker open"
        )
        assert error.server_name == "STRING"
        assert error.retry_after is None
        assert "unavailable" in str(error).lower()
        assert "STRING" in str(error)

    def test_database_unavailable_error_with_retry_after(self):
        """Test DatabaseUnavailableError with retry_after."""
        error = DatabaseUnavailableError(
            server_name="HPA",
            reason="Too many requests",
            retry_after=60
        )
        assert error.server_name == "HPA"
        assert error.retry_after == 60
        assert error.details["retry_after_seconds"] == 60


class TestMCPServerError:
    """Test MCP server error class."""

    def test_mcp_server_error_basic(self):
        """Test basic MCPServerError."""
        error = MCPServerError(
            server_name="KEGG",
            error_code=-32602,
            error_message="Invalid params"
        )
        assert error.server_name == "KEGG"
        assert error.error_code == -32602
        assert error.error_message == "Invalid params"
        assert error.tool_name is None

    def test_mcp_server_error_with_tool(self):
        """Test MCPServerError with tool name."""
        error = MCPServerError(
            server_name="Reactome",
            error_code=-32601,
            error_message="Method not found",
            tool_name="get_pathway_info"
        )
        assert error.tool_name == "get_pathway_info"
        assert "get_pathway_info" in str(error)

    def test_mcp_server_error_is_retryable_non_retryable_codes(self):
        """Test is_retryable() returns False for client errors."""
        non_retryable_codes = [-32600, -32601, -32602, -32603]

        for code in non_retryable_codes:
            error = MCPServerError(
                server_name="Test",
                error_code=code,
                error_message="Test error"
            )
            assert not error.is_retryable(), f"Code {code} should not be retryable"

    def test_mcp_server_error_is_retryable_server_errors(self):
        """Test is_retryable() returns True for server errors."""
        # Server errors (not in non-retryable list)
        error = MCPServerError(
            server_name="Test",
            error_code=-32000,  # Server error
            error_message="Internal error"
        )
        assert error.is_retryable()

        # No error code (assume retryable)
        error = MCPServerError(
            server_name="Test",
            error_code=None,
            error_message="Unknown error"
        )
        assert error.is_retryable()


class TestDataValidationErrors:
    """Test data validation exception classes."""

    def test_data_validation_error_basic(self):
        """Test basic DataValidationError."""
        error = DataValidationError("Invalid data format")
        assert "Invalid data format" in str(error)
        assert error.fallback_available is False

    def test_data_validation_error_with_details(self):
        """Test DataValidationError with full details."""
        error = DataValidationError(
            message="Score out of range",
            field="validation_score",
            value=1.5,
            expected="0.0-1.0",
            fallback_available=True
        )
        assert error.field == "validation_score"
        assert error.value == 1.5
        assert error.expected == "0.0-1.0"
        assert error.fallback_available is True
        assert "fallback_available" in str(error)

    def test_empty_result_error(self):
        """Test EmptyResultError."""
        error = EmptyResultError(
            query_type="pathway_search",
            query="cancer pathway",
            expected_min=1,
            fallback_available=True
        )
        assert error.query_type == "pathway_search"
        assert error.query == "cancer pathway"
        assert "no results" in str(error).lower()
        assert error.details["expected_min"] == 1

    def test_invalid_gene_symbol_error(self):
        """Test InvalidGeneSymbolError."""
        error = InvalidGeneSymbolError(
            gene_symbol="INVALID123",
            reason="Contains numbers"
        )
        assert error.gene_symbol == "INVALID123"
        assert error.reason == "Contains numbers"
        assert "INVALID123" in str(error)
        assert error.field == "gene_symbol"


class TestConfigurationErrors:
    """Test configuration exception classes."""

    def test_configuration_error(self):
        """Test ConfigurationError."""
        error = ConfigurationError(
            config_key="mcp_servers.kegg.path",
            message="Path does not exist",
            config_file="/path/to/config.json"
        )
        assert error.config_key == "mcp_servers.kegg.path"
        assert error.config_file == "/path/to/config.json"
        assert "Path does not exist" in str(error)

    def test_missing_configuration_error(self):
        """Test MissingConfigurationError."""
        error = MissingConfigurationError(
            config_key="api_key",
            config_file="config.yaml"
        )
        assert error.config_key == "api_key"
        assert "Missing required configuration" in str(error)
        assert "api_key" in str(error)
        assert "config.yaml" in str(error)


class TestScenarioExecutionError:
    """Test scenario execution exception class."""

    def test_scenario_execution_error_basic(self):
        """Test basic ScenarioExecutionError."""
        error = ScenarioExecutionError(
            scenario_id=3,
            scenario_name="Cancer Analysis",
            phase="network_construction"
        )
        assert error.scenario_id == 3
        assert error.scenario_name == "Cancer Analysis"
        assert error.phase == "network_construction"
        assert error.original_error is None
        assert "Scenario 3" in str(error)

    def test_scenario_execution_error_with_original_error(self):
        """Test ScenarioExecutionError with original error."""
        original = ValueError("Invalid input")
        error = ScenarioExecutionError(
            scenario_id=1,
            scenario_name="Disease Network",
            phase="marker_discovery",
            original_error=original
        )
        assert error.original_error is original
        assert error.details["original_error_type"] == "ValueError"
        assert "Invalid input" in error.details["original_error"]


class TestHelperFunctions:
    """Test helper functions."""

    def test_is_transient_error_for_transient_types(self):
        """Test is_transient_error() for transient error types."""
        transient_errors = [
            DatabaseConnectionError("Test", "Connection failed"),
            DatabaseTimeoutError("Test", 30.0, "query"),
            DatabaseUnavailableError("Test", "Down"),
            ConnectionError("Connection lost"),
            TimeoutError("Timed out"),
        ]

        for error in transient_errors:
            assert is_transient_error(error), f"{type(error).__name__} should be transient"

    def test_is_transient_error_for_non_transient_types(self):
        """Test is_transient_error() for non-transient error types."""
        non_transient_errors = [
            DataValidationError("Bad data"),
            InvalidGeneSymbolError("BAD", "Invalid"),
            ConfigurationError("key", "Missing"),
            ValueError("Bad value"),
            TypeError("Bad type"),
        ]

        for error in non_transient_errors:
            assert not is_transient_error(error), f"{type(error).__name__} should not be transient"

    def test_is_transient_error_for_mcp_server_error(self):
        """Test is_transient_error() for MCPServerError."""
        # Retryable server error
        retryable = MCPServerError("Test", -32000, "Server error")
        assert is_transient_error(retryable)

        # Non-retryable client error
        non_retryable = MCPServerError("Test", -32602, "Invalid params")
        assert not is_transient_error(non_retryable)

    def test_format_error_for_logging_basic(self):
        """Test format_error_for_logging() with basic error."""
        error = ValueError("Test error")
        formatted = format_error_for_logging(error)

        assert formatted["error_type"] == "ValueError"
        assert formatted["error_message"] == "Test error"
        assert formatted["is_transient"] is False

    def test_format_error_for_logging_omnitarget_exception(self):
        """Test format_error_for_logging() with OmniTargetException."""
        error = DatabaseConnectionError(
            "KEGG",
            "Connection failed",
            details={"host": "localhost", "port": 8080}
        )
        formatted = format_error_for_logging(error)

        assert formatted["error_type"] == "DatabaseConnectionError"
        assert formatted["is_transient"] is True
        assert formatted["server"] == "KEGG"
        assert formatted["host"] == "localhost"
        assert formatted["port"] == 8080

    def test_format_error_for_logging_with_transient_check(self):
        """Test format_error_for_logging() includes transient check."""
        transient_error = DatabaseTimeoutError("Test", 30.0, "query")
        formatted = format_error_for_logging(transient_error)
        assert formatted["is_transient"] is True

        non_transient_error = DataValidationError("Bad data")
        formatted = format_error_for_logging(non_transient_error)
        assert formatted["is_transient"] is False


class TestExceptionInheritance:
    """Test exception inheritance hierarchy."""

    def test_all_inherit_from_base(self):
        """Test all custom exceptions inherit from OmniTargetException."""
        exception_classes = [
            DatabaseError,
            DatabaseConnectionError,
            DatabaseTimeoutError,
            DatabaseUnavailableError,
            MCPServerError,
            DataValidationError,
            EmptyResultError,
            InvalidGeneSymbolError,
            ConfigurationError,
            MissingConfigurationError,
            ScenarioExecutionError,
        ]

        for exc_class in exception_classes:
            # Create instance with minimal args
            if exc_class == DatabaseConnectionError:
                instance = exc_class("Test", "Error")
            elif exc_class == DatabaseTimeoutError:
                instance = exc_class("Test", 30.0, "query")
            elif exc_class == DatabaseUnavailableError:
                instance = exc_class("Test", "Down")
            elif exc_class == MCPServerError:
                instance = exc_class("Test", -32000, "Error")
            elif exc_class == EmptyResultError:
                instance = exc_class("test", "query")
            elif exc_class == InvalidGeneSymbolError:
                instance = exc_class("GENE", "reason")
            elif exc_class == ConfigurationError:
                instance = exc_class("key", "Error")
            elif exc_class == MissingConfigurationError:
                instance = exc_class("key")
            elif exc_class == ScenarioExecutionError:
                instance = exc_class(1, "Test", "phase")
            else:
                instance = exc_class("Test error")

            assert isinstance(instance, OmniTargetException), \
                f"{exc_class.__name__} should inherit from OmniTargetException"
            assert isinstance(instance, Exception), \
                f"{exc_class.__name__} should inherit from Exception"


class TestExceptionUsagePatterns:
    """Test common exception usage patterns."""

    def test_catch_all_omnitarget_exceptions(self):
        """Test catching all OmniTarget exceptions."""
        errors = [
            DatabaseConnectionError("Test", "Error"),
            DataValidationError("Bad data"),
            MCPServerError("Test", -32000, "Error"),
        ]

        for error in errors:
            try:
                raise error
            except OmniTargetException as e:
                # Should catch all custom exceptions
                assert isinstance(e, OmniTargetException)

    def test_catch_specific_exception_type(self):
        """Test catching specific exception types."""
        error = DatabaseTimeoutError("KEGG", 30.0, "query")

        try:
            raise error
        except DatabaseTimeoutError as e:
            assert e.timeout == 30.0
            assert e.server_name == "KEGG"

    def test_exception_details_serializable(self):
        """Test exception details are JSON-serializable."""
        import json

        error = DatabaseConnectionError(
            "Test",
            "Connection failed",
            details={"host": "localhost", "port": 8080, "retries": 3}
        )

        # Should be able to serialize details
        details_json = json.dumps(error.details)
        assert '"host"' in details_json
        assert '"port"' in details_json


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
