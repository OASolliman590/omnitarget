"""
HPA Error Handling and Fallbacks

Provides robust error handling and fallback strategies for HPA queries.
Handles various HPA MCP server errors and provides alternative data sources.

Error Types:
- -32603: Programming error (response too large, buffer exceeded)
- -32700: Invalid JSON response (server crash, empty response)
- -32602: Invalid gene arguments (gene not in HPA database)
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional, Callable, Union
from enum import Enum
from functools import wraps

logger = logging.getLogger(__name__)


class HPAErrorType(Enum):
    """HPA error types."""
    BUFFER_OVERFLOW = "-32603"
    INVALID_RESPONSE = "-32700"
    INVALID_GENE = "-32602"
    TIMEOUT = "timeout"
    UNKNOWN = "unknown"


class HPAErrorInfo:
    """Information about an HPA error."""

    def __init__(
        self,
        error_type: HPAErrorType,
        error_code: Optional[str] = None,
        error_message: str = "",
        gene: Optional[str] = None
    ):
        self.error_type = error_type
        self.error_code = error_code
        self.error_message = error_message
        self.gene = gene
        self.is_retryable = error_type in [
            HPAErrorType.BUFFER_OVERFLOW,
            HPAErrorType.INVALID_RESPONSE,
            HPAErrorType.TIMEOUT
        ]
        self.is_permanent = error_type == HPAErrorType.INVALID_GENE

    def __repr__(self) -> str:
        return f"HPAErrorInfo(type={self.error_type.value}, gene={self.gene}, retryable={self.is_retryable})"


class HPAErrorHandler:
    """
    Handles HPA errors and provides fallback strategies.

    Features:
    - Automatic error classification
    - Retry with exponential backoff
    - Fallback to alternative data sources
    - Gene validation to prevent invalid gene errors
    """

    def __init__(
        self,
        max_retries: int = 3,
        base_delay: float = 1.0,
        backoff_factor: float = 2.0
    ):
        """
        Initialize HPA error handler.

        Args:
            max_retries: Maximum number of retries
            base_delay: Base delay for exponential backoff (seconds)
            backoff_factor: Multiplication factor for each retry
        """
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.backoff_factor = backoff_factor

    def parse_error(self, error: Exception, gene: Optional[str] = None) -> HPAErrorInfo:
        """
        Parse an exception and extract error information.

        Args:
            error: Exception raised
            gene: Gene symbol being queried

        Returns:
            HPAErrorInfo object
        """
        error_str = str(error)

        # Check for known error codes
        if "-32603" in error_str:
            return HPAErrorInfo(
                HPAErrorType.BUFFER_OVERFLOW,
                error_code="-32603",
                error_message=error_str,
                gene=gene
            )
        elif "-32700" in error_str:
            return HPAErrorInfo(
                HPAErrorType.INVALID_RESPONSE,
                error_code="-32700",
                error_message=error_str,
                gene=gene
            )
        elif "-32602" in error_str:
            return HPAErrorInfo(
                HPAErrorType.INVALID_GENE,
                error_code="-32602",
                error_message=error_str,
                gene=gene
            )
        elif "timeout" in error_str.lower():
            return HPAErrorInfo(
                HPAErrorType.TIMEOUT,
                error_message=error_str,
                gene=gene
            )
        else:
            return HPAErrorInfo(
                HPAErrorType.UNKNOWN,
                error_message=error_str,
                gene=gene
            )

    async def retry_with_backoff(
        self,
        func: Callable,
        *args,
        gene: Optional[str] = None,
        **kwargs
    ) -> Optional[Any]:
        """
        Execute function with retry and exponential backoff.

        Args:
            func: Async function to execute
            *args: Positional arguments for function
            gene: Gene symbol (for error context)
            **kwargs: Keyword arguments for function

        Returns:
            Function result or None if all retries failed
        """
        last_error: Optional[Exception] = None
        error_info: Optional[HPAErrorInfo] = None

        for attempt in range(self.max_retries + 1):
            try:
                result = await func(*args, **kwargs)
                if attempt > 0:
                    logger.info(f"HPA query succeeded after {attempt + 1} attempts for gene: {gene}")
                return result

            except Exception as e:
                last_error = e
                error_info = self.parse_error(e, gene)

                # Don't retry permanent errors
                if error_info.is_permanent:
                    logger.warning(
                        f"Permanent HPA error for gene {gene}: {error_info.error_type.value} - {e}"
                    )
                    break

                # Log retryable errors
                if attempt < self.max_retries:
                    delay = self.base_delay * (self.backoff_factor ** attempt)
                    logger.warning(
                        f"HPA query failed (attempt {attempt + 1}/{self.max_retries + 1}) for gene {gene}: "
                        f"{error_info.error_type.value} - {e}. Retrying in {delay:.1f}s..."
                    )
                    await asyncio.sleep(delay)
                else:
                    logger.error(
                        f"HPA query failed after {self.max_retries + 1} attempts for gene {gene}: "
                        f"{error_info.error_type.value} - {e}"
                    )

        # All retries exhausted
        return None

    def is_gene_valid(self, gene: str) -> bool:
        """
        Check if gene symbol appears to be valid format.

        This is a lightweight check to prevent obviously invalid gene symbols
        from being sent to HPA server.

        Args:
            gene: Gene symbol to validate

        Returns:
            True if gene appears valid, False otherwise
        """
        if not gene:
            return False

        # Gene symbols should be alphanumeric
        if not gene.replace('_', '').isalnum():
            return False

        # Should not be too long (typical gene symbols are 1-15 chars)
        if len(gene) > 20:
            return False

        # Should not be all numbers
        if gene.isdigit():
            return False

        return True

    def filter_valid_genes(self, genes: List[str]) -> List[str]:
        """
        Filter list of genes to only include valid-looking gene symbols.

        Args:
            genes: List of gene symbols

        Returns:
            List of valid gene symbols
        """
        valid_genes = []
        invalid_genes = []

        for gene in genes:
            if self.is_gene_valid(gene):
                valid_genes.append(gene)
            else:
                invalid_genes.append(gene)

        if invalid_genes:
            logger.debug(f"Filtered out {len(invalid_genes)} invalid gene symbols: {invalid_genes[:5]}...")

        return valid_genes

    async def get_expression_with_fallback(
        self,
        hpa_client,
        gene: str,
        fallback_data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Get tissue expression with fallback handling.

        Args:
            hpa_client: HPA client instance
            gene: Gene symbol
            fallback_data: Optional fallback data to use if HPA fails

        Returns:
            Expression data from HPA or fallback
        """
        # Validate gene first
        if not self.is_gene_valid(gene):
            logger.warning(f"Skipping invalid gene symbol: {gene}")
            return fallback_data or {"gene": gene, "expression": "N/A"}

        # Try to get from HPA with retry
        result = await self.retry_with_backoff(
            hpa_client.get_tissue_expression,
            gene=gene
        )

        if result is not None:
            return result

        # HPA failed, use fallback
        if fallback_data:
            logger.info(f"Using fallback expression data for {gene}")
            return fallback_data
        else:
            logger.warning(f"No expression data available for {gene} (HPA failed and no fallback)")
            return {"gene": gene, "expression": "N/A"}

    async def batch_get_expression_with_fallback(
        self,
        hpa_client,
        genes: List[str],
        fallback_data: Optional[Dict[str, Dict[str, Any]]] = None
    ) -> Dict[str, Dict[str, Any]]:
        """
        Get tissue expression for multiple genes with error handling.

        Args:
            hpa_client: HPA client instance
            genes: List of gene symbols
            fallback_data: Optional fallback data dict {gene: data}

        Returns:
            Dictionary mapping gene to expression data
        """
        results = {}
        fallback_data = fallback_data or {}

        # Filter invalid genes first
        valid_genes = self.filter_valid_genes(genes)

        for gene in valid_genes:
            result = await self.get_expression_with_fallback(
                hpa_client,
                gene,
                fallback_data.get(gene)
            )
            results[gene] = result

        return results

    def get_error_summary(self, errors: List[HPAErrorInfo]) -> Dict[str, Any]:
        """
        Generate summary of errors encountered.

        Args:
            errors: List of HPAErrorInfo objects

        Returns:
            Dictionary with error summary
        """
        if not errors:
            return {"total_errors": 0}

        error_counts = {}
        retryable_count = 0
        permanent_count = 0

        for error in errors:
            error_type = error.error_type.value
            error_counts[error_type] = error_counts.get(error_type, 0) + 1

            if error.is_retryable:
                retryable_count += 1
            elif error.is_permanent:
                permanent_count += 1

        return {
            "total_errors": len(errors),
            "retryable_errors": retryable_count,
            "permanent_errors": permanent_count,
            "error_breakdown": error_counts,
            "success_rate": 1.0 - (len(errors) / max(len(errors) + 1, 1))  # Estimate
        }


# Global error handler instance
_hpa_error_handler: Optional[HPAErrorHandler] = None


def get_hpa_error_handler() -> HPAErrorHandler:
    """
    Get global HPA error handler instance.

    Returns:
        HPAErrorHandler instance
    """
    global _hpa_error_handler
    if _hpa_error_handler is None:
        _hpa_error_handler = HPAErrorHandler(
            max_retries=3,
            base_delay=1.0,
            backoff_factor=2.0
        )
    return _hpa_error_handler


# Convenience decorator for HPA queries
def with_hpa_error_handling(max_retries: int = 3):
    """
    Decorator to add HPA error handling to functions.

    Args:
        max_retries: Maximum number of retries

    Example:
        @with_hpa_error_handling(max_retries=2)
        async def get_expression(hpa_client, gene):
            return await hpa_client.get_tissue_expression(gene)
    """
    def decorator(func: Callable) -> Callable:
        handler = get_hpa_error_handler()
        handler.max_retries = max_retries

        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Extract gene from kwargs for context
            gene = kwargs.get('gene')
            if not gene and len(args) > 1:
                gene = args[1]  # Assume gene is second arg

            return await handler.retry_with_backoff(func, *args, gene=gene, **kwargs)

        return wrapper
    return decorator
