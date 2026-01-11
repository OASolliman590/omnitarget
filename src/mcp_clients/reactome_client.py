"""
Reactome MCP Client

Provides type-safe interface to Reactome MCP server with all 8 tools.
Based on Reactome_MCP_Server_Test_Report.md - 100% success rate.
"""

import logging
from typing import Dict, Any, List, Optional
from .base import MCPSubprocessClient
from ..core.retry import REACTOME_RETRY_CONFIG

logger = logging.getLogger(__name__)


class ReactomeClient(MCPSubprocessClient):
    """Reactome MCP client with all 8 validated tools."""
    
    def __init__(self, server_path: str):
        # CRITICAL FIX: Increase timeout to 90s for Reactome (disease queries can be slow)
        # Analysis shows breast cancer queries exceed 60s timeout
        super().__init__(server_path, "Reactome", timeout=90)
    
    # Pathway Discovery Tools
    async def search_pathways(self, query: str, limit: int = 10) -> Dict[str, Any]:
        """Search pathways in Reactome database."""
        # CRITICAL FIX: Use REACTOME_RETRY_CONFIG for ECONNRESET resilience
        return await self.call_tool_with_retry(
            "search_pathways",
            {"query": query, "size": limit},
            retry_config=REACTOME_RETRY_CONFIG
        )
    
    async def get_pathway_details(self, pathway_id: str) -> Dict[str, Any]:
        """Get detailed pathway information."""
        return await self.call_tool_with_retry("get_pathway_details", {
            "id": pathway_id
        })
    
    async def get_pathway_hierarchy(self, pathway_id: str) -> Dict[str, Any]:
        """Get pathway hierarchy structure."""
        return await self.call_tool_with_retry("get_pathway_hierarchy", {
            "id": pathway_id
        })
    
    # Gene-Pathway Mapping Tools
    async def find_pathways_by_gene(self, gene_symbol: str) -> Dict[str, Any]:
        """Find pathways containing a specific gene."""
        # CRITICAL FIX: Use REACTOME_RETRY_CONFIG for ECONNRESET resilience
        return await self.call_tool_with_retry(
            "find_pathways_by_gene",
            {"gene": gene_symbol},
            retry_config=REACTOME_RETRY_CONFIG
        )
    
    async def find_pathways_by_disease(self, disease_name: str, size: int = 10) -> Dict[str, Any]:
        """
        Find pathways associated with a disease.

        PRODUCTION FIX: Uses smaller size limit to avoid timeout, with fallback to search_pathways.
        """
        try:
            # Limit size to 30 for broader pathway discovery (increased from 10 to improve gene coverage)
            # Per MCP documentation max is 100, but 30 provides good balance of coverage vs timeout risk
            limited_size = min(size, 30)
            # CRITICAL FIX: Use REACTOME_RETRY_CONFIG for ECONNRESET resilience
            return await self.call_tool_with_retry(
                "find_pathways_by_disease",
                {"disease": disease_name, "size": limited_size},
                retry_config=REACTOME_RETRY_CONFIG
            )
        except Exception as e:
            error_msg = str(e).lower()
            # Check if error is timeout-related
            if 'timeout' in error_msg or 'exceeded' in error_msg:
                # Fall back to search_pathways (faster, less specific)
                logger.warning(
                    f"Reactome find_pathways_by_disease timed out for '{disease_name}', "
                    f"falling back to search_pathways with limited size"
                )
                try:
                    # Use search_pathways with requested size (up to 30) for broader coverage
                    # Create a temporary client with shorter timeout for fallback
                    fallback_client = ReactomeClient(self.server_path)
                    fallback_client.timeout = 30  # Shorter timeout for fallback
                    await fallback_client.start()
                    try:
                        # CRITICAL FIX: Use REACTOME_RETRY_CONFIG for ECONNRESET resilience
                        result = await fallback_client.call_tool_with_retry(
                            "search_pathways",
                            {"query": disease_name, "size": limited_size},  # Use same limited_size as primary call
                            retry_config=REACTOME_RETRY_CONFIG
                        )
                        return result
                    finally:
                        await fallback_client.stop()
                except Exception as fallback_error:
                    logger.warning(
                        f"Reactome search_pathways fallback also failed for '{disease_name}': {fallback_error}"
                    )
                    # Return empty result structure gracefully
                    return {"pathways": []}
            else:
                # Different error - re-raise
                raise
    
    # Pathway Content Tools
    async def get_pathway_participants(self, pathway_id: str) -> Dict[str, Any]:
        """Get all participants (proteins, complexes, small molecules) in pathway."""
        return await self.call_tool_with_retry("get_pathway_participants", {
            "id": pathway_id
        })
    
    async def get_pathway_reactions(self, pathway_id: str) -> Dict[str, Any]:
        """Get all reactions in a pathway with mechanistic details."""
        return await self.call_tool_with_retry("get_pathway_reactions", {
            "id": pathway_id
        })
    
    # Protein Interaction Tools
    async def get_protein_interactions(self, pathway_id: str) -> Dict[str, Any]:
        """Get protein interactions within pathway context."""
        return await self.call_tool_with_retry("get_protein_interactions", {
            "pathwayId": pathway_id
        })
