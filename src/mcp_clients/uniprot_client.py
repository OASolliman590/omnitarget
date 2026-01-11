"""
UniProt MCP Client

Provides access to UniProt protein database via MCP server.
Based on Augmented-Nature UniProt MCP Server implementation.

ENHANCEMENT: Includes automatic reconnection logic for Python-based MCP server
stability issues (connection loss after first call).
"""

import logging
import asyncio
import json
from typing import Dict, Any, List, Optional
from .base import MCPSubprocessClient, MCPError

logger = logging.getLogger(__name__)


class UniProtClient(MCPSubprocessClient):
    """
    Client for UniProt MCP server with automatic reconnection.
    
    Provides access to:
    - Protein search and information
    - Protein features (domains, binding sites)
    - Sequence data
    - Functional annotations
    
    ENHANCEMENT: Automatically reconnects when Python MCP server loses connection.
    Supports both Python and Node.js UniProt MCP servers.
    """
    
    def __init__(self, server_path: str, server_args: Optional[List[str]] = None):
        """Initialize UniProt client with MCP server path.

        For Node.js UniProt MCP (build/index.js), pass only the path and no server_args.
        For Python-based MCP, provide the interpreter path and server_args accordingly.
        """
        super().__init__(server_path, "uniprot", 30, server_args)
        self._reconnection_count = 0
        self._max_reconnections = 3
        # Detect if this is Node.js server (ends with .js)
        self._is_node_server = server_path.endswith('.js')
    
    def _unwrap_node_response(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """
        Unwrap Node.js UniProt MCP response format.
        
        Node.js MCP wraps responses in: {"content": [{"type": "text", "text": "<JSON>"}]}
        Python MCP returns data directly.
        
        Args:
            response: Raw MCP response
            
        Returns:
            Unwrapped data dict
        """
        if not self._is_node_server:
            return response
        
        # Node.js format: unwrap content array
        content = response.get('content', [])
        if isinstance(content, list) and content:
            first_item = content[0]
            if isinstance(first_item, dict) and first_item.get('type') == 'text':
                text = first_item.get('text', '')
                # Try to parse JSON from text
                try:
                    return json.loads(text)
                except (json.JSONDecodeError, TypeError):
                    # If not JSON, return as is
                    return response
        
        return response
    
    async def _call_with_reconnection(
        self,
        tool_name: str,
        params: Dict[str, Any],
        max_attempts: int = 3
    ) -> Dict[str, Any]:
        """
        Call MCP tool with automatic reconnection on connection loss.
        
        Python-based UniProt MCP server has known issue where connection is lost
        after first call. This method automatically reconnects and retries.
        
        Args:
            tool_name: Name of MCP tool
            params: Tool parameters
            max_attempts: Maximum connection attempts
            
        Returns:
            Tool response
        """
        for attempt in range(max_attempts):
            try:
                return await self.call_tool(tool_name, params)
            except MCPError as e:
                error_msg = str(e).lower()
                
                # Check if error is connection-related
                if any(keyword in error_msg for keyword in [
                    'connection lost', 'closed connection', 'pipe closed',
                    'broken pipe', 'connection reset'
                ]):
                    if attempt < max_attempts - 1:
                        logger.warning(
                            f"UniProt MCP connection lost (attempt {attempt + 1}/{max_attempts}). "
                            f"Reconnecting..."
                        )
                        
                        # Reconnect
                        try:
                            await self.stop()
                            await asyncio.sleep(0.5)  # Brief delay before reconnect
                            await self.start()
                            self._reconnection_count += 1
                            logger.info(f"UniProt MCP reconnected (total reconnections: {self._reconnection_count})")
                        except Exception as reconnect_error:
                            logger.error(f"Reconnection failed: {reconnect_error}")
                            raise e
                    else:
                        logger.error(
                            f"UniProt MCP connection lost after {max_attempts} attempts. "
                            f"Giving up."
                        )
                        raise e
                else:
                    # Non-connection error, raise immediately
                    raise e
        
        return {}
    
    async def search_by_gene(
        self, 
        gene: str, 
        organism: str = "human"
    ) -> Dict[str, Any]:
        """
        Search for proteins by gene name.
        
        Args:
            gene: Gene name or symbol (e.g., "BRCA1")
            organism: Organism name (default: "human")
            
        Returns:
            Dict with search results including accession numbers
        """
        try:
            result = await self.call_tool("search_by_gene", {
                "gene": gene,
                "organism": organism
            })
            # Unwrap Node.js response format if needed
            unwrapped = self._unwrap_node_response(result)
            logger.debug(f"UniProt search for gene {gene}: {unwrapped}")
            return unwrapped
        except Exception as e:
            logger.warning(f"UniProt search failed for {gene}: {e}")
            return {"results": []}
    
    async def get_protein_info(self, accession: str) -> Dict[str, Any]:
        """
        Get detailed protein information with automatic reconnection.
        
        Args:
            accession: UniProt accession number (e.g., "P38398")
            
        Returns:
            Dict with protein information:
            - function/description
            - names (recommended, short)
            - subcellular location
            - protein existence level
        """
        try:
            # Use reconnection-aware call for Python MCP stability
            result = await self._call_with_reconnection("get_protein_info", {
                "accession": accession
            })
            # Unwrap Node.js response format if needed
            unwrapped = self._unwrap_node_response(result)
            logger.debug(f"UniProt info for {accession}: retrieved")
            return unwrapped
        except Exception as e:
            logger.warning(f"UniProt info failed for {accession}: {e}")
            return {}
    
    async def get_protein_features(self, accession: str) -> Dict[str, Any]:
        """
        Get protein features and domains.
        
        Args:
            accession: UniProt accession number
            
        Returns:
            Dict with protein features:
            - domains (InterPro, Pfam, SMART)
            - active sites
            - binding sites
            - regions of interest
        """
        try:
            result = await self.call_tool("get_protein_features", {
                "accession": accession
            })
            # Unwrap Node.js response format if needed
            unwrapped = self._unwrap_node_response(result)
            logger.debug(f"UniProt features for {accession}: retrieved")
            return unwrapped
        except Exception as e:
            logger.warning(f"UniProt features failed for {accession}: {e}")
            return {"features": []}
    
    async def get_protein_sequence(
        self, 
        accession: str, 
        format: str = "fasta"
    ) -> str:
        """
        Get protein amino acid sequence.
        
        Args:
            accession: UniProt accession number
            format: Output format ("fasta" or "json")
            
        Returns:
            Protein sequence in requested format
        """
        try:
            result = await self.call_tool("get_protein_sequence", {
                "accession": accession,
                "format": format
            })
            logger.debug(f"UniProt sequence for {accession}: retrieved")
            return result.get("sequence", "") if format == "json" else result
        except Exception as e:
            logger.warning(f"UniProt sequence failed for {accession}: {e}")
            return ""

    async def get_protein_domains_detailed(self, accession: str) -> Dict[str, Any]:
        """
        Get enhanced protein domain annotations (InterPro, Pfam, SMART) when supported
        by the UniProt MCP server (Node.js implementation).

        Args:
            accession: UniProt accession number

        Returns:
            Dict with detailed domain annotations when available. Falls back to
            an empty structure if the tool is not supported by the server.
        """
        try:
            result = await self.call_tool("get_protein_domains_detailed", {
                "accession": accession
            })
            logger.debug(f"UniProt detailed domains for {accession}: retrieved")
            return result
        except Exception as e:
            logger.warning(f"UniProt detailed domains failed for {accession}: {e}")
            return {"domains": []}

    async def search_proteins(
        self,
        query: str,
        limit: int = 100
    ) -> Dict[str, Any]:
        """
        Search UniProt for proteins matching a query string.

        Supports full UniProt query syntax including disease annotations,
        organism filters, and text searches.

        Examples:
            - Simple gene search: "BRCA1"
            - Disease search: "disease:breast cancer AND organism_id:9606"
            - Full-text: "(breast cancer) AND (reviewed:true) AND (organism_id:9606)"

        Args:
            query: UniProt query string (supports full UniProt query syntax)
            limit: Maximum number of results to return (default: 100)

        Returns:
            Dict with search results:
            {
                "results": [
                    {
                        "primaryAccession": "P38398",
                        "uniProtkbId": "BRCA1_HUMAN",
                        "organism": {"scientificName": "Homo sapiens"},
                        "genes": [...],
                        ...
                    },
                    ...
                ],
                "count": <int>
            }
        """
        try:
            result = await self._call_with_reconnection("search_proteins", {
                "query": query,
                "limit": limit
            })
            # Unwrap Node.js response format if needed
            unwrapped = self._unwrap_node_response(result)

            # Log results for debugging
            result_count = len(unwrapped.get('results', [])) if isinstance(unwrapped.get('results'), list) else 0
            logger.debug(f"UniProt search for '{query}': {result_count} results")

            return unwrapped
        except Exception as e:
            logger.warning(f"UniProt search failed for '{query}': {e}")
            return {"results": [], "count": 0}

