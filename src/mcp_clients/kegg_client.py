"""
KEGG MCP Client

Provides type-safe interface to KEGG MCP server with all 30 tools.
Based on KEGG_MCP_Server_Test_Report.md - 100% success rate.
"""

import logging
from typing import Dict, Any, List, Optional
from .base import MCPSubprocessClient

logger = logging.getLogger(__name__)


class KEGGClient(MCPSubprocessClient):
    """KEGG MCP client with all 30 validated tools."""
    
    def __init__(self, server_path: str):
        super().__init__(server_path, "KEGG")
        self.target_drugs_available: Optional[bool] = None
    
    # Disease and Pathway Tools
    async def search_diseases(self, query: str, limit: int = 10) -> Dict[str, Any]:
        """Search diseases in KEGG database."""
        return await self.call_tool_with_retry("search_diseases", {
            "query": query,
            "max_results": limit
        })
    
    async def get_disease_info(self, disease_id: str) -> Dict[str, Any]:
        """Get detailed disease information."""
        return await self.call_tool_with_retry("get_disease_info", {
            "disease_id": disease_id
        })
    
    async def search_pathways(self, query: str, limit: int = 10) -> Dict[str, Any]:
        """Search pathways in KEGG database."""
        return await self.call_tool_with_retry("search_pathways", {
            "query": query,
            "limit": limit
        })
    
    async def get_pathway_info(self, pathway_id: str) -> Dict[str, Any]:
        """Get pathway information."""
        return await self.call_tool_with_retry("get_pathway_info", {
            "pathway_id": pathway_id
        })
    
    async def get_pathway_genes(self, pathway_id: str) -> Dict[str, Any]:
        """Get genes in a pathway (FIXED tool)."""
        return await self.call_tool_with_retry("get_pathway_genes", {
            "pathway_id": pathway_id
        })
    
    # Gene and Protein Tools
    async def search_genes(self, query: str, organism: str = "hsa", limit: int = 10) -> Dict[str, Any]:
        """Search genes in KEGG database."""
        return await self.call_tool_with_retry("search_genes", {
            "query": query,
            "organism": organism,
            "limit": limit
        })
    
    async def get_gene_info(self, gene_id: str) -> Dict[str, Any]:
        """Get gene information."""
        return await self.call_tool_with_retry("get_gene_info", {
            "gene_id": gene_id
        })
    
    async def find_related_entries(
        self, 
        source_entries: List[str], 
        source_db: str, 
        target_db: str
    ) -> Dict[str, Any]:
        """
        Find related entries across databases.
        
        FIXED: Uses correct MCP parameter names per KEGG MCP Server documentation.
        """
        return await self.call_tool_with_retry("find_related_entries", {
            "source_entries": source_entries,  # Array format
            "source_db": source_db,
            "target_db": target_db
        })
    
    # Compound and Drug Tools
    async def search_compounds(self, query: str, limit: int = 10) -> Dict[str, Any]:
        """Search compounds in KEGG database."""
        return await self.call_tool_with_retry("search_compounds", {
            "query": query,
            "limit": limit
        })
    
    async def get_compound_info(self, compound_id: str) -> Dict[str, Any]:
        """Get compound information."""
        return await self.call_tool_with_retry("get_compound_info", {
            "compound_id": compound_id
        })
    
    async def search_drugs(self, query: str, limit: int = 10) -> Dict[str, Any]:
        """Search drugs in KEGG database."""
        return await self.call_tool_with_retry("search_drugs", {
            "query": query,
            "limit": limit
        })
    
    async def get_drug_info(self, drug_id: str) -> Dict[str, Any]:
        """Get drug information."""
        return await self.call_tool_with_retry("get_drug_info", {
            "drug_id": drug_id
        })
    
    # Database Information Tools
    async def get_database_info(self, database: str = "kegg") -> Dict[str, Any]:
        """Get database information."""
        return await self.call_tool_with_retry("get_database_info", {
            "database": database
        })
    
    async def list_organisms(self) -> Dict[str, Any]:
        """List available organisms."""
        return await self.call_tool_with_retry("list_organisms", {})
    
    # ID Conversion Tools
    async def convert_identifiers(
        self, 
        ids: List[str], 
        source_db: str, 
        target_db: str
    ) -> Dict[str, Any]:
        """Convert identifiers between databases."""
        return await self.call_tool_with_retry("convert_identifiers", {
            "ids": ids,
            "source_db": source_db,
            "target_db": target_db
        })
    
    # Batch Processing Tools
    async def batch_entry_lookup(self, entry_ids: List[str]) -> Dict[str, Any]:
        """Batch lookup of entries."""
        return await self.call_tool_with_retry("batch_entry_lookup", {
            "entry_ids": entry_ids
        })
    
    # Additional tools from test report
    async def search_enzymes(self, query: str, limit: int = 10) -> Dict[str, Any]:
        """Search enzymes."""
        return await self.call_tool_with_retry("search_enzymes", {
            "query": query,
            "limit": limit
        })
    
    async def get_enzyme_info(self, enzyme_id: str) -> Dict[str, Any]:
        """Get enzyme information."""
        return await self.call_tool_with_retry("get_enzyme_info", {
            "enzyme_id": enzyme_id
        })
    
    async def search_reactions(self, query: str, limit: int = 10) -> Dict[str, Any]:
        """Search reactions."""
        return await self.call_tool_with_retry("search_reactions", {
            "query": query,
            "limit": limit
        })
    
    async def get_reaction_info(self, reaction_id: str) -> Dict[str, Any]:
        """Get reaction information."""
        return await self.call_tool_with_retry("get_reaction_info", {
            "reaction_id": reaction_id
        })
    
    async def get_pathway_compounds(self, pathway_id: str) -> Dict[str, Any]:
        """Get compounds in pathway."""
        return await self.call_tool_with_retry("get_pathway_compounds", {
            "pathway_id": pathway_id
        })
    
    async def get_pathway_reactions(self, pathway_id: str) -> Dict[str, Any]:
        """Get reactions in pathway."""
        return await self.call_tool_with_retry("get_pathway_reactions", {
            "pathway_id": pathway_id
        })
    
    async def get_gene_orthologs(self, gene_id: str) -> Dict[str, Any]:
        """Get gene orthologs."""
        return await self.call_tool_with_retry("get_gene_orthologs", {
            "gene_id": gene_id
        })
    
    async def get_compound_structure(self, compound_id: str, format: str = "mol") -> Dict[str, Any]:
        """Get compound structure."""
        return await self.call_tool_with_retry("get_compound_structure", {
            "compound_id": compound_id,
            "format": format
        })
    
    async def get_drug_targets(self, drug_id: str) -> Dict[str, Any]:
        """Get drug targets."""
        return await self.call_tool_with_retry("get_drug_targets", {
            "drug_id": drug_id
        })
    
    async def get_target_drugs(self, target_id: str) -> Dict[str, Any]:
        """
        Get drugs targeting a protein (compatibility wrapper).

        PRODUCTION FIX: Falls back to search + filter if direct tool unavailable.
        Caches tool availability to avoid repeated API errors.
        """
        # Check cached availability
        if self.target_drugs_available is False:
            logger.debug(f"Skipping get_target_drugs tool (known unavailable) for {target_id}")
            # Go directly to fallback
        else:
            try:
                # Try direct tool first
                result = await self.call_tool_with_retry("get_target_drugs", {
                    "target_id": target_id
                })
                # If successful, mark as available
                if self.target_drugs_available is None:
                    self.target_drugs_available = True
                return result
            except Exception as e:
                error_msg = str(e)
                if 'Unknown tool' in error_msg or 'not found' in error_msg.lower() or '-32601' in error_msg:
                    # Mark as unavailable for future calls
                    if self.target_drugs_available is None:
                        self.target_drugs_available = False
                        logger.info("KEGG get_target_drugs tool not available, disabling for future calls")
                else:
                    # Re-raise other errors
                    raise

        # Fallback logic (indentation level matches the 'else' block or follows it)
        # Since we want to fall through to here if available is False OR if exception caught
        try:
            # Fall back to find_related_entries (gene → drug)
            logger.info(
                f"KEGG get_target_drugs unavailable, using find_related_entries fallback for {target_id}"
            )

            try:
                # Use find_related_entries to find drugs for gene target
                # Convert gene symbol to KEGG gene ID format if needed
                gene_id = target_id
                if not gene_id.startswith('hsa:'):
                    # Try to convert gene symbol to KEGG gene ID
                    # For now, assume it's already in correct format or add 'hsa:' prefix
                    if ':' not in gene_id:
                        gene_id = f"hsa:{gene_id}"
                
                result = await self.find_related_entries(
                    source_entries=[gene_id],
                    source_db="gene",
                    target_db="drug"
                )
                
                # Extract drugs from links dict
                drug_ids = []
                links = None
                if isinstance(result, dict) and 'links' in result:
                    links = result['links']
                    if isinstance(links, dict):
                        gene_links = links.get(gene_id, [])
                        if isinstance(gene_links, list):
                            drug_ids = gene_links
                        elif isinstance(gene_links, str):
                            drug_ids = [gene_links]
                
                if not drug_ids and links:
                    # Try alternative gene_id format
                    alt_gene_id = gene_id.replace('hsa:', '')
                    if alt_gene_id in links:
                        gene_links = links[alt_gene_id]
                        if isinstance(gene_links, list):
                            drug_ids = gene_links
                        elif isinstance(gene_links, str):
                            drug_ids = [gene_links]
                
                # Get full drug info for each drug ID
                target_drugs = []
                for drug_id in drug_ids[:20]:  # Limit to 20 to avoid timeout
                    try:
                        drug_info = await self.call_tool_with_retry("get_drug_info", {
                            "drug_id": drug_id
                        })
                        if drug_info:
                            target_drugs.append(drug_info)
                    except Exception as drug_error:
                        logger.debug(f"Failed to fetch drug {drug_id}: {drug_error}")
                        continue

                logger.info(
                    f"KEGG fallback successful: retrieved {len(target_drugs)} drugs for {target_id}"
                )
                return {
                    'target_id': target_id,
                    'drugs': target_drugs,
                    'source': 'kegg_fallback_find_related'
                }

            except Exception as fallback_error:
                logger.warning(
                    f"KEGG drug search fallback failed for {target_id}: {fallback_error}. "
                    f"Returning empty result."
                )
                return {
                    'target_id': target_id,
                    'drugs': [],
                    'source': 'kegg_fallback_failed'
                }

        except Exception as e:
            logger.error(f"Failed to get target drugs for {target_id}: {e}")
            return []
    
    async def get_pathway_diseases(self, pathway_id: str) -> Dict[str, Any]:
        """Get diseases associated with pathway."""
        return await self.call_tool_with_retry("get_pathway_diseases", {
            "pathway_id": pathway_id
        })
    
    async def get_disease_pathways(self, disease_id: str) -> Dict[str, Any]:
        """Get pathways associated with disease."""
        return await self.call_tool_with_retry("get_disease_pathways", {
            "disease_id": disease_id
        })
    
    async def get_organism_info(self, organism_code: str) -> Dict[str, Any]:
        """Get organism information."""
        return await self.call_tool_with_retry("get_organism_info", {
            "organism_code": organism_code
        })
    
    async def get_pathway_network(self, pathway_id: str) -> Dict[str, Any]:
        """Get pathway network structure."""
        return await self.call_tool_with_retry("get_pathway_network", {
            "pathway_id": pathway_id
        })
    
    async def get_compound_pathways(self, compound_id: str) -> Dict[str, Any]:
        """Get pathways containing compound."""
        return await self.call_tool_with_retry("get_compound_pathways", {
            "compound_id": compound_id
        })
    
    async def get_gene_pathways(self, gene_id: str) -> Dict[str, Any]:
        """
        Get pathways containing gene using find_related_entries.

        Note: The direct 'get_gene_pathways' tool is not available in KEGG MCP server.
        This method uses find_related_entries as the standard approach.
        
        Enhanced: Automatically converts gene symbols to KEGG gene IDs.
        """
        # Use find_related_entries (the standard approach since direct tool doesn't exist)
        logger.debug(f"Retrieving pathways for {gene_id} using find_related_entries")

        try:
            # ENHANCEMENT: Convert gene symbol to KEGG gene ID if needed
            kegg_gene_id = gene_id
            if not gene_id.startswith('hsa:') and ':' not in gene_id:
                # Likely a gene symbol, not a KEGG ID - try to resolve
                try:
                    search_result = await self.search_genes(gene_id, organism="hsa", limit=1)
                    genes = search_result.get('genes', search_result.get('results', []))
                    if genes and len(genes) > 0:
                        first_gene = genes[0]
                        if isinstance(first_gene, dict):
                            kegg_gene_id = first_gene.get('id') or first_gene.get('gene_id') or first_gene.get('entry_id', gene_id)
                        elif isinstance(first_gene, str):
                            kegg_gene_id = first_gene
                        logger.debug(f"[KEGG] Converted gene symbol '{gene_id}' to KEGG ID '{kegg_gene_id}'")
                    else:
                        # No match found, try with hsa: prefix
                        kegg_gene_id = f"hsa:{gene_id}"
                        logger.debug(f"[KEGG] No KEGG match for '{gene_id}', trying '{kegg_gene_id}'")
                except Exception as e:
                    logger.debug(f"[KEGG] Gene symbol conversion failed for '{gene_id}': {e}, using as-is")
                    kegg_gene_id = gene_id
            
            # Use find_related_entries to get pathways
            # FIXED: Use correct MCP parameter names per KEGG MCP Server documentation
            result = await self.find_related_entries(
                source_entries=[kegg_gene_id],  # Array format
                source_db="gene",
                target_db="pathway"
            )

            # DIAGNOSTIC: Log raw response structure to understand actual format
            logger.debug(f"[KEGG] find_related_entries raw response type: {type(result)}")
            if isinstance(result, dict):
                logger.debug(f"[KEGG] Response keys: {list(result.keys())}")
                # Log sample values for first few keys
                for key in list(result.keys())[:5]:
                    value = result[key]
                    value_type = type(value).__name__
                    if isinstance(value, (list, dict, str)):
                        value_len = len(value)
                    else:
                        value_len = 'N/A'
                    logger.debug(f"[KEGG] Response['{key}']: type={value_type}, "
                                 f"is_list={isinstance(value, list)}, "
                                 f"len={value_len}")
                    if isinstance(value, list) and len(value) > 0:
                        first_item = value[0]
                        if isinstance(first_item, dict):
                            logger.debug(f"[KEGG] Response['{key}'][0]: dict with keys={list(first_item.keys())[:5]}")
                        else:
                            logger.debug(f"[KEGG] Response['{key}'][0]: {type(first_item).__name__}, "
                                         f"value={str(first_item)[:100]}")
                    elif isinstance(value, dict) and len(value) > 0:
                        first_key = list(value.keys())[0]
                        logger.debug(f"[KEGG] Response['{key}']['{first_key}']: {type(value[first_key]).__name__}")
            elif isinstance(result, list):
                logger.debug(f"[KEGG] Response is list with {len(result)} items")
                if len(result) > 0:
                    logger.debug(f"[KEGG] Response[0]: type={type(result[0]).__name__}, "
                                 f"value={str(result[0])[:100] if not isinstance(result[0], dict) else 'dict'}")
                    if isinstance(result[0], dict):
                        logger.debug(f"[KEGG] Response[0] keys: {list(result[0].keys())[:5]}")
            else:
                logger.debug(f"[KEGG] Response is unexpected type: {type(result).__name__}, "
                             f"value={str(result)[:200]}")

            # Enhanced extraction with more response format handling
            pathways = []
            if isinstance(result, dict):
                # KEGG find_related_entries returns: {'source_db': 'gene', 'target_db': 'pathway', 'link_count': int, 'links': {gene_id: [pathway_ids]}}
                # Extract pathways from links[gene_id]
                link_count = result.get('link_count', 0)
                logger.debug(f"[KEGG] link_count: {link_count}, looking for gene_id: '{gene_id}'")
                if 'links' in result and isinstance(result['links'], dict):
                    links = result['links']
                    logger.debug(f"[KEGG] links dict has {len(links)} keys")
                    logger.debug(f"[KEGG] Sample links keys: {list(links.keys())[:5]}")
                    # Try direct lookup
                    gene_links = links.get(gene_id, [])
                    if isinstance(gene_links, list) and len(gene_links) > 0:
                        pathways = gene_links
                        logger.debug(f"[KEGG] Found {len(pathways)} pathways in links['{gene_id}']")
                    elif isinstance(gene_links, str):
                        # Single pathway as string
                        pathways = [gene_links]
                        logger.debug(f"[KEGG] Found single pathway in links['{gene_id}']")
                    # If not found, try key variations
                    if not pathways and link_count > 0:
                        logger.debug(f"[KEGG] Direct lookup failed, trying key variations...")
                        variations = [
                            gene_id,
                            gene_id.replace('hsa:', ''),
                            gene_id.replace(':', ''),
                            f"hsa:{gene_id}" if not gene_id.startswith('hsa:') else None,
                            gene_id.upper(),
                            gene_id.lower(),
                        ]
                        for var in variations:
                            if var and var in links:
                                gene_links = links[var]
                                if isinstance(gene_links, list) and len(gene_links) > 0:
                                    pathways = gene_links
                                    logger.debug(f"[KEGG] Found {len(pathways)} pathways with variation '{var}'")
                                    break
                                elif isinstance(gene_links, str):
                                    pathways = [gene_links]
                                    logger.debug(f"[KEGG] Found single pathway with variation '{var}'")
                                    break
                    # If still not found but link_count > 0, check all links
                    if not pathways and link_count > 0:
                        logger.debug(f"[KEGG] Key variations failed, checking all links...")
                        # Sometimes the gene_id format in links doesn't match exactly
                        # Check if any key contains the gene number
                        gene_number = gene_id.split(':')[-1] if ':' in gene_id else gene_id
                        for key, value in links.items():
                            if gene_number in key or key.endswith(gene_number):
                                if isinstance(value, list) and len(value) > 0:
                                    pathways = value
                                    logger.debug(f"[KEGG] Found {len(pathways)} pathways in links['{key}'] (matched by gene number)")
                                    break
                                elif isinstance(value, str):
                                    pathways = [value]
                                    logger.debug(f"[KEGG] Found single pathway in links['{key}'] (matched by gene number)")
                                    break
                # Fallback: Try multiple possible keys
                if not pathways:
                    pathways = (result.get('pathways') or
                                result.get('entries') or
                                result.get('related_entries') or
                                result.get('results') or
                                result.get('data') or
                                [])
                # If still empty, check if values are lists of pathway IDs
                if not pathways:
                    for key, value in result.items():
                        if isinstance(value, list) and len(value) > 0:
                            # Check if first item looks like a pathway ID
                            first_item = value[0]
                            if isinstance(first_item, str):
                                if ('path:' in first_item.lower() or
                                    'hsa' in first_item.lower() or
                                    first_item.startswith('hsa') or
                                    'map' in first_item.lower()):
                                    pathways = value
                                    logger.debug(f"[KEGG] Found pathways in key '{key}' ({len(pathways)} items)")
                                    break
                            elif isinstance(first_item, dict):
                                # Check if dict contains pathway ID
                                dict_str = str(first_item).lower()
                                if 'path:' in dict_str or 'hsa' in dict_str or 'map' in dict_str:
                                    pathways = value
                                    logger.debug(f"[KEGG] Found pathways in key '{key}' ({len(pathways)} items)")
                                    break

                logger.info(
                    f"KEGG fallback successful: retrieved {len(pathways)} pathways for {gene_id}"
                )
                if len(pathways) == 0:
                    logger.warning(
                        f"[KEGG] No pathways extracted from response. "
                        f"Response type: {type(result).__name__}, "
                        f"Keys: {list(result.keys()) if isinstance(result, dict) else 'N/A'}"
                    )
                return {
                    'gene_id': gene_id,
                    'pathways': pathways,
                    'source': 'kegg_fallback_find_related'
                }
            else:
                # If result is a list, check if items are pathway IDs
                pathways_list = []
                if isinstance(result, list):
                    # Check if list items are pathway IDs
                    for item in result:
                        if isinstance(item, str):
                            if ('path:' in item.lower() or
                                'hsa' in item.lower() or
                                item.startswith('hsa') or
                                'map' in item.lower()):
                                pathways_list.append(item)
                        elif isinstance(item, dict):
                            # Extract pathway ID from dict
                            pathway_id = (item.get('id') or
                                          item.get('pathway_id') or
                                          item.get('entry') or
                                          item.get('identifier'))
                            if pathway_id:
                                pathways_list.append(pathway_id)
                    # If no pathway-like items found, use entire list
                    if not pathways_list:
                        pathways_list = result
                logger.info(
                    f"KEGG fallback successful: retrieved {len(pathways_list)} pathways for {gene_id}"
                )
                if len(pathways_list) == 0:
                    logger.warning(
                        f"[KEGG] No pathways extracted from list response. "
                        f"List length: {len(result) if isinstance(result, list) else 0}, "
                        f"First item type: {type(result[0]).__name__ if isinstance(result, list) and len(result) > 0 else 'N/A'}"
                    )
                return {
                    'gene_id': gene_id,
                    'pathways': pathways_list,
                    'source': 'kegg_fallback_find_related'
                }

        except Exception as fallback_error:
            logger.warning(
                f"KEGG fallback failed for {gene_id}: {fallback_error}. "
                f"Returning empty result."
            )
            return {
                'gene_id': gene_id,
                'pathways': [],
                'source': 'kegg_fallback_failed'
            }
    
    async def get_pathway_hierarchy(self, pathway_id: str) -> Dict[str, Any]:
        """Get pathway hierarchy."""
        return await self.call_tool_with_retry("get_pathway_hierarchy", {
            "pathway_id": pathway_id
        })
    
    async def get_database_statistics(self) -> Dict[str, Any]:
        """Get database statistics."""
        return await self.call_tool_with_retry("get_database_statistics", {})
