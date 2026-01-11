"""
Adaptive STRING Network Builder

Provides adaptive network construction that starts conservative and expands
if the network is too small, with automatic batch processing and deduplication.
"""

import os
import logging
from typing import Dict, List, Optional, Any, Tuple, Set
import networkx as nx

logger = logging.getLogger(__name__)


class AdaptiveStringNetworkBuilder:
    """
    Adaptive STRING network builder with progressive expansion.
    
    Starts with conservative parameters and expands if network is too small.
    Enforces maximum edge cap (10000) to prevent performance issues.
    """
    
    def __init__(self, mcp_manager, data_sources: Optional[Dict[str, Any]] = None, config: Optional[Dict[str, Any]] = None):
        """
        Initialize adaptive STRING network builder.
        
        Args:
            mcp_manager: MCPClientManager instance with STRING client
            data_sources: Optional DataSourceStatus dict for tracking
            config: Optional dictionary to override default configuration parameters
        """
        self.mcp_manager = mcp_manager
        self.data_sources = data_sources
        
        # Configuration from environment variables (defaults)
        defaults = {
            'min_nodes': int(os.getenv('STRING_ADAPTIVE_MIN_NODES', '50')),
            'min_edges': int(os.getenv('STRING_ADAPTIVE_MIN_EDGES', '200')),
            'max_edges': int(os.getenv('STRING_ADAPTIVE_MAX_EDGES', '10000')),
            'initial_score': int(os.getenv('STRING_ADAPTIVE_INITIAL_SCORE', '500')),
            'initial_add_nodes': int(os.getenv('STRING_ADAPTIVE_INITIAL_ADD_NODES', '0')),
            'initial_max_genes': int(os.getenv('STRING_ADAPTIVE_INITIAL_MAX_GENES', '100')),
            'batch_size': int(os.getenv('STRING_ADAPTIVE_BATCH_SIZE', '50')),
            'score_steps': [500, 450, 400],
            'add_nodes_steps': [0, 10, 20],
            'max_genes_steps': [100, 150, 200],
            'best_effort_mode': os.getenv('STRING_ADAPTIVE_BEST_EFFORT', 'true').lower() == 'true'
        }
        
        # Override defaults with provided config
        if config:
            defaults.update(config)
            
        self.min_nodes = defaults['min_nodes']
        self.min_edges = defaults['min_edges']
        self.max_edges = defaults['max_edges']
        self.initial_score = defaults['initial_score']
        self.initial_add_nodes = defaults['initial_add_nodes']
        self.initial_max_genes = defaults['initial_max_genes']
        self.batch_size = defaults['batch_size']
        
        # Expansion parameters
        self.score_steps = defaults['score_steps']
        self.add_nodes_steps = defaults['add_nodes_steps']
        self.max_genes_steps = defaults['max_genes_steps']
        
        # Best-effort mode: continue with undersized networks instead of failing
        self.best_effort_mode = defaults['best_effort_mode']
    
    async def build_network(
        self,
        genes: List[str],
        priority_genes: Optional[List[str]] = None,
        data_sources: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Build STRING network with adaptive expansion.
        
        Args:
            genes: List of gene symbols to include in network
            priority_genes: Optional list of high-priority genes (e.g., markers)
            data_sources: Optional DataSourceStatus dict for tracking
            
        Returns:
            Dict with keys: 'nodes', 'edges', 'genes_used', 'expansion_attempts'
        """
        if not genes:
            logger.warning("Empty gene list provided to adaptive STRING builder")
            return {
                'nodes': [],
                'edges': [],
                'genes_used': [],
                'expansion_attempts': 0
            }
        
        # Select genes deterministically
        selected_genes = self._select_genes(genes, priority_genes or [], self.initial_max_genes)
        
        logger.info(
            f"Adaptive STRING network: {len(selected_genes)} genes selected "
            f"(from {len(genes)} total, {len(priority_genes or [])} priority)"
        )
        
        # Try progressive expansion
        expansion_attempt = 0
        max_attempts = len(self.score_steps) * len(self.add_nodes_steps)
        previous_edge_count = 0
        last_add_nodes = 0
        
        for score_idx, required_score in enumerate(self.score_steps):
            for add_nodes_idx, add_nodes in enumerate(self.add_nodes_steps):
                expansion_attempt += 1
                
                # Adjust max_genes if needed (only on later attempts)
                current_max_genes = self.max_genes_steps[min(score_idx, len(self.max_genes_steps) - 1)]
                if expansion_attempt > 3:
                    # Use larger gene set for later expansion attempts
                    selected_genes = self._select_genes(genes, priority_genes or [], current_max_genes)
                
                # Adaptive add_nodes reduction: reduce if previous attempt had many edges
                current_add_nodes = add_nodes
                if expansion_attempt > 1 and previous_edge_count > self.max_edges * 0.7:
                    # If we're at 70% of max_edges, reduce add_nodes
                    current_add_nodes = max(0, add_nodes - 5)
                    logger.debug(
                        f"Reducing add_nodes to {current_add_nodes} (network at {previous_edge_count} edges, "
                        f"70% of max_edges threshold)"
                    )
                last_add_nodes = current_add_nodes
                
                logger.debug(
                    f"Adaptive expansion attempt {expansion_attempt}/{max_attempts}: "
                    f"score={required_score}, add_nodes={current_add_nodes}, max_genes={current_max_genes}"
                )
                
                # Build network with current parameters
                result = await self._build_with_params(
                    selected_genes,
                    required_score,
                    current_add_nodes,
                    data_sources
                )
                
                nodes = result.get('nodes', [])
                edges = result.get('edges', [])
                previous_edge_count = len(edges)
                
                # Check if network size is adequate
                is_adequate, reason = self._check_network_size(nodes, edges)
                
                if is_adequate:
                    logger.info(
                        f"✅ Adaptive network adequate after {expansion_attempt} attempts: "
                        f"{len(nodes)} nodes, {len(edges)} edges ({reason})"
                    )
                    return {
                        'nodes': nodes,
                        'edges': edges,
                        'genes_used': selected_genes,
                        'expansion_attempts': expansion_attempt,
                        'final_score': required_score,
                        'final_add_nodes': current_add_nodes
                    }
                
                # Check if we hit max edges (stop expanding)
                if len(edges) >= self.max_edges:
                    logger.warning(
                        f"⚠️  Network hit max edges cap ({self.max_edges}) after {expansion_attempt} attempts. "
                        f"Stopping expansion: {len(nodes)} nodes, {len(edges)} edges"
                    )
                    return {
                        'nodes': nodes,
                        'edges': edges,
                        'genes_used': selected_genes,
                        'expansion_attempts': expansion_attempt,
                        'final_score': required_score,
                        'final_add_nodes': current_add_nodes
                    }
                
                logger.debug(
                    f"Network too small ({len(nodes)} nodes, {len(edges)} edges): {reason}. "
                    f"Expanding..."
                )
        
        # If we exhausted all expansion attempts, return best result (best-effort mode)
        if self.best_effort_mode:
            logger.warning(
                f"⚠️  Exhausted all expansion attempts ({max_attempts}). "
                f"Best-effort mode: returning undersized network ({len(nodes)} nodes, {len(edges)} edges). "
                f"Analysis will continue but results may be less comprehensive."
            )
        else:
            logger.warning(
                f"⚠️  Exhausted all expansion attempts ({max_attempts}). "
                f"Returning best result: {len(nodes)} nodes, {len(edges)} edges"
            )
        return {
            'nodes': nodes,
            'edges': edges,
            'genes_used': selected_genes,
            'expansion_attempts': expansion_attempt,
            'final_score': required_score,
            'final_add_nodes': last_add_nodes,
            'best_effort': len(nodes) < self.min_nodes or len(edges) < self.min_edges
        }
    
    async def _build_with_params(
        self,
        genes: List[str],
        required_score: int,
        add_nodes: int,
        data_sources: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Build network with specific parameters using batch processing.
        
        Args:
            genes: List of gene symbols
            required_score: STRING required_score parameter
            add_nodes: STRING add_nodes parameter
            data_sources: Optional DataSourceStatus dict for tracking
            
        Returns:
            Dict with 'nodes' and 'edges' lists
        """
        if not genes:
            return {'nodes': [], 'edges': []}
        
        # Batch processing for large gene lists
        if len(genes) > self.batch_size:
            batches = [
                genes[i:i + self.batch_size]
                for i in range(0, len(genes), self.batch_size)
            ]
            
            logger.info(f"Processing {len(batches)} batches of ~{self.batch_size} genes each")
            
            all_nodes_dict = {}  # Deduplicate by string_id
            all_edges_list = []  # Deduplicate by sorted tuple
            all_edges_set = set()
            
            for batch_idx, batch_genes in enumerate(batches, 1):
                batch_result = await self._call_string_api(
                    batch_genes,
                    required_score,
                    add_nodes,
                    data_sources
                )
                
                # Check if we're approaching max_edges before processing this batch
                current_edge_count = len(all_edges_list)
                batch_edge_count = len(batch_result.get('edges', []))
                
                if current_edge_count + batch_edge_count > self.max_edges:
                    # Trim batch edges to stay under max_edges
                    remaining_slots = self.max_edges - current_edge_count
                    if remaining_slots > 0:
                        batch_edges = batch_result.get('edges', [])[:remaining_slots]
                        logger.warning(
                            f"Batch {batch_idx}: Trimming {len(batch_result.get('edges', [])) - remaining_slots} "
                            f"edges to stay under max_edges ({self.max_edges})"
                        )
                    else:
                        logger.warning(
                            f"Batch {batch_idx}: Skipping batch (max_edges reached: {current_edge_count})"
                        )
                        break
                else:
                    batch_edges = batch_result.get('edges', [])
                
                # Deduplicate nodes
                for node in batch_result.get('nodes', []):
                    node_id = node.get('string_id') or node.get('preferred_name', '')
                    if node_id and node_id not in all_nodes_dict:
                        all_nodes_dict[node_id] = node
                
                # Deduplicate edges (use trimmed batch_edges)
                for edge in batch_edges:
                    source = edge.get('protein_a', '')
                    target = edge.get('protein_b', '')
                    if source and target:
                        edge_key = tuple(sorted([source, target]))
                        if edge_key not in all_edges_set:
                            all_edges_set.add(edge_key)
                            all_edges_list.append(edge)
                
                logger.debug(
                    f"  Batch {batch_idx}/{len(batches)}: "
                    f"{len(batch_result.get('nodes', []))} nodes, "
                    f"{len(batch_edges)} edges (total: {len(all_edges_list)})"
                )
                
                # Early termination if max_edges reached
                if len(all_edges_list) >= self.max_edges:
                    logger.info(f"Reached max_edges ({self.max_edges}) after batch {batch_idx}/{len(batches)}")
                    break
            
            nodes = list(all_nodes_dict.values())
            edges = all_edges_list
            
            logger.debug(f"Deduplication: {len(nodes)} unique nodes, {len(edges)} unique edges")
        else:
            # Single call for small gene lists
            result = await self._call_string_api(
                genes,
                required_score,
                add_nodes,
                data_sources
            )
            nodes = result.get('nodes', [])
            edges = result.get('edges', [])
        
        # Post-processing: Trim edges if network exceeds max_edges
        if len(edges) > self.max_edges:
            # Sort edges by confidence score (if available) and keep top max_edges
            edges_with_scores = []
            for edge in edges:
                score = edge.get('confidence_score', edge.get('score', 0))
                edges_with_scores.append((score, edge))
            
            # Sort by score descending
            edges_with_scores.sort(key=lambda x: x[0], reverse=True)
            
            # Keep top max_edges
            trimmed_edges = [edge for _, edge in edges_with_scores[:self.max_edges]]
            
            logger.warning(
                f"Post-processing: Trimmed {len(edges)} edges to {len(trimmed_edges)} "
                f"(max_edges: {self.max_edges})"
            )
            edges = trimmed_edges
        
        return {'nodes': nodes, 'edges': edges}
    
    async def _call_string_api(
        self,
        genes: List[str],
        required_score: int,
        add_nodes: int,
        data_sources: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Call STRING API with error handling.
        
        Args:
            genes: List of gene symbols
            required_score: STRING required_score parameter
            add_nodes: STRING add_nodes parameter
            data_sources: Optional DataSourceStatus dict for tracking (uses instance default if None)
            
        Returns:
            STRING API response dict
        """
        try:
            # Use instance data_sources if not provided
            tracking_sources = data_sources or self.data_sources
            
            result = await self.mcp_manager.string.get_interaction_network(
                protein_ids=genes,
                species="9606",
                required_score=required_score,
                add_nodes=add_nodes
            )
            
            return result if isinstance(result, dict) else {'nodes': [], 'edges': []}
        except Exception as e:
            logger.warning(f"STRING API call failed: {e}")
            return {'nodes': [], 'edges': []}
    
    def _select_genes(
        self,
        genes: List[str],
        priority_genes: List[str],
        max_genes: int
    ) -> List[str]:
        """
        Select genes deterministically with priority.
        
        Args:
            genes: All available genes
            priority_genes: High-priority genes (e.g., markers) to include first
            max_genes: Maximum number of genes to select
            
        Returns:
            Selected gene list (prioritized, then sorted)
        """
        if not genes:
            return []
        
        # Normalize to uppercase for comparison
        priority_set = {g.upper() for g in priority_genes}
        all_genes_upper = {g.upper() for g in genes}
        
        # Separate priority and non-priority
        priority_list = [g for g in genes if g.upper() in priority_set]
        non_priority = [g for g in genes if g.upper() not in priority_set]
        
        # Sort deterministically
        priority_list = sorted(set(priority_list))
        non_priority = sorted(set(non_priority))
        
        # Combine: all priority first, then non-priority up to max
        selected = priority_list[:max_genes]
        remaining_slots = max_genes - len(selected)
        
        if remaining_slots > 0:
            selected.extend(non_priority[:remaining_slots])
        
        return selected[:max_genes]
    
    def _check_network_size(
        self,
        nodes: List,
        edges: List
    ) -> Tuple[bool, str]:
        """
        Check if network size is adequate.
        
        Args:
            nodes: List of network nodes
            edges: List of network edges
            
        Returns:
            Tuple of (is_adequate: bool, reason: str)
        """
        node_count = len(nodes)
        edge_count = len(edges)
        
        if edge_count >= self.max_edges:
            return True, f"at max edges cap ({self.max_edges})"
        
        if node_count >= self.min_nodes and edge_count >= self.min_edges:
            return True, f"nodes ({node_count}) >= {self.min_nodes} and edges ({edge_count}) >= {self.min_edges}"
        
        if node_count < self.min_nodes:
            return False, f"nodes ({node_count}) < {self.min_nodes}"
        
        if edge_count < self.min_edges:
            return False, f"edges ({edge_count}) < {self.min_edges}"
        
        return True, "unknown"
