"""
Simplified Perturbation Simulator

Confidence-weighted breadth-first propagation with pathway context.
"""

import asyncio
import logging
import time
from typing import Dict, List, Optional, Any, Set
from collections import deque
import networkx as nx

from ...models.simulation_models import SimulationConfig, SimulationResult
from ...models.data_models import Interaction, Pathway

logger = logging.getLogger(__name__)


class SimplePerturbationSimulator:
    """
    Simplified perturbation simulator using confidence-weighted BFS propagation.
    
    Implements depth-limited propagation with confidence decay and pathway context weighting.
    """
    
    def __init__(self, network: nx.Graph, mcp_data: Dict[str, Any]):
        """
        Initialize simulator with network and MCP data.
        
        Args:
            network: NetworkX graph with protein interactions
            mcp_data: Integrated data from MCP servers
        """
        self.network = network
        self.mcp_data = mcp_data
        self.pathway_context = self._build_pathway_context()
        self.interaction_confidence = self._extract_string_confidence()
        
        # Simulation parameters from config
        self.max_depth = 3
        self.confidence_threshold = 0.4
        self.decay_factor = 0.7
        self.pathway_weights = {
            'same_pathway': 1.2,
            'connected': 0.8,
            'different': 0.4
        }
    
    def _build_pathway_context(self) -> Dict[str, Set[str]]:
        """Build pathway context mapping from MCP data."""
        pathway_context = {}
        
        # Extract pathways from KEGG and Reactome data
        if 'kegg_pathways' in self.mcp_data:
            for pathway in self.mcp_data['kegg_pathways']:
                pathway_id = pathway.get('id', '')
                genes = pathway.get('genes', [])
                for gene in genes:
                    if gene not in pathway_context:
                        pathway_context[gene] = set()
                    pathway_context[gene].add(pathway_id)
        
        if 'reactome_pathways' in self.mcp_data:
            for pathway in self.mcp_data['reactome_pathways']:
                pathway_id = pathway.get('id', '')
                genes = pathway.get('genes', [])
                for gene in genes:
                    if gene not in pathway_context:
                        pathway_context[gene] = set()
                    pathway_context[gene].add(pathway_id)
        
        return pathway_context
    
    def _extract_string_confidence(self) -> Dict[tuple, float]:
        """Extract STRING confidence scores from MCP data."""
        confidence_scores = {}
        
        if 'string_interactions' in self.mcp_data:
            for interaction in self.mcp_data['string_interactions']:
                protein_a = interaction.get('protein_a', '')
                protein_b = interaction.get('protein_b', '')
                score = interaction.get('combined_score', 0.0)
                
                # Store both directions
                confidence_scores[(protein_a, protein_b)] = score
                confidence_scores[(protein_b, protein_a)] = score
        
        return confidence_scores
    
    def _get_pathway_context(self, node_a: str, node_b: str) -> float:
        """
        Get pathway context modifier between two nodes.
        
        From Mature_development_plan.md:
        - Same pathway: 1.2
        - Connected pathways: 0.8
        - Different pathways: 0.4
        """
        pathways_a = self.pathway_context.get(node_a, set())
        pathways_b = self.pathway_context.get(node_b, set())
        
        if not pathways_a or not pathways_b:
            return self.pathway_weights['different']
        
        # Check for same pathway (exact match or share key pathways)
        if pathways_a == pathways_b:
            return self.pathway_weights['same_pathway']
        
        # Check if they share key pathways (like cancer pathways)
        key_pathways = {'hsa05224', 'R-HSA-73864'}  # Most important pathways
        shared_key_pathways = (pathways_a & pathways_b) & key_pathways
        if shared_key_pathways:
            return self.pathway_weights['same_pathway']
        
        # Check for connected pathways (shared pathways)
        if pathways_a & pathways_b:
            return self.pathway_weights['connected']
        
        # Different pathways
        return self.pathway_weights['different']
    
    def _get_interaction_confidence(self, node_a: str, node_b: str) -> float:
        """Get STRING confidence score for interaction."""
        return self.interaction_confidence.get((node_a, node_b), 0.0)
    
    async def simulate_perturbation(
        self,
        target_node: str,
        perturbation_strength: float = 0.9,
        max_depth: Optional[int] = None,
        confidence_threshold: Optional[float] = None,
        mode: str = 'inhibit'
    ) -> SimulationResult:
        """
        Simulate perturbation propagation through network.
        
        Args:
            target_node: Node to perturb
            perturbation_strength: Strength of perturbation (0-1)
            max_depth: Maximum propagation depth
            confidence_threshold: Minimum confidence for propagation
            mode: 'inhibit' or 'activate'
            
        Returns:
            SimulationResult with affected nodes and effects
        """
        start_time = time.time()
        
        # Use instance defaults if not provided
        max_depth = max_depth or self.max_depth
        confidence_threshold = confidence_threshold or self.confidence_threshold
        
        # Check if target node exists in network
        if target_node not in self.network.nodes():
            # Still include the target node in results even if not in network
            return SimulationResult(
                target_node=target_node,
                mode=mode,
                affected_nodes={target_node: perturbation_strength},
                direct_targets=[],
                downstream=[],
                upstream=[],
                feedback_loops=[],
                network_impact={'total_affected': 1, 'mean_effect': perturbation_strength, 'max_effect': perturbation_strength, 'network_coverage': 0.0, 'positive_effects': 1 if perturbation_strength > 0 else 0, 'negative_effects': 1 if perturbation_strength < 0 else 0, 'effect_ratio': 1.0 if perturbation_strength > 0 else 0.0},
                confidence_scores={target_node: 1.0},
                execution_time=0.0
            )
        
        # Initialize tracking structures
        effects = {}
        visited = set()
        queue = deque([(target_node, perturbation_strength, 0)])
        
        # Apply initial perturbation
        effects[target_node] = perturbation_strength
        visited.add(target_node)
        
        # BFS propagation
        while queue:
            current_node, effect_strength, depth = queue.popleft()
            
            if depth >= max_depth:
                continue
            
            # Get neighbors
            neighbors = list(self.network.neighbors(current_node))
            
            for neighbor in neighbors:
                if neighbor in visited:
                    continue
                
                # Get interaction confidence
                confidence = self._get_interaction_confidence(current_node, neighbor)
                
                # Get pathway context
                pathway_modifier = self._get_pathway_context(current_node, neighbor)
                
                # Calculate effective confidence
                effective_confidence = confidence * pathway_modifier
                
                if effective_confidence < confidence_threshold:
                    continue
                
                # Calculate decay by depth
                decay_factor = self.decay_factor ** depth
                
                # Calculate neighbor effect
                neighbor_effect = (
                    effect_strength * 
                    effective_confidence * 
                    decay_factor
                )
                
                # Apply mode (inhibit vs activate)
                if mode == 'inhibit':
                    neighbor_effect = -neighbor_effect
                
                # Store effect
                effects[neighbor] = neighbor_effect
                visited.add(neighbor)
                
                # Add to queue for further propagation
                queue.append((neighbor, abs(neighbor_effect), depth + 1))
        
        # Classify effects
        classification = self._classify_effects(effects, target_node)
        
        # Calculate network impact metrics
        network_impact = self._calculate_network_impact(effects, target_node)
        
        # Calculate confidence scores
        confidence_scores = self._calculate_confidence_scores(effects, target_node)
        
        execution_time = time.time() - start_time
        
        return SimulationResult(
            target_node=target_node,
            mode=mode,
            affected_nodes=effects,
            direct_targets=classification['direct_targets'],
            downstream=classification['downstream'],
            upstream=classification['upstream'],
            feedback_loops=classification['feedback_loops'],
            network_impact=network_impact,
            confidence_scores=confidence_scores,
            execution_time=execution_time
        )
    
    def _classify_effects(
        self, 
        effects: Dict[str, float], 
        target: str
    ) -> Dict[str, List[str]]:
        """
        Classify nodes as direct targets, downstream, upstream, or feedback loops.
        
        Args:
            effects: Node -> effect strength mapping
            target: Target node identifier
            
        Returns:
            Classification dictionary
        """
        classification = {
            'direct_targets': [],
            'downstream': [],
            'upstream': [],
            'feedback_loops': []
        }
        
        # Get direct neighbors
        direct_neighbors = list(self.network.neighbors(target))
        classification['direct_targets'] = [
            node for node in direct_neighbors 
            if node in effects
        ]
        
        # Classify downstream (reachable in ≤3 hops)
        for node in effects:
            if node == target:
                continue
            
            try:
                # Calculate shortest path length
                path_length = nx.shortest_path_length(
                    self.network, 
                    target, 
                    node
                )
                
                if path_length <= 3:
                    classification['downstream'].append(node)
                elif path_length > 3:
                    classification['upstream'].append(node)
                    
            except nx.NetworkXNoPath:
                # No path exists
                classification['upstream'].append(node)
        
        # Detect feedback loops (simplified)
        classification['feedback_loops'] = self._detect_simple_feedback_loops(
            target, 
            effects
        )
        
        return classification
    
    def _detect_simple_feedback_loops(
        self, 
        target: str, 
        effects: Dict[str, float]
    ) -> List[str]:
        """
        Detect simple feedback loops (2-3 node cycles).
        
        Args:
            target: Target node
            effects: Node effects
            
        Returns:
            List of nodes in feedback loops
        """
        feedback_nodes = []
        
        # Check for 2-node cycles
        for node in effects:
            if node == target:
                continue
            
            # Check if there's a path back to target
            try:
                if nx.has_path(self.network, node, target):
                    # Check if it's a short cycle (≤3 nodes)
                    path_length = nx.shortest_path_length(
                        self.network, 
                        node, 
                        target
                    )
                    if path_length <= 3:
                        feedback_nodes.append(node)
            except nx.NetworkXNoPath:
                continue
        
        return feedback_nodes
    
    def _calculate_network_impact(
        self, 
        effects: Dict[str, float], 
        target: str
    ) -> Dict[str, Any]:
        """
        Calculate network-level impact metrics.
        
        Args:
            effects: Node effects
            target: Target node
            
        Returns:
            Network impact metrics
        """
        if not effects:
            return {
                'total_affected': 0,
                'mean_effect': 0.0,
                'max_effect': 0.0,
                'network_coverage': 0.0
            }
        
        # Basic metrics
        total_affected = len(effects) - 1  # Exclude target
        effect_values = [abs(effect) for effect in effects.values()]
        
        mean_effect = sum(effect_values) / len(effect_values)
        max_effect = max(effect_values)
        
        # Network coverage
        total_nodes = len(self.network.nodes())
        network_coverage = total_affected / total_nodes if total_nodes > 0 else 0.0
        
        # Effect distribution
        positive_effects = sum(1 for effect in effects.values() if effect > 0)
        negative_effects = sum(1 for effect in effects.values() if effect < 0)
        
        return {
            'total_affected': total_affected,
            'mean_effect': mean_effect,
            'max_effect': max_effect,
            'network_coverage': network_coverage,
            'positive_effects': positive_effects,
            'negative_effects': negative_effects,
            'effect_ratio': positive_effects / (negative_effects + 1e-6)
        }
    
    def _calculate_confidence_scores(
        self, 
        effects: Dict[str, float], 
        target: str
    ) -> Dict[str, float]:
        """
        Calculate confidence scores for predictions.
        
        Args:
            effects: Node effects
            target: Target node
            
        Returns:
            Confidence scores for each affected node
        """
        confidence_scores = {}
        
        for node, effect in effects.items():
            if node == target:
                confidence_scores[node] = 1.0  # Target is certain
                continue
            
            # Get interaction confidence
            confidence = self._get_interaction_confidence(target, node)
            
            # Get pathway context
            pathway_modifier = self._get_pathway_context(target, node)
            
            # Calculate path length penalty
            try:
                path_length = nx.shortest_path_length(self.network, target, node)
                path_penalty = 0.8 ** path_length
            except nx.NetworkXNoPath:
                path_penalty = 0.1
            
            # Combined confidence score
            combined_confidence = (
                confidence * 
                pathway_modifier * 
                path_penalty
            )
            
            confidence_scores[node] = min(combined_confidence, 1.0)
        
        return confidence_scores
    
    def update_parameters(
        self,
        max_depth: Optional[int] = None,
        confidence_threshold: Optional[float] = None,
        decay_factor: Optional[float] = None,
        pathway_weights: Optional[Dict[str, float]] = None
    ) -> None:
        """
        Update simulation parameters.
        
        Args:
            max_depth: Maximum propagation depth
            confidence_threshold: Minimum confidence threshold
            decay_factor: Propagation decay factor
            pathway_weights: Pathway context weights
        """
        if max_depth is not None:
            self.max_depth = max_depth
        if confidence_threshold is not None:
            self.confidence_threshold = confidence_threshold
        if decay_factor is not None:
            self.decay_factor = decay_factor
        if pathway_weights is not None:
            self.pathway_weights.update(pathway_weights)
    
    def get_network_info(self) -> Dict[str, Any]:
        """Get network information for debugging."""
        return {
            'total_nodes': len(self.network.nodes()),
            'total_edges': len(self.network.edges()),
            'density': nx.density(self.network),
            'average_clustering': nx.average_clustering(self.network),
            'pathway_context_size': len(self.pathway_context),
            'interaction_confidence_size': len(self.interaction_confidence)
        }
