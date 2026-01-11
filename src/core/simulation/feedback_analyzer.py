"""
Feedback Loop Analyzer

Detect and classify feedback loops in biological networks.
"""

import logging
from typing import Dict, List, Optional, Any, Set, Tuple
import networkx as nx
from collections import defaultdict

from ...models.simulation_models import FeedbackLoop

logger = logging.getLogger(__name__)


class FeedbackAnalyzer:
    """
    Analyzer for detecting and classifying feedback loops in biological networks.
    """
    
    def __init__(self, network: nx.Graph):
        """
        Initialize feedback analyzer.
        
        Args:
            network: NetworkX graph to analyze
        """
        self.network = network
        self.directed_network = network.to_directed()
        
        # Cache for detected loops
        self._loop_cache = {}
        self._cycle_cache = {}
    
    def detect_feedback_loops(
        self, 
        target: str, 
        max_length: int = 5
    ) -> List[FeedbackLoop]:
        """
        Detect feedback loops containing a target node.
        
        Args:
            target: Target node to find loops for
            max_length: Maximum loop length to consider
            
        Returns:
            List of detected feedback loops
        """
        if target not in self.network.nodes():
            logger.warning(f"Target node {target} not found in network")
            return []
        
        # Check cache first
        cache_key = f"{target}_{max_length}"
        if cache_key in self._loop_cache:
            return self._loop_cache[cache_key]
        
        feedback_loops = []
        
        # Find all cycles in the network
        cycles = self._find_cycles(max_length)
        
        # Filter cycles containing the target
        target_cycles = [
            cycle for cycle in cycles 
            if target in cycle
        ]
        
        # Convert cycles to feedback loops
        for cycle in target_cycles:
            feedback_loop = self._create_feedback_loop(cycle, target)
            if feedback_loop:
                feedback_loops.append(feedback_loop)
        
        # Cache results
        self._loop_cache[cache_key] = feedback_loops
        
        return feedback_loops
    
    def _find_cycles(self, max_length: int) -> List[List[str]]:
        """Find all cycles in the network up to max_length."""
        if max_length in self._cycle_cache:
            return self._cycle_cache[max_length]
        
        cycles = []
        
        try:
            # Find simple cycles
            simple_cycles = list(nx.simple_cycles(self.directed_network))
            
            # Filter by length
            cycles = [
                cycle for cycle in simple_cycles 
                if len(cycle) <= max_length
            ]
            
        except Exception as e:
            logger.warning(f"Error finding cycles: {e}")
            # Fallback: find cycles using DFS
            cycles = self._find_cycles_dfs(max_length)
        
        # Cache results
        self._cycle_cache[max_length] = cycles
        
        return cycles
    
    def _find_cycles_dfs(self, max_length: int) -> List[List[str]]:
        """
        Find cycles using depth-first search (fallback method).
        
        Args:
            max_length: Maximum cycle length
            
        Returns:
            List of cycles
        """
        cycles = []
        visited = set()
        path = []
        
        def dfs(node, start_node, depth):
            if depth > max_length:
                return
            
            if node in path:
                # Found a cycle
                cycle_start = path.index(node)
                cycle = path[cycle_start:] + [node]
                if len(cycle) <= max_length:
                    cycles.append(cycle)
                return
            
            if node in visited:
                return
            
            visited.add(node)
            path.append(node)
            
            # Explore neighbors
            for neighbor in self.directed_network.neighbors(node):
                dfs(neighbor, start_node, depth + 1)
            
            path.pop()
            visited.remove(node)
        
        # Start DFS from each node
        for node in self.directed_network.nodes():
            if node not in visited:
                dfs(node, node, 0)
        
        return cycles
    
    def _create_feedback_loop(
        self, 
        cycle: List[str], 
        target: str
    ) -> Optional[FeedbackLoop]:
        """
        Create a FeedbackLoop object from a cycle.
        
        Args:
            cycle: List of nodes forming a cycle
            target: Target node in the cycle
            
        Returns:
            FeedbackLoop object or None
        """
        if len(cycle) < 2:
            return None
        
        # Remove duplicates while preserving order
        unique_cycle = []
        seen = set()
        for node in cycle:
            if node not in seen:
                unique_cycle.append(node)
                seen.add(node)
        
        # Classify loop type
        loop_type = self._classify_loop_type(unique_cycle)
        
        # Calculate loop strength
        strength = self._calculate_loop_strength(unique_cycle)
        
        # Get pathway context
        pathway_context = self._get_cycle_pathway_context(unique_cycle)
        
        # Infer biological function
        biological_function = self._infer_biological_function(unique_cycle)
        
        return FeedbackLoop(
            nodes=unique_cycle,
            loop_type=loop_type,
            strength=strength,
            pathway_context=pathway_context,
            biological_function=biological_function
        )
    
    def _classify_loop_type(self, cycle: List[str]) -> str:
        """
        Classify feedback loop as positive or negative.
        
        Args:
            cycle: List of nodes in the cycle
            
        Returns:
            'positive' or 'negative'
        """
        # Simplified classification based on cycle length and structure
        # In practice, this would analyze the biological function of each interaction
        
        if len(cycle) == 2:
            # Direct feedback: analyze interaction types
            return self._classify_direct_feedback(cycle)
        elif len(cycle) == 3:
            # Triangular feedback: usually positive
            return 'positive'
        else:
            # Complex feedback: analyze overall structure
            return self._classify_complex_feedback(cycle)
    
    def _classify_direct_feedback(self, cycle: List[str]) -> str:
        """Classify direct (2-node) feedback loops."""
        if len(cycle) != 2:
            return 'positive'
        
        node_a, node_b = cycle
        
        # Check for mutual interactions
        has_ab = self.directed_network.has_edge(node_a, node_b)
        has_ba = self.directed_network.has_edge(node_b, node_a)
        
        if has_ab and has_ba:
            # Mutual interaction - likely positive feedback
            return 'positive'
        else:
            # Asymmetric - could be negative feedback
            return 'negative'
    
    def _classify_complex_feedback(self, cycle: List[str]) -> str:
        """Classify complex (>3 node) feedback loops."""
        # Analyze the overall structure
        # Count positive vs negative interactions (simplified)
        
        positive_count = 0
        negative_count = 0
        
        for i in range(len(cycle)):
            node_a = cycle[i]
            node_b = cycle[(i + 1) % len(cycle)]
            
            # This is a simplified classification
            # In practice, you'd analyze the biological function
            if self._is_positive_interaction(node_a, node_b):
                positive_count += 1
            else:
                negative_count += 1
        
        return 'positive' if positive_count >= negative_count else 'negative'
    
    def _is_positive_interaction(self, node_a: str, node_b: str) -> bool:
        """
        Determine if interaction between two nodes is positive.
        
        Args:
            node_a: First node
            node_b: Second node
            
        Returns:
            True if positive interaction, False otherwise
        """
        # Simplified heuristic based on node names and network structure
        # In practice, this would use biological knowledge
        
        # Check for known positive interaction patterns
        positive_patterns = [
            'activation', 'stimulation', 'enhancement', 'promotion'
        ]
        
        # Check for known negative interaction patterns
        negative_patterns = [
            'inhibition', 'repression', 'suppression', 'blocking'
        ]
        
        # This is a placeholder - real implementation would use
        # biological databases and knowledge graphs
        return True  # Default to positive
    
    def _calculate_loop_strength(self, cycle: List[str]) -> float:
        """
        Calculate the strength of a feedback loop.
        
        Args:
            cycle: List of nodes in the cycle
            
        Returns:
            Loop strength (0-1)
        """
        if len(cycle) < 2:
            return 0.0
        
        total_strength = 0.0
        edge_count = 0
        
        for i in range(len(cycle)):
            node_a = cycle[i]
            node_b = cycle[(i + 1) % len(cycle)]
            
            if self.directed_network.has_edge(node_a, node_b):
                # Get edge weight or use default
                edge_data = self.directed_network.get_edge_data(node_a, node_b, {})
                weight = edge_data.get('weight', 0.5)  # Default weight
                total_strength += weight
                edge_count += 1
        
        if edge_count == 0:
            return 0.0
        
        return total_strength / edge_count
    
    def _get_cycle_pathway_context(self, cycle: List[str]) -> Optional[str]:
        """
        Get pathway context for a cycle.
        
        Args:
            cycle: List of nodes in the cycle
            
        Returns:
            Pathway context string or None
        """
        # This would integrate with pathway data from MCP servers
        # For now, return a simplified context
        
        if len(cycle) == 2:
            return "Direct feedback"
        elif len(cycle) == 3:
            return "Triangular feedback"
        else:
            return f"Complex {len(cycle)}-node feedback"
    
    def _infer_biological_function(self, cycle: List[str]) -> Optional[str]:
        """
        Infer the biological function of a feedback loop.
        
        Args:
            cycle: List of nodes in the cycle
            
        Returns:
            Biological function description or None
        """
        if len(cycle) < 2:
            return None
        
        # Simplified inference based on cycle structure
        if len(cycle) == 2:
            return "Direct regulatory feedback"
        elif len(cycle) == 3:
            return "Triangular regulatory circuit"
        elif len(cycle) == 4:
            return "Quadrangular regulatory circuit"
        else:
            return f"Complex {len(cycle)}-node regulatory circuit"
    
    def analyze_loop_impact(
        self, 
        feedback_loop: FeedbackLoop, 
        perturbation_strength: float
    ) -> Dict[str, Any]:
        """
        Analyze the impact of a feedback loop on perturbation.
        
        Args:
            feedback_loop: Feedback loop to analyze
            perturbation_strength: Strength of perturbation
            
        Returns:
            Impact analysis results
        """
        impact_analysis = {
            'loop_strength': feedback_loop.strength,
            'loop_type': feedback_loop.loop_type,
            'node_count': len(feedback_loop.nodes),
            'amplification_factor': self._calculate_amplification_factor(
                feedback_loop, perturbation_strength
            ),
            'stability_impact': self._assess_stability_impact(feedback_loop),
            'biological_significance': self._assess_biological_significance(feedback_loop)
        }
        
        return impact_analysis
    
    def _calculate_amplification_factor(
        self, 
        feedback_loop: FeedbackLoop, 
        perturbation_strength: float
    ) -> float:
        """Calculate amplification factor for the feedback loop."""
        if feedback_loop.loop_type == 'positive':
            # Positive feedback amplifies
            return 1.0 + (feedback_loop.strength * perturbation_strength)
        else:
            # Negative feedback dampens
            return 1.0 - (feedback_loop.strength * perturbation_strength * 0.5)
    
    def _assess_stability_impact(self, feedback_loop: FeedbackLoop) -> str:
        """Assess the stability impact of the feedback loop."""
        if feedback_loop.loop_type == 'positive':
            if feedback_loop.strength > 0.7:
                return "Destabilizing"
            else:
                return "Moderately stabilizing"
        else:
            if feedback_loop.strength > 0.7:
                return "Strongly stabilizing"
            else:
                return "Moderately stabilizing"
    
    def _assess_biological_significance(self, feedback_loop: FeedbackLoop) -> str:
        """Assess the biological significance of the feedback loop."""
        if feedback_loop.strength > 0.8:
            return "High significance"
        elif feedback_loop.strength > 0.5:
            return "Medium significance"
        else:
            return "Low significance"
    
    def get_network_feedback_summary(self) -> Dict[str, Any]:
        """Get summary of all feedback loops in the network."""
        all_cycles = self._find_cycles(max_length=5)
        
        summary = {
            'total_cycles': len(all_cycles),
            'cycle_lengths': defaultdict(int),
            'positive_loops': 0,
            'negative_loops': 0,
            'strong_loops': 0,
            'weak_loops': 0
        }
        
        for cycle in all_cycles:
            summary['cycle_lengths'][len(cycle)] += 1
            
            feedback_loop = self._create_feedback_loop(cycle, cycle[0])
            if feedback_loop:
                if feedback_loop.loop_type == 'positive':
                    summary['positive_loops'] += 1
                else:
                    summary['negative_loops'] += 1
                
                if feedback_loop.strength > 0.5:
                    summary['strong_loops'] += 1
                else:
                    summary['weak_loops'] += 1
        
        return summary
