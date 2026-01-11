"""
MRA (Modular Response Analysis) Simulator

Full Modular Response Analysis implementation with matrix operations.
"""

import asyncio
import logging
import time
from typing import Dict, List, Optional, Any, Tuple, Set
import numpy as np
import networkx as nx
from scipy.linalg import solve
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve

from ...models.simulation_models import MRASimulationResult, FeedbackLoop
from ...models.data_models import Interaction, Pathway

logger = logging.getLogger(__name__)


class MRASimulator:
    """
    Full Modular Response Analysis simulator.
    
    Implements steady-state MRA with matrix operations, convergence testing,
    and feedback loop detection.
    """
    
    def __init__(self, network: nx.Graph, mcp_data: Dict[str, Any]):
        """
        Initialize MRA simulator.
        
        Args:
            network: NetworkX graph with protein interactions
            mcp_data: Integrated data from MCP servers
        """
        self.network = network
        self.mcp_data = mcp_data
        
        # Build node index mapping
        self.node_to_idx = {node: idx for idx, node in enumerate(network.nodes())}
        self.idx_to_node = {idx: node for node, idx in self.node_to_idx.items()}
        
        # MRA parameters
        self.max_iterations = 1000
        self.convergence_threshold = 1e-6
        self.regularization_factor = 0.01
        
        # Build context data
        self.pathway_context = self._build_pathway_context()
        self.interaction_confidence = self._extract_string_confidence()
        self.expression_context = self._extract_expression_context()
        
        # Initialize matrices
        self.local_response_matrix = None
        self.global_response_matrix = None
    
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
    
    def _extract_expression_context(self) -> Dict[str, float]:
        """Extract expression context from HPA data."""
        expression_context = {}
        
        if 'hpa_expression' in self.mcp_data:
            for expression in self.mcp_data['hpa_expression']:
                gene = expression.get('gene', '')
                level = expression.get('expression_level', 'Not detected')
                
                # Convert to numerical score
                level_scores = {
                    'Not detected': 0.0,
                    'Low': 0.3,
                    'Medium': 0.6,
                    'High': 1.0
                }
                
                expression_context[gene] = level_scores.get(level, 0.0)
        
        return expression_context
    
    def _build_response_matrix(self, tissue_context: Optional[str] = None) -> np.ndarray:
        """
        Build local response coefficient matrix R.
        
        R[i,j] = influence of node j on node i
        
        Incorporates:
        - STRING confidence weighting (Level 1)
        - Pathway context modifier (Level 2)
        - Expression filtering (Level 3)
        
        Args:
            tissue_context: Optional tissue context for expression filtering
            
        Returns:
            Local response matrix R
            
        Performance: O(edges) instead of O(n²) - only processes actual interactions
        """
        n_nodes = len(self.network.nodes())
        R = np.zeros((n_nodes, n_nodes))
        
        # OPTIMIZATION: Iterate over edges directly instead of O(n²) node pairs
        # This reduces computation from n² to number of edges
        for node_i, node_j in self.network.edges():
            i = self.node_to_idx.get(node_i)
            j = self.node_to_idx.get(node_j)
            
            if i is None or j is None or i == j:
                continue
            
            # Level 1: STRING confidence weighting
            confidence = self.interaction_confidence.get((node_i, node_j), 0.0)
            
            # Level 2: Pathway context modifier
            pathway_modifier = self._get_pathway_context_modifier(node_i, node_j)
            
            # Level 3: Expression filtering
            expression_filter = self._get_expression_filter(node_i, tissue_context)
            
            # Combined response coefficient
            R[i, j] = confidence * pathway_modifier * expression_filter
        
        return R
    
    def _get_pathway_context_modifier(self, node_a: str, node_b: str) -> float:
        """Get pathway context modifier between two nodes."""
        pathways_a = self.pathway_context.get(node_a, set())
        pathways_b = self.pathway_context.get(node_b, set())
        
        if not pathways_a or not pathways_b:
            return 0.4  # Different pathways
        
        # Check for same pathway
        if pathways_a & pathways_b:
            return 1.2  # Same pathway
        
        # Check for connected pathways
        return 0.8  # Connected pathways
    
    def _get_expression_filter(self, node: str, tissue_context: Optional[str] = None) -> float:
        """Get expression filter for node."""
        if node not in self.expression_context:
            return 1.0  # No expression data, assume active
        
        expression_level = self.expression_context[node]
        
        # Apply tissue-specific filtering if context provided
        if tissue_context:
            # This would be enhanced with tissue-specific expression data
            # For now, use general expression level
            return expression_level
        
        return expression_level
    
    async def simulate_perturbation(
        self,
        target_node: str,
        perturbation_type: str = 'inhibit',
        perturbation_strength: float = 0.9,
        tissue_context: Optional[str] = None,
        timeout_seconds: float = 120.0
    ) -> MRASimulationResult:
        """
        Simulate perturbation using full MRA.
        
        Args:
            target_node: Node to perturb
            perturbation_type: 'inhibit' or 'activate'
            perturbation_strength: Strength of perturbation (0-1)
            tissue_context: Optional tissue context
            timeout_seconds: Per-target timeout in seconds (default: 30s)
            
        Returns:
            MRASimulationResult with steady-state analysis
        """
        start_time = time.time()
        logger.info(f"🔬 MRA simulation starting for target: {target_node}")
        
        try:
            # CPU-bound simulation logic - wrap in thread to allow timeout
            def _run_simulation_sync():
                # Build local response matrix
                logger.debug(f"[{target_node}] Building local response matrix...")
                self.local_response_matrix = self._build_response_matrix(tissue_context)
                
                # Initialize perturbation vector
                perturbation_vector = np.zeros(len(self.network.nodes()))
                target_idx = self.node_to_idx[target_node]
                
                # Set perturbation
                perturbation_value = (
                    -perturbation_strength if perturbation_type == 'inhibit'
                    else perturbation_strength
                )
                perturbation_vector[target_idx] = perturbation_value
                
                # Solve for steady state: (I - R)^-1 × p
                logger.debug(f"[{target_node}] Solving for steady state...")
                steady_state, convergence_info = self._solve_steady_state(
                    self.local_response_matrix,
                    perturbation_vector
                )
                
                # Build global response matrix
                logger.debug(f"[{target_node}] Building global response matrix...")
                self.global_response_matrix = self._build_global_response_matrix()
                
                # Classify upstream/downstream effects
                upstream_classification = self._classify_upstream_downstream(
                    steady_state, target_idx, 'upstream'
                )
                downstream_classification = self._classify_upstream_downstream(
                    steady_state, target_idx, 'downstream'
                )
                
                # Detect feedback loops
                logger.debug(f"[{target_node}] Detecting feedback loops...")
                feedback_loops = self._detect_feedback_loops(target_node)
                
                # Calculate tissue specificity
                tissue_specificity = self._calculate_tissue_specificity(
                    steady_state, tissue_context
                )
                
                return steady_state, convergence_info, upstream_classification, downstream_classification, feedback_loops, tissue_specificity
            
            # Run CPU-bound code in thread pool with timeout
            # asyncio.to_thread allows proper timeout since it runs in a separate thread
            result = await asyncio.wait_for(
                asyncio.to_thread(_run_simulation_sync),
                timeout=timeout_seconds
            )
            steady_state, convergence_info, upstream_classification, downstream_classification, feedback_loops, tissue_specificity = result
            
        except asyncio.TimeoutError:
            execution_time = time.time() - start_time
            logger.warning(f"⚠️  MRA simulation TIMEOUT for {target_node} after {execution_time:.1f}s (limit: {timeout_seconds}s)")
            
            # Return partial result with timeout flag
            return MRASimulationResult(
                target_node=target_node,
                mode=perturbation_type,
                steady_state={},
                local_response_matrix=[],
                global_response_matrix=[],
                convergence_info={'method': 'timeout', 'iterations': 0, 'converged': False, 'final_error': float('inf')},
                feedback_loops=[],
                upstream_classification={},
                downstream_classification={},
                tissue_specificity={},
                execution_time=execution_time
            )
        
        execution_time = time.time() - start_time
        logger.info(f"✅ MRA simulation complete for {target_node} in {execution_time:.1f}s (converged: {convergence_info.get('converged', False)})")
        
        return MRASimulationResult(
            target_node=target_node,
            mode=perturbation_type,
            steady_state={self.idx_to_node[i]: float(steady_state[i]) 
                         for i in range(len(steady_state))},
            local_response_matrix=self.local_response_matrix.tolist(),
            global_response_matrix=self.global_response_matrix.tolist(),
            convergence_info=convergence_info,
            feedback_loops=feedback_loops,
            upstream_classification=upstream_classification,
            downstream_classification=downstream_classification,
            tissue_specificity=tissue_specificity,
            execution_time=execution_time
        )
    
    def _solve_steady_state(
        self, 
        R: np.ndarray, 
        perturbation: np.ndarray
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Solve steady state: (I - R)^-1 × p
        
        Args:
            R: Local response matrix
            perturbation: Perturbation vector
            
        Returns:
            Steady state vector and convergence info
        """
        n = R.shape[0]
        I = np.eye(n)
        
        # Try direct matrix inversion first
        try:
            A = I - R
            steady_state = solve(A, perturbation)
            convergence_info = {
                'method': 'direct_inversion',
                'iterations': 1,
                'converged': True,
                'final_error': 0.0
            }
        except np.linalg.LinAlgError:
            # Use iterative solver with regularization
            steady_state, convergence_info = self._iterative_solve(
                R, perturbation
            )
        
        return steady_state, convergence_info
    
    def _iterative_solve(
        self, 
        R: np.ndarray, 
        perturbation: np.ndarray
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Iterative solver with regularization for singular matrices.
        
        Args:
            R: Local response matrix
            perturbation: Perturbation vector
            
        Returns:
            Steady state vector and convergence info
        """
        n = R.shape[0]
        I = np.eye(n)
        
        # Add regularization
        R_reg = R + self.regularization_factor * I
        A = I - R_reg
        
        # Iterative solution
        x = np.zeros(n)
        convergence_history = []
        
        for iteration in range(self.max_iterations):
            x_new = R_reg @ x + perturbation
            error = np.max(np.abs(x_new - x))
            convergence_history.append(error)
            
            if error < self.convergence_threshold:
                return x_new, {
                    'method': 'iterative',
                    'iterations': iteration + 1,
                    'converged': True,
                    'final_error': error,
                    'convergence_history': convergence_history
                }
            
            x = x_new
        
        # Did not converge
        return x, {
            'method': 'iterative',
            'iterations': self.max_iterations,
            'converged': False,
            'final_error': convergence_history[-1],
            'convergence_history': convergence_history
        }
    
    def _build_global_response_matrix(self) -> np.ndarray:
        """Build global response matrix (I - R)^-1."""
        if self.local_response_matrix is None:
            raise ValueError("Local response matrix not built")
        
        n = self.local_response_matrix.shape[0]
        I = np.eye(n)
        
        try:
            A = I - self.local_response_matrix
            return np.linalg.inv(A)
        except np.linalg.LinAlgError:
            # Use regularization for singular matrices
            A_reg = A + self.regularization_factor * I
            return np.linalg.inv(A_reg)
    
    def _classify_upstream_downstream(
        self, 
        steady_state: np.ndarray, 
        target_idx: int,
        direction: str
    ) -> Dict[str, str]:
        """
        Classify nodes as upstream or downstream.
        
        Args:
            steady_state: Steady state values
            target_idx: Target node index
            direction: 'upstream' or 'downstream'
            
        Returns:
            Node classification mapping
        """
        classification = {}
        
        for i, value in enumerate(steady_state):
            if i == target_idx:
                continue
            
            node = self.idx_to_node[i]
            
            # Use topological analysis and response magnitude
            try:
                if direction == 'downstream':
                    # Downstream: reachable from target
                    path_length = nx.shortest_path_length(
                        self.network, 
                        self.idx_to_node[target_idx], 
                        node
                    )
                    if path_length <= 3 and abs(value) > 0.1:
                        classification[node] = 'strong_downstream'
                    elif path_length <= 3:
                        classification[node] = 'weak_downstream'
                    else:
                        classification[node] = 'distant_downstream'
                
                elif direction == 'upstream':
                    # Upstream: can reach target
                    path_length = nx.shortest_path_length(
                        self.network, 
                        node, 
                        self.idx_to_node[target_idx]
                    )
                    if path_length <= 3 and abs(value) > 0.1:
                        classification[node] = 'strong_upstream'
                    elif path_length <= 3:
                        classification[node] = 'weak_upstream'
                    else:
                        classification[node] = 'distant_upstream'
                        
            except nx.NetworkXNoPath:
                classification[node] = 'unconnected'
        
        return classification
    
    def _detect_feedback_loops(self, target_node: str) -> List[FeedbackLoop]:
        """
        Detect and classify feedback loops.
        
        Args:
            target_node: Target node
            
        Returns:
            List of detected feedback loops
        """
        feedback_loops = []
        
        # Find cycles containing the target
        try:
            cycles = list(nx.simple_cycles(self.network.to_directed()))
        except:
            # If network is not directed, convert to directed
            directed_network = self.network.to_directed()
            cycles = list(nx.simple_cycles(directed_network))
        
        for cycle in cycles:
            if target_node in cycle and len(cycle) <= 5:  # Reasonable cycle length
                # Classify loop type (simplified)
                loop_type = self._classify_loop_type(cycle)
                
                # Calculate loop strength
                strength = self._calculate_loop_strength(cycle)
                
                # Get pathway context
                pathway_context = self._get_cycle_pathway_context(cycle)
                
                feedback_loop = FeedbackLoop(
                    nodes=cycle,
                    loop_type=loop_type,
                    strength=strength,
                    pathway_context=pathway_context,
                    biological_function=self._infer_biological_function(cycle)
                )
                
                feedback_loops.append(feedback_loop)
        
        return feedback_loops
    
    def _classify_loop_type(self, cycle: List[str]) -> str:
        """Classify feedback loop as positive or negative."""
        # Simplified classification based on interaction types
        # In practice, this would analyze the biological function
        return 'positive'  # Default assumption
    
    def _calculate_loop_strength(self, cycle: List[str]) -> float:
        """Calculate feedback loop strength."""
        if len(cycle) < 2:
            return 0.0
        
        total_confidence = 0.0
        for i in range(len(cycle)):
            node_a = cycle[i]
            node_b = cycle[(i + 1) % len(cycle)]
            confidence = self.interaction_confidence.get((node_a, node_b), 0.0)
            total_confidence += confidence
        
        return total_confidence / len(cycle)
    
    def _get_cycle_pathway_context(self, cycle: List[str]) -> Optional[str]:
        """Get pathway context for feedback loop."""
        pathway_sets = [self.pathway_context.get(node, set()) for node in cycle]
        common_pathways = set.intersection(*pathway_sets) if pathway_sets else set()
        
        if common_pathways:
            return list(common_pathways)[0]  # Return first common pathway
        return None
    
    def _infer_biological_function(self, cycle: List[str]) -> Optional[str]:
        """Infer biological function of feedback loop."""
        # Simplified inference based on known patterns
        if len(cycle) == 2:
            return "Direct feedback"
        elif len(cycle) == 3:
            return "Triangular feedback"
        else:
            return "Complex feedback"
    
    def _calculate_tissue_specificity(
        self, 
        steady_state: np.ndarray, 
        tissue_context: Optional[str]
    ) -> Dict[str, float]:
        """Calculate tissue-specific effect scores."""
        tissue_specificity = {}
        
        for i, value in enumerate(steady_state):
            node = self.idx_to_node[i]
            
            # Get expression level for tissue specificity
            expression_level = self.expression_context.get(node, 1.0)
            
            # Calculate tissue-specific effect
            tissue_effect = abs(value) * expression_level
            
            tissue_specificity[node] = tissue_effect
        
        return tissue_specificity
    
    def update_parameters(
        self,
        max_iterations: Optional[int] = None,
        convergence_threshold: Optional[float] = None,
        regularization_factor: Optional[float] = None
    ) -> None:
        """Update MRA parameters."""
        if max_iterations is not None:
            self.max_iterations = max_iterations
        if convergence_threshold is not None:
            self.convergence_threshold = convergence_threshold
        if regularization_factor is not None:
            self.regularization_factor = regularization_factor
    
    def get_matrix_info(self) -> Dict[str, Any]:
        """Get matrix information for debugging."""
        if self.local_response_matrix is None:
            return {'status': 'not_built'}
        
        return {
            'matrix_size': self.local_response_matrix.shape,
            'sparsity': np.count_nonzero(self.local_response_matrix) / self.local_response_matrix.size,
            'condition_number': np.linalg.cond(self.local_response_matrix),
            'max_eigenvalue': np.max(np.real(np.linalg.eigvals(self.local_response_matrix))),
            'min_eigenvalue': np.min(np.real(np.linalg.eigvals(self.local_response_matrix)))
        }
