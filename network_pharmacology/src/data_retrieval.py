"""
Network Pharmacology Data Retrieval Module

This module provides functions to retrieve gene network and drug interaction data
from multiple biological databases using MCP (Model Context Protocol) clients.

Databases supported:
- UniProt: Protein information and interactions
- Reactome: Pathway data and gene associations
- Open Targets: Drug-target associations
- STRING: Protein-protein interactions
"""

import logging
from typing import Dict, Any, List, Optional, Set
from dataclasses import dataclass
import json

logger = logging.getLogger(__name__)


@dataclass
class GeneInfo:
    """Gene/protein information."""
    symbol: str
    uniprot_id: Optional[str] = None
    description: Optional[str] = None
    pathways: List[str] = None
    interactions: List[str] = None
    centrality: Dict[str, float] = None
    
    def __post_init__(self):
        if self.pathways is None:
            self.pathways = []
        if self.interactions is None:
            self.interactions = []
        if self.centrality is None:
            self.centrality = {}


@dataclass
class DrugInfo:
    """Drug/compound information."""
    drug_id: str
    drug_name: str
    target_protein: str
    repurposing_score: float = 0.0
    safety_score: float = 0.0
    approval_status: str = "unknown"
    mechanism: str = ""


class NetworkPharmacologyRetriever:
    """
    Retrieves network pharmacology data from biological databases.
    
    This class interfaces with MCP clients to fetch:
    - Gene/protein information from UniProt
    - Pathway associations from Reactome/KEGG
    - Drug-target interactions from Open Targets
    - Protein-protein interactions for network construction
    """
    
    def __init__(self, mcp_client_manager=None):
        """
        Initialize the retriever.
        
        Args:
            mcp_client_manager: Optional MCP client manager for database access.
                               If None, will use cached/mock data.
        """
        self.mcp_client = mcp_client_manager
        self.gene_cache: Dict[str, GeneInfo] = {}
        self.drug_cache: Dict[str, DrugInfo] = {}
        
    async def fetch_gene_network(
        self,
        seed_genes: List[str],
        expand_depth: int = 2,
        min_interaction_score: float = 0.7
    ) -> Dict[str, Any]:
        """
        Fetch gene interaction network starting from seed genes.
        
        Args:
            seed_genes: List of gene symbols to start network expansion
            expand_depth: How many levels of interactions to include
            min_interaction_score: Minimum confidence score for interactions
            
        Returns:
            Dictionary containing nodes and edges of the network
        """
        nodes = {}
        edges = []
        visited = set()
        
        # Process seed genes
        current_level = set(seed_genes)
        
        for depth in range(expand_depth + 1):
            next_level = set()
            
            for gene in current_level:
                if gene in visited:
                    continue
                visited.add(gene)
                
                # Fetch gene info
                gene_info = await self._fetch_gene_info(gene)
                if gene_info:
                    nodes[gene] = {
                        'symbol': gene_info.symbol,
                        'uniprot_id': gene_info.uniprot_id,
                        'pathways': gene_info.pathways,
                        'centrality': gene_info.centrality,
                        'depth': depth,
                        'is_seed': gene in seed_genes
                    }
                    
                    # Fetch interactions
                    interactions = await self._fetch_interactions(gene, min_interaction_score)
                    for partner, score in interactions:
                        edges.append({
                            'source': gene,
                            'target': partner,
                            'score': score,
                            'type': 'protein-protein'
                        })
                        if depth < expand_depth:
                            next_level.add(partner)
            
            current_level = next_level
            
        return {
            'nodes': nodes,
            'edges': edges,
            'seed_genes': seed_genes,
            'total_nodes': len(nodes),
            'total_edges': len(edges)
        }
    
    async def fetch_drug_targets(
        self,
        target_genes: List[str],
        max_drugs_per_target: int = 10
    ) -> List[DrugInfo]:
        """
        Fetch drugs targeting the specified genes.
        
        Args:
            target_genes: List of gene symbols to find drugs for
            max_drugs_per_target: Maximum number of drugs per target
            
        Returns:
            List of DrugInfo objects
        """
        all_drugs = []
        
        for gene in target_genes:
            drugs = await self._fetch_drugs_for_target(gene, max_drugs_per_target)
            all_drugs.extend(drugs)
            
        # Sort by repurposing score
        all_drugs.sort(key=lambda d: d.repurposing_score, reverse=True)
        
        return all_drugs
    
    async def _fetch_gene_info(self, gene_symbol: str) -> Optional[GeneInfo]:
        """Fetch gene information from UniProt/databases."""
        if gene_symbol in self.gene_cache:
            return self.gene_cache[gene_symbol]
        
        if self.mcp_client:
            try:
                # Query UniProt via MCP
                result = await self.mcp_client.query(
                    'uniprot',
                    {'gene': gene_symbol, 'organism': 'human'}
                )
                
                if result:
                    gene_info = GeneInfo(
                        symbol=gene_symbol,
                        uniprot_id=result.get('primaryAccession'),
                        description=result.get('proteinName'),
                        pathways=result.get('pathways', []),
                        interactions=result.get('interactions', [])
                    )
                    self.gene_cache[gene_symbol] = gene_info
                    return gene_info
                    
            except Exception as e:
                logger.warning(f"Failed to fetch gene info for {gene_symbol}: {e}")
        
        # Return basic info if MCP unavailable
        return GeneInfo(symbol=gene_symbol)
    
    async def _fetch_interactions(
        self,
        gene_symbol: str,
        min_score: float
    ) -> List[tuple]:
        """Fetch protein-protein interactions from STRING/databases."""
        interactions = []
        
        if self.mcp_client:
            try:
                # Query STRING via MCP
                result = await self.mcp_client.query(
                    'string',
                    {'gene': gene_symbol, 'min_score': min_score}
                )
                
                if result and 'interactions' in result:
                    for inter in result['interactions']:
                        partner = inter.get('partner')
                        score = inter.get('score', 0)
                        if partner and score >= min_score:
                            interactions.append((partner, score))
                            
            except Exception as e:
                logger.warning(f"Failed to fetch interactions for {gene_symbol}: {e}")
        
        return interactions
    
    async def _fetch_drugs_for_target(
        self,
        gene_symbol: str,
        max_drugs: int
    ) -> List[DrugInfo]:
        """Fetch drugs targeting the specified gene."""
        drugs = []
        
        if self.mcp_client:
            try:
                # Query Open Targets via MCP
                result = await self.mcp_client.query(
                    'opentargets',
                    {'target': gene_symbol, 'limit': max_drugs}
                )
                
                if result and 'drugs' in result:
                    for drug_data in result['drugs'][:max_drugs]:
                        drug = DrugInfo(
                            drug_id=drug_data.get('id', ''),
                            drug_name=drug_data.get('name', 'Unknown'),
                            target_protein=gene_symbol,
                            repurposing_score=drug_data.get('score', 0.5),
                            safety_score=drug_data.get('safety', 0.7),
                            approval_status=drug_data.get('status', 'unknown'),
                            mechanism=drug_data.get('mechanism', '')
                        )
                        drugs.append(drug)
                        
            except Exception as e:
                logger.warning(f"Failed to fetch drugs for {gene_symbol}: {e}")
        
        return drugs


def load_network_from_json(filepath: str) -> Dict[str, Any]:
    """
    Load pre-computed network data from JSON file.
    
    Args:
        filepath: Path to JSON file containing network data
        
    Returns:
        Dictionary with network nodes and edges
    """
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    # Extract relevant network data
    results = data.get('results', [])
    scenarios = {r.get('scenario_id'): r.get('data', r) 
                 for r in results if isinstance(r, dict)}
    
    # Get MRA simulation data (Scenario 4)
    s4 = scenarios.get(4, {})
    
    # Get disease network data (Scenario 1)
    s1 = scenarios.get(1, {})
    
    # Get drug repurposing data (Scenario 6)
    s6 = scenarios.get(6, {})
    
    return {
        'network_nodes': s1.get('network_nodes', []),
        'mra_results': s4.get('individual_results', []),
        'drug_candidates': s6.get('candidate_drugs', []),
        'pathways': s1.get('pathways', [])
    }


def extract_gene_interactions(mra_results: List[Dict]) -> Dict[str, Any]:
    """
    Extract gene interaction network from MRA simulation results.
    
    Args:
        mra_results: List of MRA result dictionaries
        
    Returns:
        Dictionary containing nodes and edges
    """
    nodes = {}
    edges = []
    
    for result in mra_results:
        target = result.get('target_node')
        if not target:
            continue
        
        # Add target node
        nodes[target] = {
            'type': 'target',
            'effect': 1.0
        }
        
        # Add direct targets
        for gene in result.get('direct_targets', []):
            if gene not in nodes:
                affected = result.get('affected_nodes', {})
                effect = affected.get(gene, 0.5)
                if isinstance(effect, dict):
                    effect = effect.get('effect', 0.5)
                nodes[gene] = {
                    'type': 'direct',
                    'effect': float(effect)
                }
            edges.append({
                'source': target,
                'target': gene,
                'type': 'direct'
            })
        
        # Add downstream genes
        for gene in result.get('downstream', []):
            if gene not in nodes:
                affected = result.get('affected_nodes', {})
                effect = affected.get(gene, 0.3)
                if isinstance(effect, dict):
                    effect = effect.get('effect', 0.3)
                nodes[gene] = {
                    'type': 'downstream',
                    'effect': float(effect)
                }
        
        # Add feedback loops
        for loop_str in result.get('feedback_loops', []):
            if isinstance(loop_str, str):
                genes = [g.strip() for g in loop_str.split('->')]
                for i in range(len(genes) - 1):
                    src, tgt = genes[i], genes[i+1]
                    for g in [src, tgt]:
                        if g not in nodes:
                            nodes[g] = {'type': 'feedback', 'effect': 0.4}
                    edges.append({
                        'source': src,
                        'target': tgt,
                        'type': 'feedback'
                    })
    
    return {
        'nodes': nodes,
        'edges': edges,
        'summary': {
            'total_nodes': len(nodes),
            'total_edges': len(edges),
            'targets': len([n for n, d in nodes.items() if d['type'] == 'target']),
            'direct': len([n for n, d in nodes.items() if d['type'] == 'direct']),
            'downstream': len([n for n, d in nodes.items() if d['type'] == 'downstream'])
        }
    }
