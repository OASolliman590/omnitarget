"""
Scenario 2: Target-Centric Analysis

Comprehensive protein characterization and druggability assessment.
Based on Mature_development_plan.md Phase 1-5.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any
import networkx as nx

from ..core.mcp_client_manager import MCPClientManager
from ..core.exceptions import (
    DatabaseConnectionError,
    DatabaseTimeoutError,
    MCPServerError,
    EmptyResultError,
    DataValidationError,
    ScenarioExecutionError,
    format_error_for_logging
)
from ..core.data_standardizer import DataStandardizer
from ..core.validation import DataValidator
from ..core.string_network_builder import AdaptiveStringNetworkBuilder
from ..utils.batch_queries import batch_query
from ..utils.scenario_metrics import (
    summarize_expression_profiles,
    summarize_network,
    summarize_pathways,
)
from ..models.data_models import (
    Protein, Pathway, Interaction, ExpressionProfile,
    CancerMarker, NetworkNode, NetworkEdge, TargetAnalysisResult,
    DrugInfo, DrugTarget, DataSourceStatus, CompletenessMetrics
)

logger = logging.getLogger(__name__)


class TargetAnalysisScenario:
    """
    Scenario 2: Target-Centric Analysis
    
    5-phase workflow:
    1. Multi-database target resolution (STRING + HPA + KEGG)
    2. Pathway membership analysis (Reactome primary)
    3. Interaction network (STRING + Reactome mechanistic)
    4. Expression profiling (HPA tissue-specific)
    5. Druggability assessment (KEGG drugs + localization)
    """
    
    def __init__(self, mcp_manager: MCPClientManager):
        """Initialize target analysis scenario."""
        self.mcp_manager = mcp_manager
        self.standardizer = DataStandardizer()
        self.validator = DataValidator()
        self._active_data_sources: Optional[Dict[str, Any]] = None
        self.string_builder = None  # Will be initialized when data_sources available
    
    async def execute(self, target_query: str, tissue_context: Optional[str] = None) -> TargetAnalysisResult:
        """
        Execute complete target-centric analysis workflow.

        Args:
            target_query: Target protein name or identifier

        Returns:
            TargetAnalysisResult with complete analysis
        """
        logger.info(f"Starting target analysis for: {target_query}")

        # Initialize data source tracking
        data_sources = {
            'kegg': DataSourceStatus(source_name='kegg', requested=0, successful=0, failed=0, success_rate=0.0, error_types=[]),
            'reactome': DataSourceStatus(source_name='reactome', requested=0, successful=0, failed=0, success_rate=0.0, error_types=[]),
            'string': DataSourceStatus(source_name='string', requested=0, successful=0, failed=0, success_rate=0.0, error_types=[]),
            'hpa': DataSourceStatus(source_name='hpa', requested=0, successful=0, failed=0, success_rate=0.0, error_types=[]),
            'uniprot': DataSourceStatus(source_name='uniprot', requested=0, successful=0, failed=0, success_rate=0.0, error_types=[]),
            'chembl': DataSourceStatus(source_name='chembl', requested=0, successful=0, failed=0, success_rate=0.0, error_types=[])
        }

        self._active_data_sources = data_sources
        try:
            # Phase 1: Multi-database target resolution
            target_data = await self._phase1_target_resolution(target_query, data_sources)

            # Ensure target_data is a dict
            if not isinstance(target_data, dict):
                logger.error(f"Target resolution returned non-dict: {type(target_data)}")
                target_data = {'primary_target': None, 'mapping_accuracy': 0.0}

            primary_target = target_data.get('primary_target')
            if not primary_target:
                logger.error("No primary target found")
                raise ValueError("Target resolution failed: no primary target")

            # Phase 2: Pathway membership analysis
            pathway_data = await self._phase2_pathway_membership(primary_target, data_sources)

            # Phase 3: Interaction network (pass pathway_data to access Reactome pathway IDs)
            network_data = await self._phase3_interaction_network(primary_target, pathway_data, data_sources)

            # Phase 4: Expression profiling
            expression_data = await self._phase4_expression_profiling(
                primary_target,
                tissue_context,
                data_sources
            )

            # Phase 5: Druggability assessment
            network = network_data.get('network') if isinstance(network_data, dict) else None
            if not network:
                logger.warning("No network available for druggability assessment")
                network = nx.Graph()

            druggability_data = await self._phase5_druggability_assessment(
                primary_target,
                network,
                data_sources
            )
        finally:
            self._active_data_sources = None

        # Calculate completeness metrics
        completeness_metrics = self._calculate_completeness_metrics(
            target_data, pathway_data, network_data, expression_data, druggability_data
        )

        # Calculate validation score with data source tracking
        validation_score = self._calculate_validation_score(
            target_data, pathway_data, network_data, expression_data, druggability_data, data_sources
        )
        
        # Extract location strings from subcellular_locations for Pydantic model
        subcellular_locations = expression_data.get('subcellular_locations', []) if isinstance(expression_data, dict) else []
        location_strings = []
        if isinstance(subcellular_locations, list):
            for loc in subcellular_locations:
                if isinstance(loc, dict) and 'location' in loc:
                    location_strings.append(loc['location'])
                elif isinstance(loc, str):
                    location_strings.append(loc)
        
        # Convert DrugTarget objects to DrugInfo objects for known_drugs
        drug_targets = druggability_data.get('drugs', []) if isinstance(druggability_data, dict) else []
        known_drugs = []
        for drug_target in drug_targets:
            if isinstance(drug_target, DrugTarget):
                # Convert DrugTarget to DrugInfo
                drug_info = DrugInfo(
                    drug_id=drug_target.drug_id,
                    name=drug_target.drug_id,  # Use drug_id as name if not available
                    indication=None,
                    mechanism=drug_target.mechanism,
                    targets=[drug_target.target_id],
                    development_status=None,
                    drug_class=None,
                    approval_status=None
                )
                known_drugs.append(drug_info)
            elif isinstance(drug_target, DrugInfo):
                # Already a DrugInfo, use as-is
                known_drugs.append(drug_target)
            elif isinstance(drug_target, dict):
                # Try to construct from dict
                try:
                    if 'drug_id' in drug_target and 'target_id' in drug_target:
                        # It's a DrugTarget dict, convert to DrugInfo
                        drug_info = DrugInfo(
                            drug_id=drug_target['drug_id'],
                            name=drug_target.get('name', drug_target['drug_id']),
                            indication=None,
                            mechanism=drug_target.get('mechanism'),
                            targets=[drug_target['target_id']],
                            development_status=None,
                            drug_class=None,
                            approval_status=None
                        )
                        known_drugs.append(drug_info)
                    else:
                        # Assume it's a DrugInfo dict
                        try:
                            known_drugs.append(DrugInfo(**drug_target))
                        except (TypeError, ValueError) as e:
                            # Data structure errors
                            logger.debug(
                                f"Invalid drug_target structure for {target_query}: {e}",
                                extra={'drug_target_keys': list(drug_target.keys()) if isinstance(drug_target, dict) else 'not_dict'}
                            )
                            continue
                except (KeyError, AttributeError) as e:
                    # Missing required fields
                    logger.debug(
                        f"Missing required fields in drug_target for {target_query}: {e}",
                        extra={'error': str(e)}
                    )
                    continue
                except Exception as e:
                    # Unexpected errors
                    logger.debug(
                        f"Failed to convert drug_target to DrugInfo: {type(e).__name__}: {e}",
                        extra=format_error_for_logging(e)
                    )
                    continue
        
        network_summary = summarize_network(network_data.get('network'))
        expression_summary = summarize_expression_profiles(
            expression_data.get('profiles', []),
            total_gene_universe=1  # Single target context
        )
        pathway_summary = summarize_pathways(
            pathway_data.get('pathways', []),
            coverage=pathway_data.get('coverage')
        )
        prioritization_summary = self._build_prioritization_summary(
            network_summary,
            expression_summary,
            druggability_data,
            primary_target
        )

        # Build result with correct field names
        result = TargetAnalysisResult(
            target=target_data.get('primary_target') if isinstance(target_data, dict) else None,
            pathways=pathway_data.get('pathways', []) if isinstance(pathway_data, dict) else [],
            interactions=network_data.get('interactions', []) if isinstance(network_data, dict) else [],
            expression_profiles=expression_data.get('profiles', []) if isinstance(expression_data, dict) else [],
            subcellular_location=location_strings,  # List[str] as expected by model
            druggability_score=druggability_data.get('score', 0.0) if isinstance(druggability_data, dict) else 0.0,
            known_drugs=known_drugs,
            safety_profile=druggability_data.get('safety', {}) if isinstance(druggability_data, dict) else {},
            validation_score=validation_score,
            data_sources=list(data_sources.values()),
            completeness_metrics=completeness_metrics,
            network_summary=network_summary,
            expression_summary=expression_summary,
            pathway_summary=pathway_summary,
            prioritization_summary=prioritization_summary
        )
        
        logger.info(f"Target analysis completed. Validation score: {validation_score:.3f}")
        return result
    
    async def _phase1_target_resolution(self, query: str, data_sources: Dict[str, DataSourceStatus]) -> Dict[str, Any]:
        """
        Phase 1: Multi-database target resolution.

        Resolve target across STRING, HPA, and KEGG databases.
        """
        logger.info("Phase 1: Multi-database target resolution")

        # Parallel MCP calls for comprehensive target resolution
        tasks = [
            self._call_with_tracking(
                data_sources,
                'string',
                self.mcp_manager.string.search_proteins(query, limit=5)
            ),
            self._call_with_tracking(
                data_sources,
                'hpa',
                self.mcp_manager.hpa.search_proteins(query, max_results=5)
            ),
            self._call_with_tracking(
                data_sources,
                'kegg',
                self.mcp_manager.kegg.search_genes(query, limit=5)
            )
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)
        string_proteins = self._normalize_async_result(results[0], 'string')
        hpa_proteins = self._normalize_async_result(results[1], 'hpa')
        kegg_proteins = self._normalize_async_result(results[2], 'kegg')
        
        # Standardize results and track successes
        proteins = []

        if isinstance(string_proteins, dict) and string_proteins.get('proteins'):
            for protein_data in string_proteins['proteins']:
                protein = self.standardizer.standardize_string_protein(protein_data)
                if self.validator.validate_protein_confidence(protein):
                    proteins.append(protein)

        # Track HPA results (HPA returns list directly, not dict)
        hpa_success = False
        if isinstance(hpa_proteins, list):
            for protein_data in hpa_proteins:
                protein = self.standardizer.standardize_hpa_protein(protein_data)
                if self.validator.validate_protein_confidence(protein):
                    proteins.append(protein)
            hpa_success = len(hpa_proteins) > 0
        elif isinstance(hpa_proteins, dict) and hpa_proteins.get('proteins'):
            for protein_data in hpa_proteins['proteins']:
                protein = self.standardizer.standardize_hpa_protein(protein_data)
                if self.validator.validate_protein_confidence(protein):
                    proteins.append(protein)
            hpa_success = len(hpa_proteins.get('proteins', [])) > 0

        if isinstance(kegg_proteins, dict) and kegg_proteins.get('genes'):
            for gene_data in kegg_proteins['genes']:
                protein = self.standardizer.standardize_kegg_gene(gene_data)
                if self.validator.validate_protein_confidence(protein):
                    proteins.append(protein)
        
        # Get primary target (highest confidence)
        primary_target = max(proteins, key=lambda p: p.confidence) if proteins else None
        
        # Final fallback to user-provided query
        if not primary_target:
            logger.warning(f"No target found in databases for '{query}', creating fallback target")
            primary_target = Protein(
                gene_symbol=query.upper(),
                uniprot_id=None,
                string_id=None,
                confidence=0.5,
                description=f"User-provided target: {query}"
            )
            proteins.append(primary_target)
        
        logger.info(f"Primary target resolved: {primary_target.gene_symbol} (confidence: {primary_target.confidence:.2f})")
        
        # P2 Enhancement: Fetch UniProt ID if missing (needed for domain-based druggability scoring)
        if not primary_target.uniprot_id:
            if self.mcp_manager.uniprot:
                try:
                    logger.info(f"Attempting UniProt ID resolution for {primary_target.gene_symbol}...")
                    uniprot_search = await self._call_with_tracking(
                        data_sources,
                        'uniprot',
                        self.mcp_manager.uniprot.search_by_gene(
                            primary_target.gene_symbol,
                            organism="human"
                        )
                    )
                    logger.debug(f"UniProt search response: {uniprot_search}")

                    # CRITICAL FIX: Validate response type BEFORE using .get()
                    if isinstance(uniprot_search, str):
                        logger.warning(f"UniProt returned error for {primary_target.gene_symbol}: {uniprot_search}")
                    elif not isinstance(uniprot_search, dict):
                        logger.warning(f"UniProt returned unexpected type: {type(uniprot_search)}")
                    elif uniprot_search.get('results') and len(uniprot_search['results']) > 0:
                        # Extract UniProt accession from first result
                        first_result = uniprot_search['results'][0]
                        # Node.js UniProt MCP returns 'primaryAccession', not 'accession' or 'id'
                        uniprot_accession = first_result.get('primaryAccession') or first_result.get('accession') or first_result.get('id')
                        if uniprot_accession:
                            primary_target.uniprot_id = uniprot_accession
                            logger.info(f"✅ Resolved UniProt ID for {primary_target.gene_symbol}: {uniprot_accession}")
                        else:
                            logger.debug(f"No accession found in UniProt result: {first_result}")
                    else:
                        logger.debug(f"No results in UniProt search response: {uniprot_search}")
                except Exception as e:
                    logger.warning(f"UniProt ID resolution failed for {primary_target.gene_symbol}: {e}")
            else:
                logger.debug(f"UniProt MCP client not available for {primary_target.gene_symbol}")
        else:
            logger.debug(f"UniProt ID already present for {primary_target.gene_symbol}: {primary_target.uniprot_id}")
        
        # Calculate ID mapping accuracy (simplified - count resolved proteins)
        total_sources = sum([
            len(string_proteins.get('proteins', [])) if isinstance(string_proteins, dict) else 0,
            len(hpa_proteins) if isinstance(hpa_proteins, list) else 0,
            len(kegg_proteins.get('genes', [])) if isinstance(kegg_proteins, dict) else 0
        ])
        mapping_accuracy = len(proteins) / max(total_sources, 1)
        
        return {
            'proteins': proteins,
            'primary_target': primary_target,
            'string_results': string_proteins,
            'hpa_results': hpa_proteins,
            'kegg_results': kegg_proteins,
            'mapping_accuracy': mapping_accuracy
        }
    
    async def _phase2_pathway_membership(self, target: Protein, data_sources: Dict[str, DataSourceStatus]) -> Dict[str, Any]:
        """
        Phase 2: Pathway membership analysis.
        
        Get pathways containing the target from Reactome.
        """
        logger.info("Phase 2: Pathway membership analysis")
        
        if not target:
            return {'pathways': []}
        
        # Get pathways from both Reactome and KEGG
        standardized_pathways = []
        
        # Reactome pathways
        try:
            reactome_pathways = await self._call_with_tracking(
                data_sources,
                'reactome',
                self.mcp_manager.reactome.find_pathways_by_gene(
                    target.gene_symbol
                )
            )
            # CRITICAL FIX: Validate response type BEFORE using .get()
            if isinstance(reactome_pathways, str):
                logger.warning(f"Reactome returned error for {target.gene_symbol}: {reactome_pathways}")
                reactome_pathways = {'pathways': []}
            elif isinstance(reactome_pathways, dict) and reactome_pathways.get('pathways'):
                for pathway_data in reactome_pathways['pathways']:
                    pathway = self.standardizer.standardize_reactome_pathway(pathway_data)
                    standardized_pathways.append(pathway)
        except (DatabaseConnectionError, DatabaseTimeoutError, MCPServerError) as e:
            # Database/MCP errors - log and continue with KEGG only
            logger.warning(
                f"Reactome pathway search failed for {target.gene_symbol}",
                extra=format_error_for_logging(e)
            )
        except DataValidationError as e:
            # Data validation errors - log and continue
            logger.warning(
                f"Invalid pathway data from Reactome for {target.gene_symbol}",
                extra={'validation_error': str(e)}
            )
        except Exception as e:
            # Unexpected errors
            logger.warning(
                f"Reactome pathway search failed for {target.gene_symbol}: {type(e).__name__}: {e}",
                extra=format_error_for_logging(e)
            )
        
        # KEGG pathways (search for gene in KEGG) with BATCH QUERIES (P0-3: 10-20x speedup!)
        try:
            kegg_genes = await self._call_with_tracking(
                data_sources,
                'kegg',
                self.mcp_manager.kegg.search_genes(target.gene_symbol)
            )
            # CRITICAL FIX: Validate response type BEFORE using .get()
            if isinstance(kegg_genes, str):
                logger.warning(f"KEGG returned error for {target.gene_symbol}: {kegg_genes}")
                kegg_genes = {'genes': {}}
            elif isinstance(kegg_genes, dict) and kegg_genes.get('genes'):
                # Filter to human genes and collect KEGG IDs
                human_kegg_ids = []
                for kegg_id, description in kegg_genes['genes'].items():
                    if isinstance(kegg_id, str) and not kegg_id.startswith('hsa:'):
                        continue
                    if target.gene_symbol.upper() in description.upper():
                        human_kegg_ids.append(kegg_id)

                if human_kegg_ids:
                    logger.info(
                        f"Batch querying {len(human_kegg_ids)} KEGG gene pathways with parallel execution"
                    )

                    # Use batch_query to parallelize pathway retrieval for multiple genes
                    # This replaces sequential loop with parallel execution (10-20x faster!)
                    kegg_pathways_list = await batch_query(
                        lambda kegg_id: self._call_with_tracking(
                            data_sources,
                            'kegg',
                            self.mcp_manager.kegg.get_gene_pathways(kegg_id)
                        ),
                        human_kegg_ids,
                        batch_size=10
                    )

                    # Process results
                    for kegg_id, kegg_pathways in zip(human_kegg_ids, kegg_pathways_list):
                        try:
                            # Handle None results (failed queries)
                            if not kegg_pathways or isinstance(kegg_pathways, Exception):
                                logger.debug(f"No pathways or error for KEGG gene {kegg_id}")
                                continue

                            # CRITICAL FIX: Validate response type BEFORE using .get()
                            if isinstance(kegg_pathways, str):
                                logger.warning(f"KEGG pathways returned error for {kegg_id}: {kegg_pathways}")
                                continue
                            elif isinstance(kegg_pathways, dict) and kegg_pathways.get('pathways'):
                                for pathway_data in kegg_pathways['pathways']:
                                    pathway = self.standardizer.standardize_kegg_pathway(pathway_data)
                                    standardized_pathways.append(pathway)
                            else:
                                logger.debug(f"No pathways found for KEGG gene {kegg_id} (this is expected for some genes)")
                        except (DatabaseConnectionError, DatabaseTimeoutError, MCPServerError) as pathway_e:
                            # Database errors for specific gene - continue with other genes
                            logger.debug(
                                f"Failed to get pathways for {kegg_id}",
                                extra=format_error_for_logging(pathway_e)
                            )
                        except Exception as pathway_e:
                            # Unexpected errors for specific gene
                            logger.debug(
                                f"Failed to get pathways for {kegg_id}: {type(pathway_e).__name__}: {pathway_e}",
                                extra=format_error_for_logging(pathway_e)
                            )
        except (DatabaseConnectionError, DatabaseTimeoutError, MCPServerError) as e:
            # Database errors for gene search - log and continue
            logger.warning(
                f"KEGG pathway search failed for {target.gene_symbol}",
                extra=format_error_for_logging(e)
            )
        except Exception as e:
            # Unexpected errors
            logger.warning(
                f"KEGG pathway search failed for {target.gene_symbol}: {type(e).__name__}: {e}",
                extra=format_error_for_logging(e)
            )
        
        # Calculate pathway precision/recall (simplified - count pathways)
        num_pathways = len(standardized_pathways)
        pathway_metrics = self.validator.validate_pathway_precision_recall(
            true_positive=num_pathways,
            false_positive=0,
            false_negative=0
        )
        
        return {
            'pathways': standardized_pathways,
            'precision': pathway_metrics.get('precision', 0.0),
            'recall': pathway_metrics.get('recall', 0.0)
        }
    
    async def _phase3_interaction_network(
        self, 
        target: Protein, 
        pathway_data: Dict[str, Any],
        data_sources: Dict[str, DataSourceStatus]
    ) -> Dict[str, Any]:
        """
        Phase 3: Interaction network.
        
        Build interaction network using STRING and Reactome.
        
        FIXED: Now correctly uses Reactome pathway IDs from phase 2 instead of gene symbol.
        """
        logger.info("Phase 3: Interaction network")
        
        if not target:
            return {'network': nx.Graph(), 'nodes': [], 'edges': []}
        
        # Initialize adaptive STRING builder if not already done
        if self.string_builder is None:
            # P2 Tuning: Lower minimums for single-target analysis to prevent over-expansion warnings
            single_target_config = {
                'min_nodes': 15,    # Lower from default 50
                'min_edges': 50,    # Lower from default 200
                'initial_add_nodes': 10, # Start with some neighbors for context
                'add_nodes_steps': [10, 20, 30] # More aggressive neighbor addition
            }
            
            self.string_builder = AdaptiveStringNetworkBuilder(
                self.mcp_manager,
                data_sources,
                config=single_target_config
            )
        
        # Build network using adaptive builder
        try:
            network_result = await self.string_builder.build_network(
                genes=[target.gene_symbol],
                priority_genes=[target.gene_symbol],
                data_sources=data_sources
            )
            
            nodes = network_result.get('nodes', [])
            edges = network_result.get('edges', [])
            
            string_network = {
                'nodes': nodes,
                'edges': edges
            }
        except (DatabaseConnectionError, DatabaseTimeoutError, MCPServerError) as exc:
            logger.warning(
                "STRING interaction network unavailable for %s",
                target.gene_symbol,
                extra=format_error_for_logging(exc)
            )
            string_network = {'nodes': [], 'edges': []}

        # CRITICAL FIX: Validate STRING response type BEFORE using .get()
        if isinstance(string_network, str):
            logger.warning(f"STRING returned error for {target.gene_symbol}: {string_network}")
            string_network = {'nodes': [], 'edges': []}
        elif not isinstance(string_network, dict):
            logger.warning(f"STRING returned unexpected type: {type(string_network)}")
            string_network = {'nodes': [], 'edges': []}
        elif self._response_has_error(string_network):
            logger.warning("STRING MCP payload indicated error for %s", target.gene_symbol)
            string_network = {'nodes': [], 'edges': []}

        # FIXED: Get Reactome interactions from pathways (requires pathway IDs, not gene symbol)
        reactome_interactions = {'interactions': []}
        reactome_pathway_ids = []
        
        # Extract Reactome pathway IDs from phase 2 results
        if isinstance(pathway_data, dict):
            pathways = pathway_data.get('pathways', [])
            for pathway in pathways:
                # Check if it's a Pathway object or dict
                if hasattr(pathway, 'source_db') and pathway.source_db == 'reactome':
                    reactome_pathway_ids.append(pathway.id)
                elif isinstance(pathway, dict) and pathway.get('source_db') == 'reactome':
                    reactome_pathway_ids.append(pathway.get('id', ''))
        
        # Get interactions from each Reactome pathway
        if reactome_pathway_ids:
            logger.debug(f"Fetching Reactome interactions from {len(reactome_pathway_ids)} pathways")
            all_interactions = []
            for pathway_id in reactome_pathway_ids:
                if not pathway_id:
                    continue
                try:
                    pathway_interactions = await self._call_with_tracking(
                        data_sources,
                        'reactome',
                        self.mcp_manager.reactome.get_protein_interactions(pathway_id)
                    )
                    # Validate response
                    if isinstance(pathway_interactions, dict):
                        interactions = pathway_interactions.get('interactions', [])
                        if isinstance(interactions, list):
                            all_interactions.extend(interactions)
                            logger.debug(f"Retrieved {len(interactions)} interactions from pathway {pathway_id}")
                    elif isinstance(pathway_interactions, str):
                        logger.debug(f"Reactome interactions returned error for pathway {pathway_id}: {pathway_interactions}")
                except Exception as e:
                    logger.debug(f"Reactome interactions not available for pathway {pathway_id}: {e}")
            
            reactome_interactions = {'interactions': all_interactions}
            logger.info(f"Retrieved {len(all_interactions)} total Reactome interactions from {len(reactome_pathway_ids)} pathways")
        else:
            logger.debug(f"No Reactome pathways found for {target.gene_symbol}, skipping Reactome interactions")
        
        # Build network
        G = nx.Graph()
        G.add_node(target.gene_symbol, **target.dict())
        
        # Add STRING network nodes and edges
        string_nodes = string_network.get('nodes', [])
        string_edges = string_network.get('edges', [])
        
        logger.info(f"STRING returned {len(string_nodes)} nodes and {len(string_edges)} edges")
        
        # Add STRING nodes
        for node_data in string_nodes:
            node_name = node_data.get('preferred_name', node_data.get('string_id', ''))
            if node_name and node_name != target.gene_symbol:
                G.add_node(node_name, **node_data)
                logger.debug(f"Added STRING node: {node_name}")
        
        # CRITICAL FIX: Add all protein names from edges to the graph first
        edge_proteins = set()
        for edge_data in string_edges:
            source = edge_data.get('protein_a', '')
            target_node = edge_data.get('protein_b', '')
            if source:
                edge_proteins.add(source)
            if target_node:
                edge_proteins.add(target_node)
        
        # Add missing proteins as nodes
        for protein in edge_proteins:
            if protein and not G.has_node(protein):
                G.add_node(protein, protein_name=protein)
                logger.debug(f"Added edge protein node: {protein}")
        
        # Add STRING edges
        edges_added = 0
        for edge_data in string_edges:
            # Use the correct field names from STRING response
            source = edge_data.get('protein_a', '')
            target_node = edge_data.get('protein_b', '')
            score = edge_data.get('confidence_score', 0)
            evidence_types = edge_data.get('evidence_types', [])
            
            logger.debug(f"Processing edge: source='{source}', target='{target_node}', score={score}")
            
            if source and target_node and G.has_node(source) and G.has_node(target_node):
                weight = score if score <= 1.0 else score / 1000.0
                G.add_edge(source, target_node, 
                          weight=weight, 
                          source='STRING',
                          **edge_data)
                edges_added += 1
                logger.debug(f"Added STRING edge: {source} -> {target_node} (weight: {weight})")
            else:
                logger.debug(f"Skipped edge: source='{source}', target='{target_node}', has_source={G.has_node(source) if source else False}, has_target={G.has_node(target_node) if target_node else False}")
        
        logger.info(f"Added {edges_added} edges to network (from {len(string_edges)} STRING edges)")
        
        # Add Reactome interactions (if available)
        if reactome_interactions.get('interactions'):
            for interaction_data in reactome_interactions['interactions']:
                partner = interaction_data.get('partner', '')
                score = interaction_data.get('score', 0.0) / 1000.0
                
                if not G.has_node(partner):
                    G.add_node(partner, **interaction_data.get('partner_data', {}))
                
                G.add_edge(target.gene_symbol, partner,
                          weight=score,
                          source='Reactome',
                          **interaction_data)
        
        # Convert to result format
        network_nodes = []
        for node in G.nodes(data=True):
            node_id = node[0]
            node_data = node[1]
            
            network_node = NetworkNode(
                id=node_id,
                node_type='protein',
                gene_symbol=node_data.get('name', node_id),
                pathways=self._get_node_pathways(node_id),
                centrality_measures=self._calculate_centrality_measures(G, node_id)
            )
            network_nodes.append(network_node)
        
        network_edges = []
        for edge in G.edges(data=True):
            source, target, edge_data = edge
            
            network_edge = NetworkEdge(
                source=source,
                target=target,
                weight=edge_data.get('weight', 0.0),
                interaction_type=edge_data.get('interaction_type'),
                evidence_score=edge_data.get('score', 0.0) / 1000.0,
                pathway_context=self._get_edge_pathway_context(source, target)
            )
            network_edges.append(network_edge)
        
        # Extract interactions for the result
        interactions = []
        for edge in network_edges:
            interactions.append({
                'protein_a': edge.source,
                'protein_b': edge.target,
                'combined_score': edge.weight,
                'evidence_types': getattr(edge, 'evidence_types', {}),
                'interaction_type': getattr(edge, 'interaction_type', 'protein_protein'),
                'source_database': 'string'
            })

        return {
            'network': G,
            'nodes': network_nodes,
            'edges': network_edges,
            'interactions': interactions
        }
    
    async def _phase4_expression_profiling(self, target: Protein, tissue_context: Optional[str] = None, data_sources: Dict[str, DataSourceStatus] = None) -> Dict[str, Any]:
        """
        Phase 4: Expression profiling.
        
        Get tissue-specific expression data from HPA.
        """
        logger.info("Phase 4: Expression profiling")
        
        if not target:
            return {'profiles': [], 'markers': []}
        
        # Get tissue expression
        try:
            expression_data = await self._call_with_tracking(
                data_sources,
                'hpa',
                self.mcp_manager.hpa.get_tissue_expression(
                    target.gene_symbol
                )
            )
        except (DatabaseConnectionError, DatabaseTimeoutError, MCPServerError) as exc:
            logger.warning(
                "HPA expression unavailable for %s",
                target.gene_symbol,
                extra=format_error_for_logging(exc)
            )
            expression_data = {'expression': {}}

        # CRITICAL FIX: Validate HPA expression response type
        if isinstance(expression_data, str):
            logger.warning(f"HPA expression returned error for {target.gene_symbol}: {expression_data}")
            expression_data = {'expression': {}}
        elif isinstance(expression_data, dict) and self._response_has_error(expression_data):
            logger.warning("HPA expression MCP payload indicated error for %s", target.gene_symbol)
            expression_data = {'expression': {}}

        # Get subcellular location (HPA returns list of structured dicts)
        try:
            location_data = await self._call_with_tracking(
                data_sources,
                'hpa',
                self.mcp_manager.hpa.get_subcellular_location(
                    target.gene_symbol
                )
            )
        except (DatabaseConnectionError, DatabaseTimeoutError, MCPServerError) as exc:
            logger.warning(
                "HPA subcellular location unavailable for %s",
                target.gene_symbol,
                extra=format_error_for_logging(exc)
            )
            location_data = []

        # CRITICAL FIX: Validate HPA location response type
        if isinstance(location_data, str):
            logger.warning(f"HPA location returned error for {target.gene_symbol}: {location_data}")
            location_data = []
        
        # Parse HPA subcellular location format correctly - FILTER FOR TARGET GENE ONLY
        subcellular_locations = []
        if isinstance(location_data, list):
            for item in location_data:
                if isinstance(item, dict):
                    # CRITICAL FIX: Only process if this item is for the target gene
                    item_gene = item.get('Gene', '').upper()
                    if item_gene != target.gene_symbol.upper():
                        continue  # Skip locations for other genes
                    
                    # Extract structured location data (handle None values)
                    main_locations = item.get('Subcellular main location', []) or []
                    additional_locations = item.get('Subcellular additional location', []) or []
                    reliability = item.get('Reliability (IF)', 'Unknown')
                    
                    # Create structured location entries
                    for location in main_locations:
                        subcellular_locations.append({
                            'gene': target.gene_symbol,
                            'location': location,
                            'reliability': reliability,
                            'is_main_location': True,
                            'evidence': ['immunofluorescence']
                        })
                    
                    for location in additional_locations:
                        subcellular_locations.append({
                            'gene': target.gene_symbol,
                            'location': location,
                            'reliability': reliability,
                            'is_main_location': False,
                            'evidence': ['immunofluorescence']
                        })
        elif isinstance(location_data, dict):
            # Fallback for dict format
            locations = location_data.get('locations', [])
            for loc in locations:
                if isinstance(loc, str):
                    subcellular_locations.append({
                        'gene': target.gene_symbol,
                        'location': loc,
                        'reliability': 'Unknown',
                        'is_main_location': True,
                        'evidence': []
                    })
        
        # Process expression profiles using HPA parsing helpers
        expression_profiles = []
        from ..utils.hpa_parsing import _iter_expr_items, categorize_expression
        
        for tissue, ntpms in _iter_expr_items(expression_data):
            # Filter by tissue context if provided
            if tissue_context and tissue_context.lower() not in tissue.lower():
                continue
            
            # Convert nTPM to categorical using helper
            expression_level = categorize_expression(ntpms)
            
            # Create expression profile without subcellular location (keep separate)
            expression_profile = ExpressionProfile(
                gene=target.gene_symbol,  # Use the target gene symbol
                tissue=tissue,
                expression_level=expression_level,
                reliability='Approved',
                cell_type_specific=False,
                subcellular_location=[]  # Keep separate from expression profiles
            )
            expression_profiles.append(expression_profile)
        
        # Get cancer markers if available
        cancer_markers = []
        if self._is_cancer_gene(target.gene_symbol):
            try:
                pathology_response = await self._call_with_tracking(
                    data_sources,
                    'hpa',
                    self.mcp_manager.hpa.get_pathology_data([target.gene_symbol])
                )
                from ..utils.hpa_parsing import parse_pathology_data
                pathology_data = parse_pathology_data(pathology_response)
                markers_list = pathology_data.get('markers', [])
                for marker_data in markers_list:
                    try:
                        marker = self.standardizer.standardize_cancer_marker(marker_data)
                        if self.validator.validate_cancer_marker_confidence(marker):
                            cancer_markers.append(marker)
                    except Exception:
                        continue
            except Exception:
                pass
        
        # Calculate expression reproducibility
        reproducibility = self.validator.validate_expression_reproducibility(
            expression_profiles, {}
        )
        
        return {
            'profiles': expression_profiles,
            'markers': cancer_markers,
            'subcellular_locations': subcellular_locations,
            'reproducibility': reproducibility
        }
    
    async def _get_chembl_bioactivity(self, gene_symbol: str) -> Optional[Dict[str, Any]]:
        """
        Get ChEMBL bioactivity data for a target and calculate druggability metrics.

        Retrieves:
        - Target information
        - Bioactivity measurements (IC50, Ki, EC50, Kd)
        - Active compound count
        - Median potency values

        Args:
            gene_symbol: Gene symbol (e.g., "EGFR", "AXL")

        Returns:
            Dict with aggregated bioactivity metrics, or None if not found
        """
        if not self.mcp_manager.chembl:
            logger.debug("ChEMBL client not available")
            return None

        try:
            logger.info(f"🔬 Retrieving ChEMBL bioactivity data for {gene_symbol}")

            # Step 1: Search for target
            target_search = await self._call_with_tracking(
                None,
                'chembl',
                self.mcp_manager.chembl.search_targets(gene_symbol, limit=5)
            )

            if not target_search.get('targets'):
                logger.debug(f"No ChEMBL target found for {gene_symbol}")
                return None

            target = target_search['targets'][0]
            target_chembl_id = target.get('target_chembl_id')
            logger.info(f"  Found ChEMBL target: {target_chembl_id}")

            # Step 2: Get bioactivity data
            activities_result = await self._call_with_tracking(
                None,
                'chembl',
                self.mcp_manager.chembl.search_activities(
                    target_chembl_id=target_chembl_id,
                    limit=500  # Get more activities for better statistics
                )
            )

            if not activities_result.get('activities'):
                logger.debug(f"No bioactivity data found for {target_chembl_id}")
                return None

            # Step 3: Parse and aggregate bioactivities
            bioactivities = []
            for activity_raw in activities_result['activities']:
                try:
                    bioactivity = self.standardizer.standardize_chembl_bioactivity(activity_raw)
                    bioactivities.append(bioactivity)
                except Exception as e:
                    logger.debug(f"Failed to standardize bioactivity: {e}")
                    continue

            if not bioactivities:
                logger.debug(f"No valid bioactivities after standardization")
                return None

            # Step 4: Aggregate bioactivities
            aggregated = self.standardizer.aggregate_bioactivities(bioactivities, group_by="target")

            # Step 5: Calculate druggability score from bioactivity
            druggability_score = self._calculate_bioactivity_druggability_score(aggregated)

            # Step 6: Create bioactivity dict
            target_bioactivity = {
                'target_chembl_id': target_chembl_id,
                'gene_symbol': gene_symbol,
                'median_ic50': aggregated.get('median_ic50'),
                'median_ki': aggregated.get('median_ki'),
                'median_ec50': aggregated.get('median_ec50'),
                'median_kd': aggregated.get('median_kd'),
                'activity_count': aggregated.get('activity_count', 0),
                'compound_count': aggregated.get('compound_count', 0),
                'assay_count': aggregated.get('assay_count', 0),
                'data_quality': aggregated.get('data_quality', 'low'),
                'druggability_score': druggability_score
            }

            logger.info(f"  ✅ ChEMBL bioactivity: {target_bioactivity['activity_count']} activities, "
                       f"druggability={druggability_score:.3f}")

            return target_bioactivity

        except Exception as e:
            logger.warning(f"Failed to get ChEMBL bioactivity for {gene_symbol}: {e}")
            return None

    def _calculate_bioactivity_druggability_score(self, aggregated: Dict[str, Any]) -> float:
        """
        Calculate druggability score from aggregated bioactivity data.

        Scoring factors:
        1. Potency (IC50/Ki < 1000 nM): 40% weight
        2. Activity count (>10 activities): 30% weight
        3. Compound diversity (>5 compounds): 20% weight
        4. Data quality (high/medium/low): 10% weight

        Args:
            aggregated: Aggregated bioactivity metrics

        Returns:
            Druggability score (0-1)
        """
        score = 0.0

        # Factor 1: Potency (40% weight)
        median_ic50 = aggregated.get('median_ic50')
        median_ki = aggregated.get('median_ki')

        # Use best available potency measurement
        best_potency = None
        if median_ic50 is not None:
            best_potency = median_ic50
        elif median_ki is not None:
            best_potency = median_ki

        if best_potency is not None:
            if best_potency < 10:  # < 10 nM = very potent
                potency_score = 1.0
            elif best_potency < 100:  # < 100 nM = potent
                potency_score = 0.9
            elif best_potency < 1000:  # < 1000 nM = moderate
                potency_score = 0.7
            elif best_potency < 10000:  # < 10 uM = weak
                potency_score = 0.4
            else:  # > 10 uM = very weak
                potency_score = 0.2
            score += potency_score * 0.40
        else:
            # No potency data = assume moderate
            score += 0.5 * 0.40

        # Factor 2: Activity count (30% weight)
        activity_count = aggregated.get('activity_count', 0)
        if activity_count >= 50:
            activity_score = 1.0
        elif activity_count >= 10:
            activity_score = 0.8
        elif activity_count >= 5:
            activity_score = 0.6
        elif activity_count >= 1:
            activity_score = 0.4
        else:
            activity_score = 0.0
        score += activity_score * 0.30

        # Factor 3: Compound diversity (20% weight)
        compound_count = aggregated.get('compound_count', 0)
        if compound_count >= 20:
            compound_score = 1.0
        elif compound_count >= 10:
            compound_score = 0.8
        elif compound_count >= 5:
            compound_score = 0.6
        elif compound_count >= 1:
            compound_score = 0.4
        else:
            compound_score = 0.0
        score += compound_score * 0.20

        # Factor 4: Data quality (10% weight)
        data_quality = aggregated.get('data_quality', 'low')
        quality_scores = {'high': 1.0, 'medium': 0.7, 'low': 0.4}
        quality_score = quality_scores.get(data_quality, 0.4)
        score += quality_score * 0.10

        return min(score, 1.0)

    async def _phase5_druggability_assessment(
        self,
        target: Protein,
        network: nx.Graph,
        data_sources: Dict[str, DataSourceStatus] = None
    ) -> Dict[str, Any]:
        """
        Phase 5: Enhanced druggability assessment with ChEMBL bioactivity data.

        Combines:
        - KEGG drug data (existing)
        - Network properties (existing)
        - ChEMBL bioactivity metrics (NEW)

        Expected improvement: +0.2 druggability score for targets with bioactivity data.
        """
        logger.info(f"Phase 5: Enhanced druggability assessment (KEGG + ChEMBL)")

        if not target:
            return {'score': 0.0, 'drugs': [], 'chembl_bioactivity': None}

        # PART 1: Get KEGG drugs (existing functionality)
        logger.info("📚 Part 1/2: KEGG drug search...")
        try:
            drugs = await self._call_with_tracking(
                data_sources,
                'kegg',
                self.mcp_manager.kegg.search_drugs(
                    target.gene_symbol
                )
            )
        except (DatabaseConnectionError, DatabaseTimeoutError, MCPServerError) as exc:
            logger.warning(
                "KEGG drug search unavailable for %s",
                target.gene_symbol,
                extra=format_error_for_logging(exc)
            )
            drugs = {'drugs': []}

        # CRITICAL FIX: Validate KEGG response type BEFORE using .get()
        if isinstance(drugs, str):
            logger.warning(f"KEGG returned error for {target.gene_symbol}: {drugs}")
            drugs = {'drugs': []}  # Safe fallback
        elif not isinstance(drugs, dict):
            logger.warning(f"KEGG returned unexpected type: {type(drugs)}")
            drugs = {'drugs': []}  # Safe fallback
        elif self._response_has_error(drugs):
            logger.warning("KEGG MCP payload indicated error for %s", target.gene_symbol)
            drugs = {'drugs': []}

        # Get drug-target mappings
        drug_targets = []
        if isinstance(drugs, dict) and drugs.get('drugs'):
            for drug_data in drugs['drugs']:
                # Defensive parsing - handle dict/string formats
                if isinstance(drug_data, str):
                    # Drug ID string, construct dict with target_id
                    drug_info = {
                        'drug_id': drug_data,
                        'target_id': target.gene_symbol,
                        'interaction_type': 'unknown',
                        'confidence': 0.5
                    }
                elif isinstance(drug_data, dict):
                    # Already a dict, ensure it has required fields
                    drug_info = drug_data.copy()
                    if 'drug_id' not in drug_info and 'id' in drug_info:
                        drug_info['drug_id'] = drug_info['id']
                    if 'target_id' not in drug_info:
                        drug_info['target_id'] = target.gene_symbol
                else:
                    logger.debug(f"Skipping invalid drug_data type: {type(drug_data)}")
                    continue

                drug_target = await self.standardizer.standardize_drug_target(drug_info)
                if drug_target:
                    drug_targets.append(drug_target)

        logger.info(f"  ✅ KEGG: Found {len(drug_targets)} drugs for {target.gene_symbol}")

        # PART 2: Get ChEMBL bioactivity data (NEW)
        logger.info("🔬 Part 2/2: ChEMBL bioactivity analysis...")
        chembl_bioactivity = await self._get_chembl_bioactivity(target.gene_symbol)

        if chembl_bioactivity:
            activity_count = chembl_bioactivity.get('activity_count', 0)
            median_ic50 = chembl_bioactivity.get('median_ic50')
            ic50_str = f"{median_ic50:.1f}" if median_ic50 else "N/A"
            logger.info(f"  ✅ ChEMBL: {activity_count} activities, median IC50: {ic50_str} nM")
        else:
            logger.info(f"  ℹ️  ChEMBL: No bioactivity data found")

        # Calculate enhanced druggability score (combines both sources)
        druggability_score = await self._calculate_druggability_score(
            target, network, drug_targets, chembl_bioactivity
        )

        logger.info(f"✅ Druggability assessment complete: score = {druggability_score:.3f}")

        return {
            'score': druggability_score,
            'drugs': drug_targets,
            'chembl_bioactivity': chembl_bioactivity
        }


    def _normalize_async_result(self, result: Any, source_name: str) -> Any:
        """Normalize asyncio gather results (handle exceptions/error payloads)."""
        if isinstance(result, Exception):
            logger.warning(
                "%s call failed: %s",
                source_name.upper(),
                result
            )
            return None
        if isinstance(result, dict) and self._response_has_error(result):
            logger.warning("%s MCP payload indicated error", source_name.upper())
            return None
        return result

    def _response_has_error(self, payload: Any) -> bool:
        """Detect MCP payloads that encode errors."""
        if not isinstance(payload, dict):
            return False
        if payload.get('isError'):
            return True
        content = payload.get('content')
        if isinstance(content, list) and content:
            text_value = content[0].get('text')
            if isinstance(text_value, str) and text_value.lower().startswith('error'):
                return True
        return False
    
    def _is_cancer_gene(self, gene: str) -> bool:
        """Check if gene is cancer-related - removed hardcoded list for scientific validity."""
        # This method should be replaced with dynamic cancer gene detection
        # For now, return False to avoid hardcoded assumptions
        return False
    
    def _get_node_pathways(self, node_id: str) -> List[str]:
        """Get pathways for a node."""
        # Simplified pathway assignment
        return []
    
    def _calculate_centrality_measures(self, network: nx.Graph, node_id: str) -> Dict[str, float]:
        """Calculate centrality measures for a node."""
        try:
            return {
                'betweenness': nx.betweenness_centrality(network)[node_id],
                'closeness': nx.closeness_centrality(network)[node_id],
                'degree': network.degree(node_id)
            }
        except:
            return {'betweenness': 0.0, 'closeness': 0.0, 'degree': 0}
    
    def _get_edge_pathway_context(self, source: str, target: str) -> Optional[str]:
        """Get pathway context for an edge."""
        return None
    
    async def _calculate_druggability_score(
        self,
        target: Protein,
        network: nx.Graph,
        drug_targets: List[Any],
        chembl_bioactivity: Optional[Dict[str, Any]] = None
    ) -> float:
        """
        Calculate enhanced druggability score integrating KEGG, network, and ChEMBL data.

        Scoring weights (without ChEMBL):
        - Known drugs (KEGG): 40%
        - Network centrality: 30%
        - Network size: 20%
        - Protein type: 10%
        - Domain boost: up to 15%

        Enhanced scoring (with ChEMBL):
        - ChEMBL bioactivity replaces or boosts KEGG score: up to 50%
        - Network centrality: 20%
        - Network size: 15%
        - Protein type: 10%
        - Domain boost: up to 15%

        Expected improvement: +0.2 for targets with ChEMBL data.

        Args:
            target: Target protein
            network: Interaction network
            drug_targets: KEGG drug targets
            chembl_bioactivity: ChEMBL bioactivity data (optional)

        Returns:
            Druggability score (0-1+)
        """
        score = 0.0

        # STRATEGY: If ChEMBL data available, use enhanced weighting
        if chembl_bioactivity:
            logger.info(f"Using enhanced ChEMBL-based druggability scoring for {target.gene_symbol}")

            # ChEMBL bioactivity score (50% weight) - PRIMARY EVIDENCE
            chembl_score = chembl_bioactivity.get('druggability_score', 0.0)
            score += chembl_score * 0.50
            logger.info(f"  ChEMBL bioactivity score: {chembl_score:.3f} (weight: 0.50)")

            # Network centrality (20% weight)
            if network.has_node(target.gene_symbol):
                try:
                    centrality = nx.degree_centrality(network)[target.gene_symbol]
                    score += centrality * 0.20
                    logger.info(f"  Network centrality: {centrality:.3f} (weight: 0.20)")
                except Exception as e:
                    logger.warning(f"Failed to calculate centrality: {e}")

            # Network size (15% weight)
            network_size = network.number_of_nodes()
            if network_size > 0:
                size_score = min(network_size / 20.0, 1.0)
                score += size_score * 0.15
                logger.info(f"  Network size score: {size_score:.3f} (weight: 0.15)")

            # Protein type (10% weight)
            if hasattr(target, 'protein_type'):
                if target.protein_type in ['kinase', 'receptor', 'enzyme']:
                    score += 0.10
                    logger.info(f"  Druggable protein type: {target.protein_type} (weight: 0.10)")

            # Domain boost (5% weight, reduced in favor of bioactivity)
            domain_boost = await self._get_domain_boost(target)
            score += domain_boost * 0.05
            if domain_boost > 0:
                logger.info(f"  Domain boost: {domain_boost:.3f} (weight: 0.05)")

        else:
            # Original scoring (KEGG-only)
            logger.info(f"Using standard KEGG-based druggability scoring for {target.gene_symbol}")

            # Known drug targets (40% weight)
            if drug_targets:
                score += 0.40
                logger.info(f"  KEGG drugs: {len(drug_targets)} (weight: 0.40)")

            # Network centrality (30% weight)
            if network.has_node(target.gene_symbol):
                try:
                    centrality = nx.degree_centrality(network)[target.gene_symbol]
                    score += centrality * 0.30
                    logger.info(f"  Network centrality: {centrality:.3f} (weight: 0.30)")
                except Exception as e:
                    logger.warning(f"Failed to calculate centrality: {e}")

            # Network size (20% weight)
            network_size = network.number_of_nodes()
            if network_size > 0:
                size_score = min(network_size / 20.0, 1.0)
                score += size_score * 0.20
                logger.info(f"  Network size score: {size_score:.3f} (weight: 0.20)")

            # Protein type (10% weight)
            if hasattr(target, 'protein_type'):
                if target.protein_type in ['kinase', 'receptor', 'enzyme']:
                    score += 0.10
                    logger.info(f"  Druggable protein type: {target.protein_type} (weight: 0.10)")

            # Domain boost (up to 15%)
            domain_boost = await self._get_domain_boost(target)
            score += domain_boost
            if domain_boost > 0:
                logger.info(f"  Domain boost: {domain_boost:.3f}")

        # Cap at 1.0
        final_score = min(score, 1.0)
        logger.info(f"Final druggability score for {target.gene_symbol}: {final_score:.3f}")
        return final_score

    async def _get_domain_boost(self, target: Protein) -> float:
        """
        Get domain-based druggability boost from UniProt features.

        P2 Enhancement: Domain-based scoring boost for druggable features.

        Args:
            target: Target protein

        Returns:
            Domain boost score (0-0.15)
        """
        domain_boost = 0.0
        if target.uniprot_id and self.mcp_manager.uniprot:
            logger.info(f"P2: Checking domain features for {target.gene_symbol} (UniProt: {target.uniprot_id})")
            try:
                features_data = await self._call_with_tracking(
                    None,
                    'uniprot',
                    self.mcp_manager.uniprot.get_protein_features(target.uniprot_id)
                )
                logger.debug(f"P2: Features data retrieved: {bool(features_data)}, type: {type(features_data)}")

                # CRITICAL FIX: Validate response type BEFORE using .get()
                if isinstance(features_data, str):
                    logger.warning(f"UniProt returned error for {target.gene_symbol}: {features_data}")
                elif not isinstance(features_data, dict):
                    logger.warning(f"UniProt returned unexpected type: {type(features_data)}")
                elif isinstance(features_data, dict):
                    # Parse domain features
                    feature_list = features_data.get('features', [])
                    domains_list = features_data.get('domains', [])
                    active_sites = features_data.get('activeSites', [])
                    binding_sites = features_data.get('bindingSites', [])
                    
                    logger.debug(f"P2: Feature counts - features: {len(feature_list) if isinstance(feature_list, list) else 0}, "
                               f"domains: {len(domains_list) if isinstance(domains_list, list) else 0}, "
                               f"activeSites: {len(active_sites) if isinstance(active_sites, list) else 0}, "
                               f"bindingSites: {len(binding_sites) if isinstance(binding_sites, list) else 0}")
                    
                    all_features = []
                    if isinstance(feature_list, list):
                        all_features.extend(feature_list)
                    if isinstance(domains_list, list):
                        all_features.extend(domains_list)
                    if isinstance(active_sites, list):
                        all_features.extend(active_sites)
                    if isinstance(binding_sites, list):
                        all_features.extend(binding_sites)
                    
                    logger.debug(f"P2: Total features to check: {len(all_features)}")
                    
                    # Score druggable domain types
                    druggable_keywords = {
                        'kinase': 0.15,  # High druggability (kinase domains)
                        'atp': 0.12,  # ATP-binding sites
                        'catalytic': 0.10,  # Catalytic sites
                        'binding': 0.08,  # Binding sites
                        'receptor': 0.08,  # Receptor domains
                        'enzyme': 0.06,  # Enzyme domains
                        'active': 0.06,  # Active sites
                        'domain': 0.04,  # General protein domains
                    }
                    
                    for feature in all_features:
                        if isinstance(feature, dict):
                            feature_type = feature.get('type', '').lower()
                            description = feature.get('description', '').lower()
                            
                            # Check for druggable keywords
                            for keyword, boost_value in druggable_keywords.items():
                                if keyword in feature_type or keyword in description:
                                    domain_boost = max(domain_boost, boost_value)  # Take highest boost
                                    logger.debug(f"Found druggable domain: {feature_type} for {target.gene_symbol} (+{boost_value:.3f})")
                                    break  # Only count once per feature
                    
                    if domain_boost > 0:
                        logger.info(f"Domain boost for {target.gene_symbol}: +{domain_boost:.3f} (P2 enhancement)")
                    else:
                        logger.debug(f"P2: No druggable domains found for {target.gene_symbol}")
            except Exception as e:
                logger.warning(f"Domain lookup failed for {target.gene_symbol}: {e}")
                import traceback
                logger.debug(f"Traceback: {traceback.format_exc()}")
        else:
            logger.debug(f"P2: Skipping domain boost - UniProt ID: {target.uniprot_id}, UniProt client available: {bool(self.mcp_manager.uniprot)}")

        return domain_boost
    
    def _calculate_validation_score(
        self,
        target_data: Dict,
        pathway_data: Dict,
        network_data: Dict,
        expression_data: Dict,
        druggability_data: Dict,
        data_sources: Dict[str, DataSourceStatus]
    ) -> float:
        """Calculate overall validation score with data source tracking."""
        scores = {}

        # ID mapping accuracy
        scores['id_mapping_accuracy'] = target_data.get('mapping_accuracy', 0.0) if isinstance(target_data, dict) else 0.0

        # Pathway precision/recall
        scores['pathway_precision'] = pathway_data.get('precision', 0.0) if isinstance(pathway_data, dict) else 0.0
        scores['pathway_recall'] = pathway_data.get('recall', 0.0) if isinstance(pathway_data, dict) else 0.0

        # Expression reproducibility
        scores['expression_reproducibility'] = expression_data.get('reproducibility', 0.0) if isinstance(expression_data, dict) else 0.0

        # Druggability score
        scores['druggability_score'] = druggability_data.get('score', 0.0) if isinstance(druggability_data, dict) else 0.0

        # Calculate overall score with data source tracking
        validation_result = self.validator.calculate_overall_validation_score(
            scores,
            data_sources=list(data_sources.values())
        )

        # Extract the final score (float) from the result dictionary
        if isinstance(validation_result, dict):
            return validation_result.get('final_score', 0.0)
        else:
            # Backward compatibility - in case validator returns float directly
            return validation_result

    def _build_prioritization_summary(
        self,
        network_summary: Dict[str, Any],
        expression_summary: Dict[str, Any],
        druggability_data: Dict[str, Any],
        primary_target: Optional[Protein]
    ) -> Dict[str, Any]:
        """Build a composite prioritization score using real metrics."""
        network_score = 0.0
        if network_summary:
            avg_degree = network_summary.get('average_degree', 0.0)
            density = network_summary.get('density', 0.0)
            hub_fraction = network_summary.get('hub_fraction', 0.0)
            network_score = min(1.0, 0.5 * density + 0.5 * (avg_degree / 10.0 + hub_fraction))

        expression_score = expression_summary.get('expression_score', 0.0) if expression_summary else 0.0
        druggability_score = 0.0
        if isinstance(druggability_data, dict):
            druggability_score = float(druggability_data.get('score', 0.0))

        composite_priority = round(
            0.4 * druggability_score +
            0.35 * expression_score +
            0.25 * network_score,
            3
        )

        return {
            'network_score': round(network_score, 3),
            'expression_score': round(expression_score, 3),
            'druggability_score': round(druggability_score, 3),
            'composite_priority': composite_priority,
            'target_confidence': getattr(primary_target, 'confidence', 0.0) if primary_target else 0.0,
        }

    def _calculate_completeness_metrics(
        self,
        target_data: Dict,
        pathway_data: Dict,
        network_data: Dict,
        expression_data: Dict,
        druggability_data: Dict
    ) -> CompletenessMetrics:
        """Calculate data completeness metrics."""
        # Expression data completeness
        expression_count = len(expression_data.get('profiles', [])) if isinstance(expression_data, dict) else 0
        expression_data_score = min(1.0, expression_count / 10.0)  # Normalize to 10 profiles

        # Network data completeness
        network_edges = len(network_data.get('edges', [])) if isinstance(network_data, dict) else 0
        network_data_score = min(1.0, network_edges / 50.0)  # Normalize to 50 edges

        # Pathway data completeness
        pathway_count = len(pathway_data.get('pathways', [])) if isinstance(pathway_data, dict) else 0
        pathway_data_score = min(1.0, pathway_count / 5.0)  # Normalize to 5 pathways

        # Drug data completeness
        drug_count = len(druggability_data.get('drugs', [])) if isinstance(druggability_data, dict) else 0
        drug_data_score = min(1.0, drug_count / 10.0)  # Normalize to 10 drugs

        # Pathology data completeness (not applicable for target analysis)
        pathology_data_score = 0.0

        # Overall completeness
        overall_completeness = (
            expression_data_score + network_data_score + pathway_data_score +
            drug_data_score + pathology_data_score
        ) / 5.0

        return CompletenessMetrics(
            expression_data=expression_data_score,
            network_data=network_data_score,
            pathway_data=pathway_data_score,
            drug_data=drug_data_score,
            pathology_data=pathology_data_score,
            overall_completeness=overall_completeness
        )

    def _track_data_source(
        self,
        data_sources: Dict[str, DataSourceStatus],
        source_name: str,
        success: bool = True,
        error_type: Optional[str] = None,
    ) -> Optional[DataSourceStatus]:
        """Increment counters for the requested data source."""
        if not data_sources or source_name not in data_sources:
            return None

        status = data_sources[source_name]
        status.requested += 1
        if success:
            status.successful += 1
        else:
            status.failed += 1
            if error_type and error_type not in status.error_types:
                status.error_types.append(error_type)

        if status.requested:
            status.success_rate = status.successful / status.requested

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "[DATA_TRACKING] %s requested=%d successful=%d failed=%d success_rate=%.2f",
                source_name,
                status.requested,
                status.successful,
                status.failed,
                status.success_rate,
            )
        return status

    async def _call_with_tracking(
        self,
        data_sources: Optional[Dict[str, DataSourceStatus]],
        source_name: str,
        coro,
        suppress_exception: bool = False,
    ):
        """
        Await a coroutine while automatically tracking MCP request outcomes.
        """
        sources = data_sources or getattr(self, "_active_data_sources", None)
        try:
            result = await coro
            if sources:
                self._track_data_source(sources, source_name, success=True)
            return result
        except Exception as exc:
            if sources:
                self._track_data_source(
                    sources,
                    source_name,
                    success=False,
                    error_type=type(exc).__name__,
                )
            if suppress_exception:
                logger.debug(
                    "[DATA_TRACKING] Suppressed %s failure for %s",
                    source_name,
                    type(exc).__name__,
                )
                return None
            raise
