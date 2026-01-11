"""
Scenario 1: Disease Network Construction

Multi-database disease discovery and pathway mapping with network construction.
Based on Mature_development_plan.md Phase 1-5.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Set
import networkx as nx
from collections import defaultdict

from ..core.mcp_client_manager import MCPClientManager
from ..core.exceptions import (
    DatabaseConnectionError,
    DatabaseTimeoutError,
    MCPServerError,
    EmptyResultError,
    ScenarioExecutionError,
    format_error_for_logging
)
from ..core.data_standardizer import DataStandardizer
from ..core.validation import DataValidator
from ..core.simulation.simple_simulator import SimplePerturbationSimulator
from ..core.string_network_builder import AdaptiveStringNetworkBuilder
from ..utils.batch_queries import batch_query
from ..models.data_models import (
    Disease, Pathway, Protein, Interaction, ExpressionProfile,
    CancerMarker, NetworkNode, NetworkEdge, DiseaseNetworkResult,
    GeneProvenance
)

logger = logging.getLogger(__name__)


class DiseaseNetworkScenario:
    """
    Scenario 1: Disease Network Construction

    6-phase workflow:
    1. Multi-database disease discovery (KEGG + Reactome)
    1.5. Disease-gene association enrichment (UniProt) [NEW]
    2. Enhanced pathway extraction with gene merging
    3. Context-aware network construction (STRING)
    4. Expression overlay (HPA)
    5. Functional enrichment analysis
    """

    # Phase 1.5 Configuration
    PHASE1P5_ENABLED = True  # Enable/disable association enrichment
    ASSOCIATION_QUERY_LIMIT = 100  # Max proteins per disease term
    ASSOCIATION_CONFIDENCE_THRESHOLD = 0.3  # Minimum confidence score
    ASSOCIATION_MAX_TERMS = 5  # Max search terms for associations

    # Phase 2 Configuration
    GENE_MERGE_STRATEGY = 'union'  # 'union' or 'intersection'
    PROVENANCE_TRACKING_ENABLED = True  # Track gene discovery source

    def __init__(self, mcp_manager: MCPClientManager):
        """Initialize disease network scenario."""
        self.mcp_manager = mcp_manager
        self.standardizer = DataStandardizer()
        self.validator = DataValidator()
        self.gene_to_pathways = {}  # NEW: Store pathway mapping for enrichment
        self._active_data_sources: Optional[Dict[str, Any]] = None
        self.string_builder = None  # Will be initialized when data_sources available
    
    async def execute(
        self,
        disease_query: str,
        tissue_context: Optional[str] = None
    ) -> DiseaseNetworkResult:
        """
        Execute complete disease network construction workflow.

        Args:
            disease_query: Disease name or identifier
            tissue_context: Optional tissue context for expression filtering

        Returns:
            DiseaseNetworkResult with complete analysis
        """
        logger.info(f"Starting disease network analysis for: {disease_query}")

        # Initialize data source tracking
        from ..models.data_models import DataSourceStatus

        data_sources = {
            'kegg': DataSourceStatus(
                source_name="kegg",
                requested=0,
                successful=0,
                failed=0,
                success_rate=0.0,
                error_types=[]
            ),
            'reactome': DataSourceStatus(
                source_name="reactome",
                requested=0,
                successful=0,
                failed=0,
                success_rate=0.0,
                error_types=[]
            ),
            'string': DataSourceStatus(
                source_name="string",
                requested=0,
                successful=0,
                failed=0,
                success_rate=0.0,
                error_types=[]
            ),
            'hpa': DataSourceStatus(
                source_name="hpa",
                requested=0,
                successful=0,
                failed=0,
                success_rate=0.0,
                error_types=[]
            ),
            'uniprot': DataSourceStatus(
                source_name="uniprot",
                requested=0,
                successful=0,
                failed=0,
                success_rate=0.0,
                error_types=[]
            ),
            'chembl': DataSourceStatus(
                source_name="chembl",
                requested=0,
                successful=0,
                failed=0,
                success_rate=0.0,
                error_types=[]
            )
        }

        self._active_data_sources = data_sources
        try:
            # Phase 1: Multi-database disease discovery
            disease_data = await self._phase1_disease_discovery(disease_query, data_sources)

            # Phase 1.5: Disease-gene association enrichment (NEW)
            association_data = await self._phase1p5_disease_gene_associations(
                disease_query,
                disease_data.get('primary_disease'),
                data_sources
            )

            # Phase 2: Enhanced pathway extraction with gene merging
            pathway_data = await self._phase2_pathway_extraction(
                disease_data['diseases'],
                disease_data.get('reactome_results', {}).get('pathways', []),
                association_data,  # NEW: Pass association data
                data_sources
            )

            # Phase 3: Context-aware network construction
            network_data = await self._phase3_network_construction(
                pathway_data['genes'],
                pathway_data.get('gene_to_pathways', {}),  # NEW: Pass mapping
                pathway_data.get('gene_provenance', {}),  # NEW: Pass gene provenance for priority selection
                data_sources
            )

            # Phase 4: Expression overlay
            expression_data = await self._phase4_expression_overlay(
                network_data['genes'],
                tissue_context,
                data_sources
            )

            # Phase 5: Functional enrichment analysis
            enrichment_data = await self._phase5_functional_enrichment(
                network_data['network'],
                network_data['genes'],
                data_sources
            )
        finally:
            self._active_data_sources = None
        
        # Calculate validation score
        validation_score = self._calculate_validation_score(
            disease_data, pathway_data, network_data, expression_data, data_sources
        )

        # Build result with fallback if no primary disease found
        primary_disease = disease_data.get('primary_disease')
        if primary_disease is None:
            # Create a minimal disease object if none found
            from ..models.data_models import Disease
            primary_disease = Disease(
                id=f"manual_{disease_query.lower().replace(' ', '_')}",
                name=disease_query,
                confidence=0.0,
                description=f"Query: {disease_query}",
                source="manual",
                source_db="kegg"  # Use kegg as fallback since it's one of the allowed values
            )

        # Calculate completeness metrics
        from ..models.data_models import CompletenessMetrics
        expression_completeness = expression_data.get('coverage', 0.0)
        pathway_completeness = pathway_data.get('coverage', 0.0)
        network_completeness = 1.0 if network_data.get('nodes') else 0.0
        pathology_completeness = 0.0  # Scenario 1 doesn't have pathology data
        drug_completeness = 0.0  # Scenario 1 doesn't have drug data

        # Calculate overall completeness (average of available metrics)
        available_metrics = [m for m in [expression_completeness, pathway_completeness, network_completeness, pathology_completeness, drug_completeness] if m >= 0]
        overall_completeness = sum(available_metrics) / len(available_metrics) if available_metrics else 0.0

        completeness_metrics = CompletenessMetrics(
            expression_data=expression_completeness,
            pathway_data=pathway_completeness,
            network_data=network_completeness,
            pathology_data=pathology_completeness,
            drug_data=drug_completeness,
            overall_completeness=overall_completeness
        )

        # Convert gene provenance dictionaries to GeneProvenance objects
        gene_provenance_models = None
        raw_provenance = pathway_data.get('gene_provenance', {})
        if raw_provenance and self.PROVENANCE_TRACKING_ENABLED:
            gene_provenance_models = {}
            for gene_symbol, prov_dict in raw_provenance.items():
                try:
                    gene_provenance_models[gene_symbol] = GeneProvenance(
                        gene_symbol=gene_symbol,
                        source=prov_dict.get('source', 'pathway'),
                        confidence=prov_dict.get('confidence', 0.5),
                        pathway_count=prov_dict.get('pathway_count', 0),
                        association_confidence=prov_dict.get('association_confidence', 0.0)
                    )
                except Exception as e:
                    logger.warning(f"Failed to create GeneProvenance for {gene_symbol}: {e}")

        # Extract association summary
        association_summary = None
        if association_data and self.PHASE1P5_ENABLED:
            association_summary = {
                'total_associations': association_data.get('total_associations', 0),
                'filtered_associations': association_data.get('filtered_associations', 0),
                'association_genes_count': len(association_data.get('association_genes', [])),
                'source': association_data.get('source', 'uniprot'),
                'confidence_threshold': self.ASSOCIATION_CONFIDENCE_THRESHOLD
            }

        result = DiseaseNetworkResult(
            disease=primary_disease,
            pathways=pathway_data['pathways'],
            network_nodes=network_data['nodes'],
            network_edges=network_data['edges'],
            expression_profiles=expression_data['profiles'],
            cancer_markers=expression_data['markers'],
            enrichment_results=enrichment_data,
            validation_score=validation_score,
            data_sources=list(data_sources.values()),
            completeness_metrics=completeness_metrics,
            gene_provenance=gene_provenance_models,
            association_summary=association_summary
        )
        
        logger.info(f"Disease network analysis completed. Validation score: {validation_score:.3f}")
        return result
    
    async def _extract_reactome_genes(self, pathway_id: str) -> List[str]:
        """
        Extract genes from Reactome pathway using validated MCP methods.

        IMPROVED: Based on Reactome API structure analysis and comprehensive filtering.
        Focuses on actual gene/protein names, not generic pathway terms.
        Reference: Reactome_MCP_Server_Test_Report.md - 100% success rate

        Args:
            pathway_id: Reactome pathway stable identifier (e.g., R-HSA-1227990)

        Returns:
            List of validated gene symbols
        """
        # CRITICAL FIX: Validate Reactome pathway ID format before querying
        # Reactome pathway IDs must start with "R-" (e.g., R-HSA-1234567)
        # Invalid IDs (protein IDs like Q6AI08, P61968) cause 404 errors
        if not pathway_id or not isinstance(pathway_id, str) or not pathway_id.startswith('R-'):
            logger.debug(f"Skipping invalid Reactome pathway ID (not R-* format): {pathway_id}")
            return []

        genes = set()

        try:
            # Method 1: PRIMARY - Use get_pathway_participants (more reliable for genes)
            participants = await self._call_with_tracking(
                None,
                'reactome',
                self.mcp_manager.reactome.get_pathway_participants(pathway_id)
            )
            
            # DIAGNOSTIC: Log response structure for debugging
            logger.debug(f"[Reactome] get_pathway_participants response type: {type(participants)}")
            if isinstance(participants, dict):
                logger.debug(f"[Reactome] Response keys: {list(participants.keys())[:10]}")
                # Log first few keys in detail for debugging
                if len(participants) > 0:
                    first_key = list(participants.keys())[0]
                    first_value = participants[first_key]
                    logger.debug(f"[Reactome] Sample key '{first_key}': type={type(first_value)}, "
                               f"is_list={isinstance(first_value, list)}, "
                               f"is_dict={isinstance(first_value, dict)}")
                    if isinstance(first_value, list) and len(first_value) > 0:
                        logger.debug(f"[Reactome] Sample item[0] keys: {list(first_value[0].keys())[:10] if isinstance(first_value[0], dict) else 'not_dict'}")
            
            # Handle multiple possible response structures
            participant_list = []
            
            # Structure 1: {'participants': [...]}
            if isinstance(participants, dict) and participants.get('participants'):
                participant_list = participants['participants']
                logger.debug(f"[Reactome] Found {len(participant_list)} participants in 'participants' key")
            
            # Structure 2: {'entities': [...]}
            elif isinstance(participants, dict) and participants.get('entities'):
                participant_list = participants['entities']
                logger.debug(f"[Reactome] Found {len(participant_list)} entities in 'entities' key")
            
            # Structure 3: {'proteins': [...]} - NEW: Handle proteins key
            elif isinstance(participants, dict) and participants.get('proteins'):
                participant_list = participants['proteins']
                logger.debug(f"[Reactome] Found {len(participant_list)} proteins in 'proteins' key")
            
            # Structure 4: Direct list [...]
            elif isinstance(participants, list):
                participant_list = participants
                logger.debug(f"[Reactome] Response is direct list with {len(participant_list)} items")
            
            # Structure 5: Try other common keys
            elif isinstance(participants, dict):
                for key in ['results', 'data', 'items', 'components', 'proteins', 'genes']:
                    if participants.get(key) and isinstance(participants[key], list):
                        participant_list = participants[key]
                        logger.debug(f"[Reactome] Found {len(participant_list)} items in '{key}' key")
                        break
            
            # Extract genes from participant list
            if participant_list:
                logger.debug(f"[Reactome] Processing {len(participant_list)} participants/entities")
                # Log first participant structure for debugging
                if len(participant_list) > 0 and isinstance(participant_list[0], dict):
                    first_participant = participant_list[0]
                    logger.debug(f"[Reactome] First participant keys: {list(first_participant.keys())[:15]}")
                    if 'referenceEntity' in first_participant:
                        logger.debug(f"[Reactome] First participant has referenceEntity: {type(first_participant['referenceEntity'])}")
                
                for idx, participant in enumerate(participant_list):
                    if isinstance(participant, dict):
                        gene_names = self._extract_gene_names_from_entity(participant)
                        if gene_names:
                            genes.update(gene_names)
                            if idx < 3:  # Log first 3 extractions for debugging
                                logger.debug(f"[Reactome] Participant {idx}: extracted {len(gene_names)} gene candidates: {gene_names[:3]}")
                    elif isinstance(participant, str):
                        # Direct gene symbol
                        if self._is_valid_gene_symbol(participant):
                            genes.add(participant.upper())
            
            # Method 2: FALLBACK - Use get_pathway_details if participants empty
            if not genes:
                logger.debug(f"[Reactome] No genes from participants, trying get_pathway_details for {pathway_id}")
                details = await self._call_with_tracking(
                    None,
                    'reactome',
                    self.mcp_manager.reactome.get_pathway_details(pathway_id)
                )
                
                # Extract from entities (proteins, complexes, small molecules)
                if details.get('entities'):
                    for entity in details['entities']:
                        gene_names = self._extract_gene_names_from_entity(entity)
                        genes.update(gene_names)
                
                # Extract from hasEvent (reactions with participants)
                if details.get('hasEvent'):
                    for event in details['hasEvent']:
                        # Get participants from reaction
                        if event.get('participants'):
                            for participant in event['participants']:
                                gene_names = self._extract_gene_names_from_entity(participant)
                                genes.update(gene_names)

            # Filter out non-gene terms and validate
            filtered_genes = self._filter_valid_gene_symbols(genes)

            logger.info(f"✅ Reactome pathway {pathway_id}: Retrieved {len(filtered_genes)} genes (from {len(genes)} candidates)")
            
            if len(filtered_genes) == 0 and len(genes) > 0:
                logger.warning(f"⚠️ Reactome pathway {pathway_id}: {len(genes)} candidates extracted but all filtered out")

        except (DatabaseConnectionError, DatabaseTimeoutError, MCPServerError) as e:
            # Database/MCP server errors - log and return empty list (will try fallback)
            logger.warning(
                f"Database error extracting genes from Reactome pathway {pathway_id}",
                extra=format_error_for_logging(e)
            )
            return []
        except (AttributeError, KeyError, TypeError) as e:
            # Data structure errors - log and return empty
            logger.warning(
                f"Invalid data structure in Reactome pathway {pathway_id}: {type(e).__name__}: {e}",
                extra={'pathway_id': pathway_id, 'error': str(e)}
            )
            return []
        except Exception as e:
            # Unexpected errors
            logger.warning(
                f"Failed to extract genes from Reactome pathway {pathway_id}: {type(e).__name__}: {e}",
                extra=format_error_for_logging(e)
            )
            return []
        
        return list(filtered_genes)
    
    def _extract_gene_names_from_entity(self, entity: Dict[str, Any]) -> List[str]:
        """
        Extract gene names from a Reactome entity/participant.
        
        Handles various Reactome data structures:
        - Simple proteins: {'name': 'TP53', 'type': 'Protein'}
        - Complexes: {'name': 'TP53:p-S15', 'type': 'Complex'}
        - Small molecules: {'name': 'ATP', 'type': 'SmallMolecule'}
        
        Args:
            entity: Reactome entity/participant dictionary
            
        Returns:
            List of extracted gene symbol candidates
        """
        gene_names = []
        
        # Try multiple field variations based on Reactome structure
        name_fields = ['name', 'displayName', 'geneName', 'symbol', 'identifier', 'gene_symbol', 'gene']
        
        for field in name_fields:
            if field in entity and entity[field]:
                name = entity[field]
                
                # Handle different name formats
                if isinstance(name, str):
                    # Split complex names and extract gene symbols
                    candidates = self._parse_complex_name(name)
                    gene_names.extend(candidates)
                elif isinstance(name, list):
                    # Handle list of names
                    for n in name:
                        if isinstance(n, str):
                            candidates = self._parse_complex_name(n)
                            gene_names.extend(candidates)
        
        # NEW: Extract from nested referenceEntity structure
        # Reactome often nests gene info in referenceEntity
        if 'referenceEntity' in entity and isinstance(entity['referenceEntity'], dict):
            ref_entity = entity['referenceEntity']
            for field in name_fields:
                if field in ref_entity and ref_entity[field]:
                    name = ref_entity[field]
                    if isinstance(name, str):
                        candidates = self._parse_complex_name(name)
                        gene_names.extend(candidates)
        
        # NEW: Extract from components (for complexes)
        if 'components' in entity and isinstance(entity['components'], list):
            for component in entity['components']:
                if isinstance(component, dict):
                    component_genes = self._extract_gene_names_from_entity(component)
                    gene_names.extend(component_genes)
        
        # NEW: Extract from hasComponent (alternative structure)
        if 'hasComponent' in entity:
            components = entity['hasComponent']
            if isinstance(components, list):
                for component in components:
                    if isinstance(component, dict):
                        component_genes = self._extract_gene_names_from_entity(component)
                        gene_names.extend(component_genes)
        
        return gene_names
    
    def _parse_complex_name(self, name: str) -> List[str]:
        """
        Parse complex Reactome names to extract gene symbols.
        
        Examples:
        - "TP53" → ["TP53"]
        - "TP53:p-S15" → ["TP53"]
        - "CDK1 [cytosol]" → ["CDK1"]
        - "ATM kinase [nucleoplasm]" → ["ATM"]
        - "Constitutive Signaling" → [] (filtered out)
        
        Args:
            name: Raw Reactome entity name
            
        Returns:
            List of extracted gene symbol candidates
        """
        candidates = []
        
        # Remove pathway IDs (start with R-)
        if name.startswith('R-'):
            return candidates
        
        # Split on common separators
        parts = name.split()
        
        for part in parts:
            # Remove brackets and parentheses
            clean_part = part.strip('[]()')
            
            # Remove phosphorylation/modification markers
            clean_part = clean_part.split(':')[0]  # Remove :p-S15
            clean_part = clean_part.split('-')[0] if '-' in clean_part and len(clean_part.split('-')[0]) > 1 else clean_part
            
            # Check if it looks like a gene symbol (preliminary)
            if clean_part and len(clean_part) >= 2 and len(clean_part) <= 15:
                candidates.append(clean_part)
        
        return candidates
    
    def _is_valid_gene_symbol(self, symbol: str) -> bool:
        """
        Check if a string looks like a valid gene symbol.

        Criteria:
        - Length 1-20 characters (allows single-letter genes like C, P, and longer names)
        - Starts with uppercase letter (standard nomenclature)
        - Contains mostly uppercase letters and numbers
        - Not a truly generic term (reduced to ~15 terms)
        - Mostly alphanumeric (allows hyphens in names like HLA-DRB1)
        - Whitelist for known gene patterns (TP53, BRCA1, HLA-*, etc.)

        Args:
            symbol: Candidate gene symbol

        Returns:
            True if symbol passes validation criteria
        """
        if not symbol or len(symbol) < 1 or len(symbol) > 20:
            return False

        # Must start with uppercase letter (standard gene nomenclature)
        if not symbol[0].isupper():
            return False

        # COMPREHENSIVE: Exclude non-gene terms identified from pipeline analysis
        # Updated 2025-11-20 based on Reactome extraction issues
        generic_terms = {
            # Generic pathway/biological terms
            'pathway', 'disease', 'process', 'reaction', 'signaling', 'cancer',
            
            # Common English words
            'in', 'of', 'by', 'to', 'for', 'with', 'from', 'at', 'on',
            'the', 'and', 'or', 'not', 'but', 'is', 'are', 'was', 'were',
            'an', 'as', 'be', 'it', 'if', 'so', 'no', 'up', 'do',
            
            # Protein domains and regions (NEW - from pipeline analysis)
            'ECD', 'ICD', 'TMD',  # Extracellular/Intracellular/Transmembrane Domain
            'KD', 'LBD', 'ECD',  # Kinase/Ligand Binding/Extracellular Domain
            'HD', 'MAM', 'SAM', 'SH2', 'SH3', 'PDZ',  # Domain names
            'RING', 'CARD', 'DEATH', 'PH', 'C2',  # Domain families
            'PEST',  # PEST sequence
            'BINDING', 'DOMAIN', 'REGION', 'MOTIF', 'SITE',  # Domain descriptors
            
            # Technical terms (NEW - from pipeline analysis)
            'SARA',  # SMAD Anchor for Receptor Activation (ambiguous)
            'AXIN',  # Axis inhibition protein (ambiguous)
            'HR',  # Homologous Recombination (ambiguous)
            
            # Drug classes (NEW - from pipeline analysis)
            'TKIs', 'EGFRI', 'MKIs',  # Tyrosine Kinase Inhibitors, etc.
            
            # Specific drug names (NEW - from Phase 1 investigation)
            'AEE788', 'AEE78',  # EGFR/HER2 inhibitor drugs
            
            # Single letters (NEW - from pipeline analysis)
            # These appear in Reactome pathways like "Signaling by WNT in cancer" → ['P', 'I', 'A']
            'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
            'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
            
            # Non-gene abbreviations
            'nm', 'cm', 'mm', 'pm'
        }

        # Also check for mutation patterns (e.g., E17K, V600E)
        # These have format: single letter + numbers + single letter
        if (len(symbol) <= 5 and
            symbol[0].isalpha() and
            any(c.isdigit() for c in symbol[1:-1]) and
            symbol[-1].isalpha()):
            return False  # Likely a mutation notation

        if symbol.lower() in generic_terms or symbol.upper() in generic_terms:
            return False

        # Reject if it's all lowercase (likely not a gene symbol)
        if symbol.islower():
            return False

        # Must contain at least one letter
        if not any(c.isalpha() for c in symbol):
            return False

        # Allow letters, numbers, hyphens (for genes like HLA-DRB1, TP53, BRCA1)
        if not all(c.isalnum() or c == '-' for c in symbol):
            return False

        # Don't allow symbols that are mostly numbers
        if sum(c.isdigit() for c in symbol) > len(symbol) // 2:
            return False

        # Gene symbols should be mostly uppercase (at least 50%)
        upper_count = sum(1 for c in symbol if c.isupper())
        letter_count = sum(1 for c in symbol if c.isalpha())
        if letter_count > 0 and upper_count < letter_count * 0.5:
            return False

        # ADDED: Whitelist for known gene patterns
        # Pattern 1: Classic genes like TP53, BRCA1, AKT1 (letters followed by numbers)
        if any(c.isdigit() for c in symbol) and any(c.isalpha() for c in symbol):
            return True

        # Pattern 2: HLA genes (HLA-A, HLA-DRB1, etc.)
        if symbol.startswith('HLA-'):
            return True

        # Pattern 3: All uppercase letters (e.g., AXL, TNF, IL6)
        # BUT: Exclude single letters (already in generic_terms)
        if symbol.isupper() and symbol.isalpha() and len(symbol) >= 2:
            return True

        return True
    
    def _filter_valid_gene_symbols(self, genes: Set[str]) -> List[str]:
        """
        Filter and validate gene symbols with comprehensive checks.
        
        Args:
            genes: Set of candidate gene symbols
            
        Returns:
            List of validated gene symbols
        """
        valid_genes = []
        
        for gene in genes:
            if self._is_valid_gene_symbol(gene):
                # Additional validation: must start with a letter (standard gene nomenclature)
                if gene[0].isalpha():
                    valid_genes.append(gene)
        
        return valid_genes
    
    async def _enrich_nodes_with_uniprot(self, nodes: List[NetworkNode]) -> List[NetworkNode]:
        """
        Enrich network nodes with UniProt IDs, function, and subcellular location.
        
        Phase 1 (P1) Enhancement:
        - UniProt ID from HPA (existing)
        - Function/description from UniProt MCP (new)
        - Subcellular location from HPA (new)
        
        Based on:
        - Protein_Atlas_MCP_Server_Test_Report.md (HPA)
        - UniProt MCP Server documentation
        
        Args:
            nodes: List of NetworkNode objects
            
        Returns:
            List of enriched NetworkNode objects with UniProt IDs, function, location
        """
        enriched_nodes = []
        enrichment_stats = {
            'total': len(nodes),
            'uniprot_id_from_hpa': 0,
            'uniprot_id_from_search': 0,
            'function_retrieved': 0,
            'location_retrieved': 0,
            'domains_retrieved': 0,
            'failed': 0
        }
        
        # Validate UniProt MCP is available
        if not self.mcp_manager.uniprot:
            logger.warning("⚠️  UniProt MCP not available - skipping UniProt enrichment")
            return nodes
        
        for node in nodes:
            gene_symbol = node.gene_symbol or node.id
            
            if not gene_symbol or not isinstance(gene_symbol, str):
                logger.warning(f"⚠️  Invalid gene symbol for node {node.id}: {gene_symbol}")
                enriched_nodes.append(node)
                enrichment_stats['failed'] += 1
                continue
            
            uniprot_id = None
            function = None
            subcellular_location = []
            domains = []
            
            try:
                # Method 1: Get protein info from HPA (for UniProt ID)
                try:
                    protein_info = await self._call_with_tracking(
                        None,
                        'hpa',
                        self.mcp_manager.hpa.get_protein_info(gene_symbol)
                    )
                    
                    # Extract UniProt ID
                    # HPA returns list of protein info objects, get first one
                    if protein_info and isinstance(protein_info, list) and protein_info:
                        first_protein = protein_info[0]
                        if isinstance(first_protein, dict):
                            uniprot_data = first_protein.get('Uniprot', [])
                            if uniprot_data and isinstance(uniprot_data, list) and uniprot_data:
                                uniprot_id = uniprot_data[0]
                                enrichment_stats['uniprot_id_from_hpa'] += 1
                                logger.debug(f"✅ HPA UniProt ID for {gene_symbol}: {uniprot_id}")
                except Exception as e:
                    logger.debug(f"HPA protein info lookup failed for {gene_symbol}: {e}")
                
                # Method 2: Fallback to UniProt search_by_gene if HPA didn't provide ID
                if not uniprot_id and self.mcp_manager.uniprot:
                    try:
                        logger.debug(f"Trying UniProt search_by_gene for {gene_symbol}")
                        uniprot_search = await self._call_with_tracking(
                            None,
                            'uniprot',
                            self.mcp_manager.uniprot.search_by_gene(gene_symbol, organism="human")
                        )
                        
                        # Handle both Node.js and Python MCP response formats
                        if uniprot_search:
                            if isinstance(uniprot_search, dict):
                                results = uniprot_search.get('results', [])
                                if results and isinstance(results, list) and len(results) > 0:
                                    first_result = results[0]
                                    # Node.js returns 'primaryAccession', Python might return 'accession' or 'id'
                                    uniprot_id = (first_result.get('primaryAccession') or 
                                                 first_result.get('accession') or 
                                                 first_result.get('id'))
                                    if uniprot_id:
                                        enrichment_stats['uniprot_id_from_search'] += 1
                                        logger.debug(f"✅ UniProt search ID for {gene_symbol}: {uniprot_id}")
                    except Exception as e:
                        logger.debug(f"UniProt search_by_gene failed for {gene_symbol}: {e}")
                
                # P1 Enhancement: Get function from UniProt MCP if available
                if uniprot_id and self.mcp_manager.uniprot:
                    try:
                        uniprot_info = await self._call_with_tracking(
                            None,
                            'uniprot',
                            self.mcp_manager.uniprot.get_protein_info(uniprot_id)
                        )
                        if uniprot_info and isinstance(uniprot_info, dict):
                            # Extract function/description from UniProt response
                            # Node.js MCP server wraps response in 'content' list
                            function = None
                            
                            # Handle 'content' wrapper (Node.js MCP format)
                            content = uniprot_info.get('content', [])
                            if isinstance(content, list) and len(content) > 0:
                                # Content is a list, extract from first item
                                first_item = content[0]
                                if isinstance(first_item, dict):
                                    # Check if it's a text item with function info
                                    if first_item.get('type') == 'text' or 'function' in str(first_item.get('type', '')).lower():
                                        text = first_item.get('text', '')
                                        if text and text != '[object Object]':
                                            function = text
                                    # Or if it's the full protein info dict
                                    else:
                                        uniprot_info = first_item  # Use first item as the actual data
                            
                            # 1. Direct 'function' key (list or string)
                            if not function:
                                func_list = uniprot_info.get('function', [])
                                if isinstance(func_list, list) and func_list:
                                    # Join list of function strings
                                    function = ' '.join(str(f) for f in func_list if f and str(f).strip())
                                elif isinstance(func_list, str) and func_list.strip():
                                    function = func_list
                            
                            # 2. Check 'comments' array for FUNCTION type comments
                            if not function:
                                comments = uniprot_info.get('comments', [])
                                if isinstance(comments, list):
                                    for comment in comments:
                                        if isinstance(comment, dict):
                                            comment_type = comment.get('commentType', '')
                                            if comment_type == 'FUNCTION' or 'function' in str(comment_type).lower():
                                                # Extract text from comment
                                                texts = comment.get('texts', [])
                                                if isinstance(texts, list) and texts:
                                                    function = ' '.join(str(t.get('value', t)) if isinstance(t, dict) else str(t) for t in texts if t)
                                                    break
                                                elif isinstance(texts, dict):
                                                    function = texts.get('value', '')
                                                    break
                                                else:
                                                    function = comment.get('value', comment.get('text', ''))
                                                    break
                            
                            # 3. Check 'annotation' or 'annotations' fields
                            if not function:
                                annotation = uniprot_info.get('annotation', {})
                                if isinstance(annotation, dict):
                                    function = annotation.get('function', annotation.get('description', ''))
                                elif isinstance(annotation, str):
                                    function = annotation
                            
                            # 4. Fallback to description or protein name
                            if not function:
                                function = (uniprot_info.get('description') or 
                                          uniprot_info.get('protein_name') or
                                          uniprot_info.get('recommendedName', {}).get('fullName', {}).get('value') 
                                          if isinstance(uniprot_info.get('recommendedName'), dict) else None)
                            
                            # 5. Check 'gene' array for function annotation
                            if not function:
                                genes = uniprot_info.get('genes', [])
                                if isinstance(genes, list) and genes:
                                    for gene_entry in genes:
                                        if isinstance(gene_entry, dict):
                                            gene_function = gene_entry.get('function', gene_entry.get('description', ''))
                                            if gene_function:
                                                function = gene_function
                                                break
                            
                            if function and function.strip() and function != '[object Object]':
                                enrichment_stats['function_retrieved'] += 1
                                logger.debug(f"✅ UniProt function for {gene_symbol}: {function[:100]}...")
                            else:
                                logger.debug(f"⚠️  No function found in UniProt response for {gene_symbol}")
                    except Exception as e:
                        logger.warning(f"UniProt function lookup failed for {gene_symbol} ({uniprot_id}): {e}")
                
                # P1 Enhancement: Get subcellular location from HPA
                subcellular_location = []
                try:
                    location_data = await self._call_with_tracking(
                        None,
                        'hpa',
                        self.mcp_manager.hpa.get_subcellular_location(gene_symbol)
                    )
                    if location_data and isinstance(location_data, list):
                        # HPA returns data for the gene AND related proteins
                        # Filter for only the target gene
                        for loc in location_data:
                            if isinstance(loc, dict) and loc.get('Gene') == gene_symbol:
                                # Extract main location (list format from HPA)
                                main_loc = loc.get('Subcellular main location', loc.get('Subcellular location', []))
                                if main_loc and isinstance(main_loc, list):
                                    subcellular_location.extend([str(l) for l in main_loc if l])
                                elif main_loc and isinstance(main_loc, str):
                                    subcellular_location.append(main_loc)
                                break  # Found the target gene, stop looking
                        if subcellular_location:
                            enrichment_stats['location_retrieved'] += 1
                            logger.debug(f"✅ Subcellular location for {gene_symbol}: {subcellular_location}")
                except Exception as e:
                    logger.debug(f"Subcellular location lookup failed for {gene_symbol}: {e}")
                
                # P2 Enhancement: Get protein domains from UniProt MCP (Node.js version)
                domains = []
                if uniprot_id and self.mcp_manager.uniprot:
                    try:
                        features_data = await self._call_with_tracking(
                            None,
                            'uniprot',
                            self.mcp_manager.uniprot.get_protein_features(uniprot_id)
                        )
                        if features_data:
                            # Node.js UniProt MCP returns: features, domains, activeSites, bindingSites
                            
                            # Parse main features list
                            feature_list = features_data.get('features', [])
                            if isinstance(feature_list, list):
                                for feature in feature_list:
                                    if isinstance(feature, dict):
                                        feature_type = feature.get('type', '')
                                        # Focus on druggable features
                                        if any(keyword in feature_type.lower() for keyword in [
                                            'domain', 'binding', 'active', 'site', 'region', 'motif'
                                        ]):
                                            # Extract location (start/end) from location object
                                            location = feature.get('location', {})
                                            start_val = None
                                            end_val = None
                                            
                                            if isinstance(location, dict):
                                                start_obj = location.get('start', {})
                                                end_obj = location.get('end', {})
                                                if isinstance(start_obj, dict):
                                                    start_val = start_obj.get('value')
                                                elif isinstance(start_obj, (int, str)):
                                                    try:
                                                        start_val = int(start_obj)
                                                    except:
                                                        pass
                                                
                                                if isinstance(end_obj, dict):
                                                    end_val = end_obj.get('value')
                                                elif isinstance(end_obj, (int, str)):
                                                    try:
                                                        end_val = int(end_obj)
                                                    except:
                                                        pass
                                            
                                            domain_info = {
                                                'type': feature_type,
                                                'description': feature.get('description', ''),
                                                'start': start_val,
                                                'end': end_val
                                            }
                                            domains.append(domain_info)
                            
                            # Also check dedicated domains/activeSites/bindingSites fields if present
                            for field_name in ['domains', 'activeSites', 'bindingSites']:
                                additional = features_data.get(field_name, [])
                                if isinstance(additional, list):
                                    for item in additional:
                                        if isinstance(item, dict):
                                            domains.append({
                                                'type': item.get('type', field_name),
                                                'description': item.get('description', ''),
                                                'start': item.get('start'),
                                                'end': item.get('end')
                                            })
                            
                            if domains:
                                enrichment_stats['domains_retrieved'] += 1
                                logger.debug(f"✅ Protein domains for {gene_symbol}: {len(domains)} features")
                    except Exception as e:
                        logger.warning(f"Protein features lookup failed for {gene_symbol} ({uniprot_id}): {e}")
                
                # Create enriched node with P1 + P2 enhancements
                enriched_node = NetworkNode(
                    id=node.id,
                    node_type=node.node_type,
                    gene_symbol=node.gene_symbol,
                    uniprot_id=uniprot_id,
                    pathways=node.pathways,
                    centrality_measures=node.centrality_measures,
                    function=function,  # P1: protein function
                    subcellular_location=subcellular_location,  # P1: localization
                    domains=domains if domains else None  # P2: protein domains
                )
                enriched_nodes.append(enriched_node)
                
                if enriched_node.uniprot_id:
                    logger.debug(f"✅ UniProt enrichment: {gene_symbol} → {enriched_node.uniprot_id}")
                else:
                    logger.debug(f"⚠️  No UniProt ID found for {gene_symbol}")
                
            except Exception as e:
                logger.warning(f"Failed to enrich node {gene_symbol}: {type(e).__name__}: {e}")
                enrichment_stats['failed'] += 1
                # Keep original node if enrichment fails
                enriched_nodes.append(node)
            finally:
                # Throttle to avoid MCP stdio contention
                await asyncio.sleep(0.1)
        
        # Summary logging with detailed statistics
        enriched_count = sum(1 for n in enriched_nodes if n.uniprot_id)
        function_count = sum(1 for n in enriched_nodes if n.function)
        location_count = sum(1 for n in enriched_nodes if n.subcellular_location)
        domain_count = sum(1 for n in enriched_nodes if n.domains)
        
        logger.info(f"✅ UniProt enrichment complete:")
        logger.info(f"   - UniProt IDs: {enriched_count}/{len(nodes)} nodes ({100*enriched_count/len(nodes):.1f}%)")
        logger.info(f"     ↳ From HPA: {enrichment_stats['uniprot_id_from_hpa']}")
        logger.info(f"     ↳ From UniProt search: {enrichment_stats['uniprot_id_from_search']}")
        logger.info(f"   - Functions: {function_count}/{len(nodes)} nodes (P1) ({100*function_count/len(nodes):.1f}%)")
        logger.info(f"   - Locations: {location_count}/{len(nodes)} nodes (P1) ({100*location_count/len(nodes):.1f}%)")
        logger.info(f"   - Domains: {domain_count}/{len(nodes)} nodes (P2) ({100*domain_count/len(nodes):.1f}%)")
        if enrichment_stats['failed'] > 0:
            logger.warning(f"   - Failed enrichments: {enrichment_stats['failed']}/{len(nodes)} nodes")
        
        return enriched_nodes
    
    def _infer_interaction_type(self, edge_data: Dict) -> str:
        """
        Infer interaction type from STRING evidence scores.
        
        STEP 4 FIX: Based on STRING_MCP_Server_Test_Report.md:
        - experimental: Direct physical interaction evidence
        - database: Curated interaction databases
        - coexpression: Correlated expression
        - textmining: Literature co-occurrence
        
        Args:
            edge_data: Edge dictionary with STRING evidence scores
            
        Returns:
            Interaction type: 'physical', 'coexpression', 'functional', 'association', or 'predicted'
        """
        # Get evidence scores (STRING uses 0-1 scale)
        experimental = edge_data.get('experimental', 0)
        database = edge_data.get('database', 0)
        coexpression = edge_data.get('coexpression', 0)
        textmining = edge_data.get('textmining', 0)
        
        # Inference rules based on STRING documentation
        if experimental > 0.5 or database > 0.5:
            return "physical"  # Direct physical interaction
        elif coexpression > 0.7:
            return "coexpression"  # Functionally linked by expression
        elif textmining > 0.5 and coexpression > 0.3:
            return "functional"  # Functionally related
        elif textmining > 0.5:
            return "association"  # Literature association
        else:
            return "predicted"  # Computational prediction
    
    async def _phase1_disease_discovery(self, query: str, data_sources: Dict) -> Dict[str, Any]:
        """
        Phase 1: Enhanced multi-term disease discovery.

        Strategy: Expands query to multiple search terms (e.g., "breast cancer" →
        ["breast cancer", "breast neoplasm", "cancer"]) and queries each term in
        parallel across KEGG and Reactome for comprehensive pathway coverage.

        Returns deduplicated and relevance-ranked pathways.
        """
        logger.info("Phase 1: Enhanced multi-term disease discovery")

        # STEP 1: Expand query into multiple search terms
        search_terms = self._expand_disease_query(query)

        # STEP 2: Multi-term parallel search across databases
        all_kegg_diseases = {}
        all_reactome_pathways = []
        hpa_markers = None

        try:
            # Build tasks for all search terms
            tasks = []

            for term in search_terms:
                # KEGG disease search
                tasks.append(
                    self._call_with_tracking(
                        data_sources,
                        'kegg',
                        self.mcp_manager.kegg.search_diseases(term, limit=10)
                    )
                )

                # Reactome disease-specific search
                tasks.append(
                    self._call_with_tracking(
                        data_sources,
                        'reactome',
                        self.mcp_manager.reactome.find_pathways_by_disease(term, size=30)
                    )
                )

                # Reactome general pathway search (broader coverage)
                tasks.append(
                    self._call_with_tracking(
                        data_sources,
                        'reactome',
                        self.mcp_manager.reactome.search_pathways(term, limit=30)
                    )
                )

            # HPA cancer markers (original query only)
            if self._is_cancer_query(query):
                tasks.append(
                    self._call_with_tracking(
                        data_sources,
                        'hpa',
                        self.mcp_manager.hpa.search_cancer_markers(query)
                    )
                )

            # Execute all queries in parallel
            logger.info(f"Executing {len(tasks)} parallel queries across {len(search_terms)} search terms")
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # STEP 3: Aggregate results
            task_idx = 0

            for term_idx, term in enumerate(search_terms):
                # KEGG diseases
                kegg_result = results[task_idx]
                task_idx += 1
                if not isinstance(kegg_result, Exception):
                    if isinstance(kegg_result, dict) and kegg_result.get('diseases'):
                        all_kegg_diseases.update(kegg_result['diseases'])
                        logger.debug(f"  '{term}': KEGG returned {len(kegg_result.get('diseases', {}))} diseases")

                # Reactome disease-specific
                reactome_disease_result = results[task_idx]
                task_idx += 1
                if not isinstance(reactome_disease_result, Exception):
                    # Handle both dict and list responses
                    if isinstance(reactome_disease_result, list):
                        pathways = reactome_disease_result
                    elif isinstance(reactome_disease_result, dict):
                        pathways = reactome_disease_result.get('pathways', [])
                    else:
                        pathways = []
                    all_reactome_pathways.extend(pathways)
                    logger.debug(f"  '{term}': Reactome disease search returned {len(pathways)} pathways")

                # Reactome general
                reactome_general_result = results[task_idx]
                task_idx += 1
                if not isinstance(reactome_general_result, Exception):
                    # Handle both dict and list responses
                    if isinstance(reactome_general_result, list):
                        pathways = reactome_general_result
                    elif isinstance(reactome_general_result, dict):
                        pathways = reactome_general_result.get('pathways', reactome_general_result.get('results', []))
                    else:
                        pathways = []
                    all_reactome_pathways.extend(pathways)
                    logger.debug(f"  '{term}': Reactome general search returned {len(pathways)} pathways")

            # HPA markers
            if self._is_cancer_query(query):
                hpa_result = results[task_idx]
                if not isinstance(hpa_result, Exception):
                    hpa_markers = hpa_result
                    if isinstance(hpa_result, dict):
                        logger.debug(f"  HPA returned {len(hpa_result.get('markers', []))} cancer markers")
                    else:
                        logger.debug(f"  HPA returned result (type: {type(hpa_result).__name__})")

            logger.info(f"✅ Multi-term search complete:")
            logger.info(f"   KEGG: {len(all_kegg_diseases)} unique diseases")
            logger.info(f"   Reactome: {len(all_reactome_pathways)} total pathways (before deduplication)")

        except Exception as e:
            logger.error(
                f"Multi-term search failed, falling back to original query: {e}",
                extra=format_error_for_logging(e)
            )
            # Fallback to original single-term search
            try:
                all_kegg_diseases = (await self._call_with_tracking(
                    data_sources, 'kegg',
                    self.mcp_manager.kegg.search_diseases(query, limit=10)
                )).get('diseases', {})
            except Exception:
                all_kegg_diseases = {}

            try:
                reactome_result = await self._call_with_tracking(
                    data_sources, 'reactome',
                    self.mcp_manager.reactome.find_pathways_by_disease(query, size=30)
                )
                all_reactome_pathways = reactome_result.get('pathways', [])
            except Exception:
                all_reactome_pathways = []

            if self._is_cancer_query(query):
                try:
                    hpa_markers = await self._call_with_tracking(
                        data_sources, 'hpa',
                        self.mcp_manager.hpa.search_cancer_markers(query)
                    )
                except Exception:
                    hpa_markers = None

        # STEP 4: Deduplicate and rank pathways
        unique_pathways = self._deduplicate_pathways(all_reactome_pathways)
        ranked_pathways = self._rank_pathways_by_relevance(unique_pathways, query)

        logger.info(f"✅ After deduplication: {len(ranked_pathways)} unique pathways")

        # Prepare results in expected format
        kegg_diseases = {"diseases": all_kegg_diseases}
        reactome_pathways = {"pathways": ranked_pathways}
        
        # Standardize results with BATCH QUERIES (P0-3: 10-20x speedup!)
        diseases = []
        if kegg_diseases.get('diseases'):
            # KEGG returns diseases as {"disease_id": "disease_name"} dict
            disease_items = list(kegg_diseases['diseases'].items())

            if disease_items:
                logger.info(
                    f"Batch querying {len(disease_items)} disease details with parallel execution"
                )

                # Use batch_query to parallelize disease detail retrieval
                disease_infos = await batch_query(
                    lambda item: self._call_with_tracking(
                        data_sources,
                        'kegg',
                        self.mcp_manager.kegg.get_disease_info(item[0])
                    ),
                    disease_items,
                    batch_size=10
                )

                # Combine results
                for (disease_id, disease_name), disease_info in zip(disease_items, disease_infos):
                    try:
                        # Handle None results (failed queries)
                        if not disease_info or isinstance(disease_info, Exception):
                            logger.warning(f"Failed to get info for disease {disease_id}, using fallback")
                            # Fallback to basic disease object
                            disease = Disease(
                                id=disease_id,
                                name=disease_name,
                                source_db='kegg',
                                pathways=[],
                                confidence=0.7,
                                description=disease_name
                            )
                            diseases.append(disease)
                            continue

                        # Extract pathways from KEGG format
                        pathways = []
                        if 'pathway' in disease_info and isinstance(disease_info['pathway'], dict):
                            pathways = list(disease_info['pathway'].keys())

                        # Create disease object with detailed info
                        disease = Disease(
                            id=disease_id,
                            name=disease_name,
                            source_db='kegg',
                            pathways=pathways,
                            confidence=0.9,  # High confidence for direct KEGG match
                            description=disease_info.get('description', disease_name),
                            category=disease_info.get('type')
                        )

                        if self.validator.validate_disease_confidence(disease):
                            diseases.append(disease)
                    except Exception as e:
                        logger.warning(f"Failed to process disease {disease_id}: {e}")
                        # Fallback to basic disease object
                        disease = Disease(
                            id=disease_id,
                            name=disease_name,
                            source_db='kegg',
                            pathways=[],
                            confidence=0.7,
                            description=disease_name
                        )
                        diseases.append(disease)

                logger.info(f"✅ Batch disease extraction complete: {len(diseases)} diseases")
        
        # Get primary disease (highest confidence)
        primary_disease = max(diseases, key=lambda d: d.confidence) if diseases else None
        
        # Cross-database concordance scoring
        concordance = self.validator.validate_cross_database_concordance(
            kegg_diseases, reactome_pathways
        )
        
        return {
            'diseases': diseases,
            'primary_disease': primary_disease,
            'kegg_results': kegg_diseases,
            'reactome_results': reactome_pathways,
            'hpa_markers': hpa_markers,
            'concordance': concordance
        }
    
    async def _phase2_pathway_extraction(
        self,
        diseases: List[Disease],
        existing_pathways: List[Dict],
        association_data: Dict[str, Any],  # NEW: From Phase 1.5
        data_sources: Dict
    ) -> Dict[str, Any]:
        """
        Phase 2: Enhanced pathway extraction with gene merging.

        Extract pathways from KEGG and Reactome, then merge with
        association-based genes from Phase 1.5 for comprehensive coverage.

        Args:
            diseases: List of Disease objects
            existing_pathways: Pathways from Phase 1
            association_data: Gene association data from Phase 1.5
            data_sources: Data source tracking dict

        Returns:
            Dict with pathways, merged genes, coverage, gene_to_pathways,
            and gene_provenance tracking
        """
        logger.info("Phase 2: Enhanced pathway extraction with gene merging")
        
        # Get pathways for primary disease
        primary_disease = diseases[0] if diseases else None
        if not primary_disease:
            return {'pathways': [], 'genes': []}
        
        # Use pathways from disease info and Reactome results
        pathways = []
        all_genes = set()

        # Process KEGG pathways from disease with BATCH QUERIES (P0-3: 10-20x speedup!)
        if primary_disease.pathways:
            # Clean pathway IDs (remove anything in parentheses)
            clean_pathway_ids = [
                pid_raw.split('(')[0] if '(' in pid_raw else pid_raw
                for pid_raw in primary_disease.pathways
            ]

            logger.info(
                f"Batch querying {len(clean_pathway_ids)} KEGG pathways with parallel execution"
            )

            # Use batch_query to parallelize pathway info retrieval
            # This replaces sequential loop with parallel execution (10-20x faster!)
            pathway_infos = await batch_query(
                lambda pid: self._call_with_tracking(
                    data_sources,
                    'kegg',
                    self.mcp_manager.kegg.get_pathway_info(pid)
                ),
                clean_pathway_ids,
                batch_size=10  # Process 10 pathways at a time
            )

            # Use batch_query to parallelize gene retrieval
            pathway_genes_list = await batch_query(
                lambda pid: self._call_with_tracking(
                    data_sources,
                    'kegg',
                    self.mcp_manager.kegg.get_pathway_genes(pid)
                ),
                clean_pathway_ids,
                batch_size=10
            )

            # Combine results
            for pathway_id, pathway_info, pathway_genes in zip(
                clean_pathway_ids, pathway_infos, pathway_genes_list
            ):
                try:
                    # Handle None results (failed queries)
                    if not pathway_info or isinstance(pathway_info, Exception):
                        logger.warning(f"Failed to get info for pathway {pathway_id}")
                        continue
                    if not pathway_genes or isinstance(pathway_genes, Exception):
                        logger.warning(f"Failed to get genes for pathway {pathway_id}")
                        continue

                    # Extract gene list (KEGG returns array of gene objects)
                    genes = []
                    if pathway_genes.get('genes') and isinstance(pathway_genes['genes'], list):
                        genes = [gene.get('name') for gene in pathway_genes['genes'] if gene.get('name')]
                    elif pathway_genes.get('genes') and isinstance(pathway_genes['genes'], dict):
                        genes = list(pathway_genes['genes'].keys())

                    # Create pathway object
                    pathway = Pathway(
                        id=pathway_id,
                        name=pathway_info.get('name', pathway_id),
                        source_db='kegg',
                        genes=genes,
                        confidence=0.9
                    )
                    pathways.append(pathway)
                    all_genes.update(genes)

                except Exception as e:
                    logger.warning(f"Failed to process pathway {pathway_id}: {e}")

            logger.info(
                f"✅ Batch pathway extraction complete: {len(pathways)} pathways, "
                f"{len(all_genes)} unique genes"
            )
        
        # Process Reactome pathways with breast cancer-specific filtering
        if existing_pathways:
            logger.info(f"Processing {len(existing_pathways)} Reactome pathways...")
            
            # Define breast cancer-specific keywords for filtering
            breast_cancer_keywords = [
                'breast', 'brca', 'her2', 'erbb2', 'estrogen', 'er+', 'er-',
                'progesterone', 'pr+', 'pr-', 'triple negative', 'tnbc',
                'luminal', 'basal', 'mammary'
            ]
            
            for pathway_data in existing_pathways:
                try:
                    # Reactome pathways from find_pathways_by_disease
                    pathway_id = pathway_data.get('id')

                    # CRITICAL FIX: Validate Reactome pathway ID format
                    # Reactome pathway IDs must start with "R-" (e.g., R-HSA-1234567)
                    # Invalid IDs (protein IDs like Q6AI08, P61968) cause 404 errors
                    if not pathway_id or not isinstance(pathway_id, str) or not pathway_id.startswith('R-'):
                        logger.debug(f"Skipping invalid Reactome ID (not a pathway): {pathway_id}")
                        continue

                    pathway_name = pathway_data.get('name', '').lower()

                    # Filter for breast cancer relevance (but keep highly relevant cancer pathways)
                    is_breast_specific = any(keyword in pathway_name for keyword in breast_cancer_keywords)
                    is_generic_cancer = 'cancer' in pathway_name and 'signaling' in pathway_name
                    
                    # Skip pathways that are clearly not breast cancer related
                    if not is_breast_specific and not is_generic_cancer:
                        if any(x in pathway_name for x in ['lung', 'glioblastoma', 'egfrviii', 'alk', 'ltk']):
                            logger.debug(f"Skipping non-breast cancer pathway: {pathway_name}")
                            continue
                    
                    pathway_name = pathway_data.get('name')  # Get original case
                    
                    # CRITICAL FIX: Extract genes using get_pathway_details
                    # Based on Reactome_MCP_Server_Test_Report.md validation
                    genes = await self._extract_reactome_genes(pathway_id)
                    
                    # Create pathway object
                    pathway = Pathway(
                        id=pathway_id,
                        name=pathway_name,
                        source_db='reactome',
                        genes=genes,
                        confidence=0.8,
                        description=pathway_data.get('description', '')
                    )
                    pathways.append(pathway)
                    all_genes.update(genes)
                    
                except Exception as e:
                    logger.warning(f"Failed to get Reactome pathway {pathway_data.get('id')}: {e}")
        
        # Calculate pathway coverage
        expected_pathways = primary_disease.pathways if primary_disease else []
        disease_pathways = [p.id for p in pathways]
        coverage = self.validator.validate_pathway_coverage(
            expected_pathways, disease_pathways
        )
        
        # STEP 2 FIX: Build gene-to-pathways mapping for network enrichment
        gene_to_pathways = {}  # gene_symbol -> [pathway_info_dicts]
        for pathway in pathways:
            pathway_info = {
                'id': pathway.id,
                'name': pathway.name,
                'source': pathway.source_db,
                'confidence': pathway.confidence
            }
            for gene in pathway.genes:
                if gene not in gene_to_pathways:
                    gene_to_pathways[gene] = []
                gene_to_pathways[gene].append(pathway_info)
        
        logger.info(f"Built gene-to-pathway mapping for {len(gene_to_pathways)} genes across {len(pathways)} pathways")

        # NEW: Merge pathway genes with association genes from Phase 1.5
        pathway_genes = list(all_genes)
        association_genes = association_data.get('association_genes', [])
        gene_confidence = association_data.get('gene_confidence', {})

        if self.PROVENANCE_TRACKING_ENABLED and association_genes:
            logger.info(f"Merging pathway and association genes:")
            logger.info(f"  Pathway genes: {len(pathway_genes)}")
            logger.info(f"  Association genes: {len(association_genes)}")

            # Track gene provenance
            gene_provenance = {}

            # Process pathway genes
            pathway_set = set(pathway_genes)
            association_set = set(association_genes)

            for gene in pathway_genes:
                gene_provenance[gene] = {
                    'source': 'pathway',
                    'confidence': 0.9,  # High confidence from pathway membership
                    'pathway_count': len(gene_to_pathways.get(gene, [])),
                    'association_confidence': gene_confidence.get(gene, 0.0)
                }

            # Add association-only genes (not in pathways)
            for gene in association_genes:
                if gene not in pathway_set:
                    gene_provenance[gene] = {
                        'source': 'association',
                        'confidence': gene_confidence.get(gene, 0.7),
                        'pathway_count': 0,
                        'association_confidence': gene_confidence.get(gene, 0.7)
                    }

            # Mark genes found in both sources
            overlap = pathway_set & association_set
            for gene in overlap:
                gene_provenance[gene]['source'] = 'both'
                # Boost confidence for genes in both sources
                gene_provenance[gene]['confidence'] = min(
                    gene_provenance[gene]['confidence'] + 0.1,
                    1.0
                )

            # Merge and filter by confidence
            all_genes_merged = pathway_set | association_set
            confidence_threshold = self.ASSOCIATION_CONFIDENCE_THRESHOLD
            filtered_genes = [
                gene for gene in all_genes_merged
                if gene_provenance[gene]['confidence'] >= confidence_threshold
            ]

            logger.info(f"  Overlap: {len(overlap)} genes")
            logger.info(f"  Total unique: {len(all_genes_merged)} genes")
            logger.info(f"  After confidence filter (>={confidence_threshold}): {len(filtered_genes)} genes")

            # Validate coverage
            validation_result = self._validate_gene_coverage(
                pathway_genes=pathway_genes,
                association_genes=association_genes,
                merged_genes=filtered_genes,
                gene_provenance=gene_provenance,
                disease_query=primary_disease.name if primary_disease else ""
            )

            logger.info(f"Gene coverage validation:")
            logger.info(f"  Pathway-only: {validation_result['pathway_only_count']} genes")
            logger.info(f"  Association-only: {validation_result['association_only_count']} genes")
            logger.info(f"  Both sources: {validation_result['overlap_count']} genes")

            # Use merged genes
            final_genes = filtered_genes
        else:
            # No association data or provenance tracking disabled
            gene_provenance = {}
            final_genes = pathway_genes
            logger.info(f"Using pathway-only genes: {len(final_genes)}")

        return {
            'pathways': pathways,
            'genes': final_genes,
            'coverage': coverage,
            'gene_to_pathways': gene_to_pathways,
            'gene_provenance': gene_provenance if self.PROVENANCE_TRACKING_ENABLED else {}  # NEW
        }
    
    async def _phase3_network_construction(
        self,
        genes: List[str],
        gene_to_pathways: Dict[str, List[Dict]] = None,
        gene_provenance: Dict[str, Dict] = None,
        data_sources: Dict = None
    ) -> Dict[str, Any]:
        """
        Phase 3: Context-aware network construction.

        Build STRING network with pathway context weighting.

        Args:
            genes: List of gene symbols
            gene_to_pathways: Mapping from gene symbols to their pathway memberships
            gene_provenance: Provenance tracking for gene sources (association vs pathway)
        """
        logger.info("Phase 3: Context-aware network construction")

        if not genes:
            return {'network': nx.Graph(), 'nodes': [], 'edges': [], 'genes': []}

        if gene_to_pathways is None:
            gene_to_pathways = {}

        if gene_provenance is None:
            gene_provenance = {}

        # Store mapping for use in helper methods
        self.gene_to_pathways = gene_to_pathways

        # Extract high-confidence association genes as priority genes
        # These should be preserved even if they have 0 STRING edges
        priority_genes = []
        if gene_provenance:
            for gene, prov in gene_provenance.items():
                source = prov.get('source', '')
                confidence = prov.get('confidence', 0)
                # Include genes from association or both sources with confidence >= 0.5
                if source in ['association', 'both'] and confidence >= 0.5:
                    priority_genes.append(gene)

            if priority_genes:
                logger.info(
                    f"Identified {len(priority_genes)} priority genes from associations "
                    f"(will be preserved even with 0 STRING edges): {priority_genes[:5]}..."
                )

        # Initialize adaptive STRING builder if not already done
        if self.string_builder is None:
            self.string_builder = AdaptiveStringNetworkBuilder(
                self.mcp_manager,
                data_sources
            )

        # Build network using adaptive builder
        logger.info(f"Building adaptive STRING network for {len(genes)} genes")
        try:
            network_result = await self.string_builder.build_network(
                genes=genes,
                priority_genes=priority_genes,  # Pass priority genes
                data_sources=data_sources
            )
            
            all_nodes = network_result.get('nodes', [])
            all_edges = network_result.get('edges', [])
            expansion_attempts = network_result.get('expansion_attempts', 1)
            
            logger.info(
                f"Adaptive STRING network: {len(all_nodes)} nodes, {len(all_edges)} edges "
                f"after {expansion_attempts} attempt(s)"
            )
            
            # Convert to expected format
            string_network = {
                'nodes': all_nodes,
                'edges': all_edges
            }
        except Exception as e:
            logger.warning(f"STRING network construction failed: {e}")
            string_network = {
                'nodes': [],
                'edges': []
            }
        
        # STRING returns nodes and edges directly in response
        nodes = string_network.get('nodes', [])
        edges = string_network.get('edges', [])
        
        logger.info(f"STRING returned {len(nodes)} nodes and {len(edges)} edges")
        if edges and len(edges) > 0:
            logger.info(f"First edge structure: {edges[0]}")
        
        # Build NetworkX graph
        G = nx.Graph()
        
        # Add nodes (STRING format: protein_name, string_id, annotation)
        for node_data in nodes:
            node_id = node_data.get('protein_name', node_data.get('string_id', ''))
            if node_id:  # Only add non-empty nodes
                G.add_node(node_id, **node_data)
        
        logger.info(f"Added {G.number_of_nodes()} nodes to graph")
        
        # Add edges - STRING uses protein_a/protein_b field names
        edges_added = 0
        for edge_data in edges:
            # Try multiple field name variations (STRING uses protein_a/protein_b)
            source = (edge_data.get('protein_a') or
                     edge_data.get('preferredName_A') or 
                     edge_data.get('source') or 
                     edge_data.get('protein1') or
                     edge_data.get('from', ''))
            target = (edge_data.get('protein_b') or
                     edge_data.get('preferredName_B') or 
                     edge_data.get('target') or 
                     edge_data.get('protein2') or
                     edge_data.get('to', ''))
            score = edge_data.get('confidence_score', edge_data.get('score', 0))
            
            if source and target and source in G.nodes() and target in G.nodes():
                # Score is already 0-1 scale if < 1, otherwise divide by 1000
                weight = score if score <= 1.0 else score / 1000.0

                # Apply pathway context weighting
                pathway_weight = self._get_pathway_context_weight(source, target)
                weighted_weight = weight * pathway_weight

                G.add_edge(source, target, weight=weighted_weight, score=score, **edge_data)
                edges_added += 1
        
        logger.info(f"Added {edges_added} edges to graph")

        # Add priority genes that weren't returned by STRING (have 0 edges)
        # These are high-confidence association genes that should be preserved
        if priority_genes:
            added_priority = 0
            for gene in priority_genes:
                if gene not in G.nodes():
                    # Get provenance info for this gene
                    prov = gene_provenance.get(gene, {})
                    confidence = prov.get('confidence', 0.7)
                    source = prov.get('source', 'association')

                    # Add as isolated node with provenance metadata
                    G.add_node(
                        gene,
                        protein_name=gene,
                        string_id=f"synthetic_{gene}",
                        source=source,
                        confidence=confidence,
                        isolated=True  # Mark as isolated (0 STRING edges)
                    )
                    added_priority += 1
                    logger.info(
                        f"Added priority gene {gene} as isolated node "
                        f"(0 STRING edges, source: {source}, confidence: {confidence})"
                    )

            if added_priority > 0:
                logger.info(
                    f"Preserved {added_priority} priority genes with 0 STRING edges "
                    f"(e.g., TP53, PIK3CA from disease associations)"
                )

        if edges_added == 0 and len(edges) > 0:
            logger.warning(f"STRING returned {len(edges)} edges but none were added!")
            logger.warning(f"Sample edge: {edges[0] if edges else 'N/A'}")
            logger.warning(f"Graph nodes: {list(G.nodes())[:5]}")
        
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
            
            # STEP 4 FIX: Infer interaction type from evidence scores
            interaction_type = self._infer_interaction_type(edge_data)
            
            network_edge = NetworkEdge(
                source=source,
                target=target,
                weight=edge_data.get('weight', 0.0),
                interaction_type=interaction_type,  # UPDATED
                evidence_score=edge_data.get('score', 0.0) / 1000.0,
                pathway_context=self._get_edge_pathway_context(source, target)
            )
            network_edges.append(network_edge)
        
        # STEP 3 FIX: Enrich nodes with UniProt IDs from HPA
        logger.info(f"Enriching {len(network_nodes)} nodes with UniProt IDs...")
        network_nodes = await self._enrich_nodes_with_uniprot(network_nodes)
        
        return {
            'network': G,
            'nodes': network_nodes,
            'edges': network_edges,
            'genes': genes
        }
    
    async def _phase4_expression_overlay(
        self,
        genes: List[str],
        tissue_context: Optional[str],
        data_sources: Dict = None
    ) -> Dict[str, Any]:
        """
        Phase 4: Expression overlay.
        
        Get tissue expression data from HPA.
        """
        logger.info("Phase 4: Expression overlay")
        
        expression_profiles = []
        cancer_markers = []
        
        # Get expression data for each gene
        # Filter genes to prevent malformed symbols like "An" from being sent to HPA
        valid_expression_genes = [
            g for g in genes[:20]
            if g and len(g) >= 2 and g[0].isalpha() and not g.lower() in {'an', 'in', 'of', 'by', 'to', 'at', 'on', 'is'}
        ]

        logger.info(f"Retrieving expression data for {len(valid_expression_genes)} genes in tissue: {tissue_context}")
        if len(valid_expression_genes) < len(genes[:20]):
            filtered_out = set(genes[:20]) - set(valid_expression_genes)
            logger.debug(f"Filtered out {len(filtered_out)} invalid gene symbols: {filtered_out}")

        # Import HPA parsing helpers
        from ..utils.hpa_parsing import _iter_expr_items, categorize_expression

        for gene in valid_expression_genes:
            try:
                # Get tissue expression (HPA returns list directly)
                expression_data = await self._call_with_tracking(
                    data_sources,
                    'hpa',
                    self.mcp_manager.hpa.get_tissue_expression(gene)
                )
                
                # Use HPA parsing helpers to handle list/dict formats
                for tissue, ntpms in _iter_expr_items(expression_data):
                    # Filter by tissue context if provided (from YAML)
                    if tissue_context and tissue_context.lower() not in tissue.lower():
                        continue
                    
                    # Convert nTPM to categorical using helper
                    expression_level = categorize_expression(ntpms)
                    
                    expression_profile = ExpressionProfile(
                        gene=gene,
                        tissue=tissue,
                        expression_level=expression_level,
                        reliability='Approved',
                        cell_type_specific=False,
                        subcellular_location=[]
                    )
                    expression_profiles.append(expression_profile)
                
                # Get cancer markers (HPA returns list directly)
                if self._is_cancer_gene(gene):
                    markers = await self._call_with_tracking(
                        data_sources,
                        'hpa',
                        self.mcp_manager.hpa.search_cancer_markers(gene)
                    )
                    
                    if isinstance(markers, list):
                        for marker_data in markers:
                            try:
                                marker = self.standardizer.standardize_cancer_marker(marker_data)
                                if self.validator.validate_cancer_marker_confidence(marker):
                                    cancer_markers.append(marker)
                            except Exception as e:
                                logger.warning(f"Failed to standardize marker for {gene}: {e}")
                    elif isinstance(markers, dict) and markers.get('markers'):
                        for marker_data in markers['markers']:
                            try:
                                marker = self.standardizer.standardize_cancer_marker(marker_data)
                                if self.validator.validate_cancer_marker_confidence(marker):
                                    cancer_markers.append(marker)
                            except Exception as e:
                                logger.warning(f"Failed to standardize marker for {gene}: {e}")
                
            except Exception as e:
                logger.warning(f"Failed to get expression data for {gene}: {e}")
                continue
            finally:
                # Small delay to avoid MCP stdio contention across sequential requests
                await asyncio.sleep(0.1)
        
        logger.info(f"Retrieved {len(expression_profiles)} expression profiles and {len(cancer_markers)} cancer markers")
        
        # Calculate expression coverage
        covered_genes = set(ep.gene for ep in expression_profiles)
        # Use valid_expression_genes (subset) for coverage calculation to be fair
        coverage = self.validator.validate_expression_coverage(valid_expression_genes, covered_genes)
        
        return {
            'profiles': expression_profiles,
            'markers': cancer_markers,
            'coverage': coverage
        }
    
    async def _phase5_functional_enrichment(
        self,
        network: nx.Graph,
        genes: List[str],
        data_sources: Dict = None
    ) -> Dict[str, Any]:
        """
        Phase 5: Functional enrichment analysis.
        
        Perform GO enrichment using STRING.
        """
        logger.info("Phase 5: Functional enrichment analysis")
        
        if not genes:
            return {'enrichment': {}}
        
        # Get functional enrichment (use up to 20 genes for better results)
        gene_subset = genes[:20] if len(genes) > 20 else genes
        logger.info(f"Calling STRING enrichment with {len(gene_subset)} genes...")
        
        try:
            enrichment = await self._call_with_tracking(
                data_sources,
                'string',
                self.mcp_manager.string.get_functional_enrichment(
                    protein_ids=gene_subset,
                    species="9606"
                )
            )
            logger.info(f"STRING enrichment returned: {list(enrichment.keys()) if enrichment else 'empty'}")
            logger.info(f"Total enrichment terms: {enrichment.get('total_terms', 0)}")
        except (DatabaseConnectionError, DatabaseTimeoutError, MCPServerError) as e:
            # Database/MCP server errors during enrichment
            logger.warning(
                f"Functional enrichment failed for {len(genes)} genes",
                extra=format_error_for_logging(e)
            )
            return {'enrichment': {}, 'total_genes': len(genes), 'significant_terms': 0}
        except Exception as e:
            # Unexpected errors during enrichment
            logger.warning(
                f"Functional enrichment failed with unexpected error: {type(e).__name__}: {e}",
                extra=format_error_for_logging(e)
            )
            return {'enrichment': {}, 'total_genes': len(genes), 'significant_terms': 0}
        
        # STRING returns 'enrichments' (plural) not 'enrichment'
        enrichment_data = enrichment.get('enrichments', enrichment.get('enrichment', {}))
        
        # Process enrichment results
        processed_enrichment = {}
        for category, terms in enrichment_data.items():
            if isinstance(terms, list):
                processed_terms = []
                for term in terms:
                    if isinstance(term, dict):
                        processed_terms.append({
                            'term': term.get('term', ''),
                            'p_value': term.get('p_value', 1.0),
                            'fdr': term.get('fdr', 1.0),
                            'genes': term.get('genes', [])
                        })
                processed_enrichment[category] = processed_terms
        
        # Count significant terms (handle string FDR values)
        significant_count = 0
        for terms in processed_enrichment.values():
            for t in terms:
                try:
                    fdr = float(t.get('fdr', 1.0)) if t.get('fdr') else 1.0
                    if fdr < 0.05:
                        significant_count += 1
                except (ValueError, TypeError):
                    pass
        
        return {
            'enrichment': processed_enrichment,
            'total_genes': len(genes),
            'significant_terms': significant_count
        }
    
    def _is_cancer_query(self, query: str) -> bool:
        """Check if query is cancer-related."""
        cancer_keywords = ['cancer', 'tumor', 'carcinoma', 'sarcoma', 'lymphoma', 'leukemia']
        return any(keyword in query.lower() for keyword in cancer_keywords)

    def _expand_disease_query(self, disease_query: str) -> List[str]:
        """
        Generate expanded query terms for comprehensive multi-term search.

        For cancer queries: Adds parent category ("cancer") + medical synonyms.
        For non-cancer: Currently returns original query (BRITE hierarchy future enhancement).

        Args:
            disease_query: Original disease query (e.g., "breast cancer")

        Returns:
            List of search terms ordered by relevance
        """
        search_terms = [disease_query]  # Original query first (highest relevance)

        query_lower = disease_query.lower()

        # Cancer-specific expansion
        if self._is_cancer_query(disease_query):
            # Extract disease type: "breast cancer" → "breast"
            for cancer_keyword in ['cancer', 'tumor', 'carcinoma', 'sarcoma', 'lymphoma', 'leukemia']:
                if cancer_keyword in query_lower:
                    disease_type = query_lower.replace(cancer_keyword, '').strip()

                    if disease_type:  # Only add if we extracted a type
                        # Add medical synonyms
                        search_terms.extend([
                            f"{disease_type} neoplasm",   # Medical synonym
                            f"{disease_type} carcinoma",  # Specific cancer type
                            f"{disease_type} tumor",      # Alternative term
                        ])

                    # Add parent category for broader pathway discovery
                    if cancer_keyword == 'cancer':
                        search_terms.append("cancer")  # Broadest term last

                    break  # Only process first match

        # Future enhancement: For non-cancer diseases, use KEGG BRITE hierarchies
        # to discover parent/child categories dynamically

        logger.info(f"Expanded '{disease_query}' to {len(search_terms)} search terms: {search_terms}")
        return search_terms

    def _deduplicate_pathways(self, pathways: List[Dict]) -> List[Dict]:
        """
        Deduplicate pathways by ID, keeping first occurrence.

        Args:
            pathways: List of pathway dicts with 'id' field

        Returns:
            Deduplicated list of pathways
        """
        seen_ids = set()
        unique_pathways = []

        for pathway in pathways:
            pathway_id = pathway.get('id') or pathway.get('stId')  # Reactome uses 'stId'

            if pathway_id and pathway_id not in seen_ids:
                seen_ids.add(pathway_id)
                unique_pathways.append(pathway)

        logger.debug(f"Deduplicated {len(pathways)} pathways → {len(unique_pathways)} unique")
        return unique_pathways

    def _rank_pathways_by_relevance(self, pathways: List[Dict], original_query: str) -> List[Dict]:
        """
        Score and rank pathways by relevance to original query.

        Scoring:
        - Query term in pathway name: +10 points
        - Query term in pathway description: +5 points
        - Reactome source: +2 points (more detailed curation)

        Args:
            pathways: List of pathway dicts
            original_query: Original disease query for relevance scoring

        Returns:
            Pathways sorted by relevance score (descending)
        """
        query_terms = original_query.lower().split()
        scored_pathways = []

        for pathway in pathways:
            name = (pathway.get('name') or pathway.get('displayName') or '').lower()
            description = pathway.get('description', '').lower()
            source = pathway.get('source_db', '').lower()

            # Calculate relevance score
            score = 0
            for term in query_terms:
                if term in name:
                    score += 10
                if term in description:
                    score += 5

            # Bonus for Reactome (more detailed curation than KEGG)
            if 'reactome' in source or pathway.get('stId'):  # Reactome uses stId
                score += 2

            scored_pathways.append((score, pathway))

        # Sort by score (descending) and return pathways only
        scored_pathways.sort(key=lambda x: x[0], reverse=True)

        # Log top 5 pathways for debugging
        if scored_pathways:
            logger.debug(f"Top 5 ranked pathways:")
            for i, (score, pathway) in enumerate(scored_pathways[:5]):
                name = pathway.get('name') or pathway.get('displayName') or 'Unknown'
                logger.debug(f"  {i+1}. {name} (score: {score})")

        return [pathway for score, pathway in scored_pathways]

    def _is_cancer_gene(self, gene: str) -> bool:
        """Check if gene is cancer-related - removed hardcoded list for scientific validity."""
        # This method should be replaced with dynamic cancer gene detection
        # For now, return False to avoid hardcoded assumptions
        return False
    
    def _get_pathway_context_weight(self, source: str, target: str) -> float:
        """Get pathway context weight for edge."""
        # Simplified pathway context weighting
        # In practice, this would use actual pathway data
        return 1.0  # Default weight
    
    def _get_node_pathways(self, node_id: str) -> List[str]:
        """
        Get pathways for a node using gene-to-pathway mapping.
        
        STEP 2 FIX: Returns actual pathway memberships from mapping.
        """
        pathway_infos = self.gene_to_pathways.get(node_id, [])
        return [pw['id'] for pw in pathway_infos]
    
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
        """
        Get pathway context for an edge (shared pathways between nodes).
        
        STEP 2 FIX: Returns comma-separated list of shared pathway IDs.
        """
        source_pathways = set(pw['id'] for pw in self.gene_to_pathways.get(source, []))
        target_pathways = set(pw['id'] for pw in self.gene_to_pathways.get(target, []))
        shared = source_pathways & target_pathways
        
        if shared:
            return ','.join(sorted(list(shared))[:3])  # Return top 3 shared pathways
        return None
    
    def _calculate_validation_score(
        self,
        disease_data: Dict,
        pathway_data: Dict,
        network_data: Dict,
        expression_data: Dict,
        data_sources: Dict
    ) -> float:
        """Calculate overall validation score."""
        scores = {}

        # Disease confidence
        if disease_data.get('primary_disease'):
            scores['disease_confidence'] = disease_data['primary_disease'].confidence

        # Pathway coverage
        scores['pathway_coverage'] = pathway_data.get('coverage', 0.0)

        # Expression coverage
        scores['expression_coverage'] = expression_data.get('coverage', 0.0)

        # Cross-database concordance
        scores['cross_database_concordance'] = disease_data.get('concordance', 0.0)

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

    def _track_data_source(
        self,
        data_sources: Dict,
        source_name: str,
        success: bool = True,
        error_type: str = None
    ):
        """Helper method to track data source queries.

        Args:
            data_sources: Dictionary of DataSourceStatus objects
            source_name: Name of the data source (kegg, reactome, string, hpa, uniprot, chembl)
            success: Whether the query was successful
            error_type: Type of error if unsuccessful
        """
        if source_name in data_sources:
            status = data_sources[source_name]
            status.requested += 1
            if success:
                status.successful += 1
            else:
                status.failed += 1
                if error_type and error_type not in status.error_types:
                    status.error_types.append(error_type)

            if status.requested > 0:
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
        return None

    async def _call_with_tracking(
        self,
        data_sources: Optional[Dict],
        source_name: str,
        coro,
        suppress_exception: bool = False,
    ):
        """
        Await an MCP coroutine and automatically track request outcomes.

        Args:
            data_sources: Tracking dictionary (falls back to current execution context)
            source_name: Which source (kegg, reactome, string, hpa, uniprot, chembl)
            coro: Coroutine representing the MCP call
            suppress_exception: If True, swallow exceptions after tracking
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

    def _generate_disease_association_terms(
        self,
        disease_query: str,
        primary_disease: Optional[Disease]
    ) -> List[str]:
        """
        Generate disease terms for association database queries.

        Combines:
        - Original query
        - Medical synonyms from query expansion
        - Disease names from KEGG/Reactome

        Args:
            disease_query: Original disease query
            primary_disease: Primary disease from Phase 1 (optional)

        Returns:
            List of unique disease terms
        """
        terms = set()

        # Original query
        terms.add(disease_query)

        # Expand query terms (reuse existing method)
        expanded = self._expand_disease_query(disease_query)
        terms.update(expanded)

        # Add disease names from KEGG/Reactome
        if primary_disease:
            terms.add(primary_disease.name)
            # Extract base disease name (remove codes/IDs)
            base_name = self._extract_base_disease_name(primary_disease.name)
            if base_name:
                terms.add(base_name)

        # Remove too-generic terms (single words like "cancer" or "tumor")
        terms = {t for t in terms if len(t.split()) >= 2 or t.lower() not in ['cancer', 'tumor', 'neoplasm']}

        # Limit to max terms
        terms_list = list(terms)[:self.ASSOCIATION_MAX_TERMS]

        logger.debug(f"Generated {len(terms_list)} association search terms from '{disease_query}'")
        return terms_list

    def _extract_base_disease_name(self, disease_name: str) -> Optional[str]:
        """
        Extract base disease name by removing KEGG/Reactome codes.

        Examples:
            "Breast cancer (hsa:05224)" → "Breast cancer"
            "H00013:Breast cancer" → "Breast cancer"

        Args:
            disease_name: Disease name with potential codes

        Returns:
            Clean disease name or None
        """
        import re

        # Remove KEGG codes: (hsa:xxxxx) or H00xxx:
        cleaned = re.sub(r'\([^)]*\)', '', disease_name)  # Remove parentheses
        cleaned = re.sub(r'[HD]\d+:', '', cleaned)  # Remove KEGG disease codes
        cleaned = cleaned.strip()

        return cleaned if cleaned and len(cleaned) > 3 else None

    def _extract_gene_symbol_from_uniprot(self, protein_info: Dict[str, Any]) -> Optional[str]:
        """
        Extract gene symbol from UniProt protein info.

        UniProt structure:
        - genes: [{geneName: {value: "BRCA1"}, ...}]
        - proteinDescription: {recommendedName: {fullName: {value: "..."}}}

        Args:
            protein_info: UniProt protein information dict

        Returns:
            Gene symbol (uppercase) or None
        """
        try:
            # Extract from genes array (primary method)
            genes = protein_info.get('genes', [])
            if genes and isinstance(genes, list):
                first_gene = genes[0]
                if isinstance(first_gene, dict):
                    gene_name = first_gene.get('geneName', {}).get('value')
                    if gene_name:
                        return gene_name.upper()

            # Fallback: check for gene field at root level
            if 'gene' in protein_info:
                gene_data = protein_info['gene']
                if isinstance(gene_data, str):
                    return gene_data.upper()
                elif isinstance(gene_data, dict):
                    gene_name = gene_data.get('name') or gene_data.get('value')
                    if gene_name:
                        return gene_name.upper()

        except Exception as e:
            logger.debug(f"Failed to extract gene symbol: {e}")

        return None

    def _calculate_association_confidence(
        self,
        protein_info: Dict[str, Any],
        disease_query: str
    ) -> float:
        """
        Calculate confidence score for disease-gene association.

        Factors:
        - Disease mention in function/description: +0.4
        - Disease in name: +0.3
        - Clinical significance annotation: +0.2
        - Reviewed/Swiss-Prot status: +0.1

        Args:
            protein_info: UniProt protein information
            disease_query: Original disease query

        Returns:
            Confidence score (0.0-1.0)
        """
        confidence = 0.5  # Base confidence for presence in UniProt

        disease_terms = disease_query.lower().split()

        # Check disease mention in function
        function = str(protein_info.get('function', '')).lower()
        comments = protein_info.get('comments', [])
        if isinstance(comments, list):
            for comment in comments:
                if isinstance(comment, dict) and comment.get('commentType') == 'FUNCTION':
                    function += ' ' + str(comment.get('text', {}).get('value', '')).lower()

        if any(term in function for term in disease_terms):
            confidence += 0.4

        # Check disease in protein name
        protein_name = ''
        try:
            protein_desc = protein_info.get('proteinDescription', {})
            if isinstance(protein_desc, dict):
                rec_name = protein_desc.get('recommendedName', {})
                if isinstance(rec_name, dict):
                    full_name = rec_name.get('fullName', {})
                    if isinstance(full_name, dict):
                        protein_name = str(full_name.get('value', '')).lower()
        except:
            pass

        if any(term in protein_name for term in disease_terms):
            confidence += 0.3

        # Check for disease-specific comments
        if isinstance(comments, list):
            for comment in comments:
                if isinstance(comment, dict):
                    comment_type = comment.get('commentType', '')
                    if comment_type in ['DISEASE', 'INVOLVEMENT IN DISEASE']:
                        confidence += 0.2
                        break

        # Check reviewed status (Swiss-Prot = reviewed)
        if protein_info.get('reviewed') or protein_info.get('entryType') == 'Swiss-Prot':
            confidence += 0.1

        return min(confidence, 1.0)  # Cap at 1.0
    async def _phase1p5_disease_gene_associations(
        self,
        disease_query: str,
        primary_disease: Optional[Disease],
        data_sources: Dict
    ) -> Dict[str, Any]:
        """
        Phase 1.5: Disease-Gene Association Enrichment.

        Query disease-gene association databases to supplement pathway-based discovery.
        Provides disease-agnostic gene coverage independent of pathway search results.

        Strategy:
        1. Generate disease search terms from query + disease names
        2. Query UniProt for disease-associated proteins
        3. Extract gene symbols with confidence scoring
        4. Validate and filter by confidence threshold
        5. Track provenance for merge in Phase 2

        Args:
            disease_query: Original disease query (e.g., "breast cancer")
            primary_disease: Primary disease object from Phase 1
            data_sources: Data source tracking dict

        Returns:
            Dict with:
            - association_genes: List[str] - Gene symbols from associations
            - gene_confidence: Dict[str, float] - Confidence scores per gene
            - source: str - Data source used ("uniprot")
            - total_associations: int - Total associations found
            - filtered_associations: int - Associations passing validation
        """
        logger.info("Phase 1.5: Disease-gene association enrichment")

        # Check if Phase 1.5 is enabled
        if not self.PHASE1P5_ENABLED:
            logger.info("Phase 1.5 disabled by configuration")
            return {
                'association_genes': [],
                'gene_confidence': {},
                'source': None,
                'total_associations': 0,
                'filtered_associations': 0
            }

        # Check if UniProt is available
        if not self.mcp_manager.uniprot:
            logger.info("Phase 1.5: UniProt not available, skipping association enrichment")
            return {
                'association_genes': [],
                'gene_confidence': {},
                'source': None,
                'total_associations': 0,
                'filtered_associations': 0
            }

        # STEP 1: Generate disease search terms
        search_terms = self._generate_disease_association_terms(disease_query, primary_disease)
        logger.info(f"  Searching {len(search_terms)} disease terms in UniProt")

        # STEP 2: Query UniProt for disease-associated proteins
        all_proteins = {}  # accession -> protein_info

        for term in search_terms:
            try:
                # Query UniProt with disease term
                # UniProt supports full-text search with organism filter
                # Using reviewed:true to get Swiss-Prot (curated) entries only
                search_query = f'({term}) AND (organism_id:9606) AND (reviewed:true)'  # 9606 = human

                logger.debug(f"  Querying UniProt: {search_query}")

                result = await self._call_with_tracking(
                    data_sources,
                    'uniprot',
                    self.mcp_manager.uniprot.search_proteins(
                        query=search_query,
                        limit=self.ASSOCIATION_QUERY_LIMIT
                    ),
                    suppress_exception=True  # Don't fail if one term fails
                )

                if not result:
                    logger.debug(f"  No results for term '{term}'")
                    continue

                # Extract protein accessions from search results
                proteins = result.get('results', [])
                if isinstance(proteins, list):
                    for protein in proteins:
                        if isinstance(protein, dict):
                            accession = protein.get('primaryAccession') or protein.get('accession')
                            if accession and accession not in all_proteins:
                                all_proteins[accession] = protein

                    logger.debug(f"  Found {len(proteins)} proteins for '{term}' ({len(all_proteins)} total unique)")

            except Exception as e:
                logger.debug(f"  UniProt query failed for '{term}': {e}")
                continue

        if not all_proteins:
            logger.info("  No proteins found from UniProt disease search")
            return {
                'association_genes': [],
                'gene_confidence': {},
                'source': 'uniprot',
                'total_associations': 0,
                'filtered_associations': 0
            }

        logger.info(f"  Retrieved {len(all_proteins)} unique proteins from UniProt")

        # STEP 3: Extract gene symbols and calculate confidence
        association_genes = {}  # gene_symbol -> confidence
        proteins_processed = 0
        genes_extracted = 0

        for accession, protein_info in all_proteins.items():
            proteins_processed += 1

            try:
                # Extract gene symbol
                gene_symbol = self._extract_gene_symbol_from_uniprot(protein_info)

                if not gene_symbol:
                    logger.debug(f"  Could not extract gene symbol from {accession}")
                    continue

                # Calculate confidence score
                confidence = self._calculate_association_confidence(protein_info, disease_query)

                # Keep highest confidence for each gene
                if gene_symbol in association_genes:
                    association_genes[gene_symbol] = max(association_genes[gene_symbol], confidence)
                else:
                    association_genes[gene_symbol] = confidence
                    genes_extracted += 1

            except Exception as e:
                logger.debug(f"  Failed to process protein {accession}: {e}")
                continue

        logger.info(f"  Extracted {genes_extracted} unique genes from {proteins_processed} proteins")

        # STEP 4: Filter by confidence threshold
        threshold = self.ASSOCIATION_CONFIDENCE_THRESHOLD
        filtered_genes = {
            gene: conf for gene, conf in association_genes.items()
            if conf >= threshold
        }

        logger.info(f"  Confidence filtering: {len(filtered_genes)}/{len(association_genes)} genes >= {threshold}")

        # STEP 5: Validate gene symbols (basic validation)
        validated_genes = []
        for gene_symbol in filtered_genes.keys():
            # Basic validation: 1-20 chars, alphanumeric + hyphen
            if 1 <= len(gene_symbol) <= 20 and gene_symbol.replace('-', '').replace('_', '').isalnum():
                validated_genes.append(gene_symbol)
            else:
                logger.debug(f"  Filtered out invalid gene symbol: {gene_symbol}")

        logger.info(f"✅ Phase 1.5 complete: {len(validated_genes)} association genes")

        return {
            'association_genes': validated_genes,
            'gene_confidence': {g: filtered_genes[g] for g in validated_genes},
            'source': 'uniprot',
            'total_associations': len(all_proteins),
            'filtered_associations': len(validated_genes)
        }
    def _validate_gene_coverage(
        self,
        pathway_genes: List[str],
        association_genes: List[str],
        merged_genes: List[str],
        gene_provenance: Dict[str, Dict],
        disease_query: str
    ) -> Dict[str, Any]:
        """
        Validate gene coverage and identify gaps.

        Analyzes overlap between pathway-discovered and association-discovered genes
        to assess comprehensiveness of gene discovery.

        Args:
            pathway_genes: Genes discovered from pathways
            association_genes: Genes discovered from associations
            merged_genes: Combined deduplicated genes
            gene_provenance: Gene provenance tracking dict
            disease_query: Original disease query

        Returns:
            Validation metrics dict with coverage statistics
        """
        pathway_set = set(pathway_genes)
        association_set = set(association_genes)
        merged_set = set(merged_genes)

        # Calculate overlap
        overlap = pathway_set & association_set
        pathway_only = pathway_set - association_set
        association_only = association_set - pathway_set

        # Count high-confidence genes (confidence >= 0.7)
        high_confidence = [
            g for g in merged_genes
            if gene_provenance.get(g, {}).get('confidence', 0.0) >= 0.7
        ]

        # Count genes by source
        pathway_source_count = sum(
            1 for g in merged_genes
            if gene_provenance.get(g, {}).get('source') == 'pathway'
        )
        association_source_count = sum(
            1 for g in merged_genes
            if gene_provenance.get(g, {}).get('source') == 'association'
        )
        both_source_count = sum(
            1 for g in merged_genes
            if gene_provenance.get(g, {}).get('source') == 'both'
        )

        # Calculate coverage rates
        pathway_coverage = len(pathway_set) / len(merged_set) if merged_set else 0.0
        association_coverage = len(association_set) / len(merged_set) if merged_set else 0.0
        overlap_rate = len(overlap) / len(pathway_set) if pathway_set else 0.0

        return {
            'pathway_coverage': pathway_coverage,
            'association_coverage': association_coverage,
            'overlap_count': len(overlap),
            'overlap_rate': overlap_rate,
            'pathway_only_count': len(pathway_only),
            'association_only_count': len(association_only),
            'high_confidence_count': len(high_confidence),
            'pathway_source_count': pathway_source_count,
            'association_source_count': association_source_count,
            'both_source_count': both_source_count,
            'missing_count': 0,  # Placeholder for future enhancement
            'merged_total': len(merged_set),
            'pathway_total': len(pathway_set),
            'association_total': len(association_set)
        }
