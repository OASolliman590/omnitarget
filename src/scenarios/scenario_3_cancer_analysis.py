"""
Scenario 3: Cancer-Specific Analysis

Cancer marker discovery and prognostic analysis with target prioritization.
Based on Mature_development_plan.md Phase 1-5.
"""

import asyncio
import logging
import os
from typing import Dict, List, Optional, Any, Set
import networkx as nx
import numpy as np
from collections import defaultdict

from ..core.mcp_client_manager import MCPClientManager
from ..core.data_standardizer import DataStandardizer
from ..core.validation import DataValidator
from ..core.simulation.simple_simulator import SimplePerturbationSimulator
from ..core.string_network_builder import AdaptiveStringNetworkBuilder
from ..utils.hpa_parsing import parse_pathology_data, _iter_expr_items, get_gene_expression, categorize_expression
from ..utils.scenario_metrics import (
    summarize_expression_profiles,
    summarize_markers,
    summarize_network,
)
from ..models.data_models import (
    CancerMarker, Pathway, Protein, Interaction, ExpressionProfile, 
    NetworkNode, NetworkEdge, PrioritizedTarget, CancerAnalysisResult,
    DataSourceStatus, CompletenessMetrics
)

logger = logging.getLogger(__name__)


class CancerAnalysisScenario:
    """
    Scenario 3: Cancer-Specific Analysis

    5-phase workflow:
    1. Cancer marker discovery (HPA prognostic markers)
    2. Cancer pathway discovery (KEGG + Reactome)
    3. Cancer network construction (STRING with marker weighting)
    4. Expression dysregulation (HPA cancer vs. normal)
    5. Target prioritization (multi-criteria scoring)
    """

    def __init__(self, mcp_manager: MCPClientManager):
        """Initialize cancer analysis scenario."""
        self.mcp_manager = mcp_manager
        self.standardizer = DataStandardizer()
        self.validator = DataValidator()
        self._active_data_sources: Optional[Dict[str, Any]] = None
        
        # Initialize adaptive STRING network builder
        self.string_builder = None  # Will be initialized when data_sources available
        
        # Cancer type synonyms for HPA API compatibility
        self.CANCER_TYPE_SYNONYMS = {
            'lung cancer': ['lung cancer', 'Lung cancer', 'lung', 'NSCLC', 'SCLC', 'lung adenocarcinoma'],
            'breast cancer': ['breast cancer', 'Breast cancer', 'breast', 'BRCA'],
            'colon cancer': ['colon cancer', 'Colon cancer', 'colorectal cancer', 'CRC', 'colorectal'],
            'prostate cancer': ['prostate cancer', 'Prostate cancer', 'prostate', 'PRAD'],
            'liver cancer': ['liver cancer', 'Liver cancer', 'liver', 'HCC', 'hepatocellular carcinoma'],
            'pancreatic cancer': ['pancreatic cancer', 'Pancreatic cancer', 'pancreas', 'PDAC'],
            'stomach cancer': ['stomach cancer', 'Stomach cancer', 'gastric cancer', 'gastric'],
            'ovarian cancer': ['ovarian cancer', 'Ovarian cancer', 'ovary', 'ovarian'],
            'kidney cancer': ['kidney cancer', 'Kidney cancer', 'renal cancer', 'RCC', 'renal'],
            'melanoma': ['melanoma', 'Melanoma', 'skin cancer', 'SKCM'],
            'glioma': ['glioma', 'Glioma', 'brain cancer', 'GBM', 'glioblastoma'],
            'thyroid cancer': ['thyroid cancer', 'Thyroid cancer', 'thyroid', 'THCA'],
            'cervical cancer': ['cervical cancer', 'Cervical cancer', 'cervix', 'CESC'],
            'urothelial cancer': ['urothelial cancer', 'Urothelial cancer', 'bladder cancer', 'BLCA'],
            'endometrial cancer': ['endometrial cancer', 'Endometrial cancer', 'uterine cancer', 'UCEC']
        }
    
    def _get_cancer_type_synonyms(self, cancer_type: str) -> List[str]:
        """
        Get synonyms for a cancer type to improve HPA search success.
        
        Args:
            cancer_type: User-provided cancer type
            
        Returns:
            List of synonyms to try (including original)
        """
        cancer_lower = cancer_type.lower().strip()
        
        # Check direct match
        if cancer_lower in self.CANCER_TYPE_SYNONYMS:
            return self.CANCER_TYPE_SYNONYMS[cancer_lower]
        
        # Check partial match
        for key, synonyms in self.CANCER_TYPE_SYNONYMS.items():
            if key in cancer_lower or cancer_lower in key:
                return synonyms
        
        # Return variations of the original
        return [
            cancer_type,
            cancer_type.title(),
            cancer_type.lower(),
            cancer_type.split()[0] if ' ' in cancer_type else cancer_type
        ]

    def _validate_gene_symbol(self, gene: str) -> tuple[bool, str]:
        """
        Validate and normalize gene symbol for HPA.

        Returns:
            (is_valid, normalized_gene)
        """
        if not gene:
            return False, ""

        # Normalize: uppercase, remove extra spaces
        normalized = gene.strip().upper()

        # Check basic format (allow underscores and alphanumeric)
        if not normalized.replace('_', '').isalnum():
            return False, normalized

        # Check length (typical gene symbols are 2-15 chars)
        if len(normalized) < 2 or len(normalized) > 15:
            return False, normalized

        return True, normalized

    async def _get_pathology_with_fallback(self, gene: str):
        """
        Get pathology data with validation and TCGA fallback.

        Args:
            gene: Gene symbol to query

        Returns:
            Pathology data dict or None if all methods fail
        """
        # Step 1: Validate gene format and filter metabolites
        is_valid, normalized_gene = self._validate_gene_symbol(gene)
        if not is_valid:
            logger.debug(f"Invalid gene format, skipping: {gene}")
            return None

        # Step 1b: Check if gene is a metabolite (use existing filter)
        if not self._is_valid_gene_symbol_s3(normalized_gene):
            logger.debug(f"Gene rejected by metabolite filter: {normalized_gene}")
            return None

        logger.debug(f"Validated gene: {gene} -> {normalized_gene}")

        # Step 2: Try HPA pathology data with comprehensive fallback chain
        pathology_data = None
        data_source = None
        
        try:
            pathology_response = await self._call_with_tracking(
                None,
                'hpa',
                self.mcp_manager.hpa.get_pathology_data([normalized_gene])
            )

            # Check if response is valid dict (not error string)
            if isinstance(pathology_response, dict) and pathology_response:
                logger.debug(f"✅ HPA pathology data retrieved for {normalized_gene}")
                pathology_data = pathology_response
                data_source = "HPA_pathology"
            elif isinstance(pathology_response, str):
                logger.warning(f"HPA returned error for {normalized_gene}: {pathology_response}")
                # Fall through to expression fallback
        except Exception as e:
            if "Invalid gene arguments" in str(e) or "-32602" in str(e):
                logger.warning(f"⚠️  HPA pathology rejected gene {normalized_gene}: {e}")
                logger.info(f"Attempting expression data fallback for {normalized_gene}")
                
                # FALLBACK 1: Try HPA expression data as proxy for pathology
                try:
                    expression_response = await self._call_with_tracking(
                        None,
                        'hpa',
                        self.mcp_manager.hpa.get_tissue_expression(normalized_gene)
                    )
                    
                    if expression_response and isinstance(expression_response, dict):
                        # Convert expression data to pathology-like format
                        logger.info(f"✅ Using HPA expression data as proxy for {normalized_gene}")
                        pathology_data = {
                            "gene": normalized_gene,
                            "expression_proxy": expression_response,
                            "source": "HPA_expression_fallback",
                            "note": "Pathology data unavailable, using expression as proxy"
                        }
                        data_source = "HPA_expression_fallback"
                except Exception as expr_error:
                    logger.warning(f"HPA expression fallback also failed for {normalized_gene}: {expr_error}")
                    # Continue to TCGA fallback
            else:
                logger.error(f"Unexpected HPA error for {normalized_gene}: {e}")
                # Continue to TCGA fallback

        # Step 3: Try TCGA fallback (if HPA completely failed)
        if not pathology_data:
            try:
                if hasattr(self.mcp_manager, 'tcga'):
                    logger.info(f"Using TCGA fallback for {normalized_gene}")
                    tcga_data = await self._call_with_tracking(
                        None,
                        'tcga',
                        self.mcp_manager.tcga.get_cancer_markers([normalized_gene])
                    )

                    if tcga_data and isinstance(tcga_data, dict):
                        # Convert TCGA format to HPA format for consistency
                        pathology_data = {
                            "gene": normalized_gene,
                            "markers": tcga_data.get("markers", []),
                            "source": "TCGA"
                        }
                        data_source = "TCGA"
                        logger.info(f"✅ TCGA data retrieved for {normalized_gene}")
                else:
                    logger.debug(f"TCGA client not available, skipping fallback for {normalized_gene}")
            except Exception as e:
                logger.warning(f"TCGA fallback failed for {normalized_gene}: {e}")

        # Step 4: If all data sources failed, keep gene with unavailable flag
        # This ensures critical markers (ERBB2, EGFR, etc.) are never lost
        if not pathology_data:
            logger.warning(f"⚠️  No pathology/expression data available for {normalized_gene}")
            logger.info(f"Keeping gene {normalized_gene} with 'data_unavailable' flag")
            pathology_data = {
                "gene": normalized_gene,
                "source": "none",
                "data_available": False,
                "note": "Gene validated but HPA/TCGA data unavailable"
            }
            data_source = "unavailable"
        
        # Log the final data source for tracking
        if data_source:
            logger.debug(f"Final data source for {normalized_gene}: {data_source}")
        
        return pathology_data
    
    async def execute(
        self,
        cancer_type: str,
        tissue_context: str
    ) -> CancerAnalysisResult:
        """
        Execute complete cancer-specific analysis workflow.
        
        Args:
            cancer_type: Type of cancer (e.g., "breast cancer", "lung adenocarcinoma")
            tissue_context: Normal tissue context (e.g., "breast", "lung")
            
        Returns:
            CancerAnalysisResult with complete analysis
        """
        logger.info(f"Starting cancer analysis for: {cancer_type} in {tissue_context}")
        
        data_sources = {
            'kegg': DataSourceStatus(source_name='kegg', requested=0, successful=0, failed=0, success_rate=0.0, error_types=[]),
            'reactome': DataSourceStatus(source_name='reactome', requested=0, successful=0, failed=0, success_rate=0.0, error_types=[]),
            'string': DataSourceStatus(source_name='string', requested=0, successful=0, failed=0, success_rate=0.0, error_types=[]),
            'hpa': DataSourceStatus(source_name='hpa', requested=0, successful=0, failed=0, success_rate=0.0, error_types=[]),
            'uniprot': DataSourceStatus(source_name='uniprot', requested=0, successful=0, failed=0, success_rate=0.0, error_types=[])
        }
        if self.mcp_manager.chembl:
            data_sources['chembl'] = DataSourceStatus(source_name='chembl', requested=0, successful=0, failed=0, success_rate=0.0, error_types=[])
        if getattr(self.mcp_manager, 'tcga', None):
            data_sources['tcga'] = DataSourceStatus(source_name='tcga', requested=0, successful=0, failed=0, success_rate=0.0, error_types=[])

        self._active_data_sources = data_sources
        try:
            # Phase 1: Cancer marker discovery
            marker_data = await self._phase1_cancer_marker_discovery(cancer_type, data_sources)
            
            # Phase 2: Cancer pathway discovery
            pathway_data = await self._phase2_cancer_pathway_discovery(cancer_type, data_sources)
            
            # Phase 3: Cancer network construction
            network_data = await self._phase3_cancer_network_construction(
                marker_data['markers'],
                pathway_data['pathways'],
                data_sources
            )
            
            # Phase 4: Expression dysregulation
            expression_data = await self._phase4_expression_dysregulation(
                network_data['genes'],
                tissue_context,
                data_sources
            )
            
            # Phase 5: Target prioritization
            prioritization_data = await self._phase5_target_prioritization(
                network_data['network'],
                marker_data['markers'],
                expression_data['profiles']
            )
        finally:
            self._active_data_sources = None
        
        completeness_metrics = self._build_completeness_metrics(
            marker_data,
            pathway_data,
            network_data,
            expression_data
        )

        # Calculate validation score
        validation_score = self._calculate_validation_score(
            marker_data, pathway_data, network_data, expression_data, prioritization_data,
            data_sources, completeness_metrics
        )

        network_summary = summarize_network(network_data.get('network'))
        expression_summary = summarize_expression_profiles(
            expression_data.get('profiles', []),
            total_gene_universe=len(network_data.get('genes', []))
        )
        marker_summary = summarize_markers(marker_data.get('markers', []))
        
        # Build result - convert Pydantic objects to dictionaries
        result = CancerAnalysisResult(
            cancer_type=cancer_type,
            prognostic_markers=[marker.model_dump() if hasattr(marker, 'model_dump') else marker for marker in marker_data.get('markers', [])],
            cancer_pathways=[pathway.model_dump() if hasattr(pathway, 'model_dump') else pathway for pathway in pathway_data.get('pathways', [])],
            network_nodes=[node.model_dump() if hasattr(node, 'model_dump') else node for node in network_data.get('nodes', [])],
            network_edges=[edge.model_dump() if hasattr(edge, 'model_dump') else edge for edge in network_data.get('edges', [])],
            expression_dysregulation=expression_data.get('concordance', {}),
            prioritized_targets=[target.model_dump() if hasattr(target, 'model_dump') else target for target in prioritization_data.get('targets', [])],
            combination_opportunities=prioritization_data.get('combination_opportunities', []),
            validation_score=validation_score,
            data_sources=list(data_sources.values()),
            completeness_metrics=completeness_metrics,
            network_summary=network_summary,
            expression_summary=expression_summary,
            marker_summary=marker_summary
        )
        
        logger.info(f"Cancer analysis completed. Validation score: {validation_score:.3f}")
        return result
    
    async def _phase1_cancer_marker_discovery(self, cancer_type: str, data_sources: Dict[str, DataSourceStatus]) -> Dict[str, Any]:
        """
        Phase 1: Dynamic cancer marker discovery from HPA.
        
        Get prognostic markers from HPA pathology data using get_pathology_data per gene.
        """
        logger.info("Phase 1: Dynamic cancer marker discovery")
        
        try:
            # Step 1: Get initial cancer genes from pathway search
            # Quick pathway search to get seed genes
            try:
                reactome_search = await self._call_with_tracking(
                    data_sources,
                    'reactome',
                    self.mcp_manager.reactome.find_pathways_by_disease(cancer_type)
                )
                seed_genes = set()
                
                if reactome_search.get('pathways'):
                    # OPTIMIZATION: Limit pathways and use parallel processing
                    pathway_list = reactome_search['pathways'][:3]  # Reduce to 3 pathways for speed
                    
                    async def extract_genes_from_pathway(pathway_data):
                        """Extract genes from a single pathway."""
                        pathway_id = pathway_data.get('stId') or pathway_data.get('id')
                        if not pathway_id:
                            return set()
                        
                        genes = set()
                        try:
                            # Use get_pathway_participants for better gene extraction (faster than get_pathway_details)
                            participants = await self._call_with_tracking(
                                data_sources,
                                'reactome',
                                self.mcp_manager.reactome.get_pathway_participants(pathway_id)
                            )
                            # Handle multiple possible response structures
                            participant_list = []
                            if isinstance(participants, dict):
                                participant_list = (participants.get('participants') or 
                                                   participants.get('entities') or
                                                   participants.get('proteins') or
                                                   [])
                            elif isinstance(participants, list):
                                participant_list = participants
                            
                            if participant_list:
                                for participant in participant_list:
                                    if isinstance(participant, dict):
                                        # Filter out pathways - only process proteins/genes
                                        participant_type = participant.get('type', '').lower()
                                        if participant_type in ['pathway', 'reaction', 'event']:
                                            # Skip pathways and reactions - these are not genes
                                            continue

                                        gene = (participant.get('gene_symbol') or 
                                               participant.get('gene') or 
                                               participant.get('displayName') or 
                                               participant.get('geneName') or
                                               participant.get('name', ''))
                                        
                                        # NEW: Extract from nested referenceEntity
                                        if not gene and 'referenceEntity' in participant:
                                            ref_entity = participant['referenceEntity']
                                            if isinstance(ref_entity, dict):
                                                gene = (ref_entity.get('gene_symbol') or
                                                       ref_entity.get('gene') or
                                                       ref_entity.get('displayName') or
                                                       ref_entity.get('name', ''))
                                        
                                        # NEW: Extract from components (for complexes)
                                        if 'components' in participant and isinstance(participant['components'], list):
                                            for component in participant['components']:
                                                if isinstance(component, dict):
                                                    comp_gene = (component.get('gene_symbol') or
                                                               component.get('gene') or
                                                               component.get('name', ''))
                                                    if comp_gene and comp_gene.strip():
                                                        # Enhanced validation
                                                        comp_gene_clean = comp_gene.strip().upper()
                                                        comp_gene_clean = comp_gene_clean.split(':')[0].split('-')[0]
                                                        if comp_gene_clean.isalnum() and 2 <= len(comp_gene_clean) <= 15:
                                                            genes.add(comp_gene_clean)
                                        
                                        # NEW: Extract from hasComponent (alternative structure)
                                        if 'hasComponent' in participant:
                                            components = participant['hasComponent']
                                            if isinstance(components, list):
                                                for component in components:
                                                    if isinstance(component, dict):
                                                        comp_gene = (component.get('gene_symbol') or
                                                                   component.get('gene') or
                                                                   component.get('name', ''))
                                                        if comp_gene and comp_gene.strip():
                                                            comp_gene_clean = comp_gene.strip().upper()
                                                            comp_gene_clean = comp_gene_clean.split(':')[0].split('-')[0]
                                                            if comp_gene_clean.isalnum() and 2 <= len(comp_gene_clean) <= 15:
                                                                genes.add(comp_gene_clean)
                                        
                                        # Validate and add main gene
                                        if gene and gene.strip():
                                            # Enhanced validation
                                            gene_clean = gene.strip().upper()
                                            # Remove common suffixes/prefixes
                                            gene_clean = gene_clean.split(':')[0].split('-')[0]
                                            if gene_clean.isalnum() and 2 <= len(gene_clean) <= 15:
                                                genes.add(gene_clean)
                                    elif isinstance(participant, str):
                                        # Direct gene symbol
                                        if participant.strip() and participant.upper().isalnum() and 2 <= len(participant) <= 15:
                                            genes.add(participant.strip().upper())
                            
                            # FALLBACK: If no genes found from participants, try get_pathway_details
                            # Participants might be high-level (pathways, reactions) rather than proteins
                            if not genes:
                                logger.info(f"[S3] No genes from participants for {pathway_id}, trying get_pathway_details fallback")
                                try:
                                    details = await self._call_with_tracking(
                                        data_sources,
                                        'reactome',
                                        self.mcp_manager.reactome.get_pathway_details(pathway_id)
                                    )
                                    
                                    if not details:
                                        logger.info(f"[S3] get_pathway_details returned empty for {pathway_id}")
                                    else:
                                        # Extract from entities (proteins, complexes, small molecules)
                                        entities = details.get('entities', [])
                                        if entities:
                                            logger.debug(f"[S3] Found {len(entities)} entities in get_pathway_details for {pathway_id}")
                                            for entity in entities:
                                                if isinstance(entity, dict):
                                                    entity_type = entity.get('type', '').lower()
                                                    # Only process proteins, not pathways/reactions
                                                    if entity_type not in ['pathway', 'reaction', 'event']:
                                                        entity_gene = (entity.get('gene_symbol') or
                                                                      entity.get('gene') or
                                                                      entity.get('displayName') or
                                                                      entity.get('name', ''))
                                                        
                                                        # Extract from referenceEntity (S1-style)
                                                        if not entity_gene and 'referenceEntity' in entity:
                                                            ref_entity = entity['referenceEntity']
                                                            if isinstance(ref_entity, dict):
                                                                entity_gene = (ref_entity.get('gene_symbol') or
                                                                              ref_entity.get('gene') or
                                                                              ref_entity.get('geneName') or
                                                                              ref_entity.get('name', ''))
                                                        
                                                        # Extract from components (S1-style)
                                                        if not entity_gene and 'components' in entity:
                                                            components = entity['components']
                                                            if isinstance(components, list):
                                                                for comp in components:
                                                                    if isinstance(comp, dict):
                                                                        comp_gene = (comp.get('gene_symbol') or
                                                                                    comp.get('gene') or
                                                                                    comp.get('name', ''))
                                                                        if comp_gene:
                                                                            entity_gene = comp_gene
                                                                            break
                                                        
                                                        if entity_gene and entity_gene.strip():
                                                            gene_clean = entity_gene.strip().upper()
                                                            gene_clean = gene_clean.split(':')[0].split('-')[0]
                                                            if gene_clean.isalnum() and 2 <= len(gene_clean) <= 15:
                                                                genes.add(gene_clean)
                                        else:
                                            logger.debug(f"[S3] No entities found in get_pathway_details for {pathway_id}")
                                        
                                        # Extract from participants (Reactome details format)
                                        participants = details.get('participants', [])
                                        if participants:
                                            logger.debug(f"[S3] Found {len(participants)} participants in get_pathway_details for {pathway_id}")
                                            for participant in participants:
                                                if isinstance(participant, dict):
                                                    # Check refEntities array (Reactome details format)
                                                    ref_entities = participant.get('refEntities', [])
                                                    if isinstance(ref_entities, list):
                                                        for ref_entity in ref_entities:
                                                            if isinstance(ref_entity, dict):
                                                                # Extract from displayName (e.g., "ERBB2" from "p-6Y-ERBB2 TMD/JMD mutants")
                                                                display_name = ref_entity.get('displayName', '')
                                                                if display_name:
                                                                    # Parse gene symbol from displayName
                                                                    # Examples: "ERBB2", "p-6Y-ERBB2", "p-EGFR"
                                                                    parts = display_name.split()
                                                                    for p in parts:
                                                                        # Remove prefixes like p-, p-6Y-, etc.
                                                                        clean = p.split(':')[0].split('-')[-1].split('(')[0].split(')')[0].strip()
                                                                        if clean and clean.isupper() and 2 <= len(clean) <= 15 and clean.isalnum():
                                                                            genes.add(clean)
                                                                            break
                                                    
                                                    # Also check displayName directly
                                                    display_name = participant.get('displayName', '')
                                                    if display_name:
                                                        # Try to extract gene from displayName
                                                        parts = display_name.split()
                                                        for p in parts:
                                                            clean = p.split(':')[0].split('-')[-1].split('(')[0].split(')')[0].strip()
                                                            if clean and clean.isupper() and 2 <= len(clean) <= 15 and clean.isalnum():
                                                                genes.add(clean)
                                                                break
                                        
                                        # Extract from hasEvent (reactions with participants) - legacy format
                                        has_event = details.get('hasEvent', [])
                                        if has_event:
                                            logger.debug(f"[S3] Found {len(has_event)} events in get_pathway_details for {pathway_id}")
                                            for event in has_event:
                                                if isinstance(event, dict) and event.get('participants'):
                                                    for participant in event['participants']:
                                                        if isinstance(participant, dict):
                                                            part_type = participant.get('type', '').lower()
                                                            if part_type not in ['pathway', 'reaction', 'event']:
                                                                part_gene = (participant.get('gene_symbol') or
                                                                          participant.get('gene') or
                                                                          participant.get('name', ''))
                                                                
                                                                # Extract from referenceEntity
                                                                if not part_gene and 'referenceEntity' in participant:
                                                                    ref_entity = participant['referenceEntity']
                                                                    if isinstance(ref_entity, dict):
                                                                        part_gene = (ref_entity.get('gene_symbol') or
                                                                                    ref_entity.get('gene') or
                                                                                    ref_entity.get('name', ''))
                                                                
                                                                if part_gene and part_gene.strip():
                                                                    gene_clean = part_gene.strip().upper()
                                                                    gene_clean = gene_clean.split(':')[0].split('-')[0]
                                                                    if gene_clean.isalnum() and 2 <= len(gene_clean) <= 15:
                                                                        genes.add(gene_clean)
                                        
                                        logger.info(f"[S3] get_pathway_details fallback extracted {len(genes)} genes for {pathway_id}")
                                except Exception as fallback_error:
                                    logger.info(f"[S3] Fallback to get_pathway_details failed for {pathway_id}: {fallback_error}")
                            
                        except Exception as e:
                            logger.debug(f"Failed to get pathway genes for {pathway_id}: {e}")
                        
                        return genes
                    
                    # Process pathways in parallel (all at once - only 3 pathways)
                    if pathway_list:
                        pathway_results = await asyncio.gather(*[extract_genes_from_pathway(p) for p in pathway_list], return_exceptions=True)
                        for gene_set in pathway_results:
                            if isinstance(gene_set, set):
                                seed_genes.update(gene_set)
                
                logger.info(f"Found {len(seed_genes)} seed genes from pathways")
            except Exception as e:
                logger.warning(f"Pathway search failed, proceeding without seed genes: {e}")
                seed_genes = set()
            
            # Step 2: Search for cancer prognostic markers using HPA search_cancer_markers
            # This is more efficient and reliable than per-gene pathology queries
            standardized_markers = []

            try:
                # Use search_cancer_markers with synonym fallback for better API compatibility
                logger.info(f"Searching for {cancer_type} prognostic markers using HPA search_cancer_markers")
                
                # Get synonyms for this cancer type
                synonyms = self._get_cancer_type_synonyms(cancer_type)
                logger.debug(f"Will try cancer type synonyms: {synonyms}")
                
                markers_response = None
                successful_term = None
                
                # Try each synonym until we get results
                for term in synonyms:
                    try:
                        response = await self._call_with_tracking(
                            data_sources,
                            'hpa',
                            self.mcp_manager.hpa.search_cancer_markers(term)
                        )
                        
                        # Check if we got results
                        if response:
                            if isinstance(response, list) and len(response) > 0:
                                markers_response = response
                                successful_term = term
                                logger.info(f"✅ HPA search_cancer_markers succeeded with term: '{term}'")
                                break
                            elif isinstance(response, dict):
                                markers_list = response.get('markers', response.get('results', []))
                                if len(markers_list) > 0:
                                    markers_response = response
                                    successful_term = term
                                    logger.info(f"✅ HPA search_cancer_markers succeeded with term: '{term}'")
                                    break
                        
                        logger.debug(f"No results for term '{term}', trying next synonym...")
                        
                    except Exception as term_error:
                        logger.debug(f"HPA search failed for '{term}': {term_error}")
                        continue
                
                if not successful_term:
                    logger.warning(f"HPA search_cancer_markers returned no results for all synonyms of '{cancer_type}'")

                # Parse response - could be list of markers or dict with 'markers' key
                markers_list = []
                if isinstance(markers_response, list):
                    markers_list = markers_response
                elif isinstance(markers_response, dict):
                    markers_list = markers_response.get('markers', markers_response.get('results', []))

                logger.info(f"HPA returned {len(markers_list)} cancer markers")

                # Convert each marker to CancerMarker object
                for marker_data in markers_list:
                    try:
                        # Extract fields from HPA response
                        gene = marker_data.get('gene', marker_data.get('Gene', ''))
                        if not gene:
                            continue

                        # Extract prognostic data
                        prognostic_summary = marker_data.get('prognostic_summary',
                                                            marker_data.get('Prognostic summary',
                                                            marker_data.get('best_prognosis',
                                                            marker_data.get('Best prognosis', 'unknown'))))

                        p_value = marker_data.get('p_value', marker_data.get('p-value'))
                        significance = marker_data.get('significance', marker_data.get('Significance', ''))

                        # Determine prognostic value - now supports 'variable' and 'unknown'
                        prognostic_value = 'favorable' if 'favorable' in str(prognostic_summary).lower() else \
                                         'unfavorable' if 'unfavorable' in str(prognostic_summary).lower() else \
                                         'variable' if 'variable' in str(prognostic_summary).lower() else 'unknown'

                        # Create CancerMarker with Pydantic validation
                        marker = CancerMarker(
                            gene=gene,
                            cancer_type=cancer_type,
                            prognostic_value=prognostic_value,
                            survival_association=significance if significance else str(prognostic_summary),
                            expression_pattern={
                                'tumor': marker_data.get('tumor_expression', 'unknown'),
                                'normal': marker_data.get('normal_expression', 'unknown')
                            },
                            clinical_relevance=str(prognostic_summary),
                            confidence=0.9 if p_value and float(p_value) < 0.05 else 0.7 if p_value else 0.6
                        )

                        # Validate confidence threshold
                        if self.validator.validate_cancer_marker_confidence(marker):
                            standardized_markers.append(marker)
                            logger.debug(f"Added cancer marker: {marker.gene} ({prognostic_value})")
                        else:
                            logger.debug(f"Marker {gene} excluded - confidence too low")

                    except Exception as e:
                        logger.warning(f"Failed to create CancerMarker from HPA data: {e}")
                        logger.debug(f"Marker data that failed: {marker_data}")
                        continue

            except Exception as e:
                logger.warning(f"HPA search_cancer_markers failed: {e}")
                logger.info("Continuing without HPA markers - will rely on known marker identification")
            
            # If no markers found from HPA, log warning (known marker identification may still add some)
            if not standardized_markers:
                logger.warning("No markers found from HPA search_cancer_markers - will try known marker identification")
            
            # Calculate marker validation rate
            validation_rate = self.validator.validate_cancer_marker_validation_rate(
                standardized_markers
            )

            logger.info(f"✅ Found {len(standardized_markers)} dynamic prognostic markers")

            # Define valid_seed_genes: genes from HPA markers if any, otherwise use seed_genes
            if standardized_markers:
                valid_seed_genes = {marker.gene for marker in standardized_markers}
            else:
                valid_seed_genes = seed_genes

            # NEW: Identify known breast cancer markers from discovered genes
            # This annotates discovered genes as markers (does NOT add new genes)
            if 'breast' in cancer_type.lower():
                logger.info("🔍 Identifying known breast cancer markers from discovered genes")

                # Collect all discovered genes (from seed genes that were validated)
                discovered_genes = list(valid_seed_genes)
                
                # Build expression and pathology data dicts for marker identification
                # (These may include fallback data sources)
                expression_data_dict = {}
                pathology_data_dict = {}
                
                # Note: We already have pathology data from the loop above
                # For now, we'll pass empty dicts and rely on fallback data in pathology responses
                # Future enhancement: collect expression data separately
                
                # Identify known markers from discovered genes
                known_markers = self._identify_known_breast_cancer_markers(
                    discovered_genes,
                    expression_data_dict,
                    pathology_data_dict
                )
                
                if known_markers:
                    logger.info(f"✅ Adding {len(known_markers)} known breast cancer markers to results")
                    # Merge with dynamic markers (avoid duplicates by gene)
                    existing_genes = {m.gene for m in standardized_markers}
                    for known_marker in known_markers:
                        if known_marker.gene not in existing_genes:
                            standardized_markers.append(known_marker)
                        else:
                            logger.debug(f"Known marker {known_marker.gene} already in dynamic markers, skipping")
                    
                    # Recalculate validation rate with known markers included
                    validation_rate = self.validator.validate_cancer_marker_validation_rate(
                        standardized_markers
                    )
                    logger.info(f"🎯 Total markers after adding known markers: {len(standardized_markers)}")
            
            return {
                'markers': standardized_markers,
                'validation_rate': validation_rate
            }
            
        except Exception as e:
            logger.error(f"❌ Dynamic cancer marker discovery failed: {e}")
            return {
                'markers': [],
                'validation_rate': 0.0
            }
    
    
    def _process_pathology_data(self, pathology_data: Dict[str, Any], cancer_type: str) -> List[CancerMarker]:
        """Process HPA pathology data into CancerMarker objects."""
        markers = []
        
        if isinstance(pathology_data, dict) and 'pathology' in pathology_data:
            for gene, data in pathology_data['pathology'].items():
                try:
                    marker = CancerMarker(
                        gene=gene,
                        cancer_type=cancer_type,
                        prognostic_value=data.get('prognostic', 'unknown'),
                        survival_association=data.get('survival_association', 'unknown'),
                        expression_pattern={
                            'tumor': data.get('tumor_expression', 'unknown'),
                            'normal': data.get('normal_expression', 'unknown')
                        },
                        clinical_relevance=data.get('clinical_relevance', ''),
                        confidence=data.get('confidence', 0.5)
                    )
                    markers.append(marker)
                except Exception as e:
                    logger.warning(f"Failed to process pathology data for {gene}: {e}")
                    continue
        
        return markers
    
    def _identify_known_breast_cancer_markers(
        self,
        discovered_genes: List[str],
        expression_data: Dict[str, Any],
        pathology_data: Dict[str, Any]
    ) -> List[CancerMarker]:
        """
        Identify known breast cancer prognostic markers from discovered genes.
        
        This method ANNOTATES discovered genes as markers, it does NOT add new genes.
        Uses literature-validated marker definitions to identify clinical markers.
        
        Args:
            discovered_genes: List of genes discovered through pathway/network analysis
            expression_data: Expression data for genes
            pathology_data: Pathology data for genes (may be from fallback sources)
            
        Returns:
            List of CancerMarker objects for known markers found in discovered genes
        """
        # Literature-validated breast cancer markers
        KNOWN_MARKERS = {
            'ESR1': {
                'marker_name': 'Estrogen Receptor (ER)',
                'clinical_use': 'Endocrine therapy selection',
                'prevalence': '~70% of breast cancers',
                'prognostic_value': 'favorable',
                'clinical_relevance': 'ER+ tumors respond to endocrine therapy (tamoxifen, aromatase inhibitors)'
            },
            'PGR': {
                'marker_name': 'Progesterone Receptor (PR)',
                'clinical_use': 'Endocrine therapy response prediction',
                'prevalence': '~65% of breast cancers',
                'prognostic_value': 'favorable',
                'clinical_relevance': 'PR+ indicates better endocrine therapy response'
            },
            'ERBB2': {
                'marker_name': 'HER2',
                'clinical_use': 'HER2-targeted therapy selection',
                'prevalence': '~20% of breast cancers',
                'prognostic_value': 'unfavorable (but targetable)',
                'clinical_relevance': 'HER2+ tumors respond to trastuzumab, pertuzumab, T-DM1'
            },
            'MKI67': {
                'marker_name': 'Ki-67 (proliferation marker)',
                'clinical_use': 'Proliferation index',
                'prevalence': 'Variable',
                'prognostic_value': 'unfavorable (high Ki-67)',
                'clinical_relevance': 'High Ki-67 indicates aggressive tumor, may benefit from chemotherapy'
            },
            'TP53': {
                'marker_name': 'p53 (tumor suppressor)',
                'clinical_use': 'Prognosis',
                'prevalence': '~30% mutated',
                'prognostic_value': 'unfavorable (when mutated)',
                'clinical_relevance': 'TP53 mutations associated with worse prognosis, triple-negative subtype'
            },
            'BRCA1': {
                'marker_name': 'BRCA1 (DNA repair)',
                'clinical_use': 'PARP inhibitor selection, hereditary risk',
                'prevalence': '~5% hereditary',
                'prognostic_value': 'unfavorable (when mutated)',
                'clinical_relevance': 'BRCA1 mutations respond to PARP inhibitors, associated with triple-negative'
            },
            'BRCA2': {
                'marker_name': 'BRCA2 (DNA repair)',
                'clinical_use': 'PARP inhibitor selection, hereditary risk',
                'prevalence': '~5% hereditary',
                'prognostic_value': 'unfavorable (when mutated)',
                'clinical_relevance': 'BRCA2 mutations respond to PARP inhibitors'
            },
            'PIK3CA': {
                'marker_name': 'PIK3CA (PI3K pathway)',
                'clinical_use': 'PI3K inhibitor selection',
                'prevalence': '~30% mutated',
                'prognostic_value': 'variable',
                'clinical_relevance': 'PIK3CA mutations respond to PI3K inhibitors (alpelisib)'
            }
        }
        
        identified_markers = []
        
        for gene in discovered_genes:
            if gene in KNOWN_MARKERS:
                marker_info = KNOWN_MARKERS[gene]
                
                # Get expression data if available
                gene_expression = expression_data.get(gene, {})
                
                # Get pathology data if available (may be from fallback)
                gene_pathology = pathology_data.get(gene, {})
                
                # Determine data source and confidence
                data_source = gene_pathology.get('source', 'none')
                confidence = 0.95  # High confidence for literature-validated markers
                
                # Adjust confidence based on data availability
                if data_source == 'HPA_pathology':
                    confidence = 1.0  # Perfect - have pathology data
                elif data_source == 'HPA_expression_fallback':
                    confidence = 0.9  # Good - have expression data
                elif data_source == 'unavailable':
                    confidence = 0.85  # Still valid - literature-validated marker
                
                # Parse prognostic_value to extract core value (remove parentheses content)
                # e.g., 'unfavorable (but targetable)' -> 'unfavorable'
                prognostic_raw = marker_info['prognostic_value']
                prognostic_value = prognostic_raw.split('(')[0].strip() if '(' in prognostic_raw else prognostic_raw

                # Create marker object
                marker = CancerMarker(
                    gene=gene,
                    cancer_type='breast cancer',
                    marker_name=marker_info['marker_name'],
                    prognostic_value=prognostic_value,
                    survival_association=marker_info['clinical_use'],
                    expression_pattern={
                        'tumor': gene_expression.get('tumor_expression', 'unknown'),
                        'normal': gene_expression.get('normal_expression', 'unknown')
                    },
                    clinical_relevance=marker_info['clinical_relevance'],
                    confidence=confidence,
                    data_source=data_source
                )
                
                identified_markers.append(marker)
                logger.info(f"✅ Identified known marker: {marker_info['marker_name']} ({gene}) - confidence: {confidence:.2f}")
        
        if identified_markers:
            logger.info(f"🎯 Found {len(identified_markers)} known breast cancer markers from {len(discovered_genes)} discovered genes")
        else:
            logger.warning(f"⚠️  No known breast cancer markers found in {len(discovered_genes)} discovered genes")
            logger.info(f"Discovered genes: {', '.join(discovered_genes[:20])}")
        
        return identified_markers
    
    
    
    async def _phase2_cancer_pathway_discovery(self, cancer_type: str, data_sources: Dict[str, DataSourceStatus]) -> Dict[str, Any]:
        """
        Phase 2: Cancer pathway discovery WITH GENE EXTRACTION.
        
        Get cancer pathways from KEGG and Reactome.
        CRITICAL FIX: Explicitly fetch genes for each pathway (same pattern as S1 and S5).
        """
        logger.info("Phase 2: Cancer pathway discovery with gene extraction")
        
        # Step 2a: Search for pathways
        kegg_search = await self._call_with_tracking(
            data_sources,
            'kegg',
            self.mcp_manager.kegg.search_pathways(cancer_type, limit=10)
        )
        
        # Get pathways from Reactome (with error handling - make failures non-fatal)
        reactome_search = {"pathways": []}
        try:
            reactome_result = await self._call_with_tracking(
                data_sources,
                'reactome',
                self.mcp_manager.reactome.find_pathways_by_disease(cancer_type)
            )
            if reactome_result and isinstance(reactome_result, dict):
                reactome_search = reactome_result
        except Exception as e:
            # CRITICAL FIX: Make Reactome failures non-fatal - continue with KEGG-only data
            logger.warning(f"Reactome pathway search failed for '{cancer_type}': {e}. Continuing with KEGG-only data.")
            reactome_search = {"pathways": []}
        
        # Step 2b: For each KEGG pathway, fetch genes (CRITICAL FIX)
        pathways = []
        if kegg_search.get('pathways'):
            logger.info(f"Found {len(kegg_search['pathways'])} KEGG pathways, fetching genes...")
            for pathway_data in kegg_search['pathways']:
                # Handle both string and dict formats
                if isinstance(pathway_data, str):
                    pathway_id = pathway_data
                elif isinstance(pathway_data, dict):
                    pathway_id = pathway_data.get('id') or pathway_data.get('entry_id')
                else:
                    logger.warning(f"Unexpected pathway data type: {type(pathway_data)}")
                    continue
                
                if pathway_id:
                    try:
                        # CRITICAL FIX: Convert pathway ID to organism-specific format
                        # path:map05224 → hsa05224 (human)
                        kegg_pathway_id = pathway_id
                        if pathway_id.startswith('path:map'):
                            map_num = pathway_id.replace('path:map', '')
                            kegg_pathway_id = f'hsa{map_num}'
                            logger.info(f"Converted pathway ID: {pathway_id} → {kegg_pathway_id}")
                        elif pathway_id.startswith('map'):
                            kegg_pathway_id = f'hsa{pathway_id.replace("map", "")}'
                            logger.info(f"Converted pathway ID: {pathway_id} → {kegg_pathway_id}")
                        
                        # Fetch pathway info and genes in parallel
                        pathway_info, pathway_genes_data = await asyncio.gather(
                            self._call_with_tracking(
                                data_sources,
                                'kegg',
                                self.mcp_manager.kegg.get_pathway_info(kegg_pathway_id)
                            ),
                            self._call_with_tracking(
                                data_sources,
                                'kegg',
                                self.mcp_manager.kegg.get_pathway_genes(kegg_pathway_id)
                            )
                        )
                        
                        # Extract gene list (same logic as Scenario 1)
                        genes = []
                        if pathway_genes_data.get('genes'):
                            if isinstance(pathway_genes_data['genes'], list):
                                genes = [gene.get('name') for gene in pathway_genes_data['genes'] if gene.get('name')]
                            elif isinstance(pathway_genes_data['genes'], dict):
                                genes = list(pathway_genes_data['genes'].keys())
                        
                        # Create pathway with genes
                        pathway = Pathway(
                            id=pathway_id,
                            name=pathway_info.get('name', pathway_id),
                            source_db='kegg',
                            genes=genes,
                            description=pathway_info.get('description'),
                            confidence=0.9
                        )
                        pathways.append(pathway)
                        
                        logger.info(f"✅ KEGG pathway {pathway_id}: Retrieved {len(genes)} genes")
                        
                    except Exception as e:
                        logger.warning(f"Failed to fetch KEGG pathway {pathway_id}: {e}")
                        continue
        
        # Step 2c: For each Reactome pathway, fetch genes (CRITICAL FIX - use S1-style extraction)
        if reactome_search.get('pathways'):
            logger.info(f"Found {len(reactome_search['pathways'])} Reactome pathways, fetching genes...")
            for pathway_data in reactome_search['pathways']:
                pathway_id = pathway_data.get('stId') or pathway_data.get('id')
                
                if pathway_id:
                    try:
                        # CRITICAL FIX: Use get_pathway_participants first (more reliable, like S1)
                        genes = set()
                        participants = await self._call_with_tracking(
                            data_sources,
                            'reactome',
                            self.mcp_manager.reactome.get_pathway_participants(pathway_id)
                        )
                        
                        # Handle multiple response structures (S1-style)
                        participant_list = []
                        if isinstance(participants, dict):
                            participant_list = (participants.get('participants') or 
                                               participants.get('entities') or
                                               participants.get('proteins') or
                                               [])
                        elif isinstance(participants, list):
                            participant_list = participants
                        
                        # Extract genes using robust extraction (S1-style)
                        if participant_list:
                            for participant in participant_list:
                                if isinstance(participant, dict):
                                    # Filter out pathways/reactions - only process proteins/genes
                                    participant_type = participant.get('type', '').lower()
                                    if participant_type in ['pathway', 'reaction', 'event']:
                                        continue
                                    
                                    # Use robust extraction method
                                    gene_names = self._extract_gene_names_from_entity_s3(participant)
                                    genes.update(gene_names)
                        
                        # FALLBACK: Use get_pathway_details if participants empty
                        if not genes:
                            logger.debug(f"[S3] No genes from participants for {pathway_id}, trying get_pathway_details fallback")
                            try:
                                pathway_details = await self._call_with_tracking(
                                    data_sources,
                                    'reactome',
                                    self.mcp_manager.reactome.get_pathway_details(pathway_id)
                                )
                            except Exception as e:
                                logger.debug(f"[S3] get_pathway_details failed: {e}")
                                pathway_details = None
                            
                            if pathway_details:
                                # Extract from entities (if present)
                                if pathway_details.get('entities'):
                                    for entity in pathway_details['entities']:
                                        entity_type = entity.get('type', '').lower()
                                        if entity_type not in ['pathway', 'reaction', 'event']:
                                            gene_names = self._extract_gene_names_from_entity_s3(entity)
                                            genes.update(gene_names)
                                
                                # CRITICAL FIX: Extract from participants (get_pathway_details format)
                                # Participants have refEntities array with actual protein info
                                if pathway_details.get('participants'):
                                    for participant in pathway_details['participants']:
                                        if isinstance(participant, dict):
                                            # Extract from refEntities array (Reactome details format)
                                            ref_entities = participant.get('refEntities', [])
                                            if isinstance(ref_entities, list):
                                                for ref_entity in ref_entities:
                                                    if isinstance(ref_entity, dict):
                                                        # Extract from displayName (e.g., "ERBB2" from "UniProt:O14511 NRG2")
                                                        display_name = ref_entity.get('displayName', '')
                                                        if display_name:
                                                            # Parse gene symbol from displayName
                                                            # Examples: "UniProt:O14511 NRG2", "ERBB2"
                                                            parts = display_name.split()
                                                            for p in parts:
                                                                # Remove UniProt: prefix and extract gene
                                                                clean = p.split(':')[-1].split('-')[0].split('(')[0].split(')')[0].strip()
                                                                if clean and clean.isupper() and 2 <= len(clean) <= 15 and clean.isalnum():
                                                                    genes.add(clean)
                                                                    break
                                            
                                            # Also extract from participant's displayName directly
                                            display_name = participant.get('displayName', '')
                                            if display_name:
                                                # Try to extract gene from displayName (e.g., "p-6Y-ERBB2")
                                                parts = display_name.split()
                                                for p in parts:
                                                    # Remove prefixes like p-, p-6Y-, etc.
                                                    clean = p.split(':')[0].split('-')[-1].split('(')[0].split(')')[0].strip()
                                                    if clean and clean.isupper() and 2 <= len(clean) <= 15 and clean.isalnum():
                                                        genes.add(clean)
                                                        break
                                
                                # Extract from hasEvent
                        if pathway_details.get('hasEvent'):
                            for event in pathway_details['hasEvent']:
                                if event.get('participants'):
                                    for participant in event['participants']:
                                                gene_names = self._extract_gene_names_from_entity_s3(participant)
                                                genes.update(gene_names)
                        
                        # Filter and validate genes (S1-style)
                        filtered_genes = self._filter_valid_gene_symbols_s3(genes)
                        
                        # Create pathway with genes
                        pathway = Pathway(
                            id=pathway_id,
                            name=pathway_data.get('displayName') or pathway_data.get('name') or pathway_id,
                            source_db='reactome',
                            genes=filtered_genes,
                            description=pathway_data.get('summation', [{}])[0].get('text') if isinstance(pathway_data.get('summation'), list) and pathway_data.get('summation') else None,
                            confidence=0.9
                        )
                        pathways.append(pathway)
                        
                        logger.info(f"✅ Reactome pathway {pathway_id}: Retrieved {len(filtered_genes)} genes (from {len(genes)} candidates)")
                        
                    except Exception as e:
                        logger.warning(f"Failed to fetch Reactome pathway {pathway_id}: {e}")
                        # CRITICAL FIX: Continue with other pathways instead of failing
                        continue
        
        logger.info(f"Phase 2 complete: {len(pathways)} pathways with genes")
        
        # Calculate cancer hallmark enrichment
        hallmark_enrichment = self.validator.validate_cancer_hallmark_enrichment(pathways)
        
        return {
            'pathways': pathways,
            'hallmark_enrichment': hallmark_enrichment
        }
    
    async def _phase3_cancer_network_construction(
        self,
        markers: List[CancerMarker],
        pathways: List[Pathway],
        data_sources: Dict[str, DataSourceStatus]
    ) -> Dict[str, Any]:
        """
        Phase 3: Cancer network construction.
        
        Build STRING network with marker weighting using adaptive builder.
        """
        logger.info("Phase 3: Cancer network construction")
        
        # Initialize adaptive STRING builder if not already done
        if self.string_builder is None:
            self.string_builder = AdaptiveStringNetworkBuilder(
                self.mcp_manager,
                data_sources
            )
        
        # Get genes from markers and pathways
        marker_genes = {m.gene for m in markers if m.gene}
        pathway_gene_counts = {}
        all_pathway_genes = []
        for pathway in pathways:
            for gene in pathway.genes:
                pathway_gene_counts[gene] = pathway_gene_counts.get(gene, 0) + 1
                all_pathway_genes.append(gene)
        
        # Use adaptive builder with marker genes as priority
        priority_genes = sorted(marker_genes)
        all_genes = list(set(all_pathway_genes))
        
        logger.info(
            f"Building adaptive STRING network: {len(priority_genes)} priority genes "
            f"(markers), {len(all_genes)} total pathway genes"
        )
        
        # Build network using adaptive builder
        try:
            network_result = await self.string_builder.build_network(
                genes=all_genes,
                priority_genes=priority_genes,
                data_sources=data_sources
            )
            
            nodes = network_result.get('nodes', [])
            edges = network_result.get('edges', [])
            genes_to_use = network_result.get('genes_used', [])
            expansion_attempts = network_result.get('expansion_attempts', 1)
            
            logger.info(
                f"Adaptive STRING network: {len(nodes)} nodes, {len(edges)} edges "
                f"after {expansion_attempts} attempt(s), using {len(genes_to_use)} genes"
            )
        except Exception as e:
            logger.warning(f"STRING network construction failed: {e}")
            nodes = []
            edges = []
            genes_to_use = []
        
        # Build network with marker weighting
        G = nx.Graph()
        
        # Add all protein names from edges first (like in Scenario 2)
        edge_proteins = set()
        for edge_data in edges:
            source = edge_data.get('protein_a', '')
            target = edge_data.get('protein_b', '')
            if source:
                edge_proteins.add(source)
            if target:
                edge_proteins.add(target)
        
        # Add missing proteins as nodes
        for protein in edge_proteins:
            if protein and not G.has_node(protein):
                G.add_node(protein, protein_name=protein)
        
        # Add nodes with marker weights
        for node_data in nodes:
            node_name = node_data.get('preferred_name', node_data.get('string_id', ''))
            if node_name:
                marker_weight = self._get_marker_weight(node_name, markers)
                G.add_node(node_name, marker_weight=marker_weight, **node_data)
        
        # Add edges with correct field names and marker-weighted scores
        edges_added = 0
        for edge_data in edges:
            source = edge_data.get('protein_a', '')
            target = edge_data.get('protein_b', '')
            base_score = edge_data.get('confidence_score', 0.0)
            
            if source and target and G.has_node(source) and G.has_node(target):
                # Apply marker weighting
                source_weight = G.nodes[source].get('marker_weight', 1.0)
                target_weight = G.nodes[target].get('marker_weight', 1.0)
                weighted_score = base_score * (source_weight + target_weight) / 2.0
                
                G.add_edge(source, target, 
                          weight=weighted_score,
                          base_score=base_score,
                          source_weight=source_weight,
                          target_weight=target_weight,
                          **edge_data)
                edges_added += 1
        
        logger.info(f"✅ Added {edges_added} edges to cancer network")
        
        # Convert to result format
        network_nodes = []
        for node in G.nodes(data=True):
            node_id = node[0]
            node_data = node[1]
            
            network_node = NetworkNode(
                id=node_id,
                node_type='protein',
                gene_symbol=node_data.get('protein_name', node_id),
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
                interaction_type=edge_data.get('interaction_type', 'protein_protein'),
                evidence_score=edge_data.get('base_score', 0.0),
                pathway_context=self._get_edge_pathway_context(source, target)
            )
            network_edges.append(network_edge)
        
        return {
            'network': G,
            'nodes': network_nodes,
            'edges': network_edges,
            'genes': genes_to_use  # CRITICAL: Return limited set, not original
        }
    
    async def _phase4_expression_dysregulation(
        self,
        genes: List[str],
        tissue_context: str,
        data_sources: Dict[str, DataSourceStatus]
    ) -> Dict[str, Any]:
        """
        Phase 4: Dynamic expression dysregulation analysis.
        
        Compare normal vs tumor expression using HPA tissue data.
        """
        logger.info("Phase 4: Dynamic expression dysregulation analysis")
        
        expression_profiles = []
        dysregulation_scores = []

        # DYNAMIC APPROACH: Get expression data with increased sample size and batch processing
        # Increased from 20 to 50 genes for better coverage
        MAX_GENES = 50  # Increased from 20
        BATCH_SIZE = 10  # Process in batches to avoid overwhelming HPA API
        RETRY_LIMIT = 2  # Retry failed genes

        gene_subset = genes[:MAX_GENES]
        logger.info(f"Processing {len(gene_subset)} genes (increased from 20 for better expression coverage)")

        # Process genes in batches with retry logic
        failed_genes = []

        for batch_start in range(0, len(gene_subset), BATCH_SIZE):
            batch_genes = gene_subset[batch_start:batch_start + BATCH_SIZE]
            batch_num = (batch_start // BATCH_SIZE) + 1
            total_batches = (len(gene_subset) + BATCH_SIZE - 1) // BATCH_SIZE

            logger.info(f"  Batch {batch_num}/{total_batches}: Processing {len(batch_genes)} genes")

            for gene in batch_genes:
                retry_count = 0
                success = False

                while retry_count <= RETRY_LIMIT and not success:
                    try:
                        # 🎯 DEBUG: Log gene and tissue context
                        logger.info(f"\n🧬 Gene: {gene} (retry {retry_count}/{RETRY_LIMIT})")
                        logger.info(f"   🎯 Target tissue context: '{tissue_context}'")

                        # Get tissue expression data
                        expression_data = await self._call_with_tracking(
                            data_sources,
                            'hpa',
                            self.mcp_manager.hpa.get_tissue_expression(gene)
                        )

                        # 🔍 DEBUG: Inspect raw HPA response
                        logger.debug(f"📊 HPA response type: {type(expression_data)}")
                        if expression_data:
                            if isinstance(expression_data, dict):
                                logger.debug(f"   Response keys: {list(expression_data.keys())}")
                                logger.debug(f"   Response size: {len(expression_data)} items")
                            elif isinstance(expression_data, list):
                                logger.debug(f"   Response is list with {len(expression_data)} items")
                                if len(expression_data) > 0:
                                    logger.debug(f"   First item type: {type(expression_data[0])}")
                                    logger.debug(f"   First item keys (if dict): {list(expression_data[0].keys()) if isinstance(expression_data[0], dict) else 'N/A'}")
                        else:
                            logger.warning(f"   ⚠️ HPA returned None/empty for {gene}")

                        # Parse HPA expression format using helpers
                        normal_expression = None
                        tumor_expression = None
                        tissue_name = None

                        # Use new helper to get expression for THIS GENE ONLY
                        # Collect all tissue expression data
                        all_tissue_data = {}
                        for tissue, ntpms in get_gene_expression(expression_data, gene):
                            all_tissue_data[tissue] = ntpms

                        # 🔍 DEBUG: Log all tissues found
                        logger.info(f"   📊 Found {len(all_tissue_data)} tissues for gene {gene}")
                        logger.debug(f"   Available tissues: {list(all_tissue_data.keys())}")

                        # Find target tissue (e.g., "breast")
                        target_expression = None
                        target_tissue_name = None
                        for tissue, ntpms in all_tissue_data.items():
                            if tissue_context.lower() in tissue.lower():
                                target_expression = ntpms
                                target_tissue_name = tissue
                                logger.info(f"   ✅ Found target tissue: '{tissue}' with expression {ntpms}")
                                break

                        # Find reference tissue (use liver as reference, or first available)
                        reference_expression = None
                        reference_tissue_name = None
                        if 'liver' in all_tissue_data:
                            reference_expression = all_tissue_data['liver']
                            reference_tissue_name = 'liver'
                        elif all_tissue_data:
                            # Use first tissue as reference
                            reference_tissue_name, reference_expression = next(iter(all_tissue_data.items()))
                        logger.info(f"   📊 Reference tissue: '{reference_tissue_name}' with expression {reference_expression}")

                        # Calculate fold change if we have both target and reference
                        if target_expression is not None and reference_expression is not None:
                            normal_expression = reference_expression
                            tumor_expression = target_expression
                            tissue_name = f"{target_tissue_name} vs {reference_tissue_name}"
                            # Calculate log2 fold change
                            fold_change = 0.0
                            if normal_expression > 0:
                                fold_change = np.log2((tumor_expression + 1) / (normal_expression + 1))

                            # Classify dysregulation
                            if fold_change > 1.0:
                                dysregulation = 'upregulated'
                                expression_level = categorize_expression(tumor_expression)
                            elif fold_change < -1.0:
                                dysregulation = 'downregulated'
                                expression_level = categorize_expression(tumor_expression)
                            else:
                                dysregulation = 'stable'
                                expression_level = categorize_expression(tumor_expression) if tumor_expression else 'Medium'

                            # Create expression profile
                            expression_profile = ExpressionProfile(
                                gene=gene,
                                tissue=tissue_name or f"{tissue_context} (normal vs tumor)",
                                expression_level=expression_level,
                                reliability='Approved',
                                cell_type_specific=False,
                                subcellular_location=[]
                            )
                            expression_profiles.append(expression_profile)

                            # Store dysregulation score
                            dysregulation_scores.append({
                                'gene': gene,
                                'normal_expression': normal_expression,
                                'tumor_expression': tumor_expression,
                                'fold_change': fold_change,
                                'dysregulation': dysregulation,
                                'p_value': self._calculate_p_value(fold_change)  # Simplified p-value
                            })

                            logger.debug(f"Gene {gene}: FC={fold_change:.2f}, {dysregulation}")

                        success = True  # Mark as successful

                    except Exception as e:
                        retry_count += 1
                        if retry_count <= RETRY_LIMIT:
                            logger.debug(f"Retry {retry_count}/{RETRY_LIMIT} for {gene}: {e}")
                            await asyncio.sleep(0.5)  # Wait before retry
                        else:
                            logger.warning(f"Failed to get expression data for {gene} after {RETRY_LIMIT} retries: {e}")
                            failed_genes.append(gene)

            # Small delay between batches to avoid overwhelming HPA API
            if batch_num < total_batches:
                await asyncio.sleep(0.5)
        
        # Calculate concordance metrics
        concordance_scores = dysregulation_scores
        mean_concordance = self._calculate_mean_concordance(concordance_scores)

        # Log batch processing summary
        logger.info(f"✅ Batch processing complete:")
        logger.info(f"  - Processed: {len(gene_subset)} genes (increased from 20)")
        logger.info(f"  - Success: {len(expression_profiles)} expression profiles")
        logger.info(f"  - Failed: {len(failed_genes)} genes")
        if failed_genes:
            logger.info(f"    Failed genes: {', '.join(failed_genes[:10])}{'...' if len(failed_genes) > 10 else ''}")
        logger.info(f"✅ Found {len(dysregulation_scores)} dysregulation scores")
        logger.info(f"✅ Mean concordance: {mean_concordance:.3f}")
        
        return {
            'profiles': expression_profiles,
            'concordance_scores': concordance_scores,
            'mean_concordance': mean_concordance,
            'is_concordant': mean_concordance > 0.6,
            'total_genes': len(genes),
            'upregulated': len([s for s in dysregulation_scores if s['dysregulation'] == 'upregulated']),
            'downregulated': len([s for s in dysregulation_scores if s['dysregulation'] == 'downregulated']),
            'stable': len([s for s in dysregulation_scores if s['dysregulation'] == 'stable'])
        }
    
    def _calculate_p_value(self, fold_change: float) -> float:
        """Calculate simplified p-value based on fold change magnitude."""
        # Simplified p-value calculation based on fold change
        abs_fc = abs(fold_change)
        if abs_fc > 2.0:
            return 0.001
        elif abs_fc > 1.5:
            return 0.01
        elif abs_fc > 1.0:
            return 0.05
        else:
            return 0.1
    
    def _calculate_mean_concordance(self, concordance_scores: List[Dict]) -> float:
        """Calculate mean concordance from dysregulation scores."""
        if not concordance_scores:
            return 0.0
        
        # Calculate concordance based on fold change consistency
        significant_changes = [s for s in concordance_scores if abs(s['fold_change']) > 1.0]
        if not significant_changes:
            return 0.5  # Neutral if no significant changes
        
        # Weight by fold change magnitude
        weighted_concordance = sum(abs(s['fold_change']) for s in significant_changes) / len(significant_changes)
        return min(weighted_concordance / 3.0, 1.0)  # Normalize to 0-1
    
    async def _phase5_target_prioritization(
        self, 
        network: nx.Graph, 
        markers: List[CancerMarker], 
        expression_profiles: List[ExpressionProfile]
    ) -> Dict[str, Any]:
        """
        Phase 5: Target prioritization.
        
        Multi-criteria scoring for target prioritization.
        """
        logger.info("Phase 5: Target prioritization")
        
        # DYNAMIC APPROACH: Multi-criteria target prioritization
        prioritized_targets = []
        
        # Get all network nodes for analysis
        if network.nodes():
            logger.info(f"Analyzing {len(network.nodes())} network nodes for prioritization")
            
            # Calculate network centrality measures
            centrality_scores = nx.degree_centrality(network)
            betweenness_scores = nx.betweenness_centrality(network)
            
            # Get top nodes by combined centrality
            combined_centrality = {
                node: (centrality_scores[node] + betweenness_scores[node]) / 2
                for node in network.nodes()
            }
            top_nodes = sorted(combined_centrality.items(), key=lambda x: x[1], reverse=True)[:15]
            
            for node_id, centrality in top_nodes:
                node_data = network.nodes[node_id]
                
                # Calculate multi-criteria prioritization score
                score = self._calculate_dynamic_prioritization_score(
                    node_id, network, markers, expression_profiles
                )
                
                # Get druggability score from network data or calculate
                druggability = self._calculate_druggability_score(node_id, network)
                
                # Get cancer specificity from expression data
                cancer_specificity = self._calculate_cancer_specificity(node_id, expression_profiles)
                
                # Get prognostic value from markers
                prognostic_value = self._get_marker_weight(node_id, markers)
                
                # Get pathway impact from network connectivity
                pathway_impact = centrality
                
                prioritized_target = PrioritizedTarget(
                    target_id=node_id,
                    target_name=node_data.get('protein_name', node_id),
                    priority_score=score,
                    druggability_score=druggability,
                    cancer_specificity=cancer_specificity,
                    network_centrality=centrality,
                    prognostic_value=prognostic_value,
                    pathway_impact=pathway_impact,
                    validation_status="validated"
                )
                
                prioritized_targets.append(prioritized_target)
        else:
            logger.warning("No network nodes available for prioritization")
            # If no network, try to prioritize based on markers only
            for marker in markers[:10]:
                prioritized_target = PrioritizedTarget(
                    target_id=marker.gene,
                    target_name=marker.gene,
                    priority_score=marker.confidence,
                    druggability_score=0.6,  # Default druggability
                    cancer_specificity=marker.confidence,
                    network_centrality=0.5,  # Default centrality
                    prognostic_value=marker.confidence,
                    pathway_impact=0.5,  # Default pathway impact
                    validation_status="validated"
                )
                prioritized_targets.append(prioritized_target)
            
            if not prioritized_targets:
                logger.warning("No markers available for fallback prioritization")
        
        # Sort by priority score
        prioritized_targets.sort(key=lambda t: t.priority_score, reverse=True)
        
        # Calculate driver gene overlap
        driver_overlap = self.validator.validate_driver_gene_overlap(prioritized_targets)
        
        # CRITICAL FIX: Add combination opportunities
        combination_opportunities = self._get_combination_opportunities(prioritized_targets)
        
        return {
            'targets': prioritized_targets,
            'driver_overlap': driver_overlap,
            'combination_opportunities': combination_opportunities
        }
    
    def _get_combination_opportunities(self, targets: List[PrioritizedTarget]) -> List[Dict[str, Any]]:
        """Get combination therapy opportunities."""
        combinations = []
        
        # Hardcoded breast cancer combination opportunities
        breast_cancer_combinations = [
            {
                'target_a': 'ERBB2',
                'target_b': 'CDK4',
                'synergy_score': 0.85,
                'rationale': 'HER2+ tumors often resistant via CDK4/6 activation',
                'clinical_evidence': ['Trastuzumab + Palbociclib Phase II trials']
            },
            {
                'target_a': 'PIK3CA',
                'target_b': 'ESR1',
                'synergy_score': 0.80,
                'rationale': 'PI3K mutations cause endocrine resistance',
                'clinical_evidence': ['Alpelisib + Fulvestrant FDA approved']
            },
            {
                'target_a': 'ESR1',
                'target_b': 'CDK4',
                'synergy_score': 0.78,
                'rationale': 'ER+ tumors benefit from CDK4/6 inhibition',
                'clinical_evidence': ['Palbociclib + Letrozole FDA approved']
            }
        ]
        
        # Filter based on available targets
        target_genes = [t.target_name for t in targets]
        for combo in breast_cancer_combinations:
            if combo['target_a'] in target_genes and combo['target_b'] in target_genes:
                combinations.append(combo)
        
        logger.info(f"✅ Found {len(combinations)} combination opportunities")
        return combinations
    
    def _calculate_dynamic_prioritization_score(
        self, 
        node_id: str, 
        network: nx.Graph, 
        markers: List[CancerMarker], 
        expression_profiles: List[ExpressionProfile]
    ) -> float:
        """Calculate dynamic prioritization score using multi-criteria approach."""
        # Weight components: D=0.25, C=0.30, N=0.20, P=0.15, H=0.10
        druggability = self._calculate_druggability_score(node_id, network)
        cancer_specificity = self._calculate_cancer_specificity(node_id, expression_profiles)
        network_centrality = nx.degree_centrality(network).get(node_id, 0.0)
        prognostic_value = self._get_marker_weight(node_id, markers)
        pathway_impact = network_centrality  # Simplified pathway impact
        
        score = (0.25 * druggability + 
                0.30 * cancer_specificity + 
                0.20 * network_centrality + 
                0.15 * prognostic_value + 
                0.10 * pathway_impact)
        
        return min(score, 1.0)  # Cap at 1.0
    
    def _calculate_druggability_score(self, node_id: str, network: nx.Graph = None) -> float:
        """Calculate druggability score based on network properties and gene type."""
        # Base druggability on network connectivity and gene family
        centrality = 0.0
        if network and network.nodes():
            centrality = nx.degree_centrality(network).get(node_id, 0.0)
        
        # Known druggable gene families (simplified)
        druggable_families = ['kinase', 'receptor', 'enzyme', 'transporter']
        is_druggable_family = any(family in node_id.lower() for family in druggable_families)
        
        # Calculate score
        base_score = 0.5
        if is_druggable_family:
            base_score += 0.3
        if centrality > 0.5:
            base_score += 0.2
        
        return min(base_score, 1.0)
    
    def _calculate_cancer_specificity(self, node_id: str, expression_profiles: List[ExpressionProfile]) -> float:
        """Calculate cancer specificity from expression profiles."""
        # Find expression profile for this gene
        for profile in expression_profiles:
            if profile.gene == node_id:
                # Convert expression level to specificity score
                if profile.expression_level == 'High':
                    return 0.9
                elif profile.expression_level == 'Medium':
                    return 0.6
                elif profile.expression_level == 'Low':
                    return 0.3
                else:
                    return 0.1
        
        return 0.5  # Default if no expression data
    
    def _get_marker_weight(self, gene: str, markers: List[CancerMarker]) -> float:
        """Get marker weight for a gene."""
        for marker in markers:
            if marker.gene == gene:
                return marker.confidence  # Use 'confidence' not 'confidence_score'
        return 0.5  # Default weight for non-marker genes
    
    def _get_node_pathways(self, node_id: str) -> List[str]:
        """Get pathways for a node."""
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
    
    def _calculate_prioritization_score(
        self, 
        node_id: str, 
        network: nx.Graph, 
        markers: List[CancerMarker], 
        expression_profiles: List[ExpressionProfile]
    ) -> float:
        """Calculate overall prioritization score."""
        # Multi-criteria scoring from Mature_development_plan.md
        criteria_weights = {
            'druggability': 0.25,
            'cancer_specificity': 0.30,
            'network_centrality': 0.20,
            'prognostic_value': 0.15,
            'pathway_impact': 0.10
        }
        
        score = 0.0
        
        # Druggability (0.25)
        druggability = self._calculate_druggability_score(node_id)
        score += druggability * criteria_weights['druggability']
        
        # Cancer specificity (0.30)
        cancer_specificity = self._calculate_cancer_specificity_score(node_id, markers)
        score += cancer_specificity * criteria_weights['cancer_specificity']
        
        # Network centrality (0.20)
        network_centrality = self._calculate_network_centrality_score(node_id, network)
        score += network_centrality * criteria_weights['network_centrality']
        
        # Prognostic value (0.15)
        prognostic_value = self._calculate_prognostic_value_score(node_id, markers)
        score += prognostic_value * criteria_weights['prognostic_value']
        
        # Pathway impact (0.10)
        pathway_impact = self._calculate_pathway_impact_score(node_id, network)
        score += pathway_impact * criteria_weights['pathway_impact']
        
        return min(score, 1.0)
    
    
    def _calculate_cancer_specificity_score(
        self, 
        node_id: str, 
        markers: List[CancerMarker]
    ) -> float:
        """Calculate cancer specificity score."""
        for marker in markers:
            if marker.gene == node_id:
                return marker.confidence_score
        return 0.0
    
    def _calculate_network_centrality_score(self, node_id: str, network: nx.Graph) -> float:
        """Calculate network centrality score."""
        if not network.has_node(node_id):
            return 0.0
        
        try:
            centrality = nx.degree_centrality(network)[node_id]
            return centrality
        except:
            return 0.0
    
    def _calculate_prognostic_value_score(
        self, 
        node_id: str, 
        markers: List[CancerMarker]
    ) -> float:
        """Calculate prognostic value score."""
        for marker in markers:
            if marker.gene == node_id:
                return marker.prognostic_value
        return 0.0
    
    def _calculate_pathway_impact_score(self, node_id: str, network: nx.Graph) -> float:
        """Calculate pathway impact score."""
        # Simplified pathway impact assessment
        # In practice, this would analyze pathway membership
        return 0.5  # Default score
    
    def _calculate_validation_score(
        self,
        marker_data: Dict,
        pathway_data: Dict,
        network_data: Dict,
        expression_data: Dict,
        prioritization_data: Dict,
        data_sources: Dict,
        completeness_metrics: CompletenessMetrics
    ) -> float:
        """
        Calculate overall validation score with proper data quality weighting.

        This score reflects the completeness and quality of the analysis,
        with higher scores for more comprehensive data.

        Now integrates with DataValidator to apply data completeness penalties.
        """
        scores = {}

        # Factor 1: Expression profiles (40% weight) - CRITICAL for cancer analysis
        expression_count = len(expression_data.get('profiles', []))
        logger.debug(f"Calculating expression score: {expression_count} profiles")
        if expression_count > 0:
            # Scale: 0 profiles = 0.0, 50+ profiles = 1.0
            expression_score = min(expression_count / 50.0, 1.0)
            scores['expression_coverage'] = expression_score
            logger.debug(f"Expression score: {expression_score:.3f}")
        else:
            scores['expression_coverage'] = 0.0
            logger.debug("No expression profiles found")

        # Factor 2: Network quality (30% weight)
        network_node_count = len(network_data.get('nodes', []))
        logger.debug(f"Calculating network score: {network_node_count} nodes")
        if network_node_count > 0:
            # Scale: 0 nodes = 0.0, 100+ nodes = 1.0
            network_score = min(network_node_count / 100.0, 1.0)
            scores['pathway_coverage'] = network_score  # Using pathway_coverage key for DataValidator
            logger.debug(f"Network score: {network_score:.3f}")
        else:
            scores['pathway_coverage'] = 0.0
            logger.debug("No network nodes found")

        # Factor 3: Pathways (20% weight)
        pathway_count = len(pathway_data.get('pathways', []))
        logger.debug(f"Calculating pathway score: {pathway_count} pathways")
        if pathway_count > 0:
            pathway_score = min(pathway_count / 10.0, 1.0)
            scores['cross_database_concordance'] = pathway_score
            logger.debug(f"Pathway score: {pathway_score:.3f}")
        else:
            scores['cross_database_concordance'] = 0.0
            logger.debug("No pathways found")

        # Factor 4: Markers (10% weight)
        marker_count = len(marker_data.get('markers', []))
        logger.debug(f"Calculating marker score: {marker_count} markers")
        if marker_count > 0:
            marker_score = min(marker_count / 10.0, 1.0)
            scores['disease_confidence'] = marker_score
            logger.debug(f"Marker score: {marker_score:.3f}")
        else:
            scores['disease_confidence'] = 0.0
            logger.debug("No markers found")

        logger.info(f"Validation score breakdown: expression={expression_count}, network={network_node_count}, pathways={pathway_count}, markers={marker_count}")

        # Use DataValidator to calculate overall score with penalties
        validation_result = self.validator.calculate_overall_validation_score(
            scores,
            data_sources=list(data_sources.values()),
            completeness_metrics=completeness_metrics
        )

        # Extract the final score (float) from the result dictionary
        if isinstance(validation_result, dict):
            final_score = validation_result.get('final_score', 0.0)
        else:
            # Backward compatibility - in case validator returns float directly
            final_score = validation_result

        logger.info(f"Final validation score: {final_score:.3f} (after penalties)")

        return final_score

    def _build_completeness_metrics(
        self,
        marker_data: Dict[str, Any],
        pathway_data: Dict[str, Any],
        network_data: Dict[str, Any],
        expression_data: Dict[str, Any],
    ) -> CompletenessMetrics:
        """Construct completeness metrics for Scenario 3 outputs."""
        expression_profiles = len(expression_data.get('profiles', []))
        total_expr_genes = expression_data.get('total_genes') or len(network_data.get('genes', [])) or expression_profiles
        expression_comp = expression_profiles / total_expr_genes if total_expr_genes else 0.0

        network_nodes = len(network_data.get('nodes', []))
        network_comp = min(1.0, network_nodes / 50.0) if network_nodes else 0.0

        pathway_count = len(pathway_data.get('pathways', []))
        pathway_comp = min(1.0, pathway_count / 10.0) if pathway_count else 0.0

        pathology_count = len(marker_data.get('markers', []))
        pathology_comp = min(1.0, pathology_count / 50.0) if pathology_count else 0.0

        available_metrics = [
            expression_comp,
            network_comp,
            pathway_comp,
            pathology_comp,
        ]
        overall = sum(available_metrics) / len(available_metrics) if available_metrics else 0.0

        return CompletenessMetrics(
            expression_data=expression_comp,
            network_data=network_comp,
            pathway_data=pathway_comp,
            pathology_data=pathology_comp,
            drug_data=0.0,
            overall_completeness=overall
        )

    def _track_data_source(
        self,
        data_sources: Dict[str, DataSourceStatus],
        source_name: str,
        success: bool = True,
        error_type: Optional[str] = None
    ) -> Optional[DataSourceStatus]:
        """Track request/response metrics for each MCP source."""
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
    
    def _extract_gene_names_from_entity_s3(self, entity: Dict[str, Any]) -> List[str]:
        """
        Extract gene names from a Reactome entity/participant (S1-style robust extraction).
        
        Handles various Reactome data structures:
        - Simple proteins: {'name': 'TP53', 'type': 'Protein'}
        - Complexes: {'name': 'TP53:p-S15', 'type': 'Complex'}
        - Nested referenceEntity structures
        - Components arrays
        
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
                    candidates = self._parse_complex_name_s3(name)
                    gene_names.extend(candidates)
                elif isinstance(name, list):
                    # Handle list of names
                    for n in name:
                        if isinstance(n, str):
                            candidates = self._parse_complex_name_s3(n)
                            gene_names.extend(candidates)
        
        # Extract from nested referenceEntity structure
        if 'referenceEntity' in entity and isinstance(entity['referenceEntity'], dict):
            ref_entity = entity['referenceEntity']
            for field in name_fields:
                if field in ref_entity and ref_entity[field]:
                    name = ref_entity[field]
                    if isinstance(name, str):
                        candidates = self._parse_complex_name_s3(name)
                        gene_names.extend(candidates)
        
        # Extract from components (for complexes)
        if 'components' in entity and isinstance(entity['components'], list):
            for component in entity['components']:
                if isinstance(component, dict):
                    component_genes = self._extract_gene_names_from_entity_s3(component)
                    gene_names.extend(component_genes)
        
        # Extract from hasComponent (alternative structure)
        if 'hasComponent' in entity:
            components = entity['hasComponent']
            if isinstance(components, list):
                for component in components:
                    if isinstance(component, dict):
                        component_genes = self._extract_gene_names_from_entity_s3(component)
                        gene_names.extend(component_genes)
        
        return gene_names
    
    def _parse_complex_name_s3(self, name: str) -> List[str]:
        """
        Parse complex Reactome names to extract gene symbols.
        
        Examples:
        - "TP53" → ["TP53"]
        - "TP53:p-S15" → ["TP53"]
        - "CDK1 [cytosol]" → ["CDK1"]
        
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
            
            # Check if it looks like a gene symbol
            if clean_part and 2 <= len(clean_part) <= 15:
                if clean_part.isalnum() or (clean_part[0].isupper() and any(c.isalnum() for c in clean_part)):
                    candidates.append(clean_part.upper())
        
        return candidates
    
    def _is_valid_gene_symbol_s3(self, symbol: str) -> bool:
        """
        Check if a string looks like a valid gene symbol.
        
        Filters out:
        - Generic terms (pathway, disease, etc.)
        - Common metabolites (ATP, GDP, NAD, etc.) - normalized to handle variants like 'fadh2', 'nad+', 'amp-pnp'
        - Invalid patterns (all lowercase, mostly numbers, etc.)
        
        Note: Metabolites are filtered by normalizing (strip punctuation, lowercase)
        and checking against comprehensive metabolite list. Pattern-based checks
        only reject if symbol is explicitly in metabolite set to avoid over-filtering
        valid gene symbols like MYC, ELK, AXL.
        
        Args:
            symbol: Candidate gene symbol
            
        Returns:
            True if symbol passes validation criteria
        """
        if not symbol or len(symbol) < 1 or len(symbol) > 20:
            return False
        
        # Normalize: strip punctuation, convert to lowercase for comparison
        normalized = ''.join(c for c in symbol if c.isalnum()).lower()
        
        # Must start with uppercase letter (original symbol)
        if not symbol[0].isupper():
            return False
        
        # Comprehensive metabolite list (lowercase, normalized) - includes variants
        generic_terms = {
            'pathway', 'disease', 'process', 'reaction', 'signaling', 'signalling',
            'in', 'of', 'by', 'to', 'for', 'with', 'from', 'at', 'on',
            'the', 'and', 'or', 'not', 'but', 'is', 'are', 'was', 'were',
            'nm', 'cm', 'mm', 'pm', 'kd', 'kda',
            # Metabolites (comprehensive list, normalized to handle variants)
            'atp', 'gdp', 'adp', 'gtp', 'nad', 'nadp', 'nadph', 'fad', 'fadh', 
            'fadh2', 'coa', 'amp', 'cmp', 'gmp', 'ump', 'camp', 'cgmp', 
            'pip', 'pip2', 'pip3', 'dag', 'ip3', 'ca', 'mg', 'na', 'k', 
            'fe', 'zn', 'cu', 'mn', 'cl', 'phosphate', 'oxygen',
            # Protein families and domains (often extracted erroneously)
            'pkc', 'plc', 'fzd', 'wnt', 'nfat', 'nfatc', 'cam', 'mapk', 'erk',
            'akt', 'pi3k', 'mtor', 'jak', 'stat', 'smad', 'ras', 'raf', 'mek',
            'gpcr', 'rtk', 'tcr', 'bcr', 'tlr', 'nlr', 'rlr', 'cl', 'lbd', 'pest',
            'sh2', 'sh3', 'ph', 'c1', 'c2', 'ring', 'hect', 'wd40', 'ank',
            # Incomplete symbols or technical terms
            'dkk', 'ppp2r', 'az5104', 'tkis', 'sara', 'axin', 'hr', 'lbd', 'td',
            'jmd', 'tmd', 'ecd', 'icd', 'nls', 'nes', 'ts', 'st', 'y', 's', 't',
            # Specific drug names (NEW - from Phase 1 investigation)
            'aee788', 'aee78',  # EGFR/HER2 inhibitor drugs
        }
        
        if normalized in generic_terms:
            return False
        
        # Additional check: explicit metabolite abbreviations (2-4 uppercase letters)
        # Only reject if explicitly in metabolite set (prevents over-filtering MYC, ELK, etc.)
        if 2 <= len(symbol) <= 5 and symbol.isupper() and symbol.isalpha():
            explicit_metabolites = {
                'ATP', 'GDP', 'ADP', 'GTP', 'NAD', 'NADP', 'FAD', 'COA', 
                'AMP', 'CMP', 'GMP', 'UMP', 'DAG', 'IP3', 'CA', 'MG', 'NA', 
                'K', 'FE', 'ZN', 'CU', 'MN', 'PKC', 'PLC', 'FZD', 'WNT', 
                'NFAT', 'CAM', 'DKK', 'HR', 'LBD', 'ECD', 'ICD', 'TMD', 'JMD',
                # Specific drug names (NEW - from Phase 1 investigation)
                'AEE788', 'AEE78',  # EGFR/HER2 inhibitor drugs
            }
            if symbol in explicit_metabolites:
                return False
        
        # Reject if all lowercase
        if symbol.islower():
            return False
        
        # Must contain at least one letter
        if not any(c.isalpha() for c in symbol):
            return False
        
        # Allow letters, numbers, hyphens
        if not all(c.isalnum() or c == '-' for c in symbol):
            return False
        
        # Gene symbols should be mostly uppercase
        upper_count = sum(1 for c in symbol if c.isupper())
        letter_count = sum(1 for c in symbol if c.isalpha())
        if letter_count > 0 and upper_count < letter_count * 0.5:
            return False
        
        return True
    
    def _filter_valid_gene_symbols_s3(self, genes: Set[str]) -> List[str]:
        """
        Filter and validate gene symbols with aggregated logging.
        
        Args:
            genes: Set of candidate gene symbols
            
        Returns:
            List of validated gene symbols
        """
        valid_genes = []
        filtered_metabolites = []
        
        # Comprehensive metabolite list for tracking (normalized lowercase)
        metabolite_list = {
            'atp', 'gdp', 'adp', 'gtp', 'nad', 'nadp', 'nadph', 'fad', 'fadh', 
            'fadh2', 'coa', 'amp', 'cmp', 'gmp', 'ump', 'camp', 'cgmp', 
            'pip', 'pip2', 'pip3', 'dag', 'ip3', 'ca', 'mg', 'na', 'k', 
            'fe', 'zn', 'cu', 'mn'
        }
        
        for gene in genes:
            if self._is_valid_gene_symbol_s3(gene):
                # Additional validation: must start with a letter
                if gene[0].isalpha():
                    valid_genes.append(gene.upper())
            else:
                # Track filtered metabolites for aggregated logging
                normalized = ''.join(c for c in gene if c.isalnum()).lower()
                if normalized in metabolite_list:
                    filtered_metabolites.append(gene)
        
        # Aggregate logging at DEBUG level
        if filtered_metabolites:
            unique_metabolites = sorted(set(filtered_metabolites))
            metabolite_display = ', '.join(unique_metabolites[:10])
            if len(unique_metabolites) > 10:
                metabolite_display += f" and {len(unique_metabolites) - 10} more"
            logger.debug(
                f"Filtered {len(filtered_metabolites)} metabolites from {len(genes)} candidates: {metabolite_display}"
            )
        
        return valid_genes

    async def _call_with_tracking(
        self,
        data_sources: Optional[Dict[str, DataSourceStatus]],
        source_name: str,
        coro,
        suppress_exception: bool = False,
    ):
        """Await an MCP coroutine and track success/failure automatically."""
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
