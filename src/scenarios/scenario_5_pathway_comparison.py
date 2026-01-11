"""
Scenario 5: Pathway Comparison and Validation

Cross-database pathway validation with overlap analysis and mechanistic details.
Based on Mature_development_plan.md Phase 1-5 and OmniTarget_Development_Plan.md.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Set, Tuple, Iterable
import networkx as nx
import numpy as np
from collections import defaultdict

from ..core.mcp_client_manager import MCPClientManager
from ..core.data_standardizer import DataStandardizer
from ..core.validation import DataValidator
from ..core.string_network_builder import AdaptiveStringNetworkBuilder
from ..utils.hpa_parsing import _iter_expr_items, categorize_expression
from ..models.data_models import (
    Pathway, Protein, Interaction, ExpressionProfile, 
    NetworkNode, NetworkEdge, DataSourceStatus, CompletenessMetrics
)
from ..models.simulation_models import PathwayComparisonResult

logger = logging.getLogger(__name__)

CROSSTALK_MODULES = [
    {
        'name': 'PI3K/AKT survival',
        'genes': {'PIK3CA', 'PIK3CB', 'PIK3R1', 'PIK3R2', 'AKT1', 'AKT2', 'AKT3', 'PTEN', 'MTOR'},
        'mechanism': 'Drives PI3K/AKT-mediated survival and endocrine resistance',
    },
    {
        'name': 'TGF-β / EMT',
        'genes': {'TGFB1', 'TGFB2', 'TGFBR1', 'TGFBR2', 'SMAD2', 'SMAD3', 'SMAD4', 'SNAI1', 'SNAI2', 'VIM'},
        'mechanism': 'Induces EMT and metastatic spread via TGF-β signalling',
    },
    {
        'name': 'MAPK compensation',
        'genes': {'MAPK1', 'MAPK3', 'MAPK14', 'RAF1', 'BRAF', 'KRAS', 'NRAS'},
        'mechanism': 'MAPK cascade reactivation that bypasses upstream inhibition',
    },
    {
        'name': 'DNA damage response',
        'genes': {'BRCA1', 'BRCA2', 'ATM', 'ATR', 'CHEK1', 'CHEK2', 'RAD51'},
        'mechanism': 'DNA repair rewiring that reduces apoptotic priming',
    },
    {
        'name': 'Immune/Checkpoint',
        'genes': {'PDCD1', 'CD274', 'CTLA4', 'HAVCR2', 'LAG3'},
        'mechanism': 'Immune evasive signalling reducing anti-tumour immunity',
    },
]

MULTI_RTK_GENES = {'AXL', 'MET', 'EGFR', 'ERBB2', 'FGFR1', 'PDGFRA', 'PDGFRB', 'IGF1R'}
PI3K_FEEDBACK_GENES = {'PIK3CA', 'PIK3CB', 'PIK3CD', 'PIK3R1', 'PIK3R2', 'AKT1', 'AKT2', 'AKT3', 'MTOR'}
TGF_EMT_GENES = {'TGFB1', 'TGFB2', 'TGFB3', 'TGFBR1', 'TGFBR2', 'SMAD2', 'SMAD3', 'SMAD4', 'SNAI1', 'SNAI2'}
IMMUNE_EVASION_GENES = {'PDCD1', 'CD274', 'CTLA4', 'LAG3', 'HAVCR2', 'TIGIT'}


class PathwayComparisonScenario:
    """
    Scenario 5: Pathway Comparison and Validation
    
    7-step workflow:
    1. Parallel search (KEGG + Reactome)
    2. ID mapping (KEGG convert_identifiers)
    3. Gene extraction
    4. Overlap analysis (Jaccard similarity ≥0.4)
    5. Mechanistic details (Reactome reactions)
    6. Interaction validation (STRING)
    7. Expression context (HPA)
    """
    
    def __init__(self, mcp_manager: MCPClientManager):
        """Initialize pathway comparison scenario."""
        self.mcp_manager = mcp_manager
        self.standardizer = DataStandardizer()
        self.validator = DataValidator()
        self._active_data_sources: Optional[Dict[str, DataSourceStatus]] = None
        self.string_builder = None  # Will be initialized when data_sources available
    
    async def execute(
        self,
        pathway_query: str,
        tissue_context: Optional[str] = None
    ) -> PathwayComparisonResult:
        """
        Execute complete pathway comparison workflow.

        Args:
            pathway_query: Pathway name or identifier
            tissue_context: Optional tissue context (accepted for compatibility with global_params, currently unused)

        Returns:
            PathwayComparisonResult with complete analysis
        """
        logger.info(f"Starting pathway comparison for: {pathway_query}")
        
        data_sources = {
            'kegg': DataSourceStatus(source_name='kegg', requested=0, successful=0, failed=0, success_rate=0.0, error_types=[]),
            'reactome': DataSourceStatus(source_name='reactome', requested=0, successful=0, failed=0, success_rate=0.0, error_types=[]),
            'string': DataSourceStatus(source_name='string', requested=0, successful=0, failed=0, success_rate=0.0, error_types=[]),
            'hpa': DataSourceStatus(source_name='hpa', requested=0, successful=0, failed=0, success_rate=0.0, error_types=[])
        }
        self._active_data_sources = data_sources
        
        try:
            # Step 1: Parallel search
            search_data = await self._step1_parallel_search(pathway_query)
            
            # Step 2: ID mapping
            mapping_data = await self._step2_id_mapping(
                search_data['kegg_pathways'],
                search_data['reactome_pathways']
            )
            
            # Step 3: Gene extraction
            gene_data = await self._step3_gene_extraction(
                search_data['kegg_pathways'],
                search_data['reactome_pathways']
            )
            
            # Step 4: Overlap analysis
            overlap_data = await self._step4_overlap_analysis(
                gene_data['kegg_genes'],
                gene_data['reactome_genes']
            )
            
            # Step 5: Mechanistic details
            mechanistic_data = await self._step5_mechanistic_details(
                search_data['reactome_pathways']
            )
            
            # Step 6: Interaction validation
            interaction_data = await self._step6_interaction_validation(
                gene_data['all_genes']
            )
            
            # Step 7: Expression context
            expression_data = await self._step7_expression_context(
                gene_data['all_genes']
            )
            
            # Build result with correct field mapping for simulation_models
            kegg_pathways = search_data.get('kegg_pathways', [])
            reactome_pathways = search_data.get('reactome_pathways', [])

            # Convert pathways to dictionaries
            kegg_pathways_dict = []
            for pathway in kegg_pathways:
                if hasattr(pathway, 'model_dump'):
                    kegg_pathways_dict.append(pathway.model_dump())
                else:
                    kegg_pathways_dict.append(pathway)

            reactome_pathways_dict = []
            for pathway in reactome_pathways:
                if hasattr(pathway, 'model_dump'):
                    reactome_pathways_dict.append(pathway.model_dump())
                else:
                    reactome_pathways_dict.append(pathway)

            # Extract data from various steps
            overlap_analysis = overlap_data.get('overlap_analysis', {})
            modules = overlap_analysis.get('functional_modules', [])
            resistance_mechanisms = overlap_analysis.get('resistance_mechanisms', [])
            mechanistic_details = mechanistic_data.get('reactions', [])
            raw_expression_profiles = expression_data.get('profiles', [])
            expression_profiles = []
            for profile in raw_expression_profiles[:20]:
                if hasattr(profile, 'model_dump'):
                    expression_profiles.append(profile.model_dump())
                elif isinstance(profile, dict):
                    expression_profiles.append(profile)
                else:
                    expression_profiles.append({'value': str(profile)})

            # Get consensus pathways (pathways that appear in both databases)
            kegg_ids = [p.get('id', '') for p in kegg_pathways_dict if isinstance(p, dict)]
            reactome_ids = [p.get('id', '') for p in reactome_pathways_dict if isinstance(p, dict)]
            consensus_pathways = list(set(kegg_ids) & set(reactome_ids))

            # Database-specific insights
            database_specific_insights = {
                'kegg_unique': [
                    p.get('id', '')
                    for p in kegg_pathways_dict
                    if isinstance(p, dict) and p.get('id', '') not in reactome_ids
                ],
                'reactome_unique': [
                    p.get('id', '')
                    for p in reactome_pathways_dict
                    if isinstance(p, dict) and p.get('id', '') not in kegg_ids
                ]
            }

            completeness_metrics = self._build_completeness_metrics(
                kegg_pathways_dict,
                reactome_pathways_dict,
                interaction_data,
                expression_data
            )

            # Calculate validation score
            validation_score = self._calculate_validation_score(
                search_data, mapping_data, gene_data, overlap_data,
                mechanistic_data, interaction_data, expression_data,
                data_sources, completeness_metrics
            )

            mechanistic_differences = modules + resistance_mechanisms
            if not mechanistic_differences:
                mechanistic_differences = self._summarize_reactome_themes(reactome_pathways_dict)

            result = PathwayComparisonResult(
                pathway_query=pathway_query,
                kegg_pathways=kegg_pathways_dict,
                reactome_pathways=reactome_pathways_dict,
                pathway_overlap=overlap_analysis,
                gene_overlap={
                    'common_genes': overlap_analysis.get('common_genes', []),
                    'kegg_unique_genes': overlap_analysis.get('kegg_unique_genes', []),
                    'reactome_unique_genes': overlap_analysis.get('reactome_unique_genes', []),
                    'jaccard_similarity': overlap_analysis.get('jaccard_similarity', 0.0),
                    'overlap_percentage': overlap_analysis.get('overlap_percentage', 0.0)
                },
                mechanistic_differences=mechanistic_differences,
                expression_context={
                    'expression_profiles': expression_profiles,
                    'coverage': expression_data.get('coverage', 0.0)
                },
                consensus_pathways=consensus_pathways,
                database_specific_insights=database_specific_insights,
                validation_score=validation_score,
                data_sources=list(data_sources.values()),
                completeness_metrics=completeness_metrics
            )

            logger.info(f"Pathway comparison completed. Validation score: {validation_score:.3f}")
            return result
        finally:
            self._active_data_sources = None
    
    def _extract_gene_from_query(self, query: str) -> Optional[str]:
        """
        Extract gene symbol from query string.
        
        Handles queries like:
        - "AXL pathway" → "AXL"
        - "AXL" → "AXL"
        - "breast cancer" → None (not a gene)
        
        Args:
            query: Query string that may contain a gene name
            
        Returns:
            Gene symbol if found, None otherwise
        """
        # Remove common pathway-related words
        query_clean = query.strip()
        for word in ['pathway', 'pathways', 'signaling', 'signalling', 'network', 'analysis']:
            query_clean = query_clean.replace(word, '').strip()
        
        # Check if remaining string looks like a gene symbol
        # Gene symbols are typically 2-15 uppercase alphanumeric characters
        query_clean = query_clean.split()[0] if query_clean.split() else query_clean
        if query_clean and query_clean.isalnum() and 2 <= len(query_clean) <= 15:
            # Check if it's uppercase (most gene symbols are)
            if query_clean.isupper() or (query_clean[0].isupper() and query_clean[1:].isalnum()):
                return query_clean.upper()
        
        return None
    
    async def _get_kegg_gene_id(self, gene_symbol: str) -> Optional[str]:
        """
        Convert gene symbol to KEGG gene ID.
        
        Args:
            gene_symbol: Human gene symbol (e.g., "AXL")
            
        Returns:
            KEGG gene ID (e.g., "hsa:91464") or None if not found
        """
        # Strategy 1: Try direct construction (e.g., "hsa:AXL")
        direct_id = f"hsa:{gene_symbol}"
        try:
            gene_info = await self._call_with_tracking(
                None,
                'kegg',
                self.mcp_manager.kegg.get_gene_info(direct_id)
            )
            if gene_info:
                return direct_id
        except Exception:
            pass
        
        # Strategy 2: Search for gene
        try:
            search_result = await self._call_with_tracking(
                None,
                'kegg',
                self.mcp_manager.kegg.search_genes(gene_symbol, limit=5)
            )
            if search_result.get('genes'):
                # Find human gene (hsa: prefix)
                for gene_id, description in search_result['genes'].items():
                    if gene_id.startswith('hsa:') and gene_symbol.upper() in description.upper():
                        return gene_id
        except Exception:
            pass
        
        return None
    
    def _classify_query_type(self, query: str) -> str:
        """
        Classify query type to optimize search strategy.

        Returns:
            'gene' - Gene-related query (e.g., "AXL pathway", "BRCA1 signaling")
            'disease' - Disease query (e.g., "breast cancer", "diabetes")
            'pathway' - Direct pathway name (e.g., "PI3K-Akt pathway")
        """
        query_lower = query.lower()

        # Check for gene pattern first (most specific)
        gene_symbol = self._extract_gene_from_query(query)
        if gene_symbol:
            logger.info(f"Query classified as 'gene' (extracted: {gene_symbol})")
            return 'gene'

        # Check for disease keywords
        disease_keywords = [
            'cancer', 'disease', 'syndrome', 'disorder', 'tumor', 'carcinoma',
            'leukemia', 'lymphoma', 'melanoma', 'sarcoma', 'adenocarcinoma',
            'diabetes', 'alzheimer', 'parkinson', 'hypertension', 'obesity'
        ]
        if any(keyword in query_lower for keyword in disease_keywords):
            logger.info(f"Query classified as 'disease' (matched keyword)")
            return 'disease'

        # Default to pathway
        logger.info(f"Query classified as 'pathway' (default)")
        return 'pathway'

    async def _step1_parallel_search(self, query: str) -> Dict[str, Any]:
        """
        Step 1: Parallel search WITH GENE EXTRACTION AND QUERY CLASSIFICATION.

        Search pathways in KEGG and Reactome databases.
        Uses query classification to avoid sequential failed searches.
        """
        logger.info("Step 1: Optimized pathway search with query classification")

        # Classify query to determine optimal search strategy
        query_type = self._classify_query_type(query)

        kegg_search: Dict[str, Any] = {'pathways': []}

        # Route to appropriate search strategy based on query type
        if query_type == 'gene':
            # Gene-based search (fastest for gene queries)
            logger.info("Using gene-based search strategy")
            gene_symbol = self._extract_gene_from_query(query)

            if gene_symbol:
                logger.info(f"Extracted gene symbol: {gene_symbol}")
                kegg_gene_id = await self._get_kegg_gene_id(gene_symbol)

                if kegg_gene_id:
                    logger.info(f"Found KEGG gene ID: {kegg_gene_id}")
                    try:
                        pathways_result = await self._call_with_tracking(
                            None,
                            'kegg',
                            self.mcp_manager.kegg.find_related_entries(
                                source_entries=[kegg_gene_id],
                                source_db="gene",
                                target_db="pathway"
                            )
                        )

                        # Extract pathways from links dict
                        pathway_ids = []
                        if isinstance(pathways_result, dict) and 'links' in pathways_result:
                            links = pathways_result['links']
                            if isinstance(links, dict):
                                gene_links = links.get(kegg_gene_id, [])
                                if isinstance(gene_links, list):
                                    pathway_ids = gene_links
                                elif isinstance(gene_links, str):
                                    pathway_ids = [gene_links]

                                # Try alternative format if needed
                                if not pathway_ids:
                                    alt_id = kegg_gene_id.replace('hsa:', '')
                                    if alt_id in links:
                                        gene_links = links[alt_id]
                                        if isinstance(gene_links, list):
                                            pathway_ids = gene_links
                                        elif isinstance(gene_links, str):
                                            pathway_ids = [gene_links]

                        if pathway_ids:
                            logger.info(f"✅ Gene-based discovery found {len(pathway_ids)} KEGG pathways")
                            kegg_search = {'pathways': pathway_ids}
                        else:
                            logger.warning(f"Gene-based discovery returned 0 pathways for {gene_symbol}")
                    except Exception as e:
                        logger.warning(f"Gene-based KEGG pathway discovery failed: {e}")

        elif query_type == 'disease':
            # Disease-based search
            logger.info("Using disease-based search strategy")
            try:
                disease_result = await self._call_with_tracking(
                    None,
                    'kegg',
                    self.mcp_manager.kegg.search_diseases(query, limit=10)
                )
                disease_pathways = []

                if disease_result.get('diseases'):
                    disease = disease_result['diseases'][0]
                    disease_pathways = disease.get('pathways', [])
                    logger.info(f"Found disease with {len(disease_pathways)} associated KEGG pathways")

                kegg_search = {'pathways': disease_pathways}

            except Exception as e:
                logger.info(f"Disease search failed: {e}, falling back to pathway search")
                kegg_search = await self._call_with_tracking(
                    None,
                    'kegg',
                    self.mcp_manager.kegg.search_pathways(query, limit=10)
                )

        else:  # query_type == 'pathway'
            # Direct pathway search
            logger.info("Using pathway-based search strategy")
            try:
                kegg_search = await self._call_with_tracking(
                    None,
                    'kegg',
                    self.mcp_manager.kegg.search_pathways(query, limit=10)
                )
            except Exception as e:
                logger.info(f"Pathway search failed: {e}")
                kegg_search = {'pathways': []}
        
        # Step 1b: Search Reactome (use find_pathways_by_disease like S1, S3, S6)
        reactome_search = {'pathways': []}
        try:
            reactome_search_result = await self._call_with_tracking(
                None,
                'reactome',
                self.mcp_manager.reactome.find_pathways_by_disease(query)
            )
            if reactome_search_result and isinstance(reactome_search_result, dict):
                reactome_search = reactome_search_result
        except Exception as e:
            logger.warning(f"Reactome disease search failed: {e}")
        
        # CRITICAL FIX: Gene-based Reactome fallback if disease search failed
        if not reactome_search.get('pathways'):
            logger.info("Reactome disease search failed; trying gene-based discovery")
            gene_symbol = self._extract_gene_from_query(query)
            
            if gene_symbol:
                logger.info(f"Trying Reactome find_pathways_by_gene for: {gene_symbol}")
                try:
                    reactome_gene_result = await self._call_with_tracking(
                        None,
                        'reactome',
                        self.mcp_manager.reactome.find_pathways_by_gene(gene_symbol)
                    )
                    if reactome_gene_result and isinstance(reactome_gene_result, dict):
                        reactome_search = reactome_gene_result
                        if reactome_search.get('pathways'):
                            logger.info(f"✅ Gene-based Reactome discovery found {len(reactome_search['pathways'])} pathways")
                except Exception as e:
                    logger.warning(f"Reactome gene-based search failed: {e}")
        
        # Step 1b: For each KEGG pathway, fetch genes (CRITICAL FIX)
        standardized_kegg = []
        if kegg_search.get('pathways'):
            logger.info(f"Found {len(kegg_search['pathways'])} KEGG pathways, fetching genes...")
            standardized_kegg.extend(
                await self._fetch_kegg_pathways(kegg_search['pathways'])
            )

        if not standardized_kegg:
            logger.warning(
                "No KEGG pathways returned for '%s' – skipping KEGG overlap analysis.",
                query
            )
        
        # Step 1c: For each Reactome pathway, fetch genes (ENHANCED - same logic as S1)
        standardized_reactome = []
        if reactome_search.get('pathways'):
            logger.info(f"Found {len(reactome_search['pathways'])} Reactome pathways, fetching genes...")
            for pathway_data in reactome_search['pathways']:
                pathway_id = pathway_data.get('stId') or pathway_data.get('stableIdentifier') or pathway_data.get('id')
                species_name = (pathway_data.get('speciesName') or pathway_data.get('species') or '').lower()
                # Keep Homo sapiens pathways only (Reactome human prefix R-HSA)
                is_human_id = isinstance(pathway_id, str) and pathway_id.startswith('R-HSA-')
                is_human_species = 'homo sapiens' in species_name or species_name == 'human'
                if not (is_human_id or is_human_species):
                    logger.debug(
                        "Skipping non-human Reactome pathway %s (%s)",
                        pathway_id,
                        species_name or 'unknown species'
                    )
                    continue
                pathway_id = pathway_data.get('stId') or pathway_data.get('id')
                
                if pathway_id:
                    if not str(pathway_id).startswith('R-HSA-'):
                        logger.debug(f"Skipping non-human Reactome pathway {pathway_id}")
                        continue
                    try:
                        # Use enhanced gene extraction (same as S1)
                        genes = await self._extract_reactome_genes(pathway_id)
                        
                        # Get pathway name from pathway_data or pathway_details
                        pathway_name = pathway_data.get('displayName') or pathway_data.get('name') or pathway_id
                        if pathway_name == pathway_id:
                            # Try to get display name from pathway details
                            try:
                                pathway_details = await self._call_with_tracking(
                                    None,
                                    'reactome',
                                    self.mcp_manager.reactome.get_pathway_details(pathway_id)
                                )
                                pathway_name = pathway_details.get('displayName') or pathway_details.get('name') or pathway_id
                            except:
                                pass
                        
                        # Get description from pathway_data or pathway_details
                        description = pathway_data.get('description') or pathway_data.get('summation')
                        if not description:
                            try:
                                pathway_details = await self._call_with_tracking(
                                    None,
                                    'reactome',
                                    self.mcp_manager.reactome.get_pathway_details(pathway_id)
                                )
                                if pathway_details.get('summation'):
                                    summation = pathway_details['summation']
                                    if isinstance(summation, list) and summation:
                                        description = summation[0].get('text') if isinstance(summation[0], dict) else str(summation[0])
                                    elif isinstance(summation, dict):
                                        description = summation.get('text')
                            except:
                                pass
                        
                        # Create pathway with genes
                        pathway = Pathway(
                            id=pathway_id,
                            name=pathway_name,
                            source_db='reactome',
                            genes=genes,
                            description=description,
                            confidence=0.9
                        )
                        standardized_reactome.append(pathway)
                        
                        logger.info(f"✅ Reactome pathway {pathway_id}: Retrieved {len(genes)} genes")
                        
                    except Exception as e:
                        logger.warning(f"Failed to fetch Reactome pathway {pathway_id}: {e}")
                        continue
        
        logger.info(f"Step 1 complete: {len(standardized_kegg)} KEGG pathways, {len(standardized_reactome)} Reactome pathways")
        
        return {
            'kegg_pathways': standardized_kegg,
            'reactome_pathways': standardized_reactome
        }

    async def _fetch_kegg_pathways(self, pathway_items: Iterable[Any]) -> List[Pathway]:
        """Fetch KEGG pathway definitions and gene members for the given IDs."""
        fetched: List[Pathway] = []
        for item in pathway_items:
            if isinstance(item, str):
                raw_id = item
            elif isinstance(item, dict):
                raw_id = item.get('id') or item.get('entry_id') or item.get('pathway_id')
            else:
                logger.debug(f"Unsupported KEGG pathway item type: {type(item)}")
                continue

            kegg_id = self._normalize_kegg_pathway_id(raw_id)
            if not kegg_id:
                logger.debug(f"Unable to normalize KEGG pathway id from '{raw_id}'")
                continue

            try:
                pathway_info, pathway_genes_data = await asyncio.gather(
                    self._call_with_tracking(
                        None,
                        'kegg',
                        self.mcp_manager.kegg.get_pathway_info(kegg_id)
                    ),
                    self._call_with_tracking(
                        None,
                        'kegg',
                        self.mcp_manager.kegg.get_pathway_genes(kegg_id)
                    )
                )
            except Exception as exc:
                logger.warning(f"Failed to fetch KEGG pathway {raw_id}: {exc}")
                continue

            genes: List[str] = []
            if isinstance(pathway_genes_data, dict) and pathway_genes_data.get('genes'):
                raw_genes = pathway_genes_data['genes']
                if isinstance(raw_genes, list):
                    genes = [gene.get('name') for gene in raw_genes if isinstance(gene, dict) and gene.get('name')]
                elif isinstance(raw_genes, dict):
                    genes = list(raw_genes.keys())
            genes = list(self._filter_valid_gene_symbols(set(genes)))

            pathway = Pathway(
                id=kegg_id,
                name=pathway_info.get('name', kegg_id) if isinstance(pathway_info, dict) else kegg_id,
                source_db='kegg',
                genes=genes,
                description=(pathway_info.get('description') if isinstance(pathway_info, dict) else None),
                confidence=0.9
            )
            fetched.append(pathway)
            logger.info(f"✅ KEGG pathway {kegg_id}: Retrieved {len(genes)} genes")

        return fetched

    def _normalize_kegg_pathway_id(self, pathway_id: Optional[str]) -> Optional[str]:
        """Normalize KEGG IDs (path:map05224, map05200, 5200) to hsaXXXXX format."""
        if not pathway_id:
            return None
        pid = pathway_id.strip()
        if ':' in pid:
            pid = pid.split(':', 1)[-1]
        pid = pid.replace(' ', '')
        if pid.startswith('map') and len(pid) > 3 and pid[3:].isdigit():
            pid = f"hsa{pid[3:]}"
        elif pid.isdigit():
            pid = f"hsa{pid}"
        if not pid:
            return None
        return pid
    
    async def _step2_id_mapping(
        self, 
        kegg_pathways: List[Pathway], 
        reactome_pathways: List[Pathway]
    ) -> Dict[str, Any]:
        """
        Step 2: ID mapping.
        
        Map identifiers using KEGG convert_identifiers.
        """
        logger.info("Step 2: ID mapping")
        
        mapping_results = []
        
        # Map genes from each pathway
        for pathway in kegg_pathways + reactome_pathways:
            for gene in pathway.genes:
                try:
                    # Use KEGG convert_identifiers
                    mapping = await self._call_with_tracking(
                        None,
                        'kegg',
                        self.mcp_manager.kegg.convert_identifiers(
                            ids=[gene],
                            source_db='kegg',
                            target_db='uniprot'
                        )
                    )
                    
                    if mapping.get('mappings'):
                        mapping_results.append({
                            'original_id': gene,
                            'pathway_id': pathway.id,
                            'mappings': mapping['mappings']
                        })
                
                except Exception as e:
                    logger.warning(f"ID mapping failed for {gene}: {e}")
                    continue
        
        # Calculate mapping success rate
        total_genes = sum(len(p.genes) for p in kegg_pathways + reactome_pathways)
        mapped_genes = len(mapping_results)
        mapping_success_rate = mapped_genes / total_genes if total_genes > 0 else 0.0
        
        return {
            'mapping_results': mapping_results,
            'success_rate': mapping_success_rate
        }
    
    async def _step3_gene_extraction(
        self, 
        kegg_pathways: List[Pathway], 
        reactome_pathways: List[Pathway]
    ) -> Dict[str, Any]:
        """
        Step 3: Gene extraction.
        
        Extract genes from pathways.
        """
        logger.info("Step 3: Gene extraction")
        
        # Extract genes from KEGG pathways
        kegg_genes = set()
        for pathway in kegg_pathways:
            kegg_genes.update(pathway.genes)
        
        # Extract genes from Reactome pathways
        reactome_genes = set()
        for pathway in reactome_pathways:
            reactome_genes.update(pathway.genes)
        
        # Combine all genes
        all_genes = kegg_genes | reactome_genes
        
        return {
            'kegg_genes': list(kegg_genes),
            'reactome_genes': list(reactome_genes),
            'all_genes': list(all_genes)
        }
    
    async def _step4_overlap_analysis(
        self, 
        kegg_genes: List[str], 
        reactome_genes: List[str]
    ) -> Dict[str, Any]:
        """
        Step 4: Overlap analysis.
        
        Calculate Jaccard similarity and other overlap metrics.
        """
        logger.info("Step 4: Overlap analysis")
        
        kegg_set = set(g.upper() for g in kegg_genes if g)
        reactome_set = set(g.upper() for g in reactome_genes if g)
        
        intersection = kegg_set & reactome_set
        union = kegg_set | reactome_set
        kegg_unique = kegg_set - reactome_set
        reactome_unique = reactome_set - kegg_set
        
        jaccard_similarity = len(intersection) / len(union) if union else 0.0
        overlap_percentage = len(intersection) / len(kegg_set) if kegg_set else 0.0
        meets_criteria = jaccard_similarity >= 0.4

        modules, resistance_mechanisms = self._analyze_crosstalk(intersection, union)
        
        overlap_detail = {
            'kegg_genes': len(kegg_set),
            'reactome_genes': len(reactome_set),
            'common_genes': sorted(intersection),
            'kegg_unique_genes': sorted(kegg_unique),
            'reactome_unique_genes': sorted(reactome_unique),
            'jaccard_similarity': jaccard_similarity,
            'overlap_percentage': overlap_percentage,
            'functional_modules': modules,
            'resistance_mechanisms': resistance_mechanisms
        }
        
        return {
            'jaccard_similarity': jaccard_similarity,
            'overlap_percentage': overlap_percentage,
            'intersection_size': len(intersection),
            'union_size': len(union),
            'meets_criteria': meets_criteria,
            'overlap_analysis': overlap_detail,
            'common_genes': overlap_detail['common_genes'],
            'kegg_unique_genes': overlap_detail['kegg_unique_genes'],
            'reactome_unique_genes': overlap_detail['reactome_unique_genes'],
            'functional_modules': modules,
            'resistance_mechanisms': resistance_mechanisms
        }
    
    async def _step5_mechanistic_details(self, reactome_pathways: List[Pathway]) -> Dict[str, Any]:
        """
        Step 5: Mechanistic details.
        
        Get Reactome reactions for mechanistic analysis.
        """
        logger.info("Step 5: Mechanistic details")
        
        all_reactions = []
        
        # Get reactions for each Reactome pathway
        for pathway in reactome_pathways:
            try:
                reactions = await self._call_with_tracking(
                    None,
                    'reactome',
                    self.mcp_manager.reactome.get_pathway_reactions(
                        pathway.id
                    )
                )
                
                if reactions.get('reactions'):
                    for reaction_data in reactions['reactions']:
                        reaction = await self.standardizer.standardize_reactome_reaction(
                            reaction_data
                        )
                        if reaction:  # Only append if reaction is not None
                            all_reactions.append(reaction)
                
            except Exception as e:
                logger.warning(f"Failed to get reactions for pathway {pathway.id}: {e}")
                continue
        
        # Calculate mechanistic complexity
        complexity_metrics = self._calculate_mechanistic_complexity(all_reactions)
        
        return {
            'reactions': all_reactions,
            'complexity_metrics': complexity_metrics
        }
    
    async def _step6_interaction_validation(self, genes: List[str]) -> Dict[str, Any]:
        """
        Step 6: Interaction validation.
        
        Validate interactions using STRING.
        """
        logger.info("Step 6: Interaction validation")
        
        if not genes:
            return {'validation': {}, 'confidence_score': 0.0}
        
        # Initialize adaptive STRING builder if not already done
        if self.string_builder is None:
            self.string_builder = AdaptiveStringNetworkBuilder(
                self.mcp_manager,
                self._active_data_sources
            )
        
        # Build network using adaptive builder
        try:
            network_result = await self.string_builder.build_network(
                genes=genes,
                priority_genes=None,
                data_sources=self._active_data_sources
            )
            
            nodes = network_result.get('nodes', [])
            edges = network_result.get('edges', [])
            
            # S5 expects nested structure
            string_network = {
                'network': {
                    'edges': edges
                }
            }
            interactions = edges
        except Exception as e:
            logger.warning(f"STRING network construction failed: {e}")
            interactions = []
        
        # Calculate interaction confidence
        # STRING returns 'confidence_score', but some code expects 'score'
        confidence_scores = [
            edge.get('confidence_score', edge.get('score', 0)) 
            for edge in interactions
        ]
        median_confidence = np.median(confidence_scores) if confidence_scores else 0
        
        # Calculate network density
        network_density = len(interactions) / (len(genes) * (len(genes) - 1) / 2) if len(genes) > 1 else 0
        
        return {
            'validation': {
                'interactions': interactions,
                'network_density': network_density,
                'median_confidence': median_confidence
            },
            'confidence_score': median_confidence / 1000.0
        }
    
    async def _step7_expression_context(self, genes: List[str]) -> Dict[str, Any]:
        """
        Step 7: Expression context.
        
        Get expression context using HPA.
        """
        logger.info("Step 7: Expression context")
        
        expression_profiles = []
        
        # Get expression for each gene
        for gene in genes[:20]:  # Limit for performance
            try:
                expression_data = await self._call_with_tracking(
                    None,
                    'hpa',
                    self.mcp_manager.hpa.get_tissue_expression(gene)
                )
                
                # Use helper to parse HPA expression (handles list/dict formats)
                for tissue, ntpms in _iter_expr_items(expression_data):
                        expression_profile = ExpressionProfile(
                            gene=gene,
                            tissue=tissue,
                        expression_level=categorize_expression(ntpms),
                            reliability='Approved',
                            cell_type_specific=False,
                            subcellular_location=[]
                        )
                        expression_profiles.append(expression_profile)
                
            except Exception as e:
                logger.warning(f"Failed to get expression for {gene}: {e}")
                continue
        
        # Calculate expression coverage
        covered_genes = set(ep.gene for ep in expression_profiles)
        coverage = len(covered_genes) / len(genes) if genes else 0.0
        
        return {
            'profiles': expression_profiles,
            'coverage': coverage
        }

    def _analyze_crosstalk(
        self,
        shared_genes: Set[str],
        union_genes: Set[str]
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Identify crosstalk modules and resistance implications."""
        shared_upper = {gene.upper() for gene in shared_genes}
        union_upper = {gene.upper() for gene in union_genes}

        modules = []
        for module in CROSSTALK_MODULES:
            hits = sorted(g for g in shared_upper if g in module['genes'])
            if hits:
                modules.append({
                    'module': module['name'],
                    'genes': hits,
                    'mechanism': module['mechanism']
                })

        resistance: List[Dict[str, Any]] = []
        multi_rtk_hits = sorted(MULTI_RTK_GENES & union_upper)
        if len(multi_rtk_hits) >= 2:
            resistance.append({
                'name': 'Multi-RTK redundancy',
                'genes': multi_rtk_hits,
                'implication': 'Parallel RTK signalling can bypass single-agent inhibition'
            })

        pi3k_hits = sorted(PI3K_FEEDBACK_GENES & union_upper)
        if len(pi3k_hits) >= 2:
            resistance.append({
                'name': 'PI3K/AKT feedback',
                'genes': pi3k_hits,
                'implication': 'PI3K-AKT axis can reactivate survival pathways after inhibitor withdrawal'
            })

        tgf_hits = sorted(TGF_EMT_GENES & union_upper)
        if len(tgf_hits) >= 2:
            resistance.append({
                'name': 'TGF-β driven EMT',
                'genes': tgf_hits,
                'implication': 'TGF-β signalling promotes EMT and metastatic escape'
            })

        immune_hits = sorted(IMMUNE_EVASION_GENES & union_upper)
        if len(immune_hits) >= 2:
            resistance.append({
                'name': 'Immune evasion signalling',
                'genes': immune_hits,
                'implication': 'Checkpoint upregulation can blunt anti-tumour immunity'
            })

        return modules, resistance

    def _summarize_reactome_themes(self, reactome_pathways: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Provide simple thematic summaries for Reactome pathways."""
        summaries = []
        for pathway in reactome_pathways[:5]:
            if isinstance(pathway, dict):
                pathway_id = pathway.get('id')
                pathway_name = pathway.get('name') or pathway_id
                gene_count = len(pathway.get('genes', []))
            else:
                pathway_id = str(pathway)
                pathway_name = str(pathway)
                gene_count = 0
            summaries.append({
                'pathway_id': pathway_id,
                'pathway_name': pathway_name,
                'theme': self._infer_pathway_theme(pathway_name or ''),
                'gene_count': gene_count,
            })
        return summaries

    def _infer_pathway_theme(self, name: str) -> str:
        """Infer mechanistic theme from pathway name."""
        lowered = name.lower()
        if 'pi3k' in lowered or 'akt' in lowered:
            return 'PI3K/AKT survival'
        if 'tgf' in lowered or 'smad' in lowered:
            return 'TGF-β signalling / EMT'
        if 'wnt' in lowered:
            return 'WNT stemness / plasticity'
        if 'mapk' in lowered or 'erk' in lowered:
            return 'MAPK proliferation cascade'
        if 'immune' in lowered or 'checkpoint' in lowered:
            return 'Immune evasion'
        if 'dna' in lowered or 'repair' in lowered:
            return 'DNA damage response'
        return 'General signalling'
    
    async def _extract_reactome_genes(self, pathway_id: str) -> List[str]:
        """
        Extract genes from Reactome pathway using validated MCP methods.
        
        ENHANCED: Based on S1's successful implementation with comprehensive filtering.
        Focuses on actual gene/protein names, not generic pathway terms.
        
        Args:
            pathway_id: Reactome pathway stable identifier (e.g., R-HSA-1227990)
            
        Returns:
            List of validated gene symbols
        """
        genes = set()
        
        try:
            # Method 1: Get pathway details and extract from entities
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
            
            # Method 2: Backup with get_pathway_participants if needed
            if not genes:
                participants = await self._call_with_tracking(
                    None,
                    'reactome',
                    self.mcp_manager.reactome.get_pathway_participants(pathway_id)
                )
                if participants.get('participants'):
                    for participant in participants['participants']:
                        gene_names = self._extract_gene_names_from_entity(participant)
                        genes.update(gene_names)
            
            # Filter out non-gene terms and validate
            filtered_genes = self._filter_valid_gene_symbols(genes)
            
            logger.info(f"✅ Reactome pathway {pathway_id}: Extracted {len(filtered_genes)} valid gene symbols (from {len(genes)} candidates)")
            
        except Exception as e:
            logger.warning(f"Failed to extract genes from Reactome pathway {pathway_id}: {e}")
            return []
        
        return list(filtered_genes)

    def _build_completeness_metrics(
        self,
        kegg_pathways: List[Dict[str, Any]],
        reactome_pathways: List[Dict[str, Any]],
        interaction_data: Dict[str, Any],
        expression_data: Dict[str, Any],
    ) -> CompletenessMetrics:
        """Construct completeness metrics for Scenario 5 outputs."""
        expression_comp = expression_data.get('coverage')
        if expression_comp is None:
            profile_count = len(expression_data.get('profiles', []))
            expression_comp = min(1.0, profile_count / 20.0) if profile_count else 0.0
        expression_comp = max(0.0, min(1.0, expression_comp))

        network_density = interaction_data.get('validation', {}).get('network_density')
        if network_density is None:
            interaction_count = len(interaction_data.get('interactions', []))
            network_density = min(1.0, interaction_count / 200.0) if interaction_count else 0.0
        network_comp = max(0.0, min(1.0, network_density))

        total_pathways = len(kegg_pathways) + len(reactome_pathways)
        pathway_comp = min(1.0, total_pathways / 20.0) if total_pathways else 0.0

        values = [metric for metric in (expression_comp, network_comp, pathway_comp) if metric is not None]
        overall = sum(values) / len(values) if values else 0.0

        return CompletenessMetrics(
            expression_data=expression_comp,
            network_data=network_comp,
            pathway_data=pathway_comp,
            drug_data=0.0,
            pathology_data=None,
            overall_completeness=overall
        )
    
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
        name_fields = ['name', 'displayName', 'geneName', 'symbol', 'identifier']
        
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
        - Length 2-15 characters (typical gene symbol length)
        - Starts with uppercase letter (standard nomenclature)
        - Contains mostly uppercase letters and numbers
        - Not a common English word or generic term
        - Mostly alphanumeric (allows hyphens in names like HLA-DRB1)
        
        Args:
            symbol: Candidate gene symbol
            
        Returns:
            True if symbol passes validation criteria
        """
        if not symbol or len(symbol) < 2 or len(symbol) > 15:
            return False
        
        # Must start with uppercase letter (standard gene nomenclature)
        if not symbol[0].isupper():
            return False
        
        # Exclude generic terms, common words, and biological process terms (case-insensitive)
        # Updated 2025-11-20 based on pipeline analysis
        generic_terms = {
            # Generic biological terms
            'constitutive', 'signaling', 'drug', 'pathway', 'cancer', 'disease',
            'protein', 'complex', 'molecule', 'reaction', 'process', 'pathway',
            'regulation', 'activation', 'inhibition', 'binding', 'transport',
            'kinase', 'receptor', 'factor', 'domain', 'site', 'motif',
            'phosphorylation', 'ubiquitination', 'methylation', 'acetylation',
            'nucleus', 'cytoplasm', 'membrane', 'mitochondrial', 'nuclear',
            # Common English words that appear in pathway descriptions
            'mutants', 'activated', 'promotes', 'recruits', 'anchoring', 
            'does', 'recruit', 'function', 'aberrant', 'loss', 'resistance',
            'ligand', 'toxins', 'catalyzes', 'enhances', 'prevents', 'inhibits',
            # Common abbreviations that aren't genes
            'tn', 'hd', 'lbd', 'ecd', 'kd', 'nm', 'cm', 'mm', 'pm', 'an',
            'tkis', 'pest', 'sara', 'axin', 'hr', 'egfri', 'mkis',
            # Prepositions and articles
            'in', 'of', 'by', 'to', 'for', 'with', 'from', 'at', 'on',
            'the', 'and', 'or', 'not', 'but', 'is', 'are', 'was', 'were',
            # Chemical/modification terms
            'phospho', 'acetyl', 'methyl', 'ubiq', 'sumo',
            # Drug names (common inhibitors)
            'xav939', 'lgk974', 'az5104',
            
            # Specific drug names (NEW - from Phase 1 investigation)
            'aee788', 'aee78',  # EGFR/HER2 inhibitor drugs
            # Metabolites & nucleotides
            'atp', 'gdp', 'gtp', 'dna', 'rna', 'amp', 'adp', 'nad', 'nadh',
            'fad', 'coa', 'gmp', 'ump', 'cmp', 'tmp',
            'pi', 'pip', 'pip2', 'pip3', 'dag', 'ip3',
            'ca', 'mg', 'zn', 'fe', 'cu', 'mn',
            # Protein family abbreviations (common in KEGG diagrams)
            'pkc', 'pka', 'pkg', 'prkg', 'plc', 'pld', 'pla', 'pla2',
            'pde', 'pde6', 'fzd', 'wnt', 'cam', 'calm', 'nfat', 'nos', 'mapk',
            # Incomplete gene symbols
            'dkk', 'ppp2r',
            # Single letters (NEW - from pipeline analysis)
            # These appear in Reactome pathways like "Signaling by WNT in cancer" → ['P', 'I', 'A']
            'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm',
            'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z',
        }
        
        # Also check for mutation patterns (e.g., E17K, V600E)
        # These have format: single letter + numbers + single letter
        if (len(symbol) <= 5 and 
            symbol[0].isalpha() and 
            any(c.isdigit() for c in symbol[1:-1]) and 
            symbol[-1].isalpha()):
            return False  # Likely a mutation notation
        
        if symbol.lower() in generic_terms:
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
    
    def _calculate_mechanistic_complexity(self, reactions: List[Any]) -> Dict[str, Any]:
        """Calculate mechanistic complexity metrics."""
        if not reactions:
            return {'total_reactions': 0, 'complexity_score': 0.0}
        
        total_reactions = len(reactions)
        
        # Calculate complexity score based on reaction types
        complexity_score = 0.0
        for reaction in reactions:
            # Simplified complexity scoring
            if hasattr(reaction, 'reaction_type'):
                if reaction.reaction_type == 'catalysis':
                    complexity_score += 1.0
                elif reaction.reaction_type == 'binding':
                    complexity_score += 0.5
                else:
                    complexity_score += 0.3
        
        return {
            'total_reactions': total_reactions,
            'complexity_score': complexity_score,
            'average_complexity': complexity_score / total_reactions if total_reactions > 0 else 0.0
        }

    def _track_data_source(
        self,
        data_sources: Dict[str, DataSourceStatus],
        source_name: str,
        success: bool = True,
        error_type: Optional[str] = None
    ) -> Optional[DataSourceStatus]:
        """Track MCP data source usage."""
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
        """Await an MCP coroutine and update tracking counters automatically."""
        sources = data_sources or self._active_data_sources
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
    
    def _calculate_validation_score(
        self,
        search_data: Dict,
        mapping_data: Dict,
        gene_data: Dict,
        overlap_data: Dict,
        mechanistic_data: Dict,
        interaction_data: Dict,
        expression_data: Dict,
        data_sources: Dict,
        completeness_metrics: CompletenessMetrics
    ) -> float:
        """Calculate overall validation score with data completeness penalties."""
        scores = {}

        # Pathway search success
        scores['search_success'] = 1.0 if (search_data['kegg_pathways'] or search_data['reactome_pathways']) else 0.0

        # ID mapping success rate
        scores['mapping_success'] = mapping_data.get('success_rate', 0.0)

        # Gene overlap (Jaccard similarity)
        scores['gene_overlap'] = overlap_data.get('jaccard_similarity', 0.0)

        # Mechanistic complexity
        scores['mechanistic_complexity'] = mechanistic_data.get('complexity_metrics', {}).get('average_complexity', 0.0)

        # Interaction confidence
        scores['interaction_confidence'] = interaction_data.get('confidence_score', 0.0)

        # Expression coverage
        scores['expression_coverage'] = expression_data.get('coverage', 0.0)

        # Calculate overall score with data source penalties
        validation_result = self.validator.calculate_overall_validation_score(
            scores,
            data_sources=list(data_sources.values()),
            completeness_metrics=completeness_metrics
        )

        # Extract the final score (float) from the result dictionary
        if isinstance(validation_result, dict):
            return validation_result.get('final_score', 0.0)
        else:
            # Backward compatibility - in case validator returns float directly
            return validation_result
