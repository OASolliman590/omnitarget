"""
Data Standardizer

Normalizes MCP outputs to unified data models with cross-database integration.
"""

import logging
from typing import Dict, List, Any, Optional
from .exceptions import DataValidationError, format_error_for_logging
from ..models.data_models import (
    Disease, Pathway, Protein, Interaction, ExpressionProfile,
    CancerMarker, DrugInfo, DrugTarget, NetworkNode, NetworkEdge,
    Compound, Bioactivity, TargetBioactivity, DrugLikenessAssessment, MolecularDescriptors
)

logger = logging.getLogger(__name__)


class DataStandardizer:
    """Standardizes MCP outputs to unified data models."""
    
    def __init__(self):
        """Initialize data standardizer."""
        self.expression_level_mapping = {
            'Not detected': 0.0,
            'Low': 0.3,
            'Medium': 0.6,
            'High': 1.0
        }
    
    def standardize_kegg_disease(self, raw: Dict[str, Any]) -> Disease:
        """Standardize KEGG disease data."""
        try:
            return Disease(
                id=raw.get('id', ''),
                name=raw.get('name', ''),
                source_db='kegg',
                pathways=raw.get('pathways', []),
                confidence=raw.get('confidence', 0.8),
                description=raw.get('description'),
                category=raw.get('category')
            )
        except (KeyError, TypeError, ValueError) as e:
            # Catch validation errors during model creation
            logger.error(
                "Failed to standardize KEGG disease - validation error",
                extra={
                    **format_error_for_logging(e),
                    'raw_data_keys': list(raw.keys()) if isinstance(raw, dict) else 'not_dict',
                    'field': str(e).split("'")[1] if "'" in str(e) else 'unknown'
                }
            )
            raise DataValidationError(
                message=f"Invalid disease data structure: {e}",
                field='disease_data',
                value=type(raw).__name__,
                expected='Dict with id, name fields'
            )
        except Exception as e:
            logger.error(
                "Failed to standardize KEGG disease - unexpected error",
                extra=format_error_for_logging(e)
            )
            raise
    
    def standardize_reactome_disease(self, raw: Dict[str, Any]) -> Disease:
        """Standardize Reactome disease data."""
        try:
            return Disease(
                id=raw.get('id', ''),
                name=raw.get('name', ''),
                source_db='reactome',
                pathways=raw.get('pathways', []),
                confidence=raw.get('confidence', 0.7),
                description=raw.get('description'),
                category=raw.get('category')
            )
        except (KeyError, TypeError, ValueError) as e:
            # Catch validation errors during model creation
            logger.error(
                "Failed to standardize Reactome disease - validation error",
                extra={
                    **format_error_for_logging(e),
                    'raw_data_keys': list(raw.keys()) if isinstance(raw, dict) else 'not_dict',
                    'field': str(e).split("'")[1] if "'" in str(e) else 'unknown'
                }
            )
            raise DataValidationError(
                message=f"Invalid disease data structure: {e}",
                field='disease_data',
                value=type(raw).__name__,
                expected='Dict with id, name fields'
            )
        except Exception as e:
            logger.error(
                "Failed to standardize Reactome disease - unexpected error",
                extra=format_error_for_logging(e)
            )
            raise
    
    def standardize_kegg_pathway(self, raw: Dict[str, Any]) -> Pathway:
        """Standardize KEGG pathway data."""
        try:
            # Handle case where raw is a string instead of dict
            if isinstance(raw, str):
                return Pathway(
                    id=raw,
                    name=raw,
                    source_db='kegg',
                    genes=[],
                    hierarchy_level=None,
                    parent_pathway=None,
                    description=None,
                    confidence=0.5
                )
            
            return Pathway(
                id=raw.get('id', ''),
                name=raw.get('name', ''),
                source_db='kegg',
                genes=raw.get('genes', []),
                hierarchy_level=None,  # KEGG doesn't have hierarchy
                parent_pathway=None,
                description=raw.get('description'),
                confidence=raw.get('confidence', 0.9)
            )
        except (KeyError, TypeError, ValueError) as e:
            # Catch validation errors during model creation
            logger.error(
                "Failed to standardize KEGG pathway - validation error",
                extra={
                    **format_error_for_logging(e),
                    'raw_data_keys': list(raw.keys()) if isinstance(raw, dict) else 'not_dict',
                    'field': str(e).split("'")[1] if "'" in str(e) else 'unknown'
                }
            )
            raise DataValidationError(
                message=f"Invalid pathway data structure: {e}",
                field='pathway_data',
                value=type(raw).__name__,
                expected='Dict with id, name fields'
            )
        except Exception as e:
            logger.error(
                "Failed to standardize KEGG pathway - unexpected error",
                extra=format_error_for_logging(e)
            )
            raise
    
    def standardize_reactome_pathway(self, raw: Dict[str, Any]) -> Pathway:
        """Standardize Reactome pathway data."""
        try:
            # Handle name as list (Reactome returns name as list)
            name = raw.get('name', '')
            if isinstance(name, list) and len(name) > 0:
                name = name[0]  # Take first name from list
            elif not isinstance(name, str):
                name = str(name)
            
            return Pathway(
                id=raw.get('id', ''),
                name=name,
                source_db='reactome',
                genes=raw.get('genes', []),
                hierarchy_level=raw.get('hierarchy_level'),
                parent_pathway=raw.get('parent_pathway'),
                description=raw.get('description'),
                confidence=raw.get('confidence', 0.8)
            )
        except (KeyError, TypeError, ValueError) as e:
            # Catch validation errors during model creation
            logger.error(
                "Failed to standardize Reactome pathway - validation error",
                extra={
                    **format_error_for_logging(e),
                    'raw_data_keys': list(raw.keys()) if isinstance(raw, dict) else 'not_dict',
                    'field': str(e).split("'")[1] if "'" in str(e) else 'unknown'
                }
            )
            raise DataValidationError(
                message=f"Invalid pathway data structure: {e}",
                field='pathway_data',
                value=type(raw).__name__,
                expected='Dict with id, name fields'
            )
        except Exception as e:
            logger.error(
                "Failed to standardize Reactome pathway - unexpected error",
                extra=format_error_for_logging(e)
            )
            raise
    
    def standardize_string_interaction(self, raw: Dict[str, Any]) -> Interaction:
        """Standardize STRING interaction data."""
        try:
            return Interaction(
                protein_a=raw.get('protein_a', ''),
                protein_b=raw.get('protein_b', ''),
                combined_score=raw.get('combined_score', 0.0) / 1000.0,  # Convert to 0-1 scale
                evidence_types=raw.get('evidence_types', {}),
                pathway_context=raw.get('pathway_context'),
                interaction_type=raw.get('interaction_type'),
                source_database='string'
            )
        except (KeyError, TypeError, ValueError) as e:
            # Catch validation errors during model creation
            logger.error(
                "Failed to standardize STRING interaction - validation error",
                extra={
                    **format_error_for_logging(e),
                    'raw_data_keys': list(raw.keys()) if isinstance(raw, dict) else 'not_dict',
                    'field': str(e).split("'")[1] if "'" in str(e) else 'unknown'
                }
            )
            raise DataValidationError(
                message=f"Invalid interaction data structure: {e}",
                field='interaction_data',
                value=type(raw).__name__,
                expected='Dict with protein_a, protein_b fields'
            )
        except Exception as e:
            logger.error(
                "Failed to standardize STRING interaction - unexpected error",
                extra=format_error_for_logging(e)
            )
            raise
    
    def standardize_reactome_interaction(self, raw: Dict[str, Any]) -> Interaction:
        """Standardize Reactome interaction data."""
        try:
            return Interaction(
                protein_a=raw.get('protein_a', ''),
                protein_b=raw.get('protein_b', ''),
                combined_score=raw.get('confidence', 0.8),
                evidence_types={'mechanistic': 1.0},
                pathway_context=raw.get('pathway_context'),
                interaction_type=raw.get('interaction_type'),
                source_database='reactome'
            )
        except Exception as e:
            logger.error(f"Failed to standardize Reactome interaction: {e}")
            raise
    
    def standardize_hpa_expression(self, raw: Dict[str, Any]) -> ExpressionProfile:
        """Standardize HPA expression data."""
        try:
            return ExpressionProfile(
                gene=raw.get('gene', ''),
                tissue=raw.get('tissue', ''),
                expression_level=raw.get('expression_level', 'Not detected'),
                reliability=raw.get('reliability', 'Uncertain'),
                cell_type_specific=raw.get('cell_type_specific', False),
                subcellular_location=raw.get('subcellular_location', [])
            )
        except Exception as e:
            logger.error(f"Failed to standardize HPA expression: {e}")
            raise
    
    def standardize_cancer_marker(self, raw: Dict[str, Any]) -> CancerMarker:
        """Standardize cancer marker data."""
        try:
            return CancerMarker(
                gene=raw.get('gene', ''),
                cancer_type=raw.get('cancer_type', ''),
                prognostic_value=raw.get('prognostic_value', 'unfavorable'),
                survival_association=raw.get('survival_association', ''),
                expression_pattern=raw.get('expression_pattern', {}),
                clinical_relevance=raw.get('clinical_relevance'),
                confidence=raw.get('confidence', 0.7)
            )
        except Exception as e:
            logger.error(f"Failed to standardize cancer marker: {e}")
            raise
    
    def standardize_kegg_drug(self, raw: Dict[str, Any]) -> DrugInfo:
        """Standardize KEGG drug data."""
        try:
            return DrugInfo(
                drug_id=raw.get('drug_id', ''),
                name=raw.get('name', ''),
                indication=raw.get('indication'),
                mechanism=raw.get('mechanism'),
                targets=raw.get('targets', []),
                development_status=raw.get('development_status'),
                drug_class=raw.get('drug_class'),
                approval_status=raw.get('approval_status')
            )
        except Exception as e:
            logger.error(f"Failed to standardize KEGG drug: {e}")
            raise
    
    def create_network_node(self, protein: Protein, network_data: Dict[str, Any]) -> NetworkNode:
        """Create network node from protein and network data."""
        try:
            return NetworkNode(
                id=protein.gene_symbol,
                node_type='protein',
                gene_symbol=protein.gene_symbol,
                uniprot_id=protein.uniprot_id,
                pathways=network_data.get('pathways', []),
                expression_level=network_data.get('expression_level'),
                centrality_measures=network_data.get('centrality_measures', {})
            )
        except Exception as e:
            logger.error(f"Failed to create network node: {e}")
            raise
    
    def create_network_edge(self, interaction: Interaction) -> NetworkEdge:
        """Create network edge from interaction data."""
        try:
            return NetworkEdge(
                source=interaction.protein_a,
                target=interaction.protein_b,
                weight=interaction.combined_score,
                interaction_type=interaction.interaction_type,
                evidence_score=interaction.combined_score,
                pathway_context=interaction.pathway_context
            )
        except Exception as e:
            logger.error(f"Failed to create network edge: {e}")
            raise
    
    def merge_cross_database_results(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Merge results from multiple databases."""
        try:
            merged = {
                'diseases': [],
                'pathways': [],
                'interactions': [],
                'expression_profiles': [],
                'cancer_markers': [],
                'drugs': []
            }
            
            # Merge diseases
            if 'kegg_diseases' in results:
                merged['diseases'].extend([
                    self.standardize_kegg_disease(d) for d in results['kegg_diseases']
                ])
            if 'reactome_diseases' in results:
                merged['diseases'].extend([
                    self.standardize_reactome_disease(d) for d in results['reactome_diseases']
                ])
            
            # Merge pathways
            if 'kegg_pathways' in results:
                merged['pathways'].extend([
                    self.standardize_kegg_pathway(p) for p in results['kegg_pathways']
                ])
            if 'reactome_pathways' in results:
                merged['pathways'].extend([
                    self.standardize_reactome_pathway(p) for p in results['reactome_pathways']
                ])
            
            # Merge interactions
            if 'string_interactions' in results:
                merged['interactions'].extend([
                    self.standardize_string_interaction(i) for i in results['string_interactions']
                ])
            if 'reactome_interactions' in results:
                merged['interactions'].extend([
                    self.standardize_reactome_interaction(i) for i in results['reactome_interactions']
                ])
            
            # Add other data types
            if 'hpa_expression' in results:
                merged['expression_profiles'].extend([
                    self.standardize_hpa_expression(e) for e in results['hpa_expression']
                ])
            
            if 'hpa_cancer_markers' in results:
                merged['cancer_markers'].extend([
                    self.standardize_cancer_marker(m) for m in results['hpa_cancer_markers']
                ])
            
            if 'kegg_drugs' in results:
                merged['drugs'].extend([
                    self.standardize_kegg_drug(d) for d in results['kegg_drugs']
                ])
            
            return merged
            
        except Exception as e:
            logger.error(f"Failed to merge cross-database results: {e}")
            raise
    
    def calculate_expression_score(self, expression_level: str) -> float:
        """Convert expression level to numerical score."""
        return self.expression_level_mapping.get(expression_level, 0.0)
    
    def validate_data_quality(self, data: Any, data_type: str) -> bool:
        """Validate data quality based on success metrics."""
        try:
            if data_type == 'disease':
                return data.confidence >= 0.6
            elif data_type == 'interaction':
                return data.combined_score >= 0.4  # 400/1000 scale
            elif data_type == 'expression':
                return data.reliability in ['Approved', 'Supported']
            elif data_type == 'cancer_marker':
                return data.confidence >= 0.7
            else:
                return True
        except Exception as e:
            logger.error(f"Data validation failed for {data_type}: {e}")
            return False

    def standardize_string_protein(self, data: Dict[str, Any], source_db: str = "string") -> Optional[Protein]:
        """Standardizes STRING protein data into a Protein Pydantic model."""
        if not data:
            return None

        protein_id = data.get('stringId') or data.get('string_id') or data.get('id')
        gene_symbol = (data.get('preferredName') or 
                      data.get('preferred_name') or 
                      data.get('gene_symbol') or 
                      'UNKNOWN')
        
        return Protein(
            gene_symbol=gene_symbol,
            uniprot_id=data.get('uniprotId'),
            string_id=protein_id,
            kegg_id=data.get('keggId'),
            ensembl_id=data.get('ensemblId'),
            hpa_id=data.get('hpaId'),
            description=data.get('description'),
            molecular_weight=data.get('molecularWeight'),
            protein_class=data.get('proteinClass'),
            confidence=0.8  # Default confidence for STRING data
        )

    def standardize_hpa_protein(self, data: Dict[str, Any]) -> Protein:
        """Standardizes HPA protein data into a Protein Pydantic model."""
        if not data:
            return None
        
        gene_symbol = data.get('Gene') or data.get('gene_name') or data.get('gene') or data.get('protein_name') or 'UNKNOWN'
        
        return Protein(
            gene_symbol=gene_symbol,
            uniprot_id=data.get('uniprot_id'),
            string_id=data.get('string_id'),
            kegg_id=None,
            ensembl_id=data.get('ensembl_id'),
            hpa_id=data.get('hpa_id') or data.get('id'),
            description=data.get('annotation') or data.get('description'),
            molecular_weight=None,
            protein_class=data.get('protein_class'),
            confidence=0.8  # Default confidence for HPA data
        )
    
    def standardize_kegg_gene(self, data: Dict[str, Any]) -> Protein:
        """Standardizes KEGG gene data into a Protein Pydantic model."""
        if not data:
            return None
        
        # Handle string format (just gene symbol)
        if isinstance(data, str):
            return Protein(
                gene_symbol=data,
                uniprot_id=None,
                string_id=None,
                kegg_id=None,
                ensembl_id=None,
                hpa_id=None,
                description=None,
                molecular_weight=None,
                protein_class=None,
                confidence=0.75
            )
        
        # KEGG gene format: {"gene_id": "hsa:123", "symbol": "GENE", "name": "Gene name"}
        gene_id = data.get('gene_id') or data.get('id')
        gene_symbol = data.get('symbol') or data.get('gene_symbol') or 'UNKNOWN'
        
        return Protein(
            gene_symbol=gene_symbol,
            uniprot_id=None,
            string_id=None,
            kegg_id=gene_id,
            ensembl_id=None,
            hpa_id=None,
            description=data.get('name') or data.get('description'),
            molecular_weight=None,
            protein_class=None,
            confidence=0.75  # Default confidence for KEGG data
        )

    async def standardize_drug_target(self, data: Dict[str, Any], source_db: str = "kegg") -> Optional[DrugTarget]:
        """Standardizes drug-target interaction data into a DrugTarget Pydantic model."""
        if not data:
            return None

        # Extract drug_id with fallback to 'id' field
        drug_id = data.get('drug_id') or data.get('id')
        if not drug_id:
            return None

        # Extract target_id
        target_id = data.get('target_id')
        if not target_id:
            return None

        target_symbol = str(
            data.get('target_symbol')
            or data.get('target_gene')
            or target_id
        )

        return DrugTarget(
            drug_id=str(drug_id),
            target_id=str(target_id),
            target_protein=target_symbol,
            interaction_type=data.get('interaction_type', 'unknown'),
            affinity=data.get('affinity'),
            mechanism=data.get('mechanism'),
            confidence=data.get('confidence', 0.5)
        )

    async def standardize_reactome_reaction(self, data: Dict[str, Any], source_db: str = "reactome") -> Optional[Dict[str, Any]]:
        """Standardizes Reactome reaction data."""
        if not data or not data.get('id'):
            return None

        return {
            'id': data['id'],
            'name': data.get('name', ''),
            'participants': data.get('participants', []),
            'direction': data.get('direction', 'unknown'),
            'confidence': data.get('confidence', 0.8)
        }

    # ========================================================================
    # ChEMBL-Specific Standardization Methods (Phase 2)
    # ========================================================================

    def standardize_chembl_compound(self, raw: Dict[str, Any]) -> Compound:
        """
        Standardize ChEMBL compound data to Compound model.

        Args:
            raw: Raw compound data from ChEMBL API

        Returns:
            Compound: Standardized compound model

        Example:
            >>> raw = {"molecule_chembl_id": "CHEMBL25", "pref_name": "ASPIRIN", ...}
            >>> compound = standardizer.standardize_chembl_compound(raw)
        """
        try:
            # Extract molecule properties
            props = raw.get('molecule_properties', {})
            structures = raw.get('molecule_structures', {})

            return Compound(
                chembl_id=raw.get('molecule_chembl_id', ''),
                name=raw.get('pref_name'),

                # Chemical structure
                smiles=structures.get('canonical_smiles'),
                inchi=structures.get('standard_inchi'),
                inchi_key=structures.get('standard_inchi_key'),
                molecular_formula=raw.get('molecule_formula') or props.get('full_molformula'),

                # Physicochemical properties
                molecular_weight=props.get('molecular_weight') or props.get('full_mwt'),
                alogp=props.get('alogp'),
                hba=props.get('hba') or props.get('hba_lipinski'),
                hbd=props.get('hbd') or props.get('hbd_lipinski'),
                psa=props.get('psa'),
                rtb=props.get('rtb') or props.get('num_ro5_violations'),
                ro5_violations=props.get('num_ro5_violations', 0),

                # Metadata
                molecule_type=raw.get('molecule_type', 'Small molecule'),
                source_db='chembl',
                confidence=self.calculate_compound_confidence(raw, 'chembl'),
                synonyms=raw.get('molecule_synonyms', [])
            )
        except (KeyError, TypeError, ValueError) as e:
            # Catch validation errors during model creation
            logger.error(
                "Failed to standardize ChEMBL compound - validation error",
                extra={
                    **format_error_for_logging(e),
                    'raw_data_keys': list(raw.keys()) if isinstance(raw, dict) else 'not_dict',
                    'field': str(e).split("'")[1] if "'" in str(e) else 'unknown'
                }
            )
            raise DataValidationError(
                message=f"Invalid compound data structure: {e}",
                field='compound_data',
                value=type(raw).__name__,
                expected='Dict with chembl_id, smiles fields'
            )
        except Exception as e:
            logger.error(
                "Failed to standardize ChEMBL compound - unexpected error",
                extra=format_error_for_logging(e)
            )
            raise

    def standardize_chembl_bioactivity(self, raw: Dict[str, Any]) -> Bioactivity:
        """
        Standardize ChEMBL bioactivity data to Bioactivity model.

        Args:
            raw: Raw bioactivity data from ChEMBL API

        Returns:
            Bioactivity: Standardized bioactivity model

        Example:
            >>> raw = {"activity_id": "12345", "standard_type": "IC50", ...}
            >>> bioactivity = standardizer.standardize_chembl_bioactivity(raw)
        """
        try:
            # Convert activity value to standard units (nM)
            activity_value = raw.get('standard_value')
            activity_units = raw.get('standard_units')

            # DEFENSIVE: Check and handle different activity_value types
            if activity_value is not None:
                # Handle list/tuple (take first element if possible)
                if isinstance(activity_value, (list, tuple)):
                    if len(activity_value) > 0:
                        activity_value = activity_value[0]
                    else:
                        activity_value = None
                elif isinstance(activity_value, str):
                    # Try to parse string to float
                    try:
                        activity_value = float(activity_value)
                    except ValueError:
                        activity_value = None
                # If it's already a number, ensure it's a float
                elif not isinstance(activity_value, (int, float)):
                    try:
                        activity_value = float(activity_value)
                    except (ValueError, TypeError):
                        activity_value = None

            if activity_value and activity_units:
                activity_value = self.convert_bioactivity_units(
                    float(activity_value),
                    activity_units,
                    'nM'
                )
                activity_units = 'nM'

            # Calculate pChEMBL value if not present
            pchembl_value = raw.get('pchembl_value')
            if not pchembl_value and activity_value:
                try:
                    # pChEMBL = -log10(activity in M)
                    activity_m = activity_value * 1e-9  # nM to M
                    import math
                    pchembl_value = -math.log10(activity_m)
                except (ValueError, ZeroDivisionError):
                    pchembl_value = None

            return Bioactivity(
                activity_id=str(raw.get('activity_id', '')),
                assay_chembl_id=raw.get('assay_chembl_id', ''),
                target_chembl_id=raw.get('target_chembl_id'),
                molecule_chembl_id=raw.get('molecule_chembl_id', ''),

                activity_type=raw.get('standard_type', raw.get('type', 'Unknown')),
                activity_value=activity_value,
                activity_units=activity_units,
                activity_relation=raw.get('standard_relation', '='),

                assay_type=raw.get('assay_type'),
                assay_organism=raw.get('assay_organism'),

                confidence=self._assess_bioactivity_confidence(raw),
                pchembl_value=pchembl_value,

                activity_comment=raw.get('activity_comment'),
                data_validity_comment=raw.get('data_validity_comment')
            )
        except (KeyError, TypeError, ValueError) as e:
            # Enhanced error logging with compound and target IDs
            compound_id = raw.get('molecule_chembl_id', 'UNKNOWN')
            target_id = raw.get('target_chembl_id', 'UNKNOWN')
            activity_type = raw.get('standard_type', raw.get('type', 'UNKNOWN'))
            activity_value = raw.get('standard_value')

            logger.error(
                f"Failed to standardize ChEMBL bioactivity for {compound_id} (target: {target_id}, type: {activity_type}, value: {activity_value}) - validation error",
                extra={
                    **format_error_for_logging(e),
                    'raw_data_keys': list(raw.keys()) if isinstance(raw, dict) else 'not_dict',
                    'field': str(e).split("'")[1] if "'" in str(e) else 'unknown',
                    'compound_id': compound_id,
                    'target_id': target_id,
                    'activity_type': activity_type,
                    'activity_value': activity_value,
                    'activity_value_type': type(activity_value).__name__
                }
            )
            raise DataValidationError(
                message=f"Invalid bioactivity data structure for {compound_id}: {e}",
                field='bioactivity_data',
                value=type(raw).__name__,
                expected='Dict with chembl_id, standard_value fields'
            )
        except Exception as e:
            logger.error(
                "Failed to standardize ChEMBL bioactivity - unexpected error",
                extra=format_error_for_logging(e)
            )
            raise

    def standardize_chembl_target(self, raw: Dict[str, Any]) -> Protein:
        """
        Standardize ChEMBL target data to Protein model.

        Args:
            raw: Raw target data from ChEMBL API

        Returns:
            Protein: Standardized protein model

        Example:
            >>> raw = {"target_chembl_id": "CHEMBL2095173", "pref_name": "AXL receptor", ...}
            >>> protein = standardizer.standardize_chembl_target(raw)
        """
        try:
            # Extract gene symbol from target name or components
            gene_symbol = raw.get('gene_symbol', '')
            if not gene_symbol and 'target_components' in raw:
                components = raw['target_components']
                if components and len(components) > 0:
                    gene_symbol = components[0].get('component_symbol', '')

            # Extract UniProt ID
            uniprot_id = None
            if 'target_components' in raw:
                components = raw['target_components']
                if components and len(components) > 0:
                    uniprot_id = components[0].get('accession')

            return Protein(
                gene_symbol=gene_symbol or raw.get('pref_name', 'Unknown'),
                uniprot_id=uniprot_id,
                string_id=None,  # ChEMBL doesn't provide STRING IDs
                kegg_id=None,
                ensembl_id=None,
                hpa_id=None,
                description=raw.get('pref_name'),
                molecular_weight=None,  # Not typically in ChEMBL target data
                protein_class=raw.get('target_type'),
                confidence=0.85  # High confidence for ChEMBL targets
            )
        except Exception as e:
            logger.error(f"Failed to standardize ChEMBL target: {e}")
            raise

    def merge_kegg_chembl_drugs(
        self,
        kegg_drugs: List[DrugInfo],
        chembl_drugs: List[Compound]
    ) -> List[DrugInfo]:
        """
        Merge and deduplicate drugs from KEGG and ChEMBL sources.

        Args:
            kegg_drugs: List of DrugInfo objects from KEGG
            chembl_drugs: List of Compound objects from ChEMBL

        Returns:
            List[DrugInfo]: Merged and deduplicated drug list

        Strategy:
            1. Use KEGG drugs as base (they have indication/mechanism data)
            2. Enhance with ChEMBL compounds not in KEGG
            3. Deduplicate by drug ID/name matching
        """
        logger.info(f"\n🔄 DRUG MERGE PIPELINE START")
        logger.info(f"   Input KEGG drugs: {len(kegg_drugs)}")
        logger.info(f"   Input ChEMBL compounds: {len(chembl_drugs)}")

        # Log sample inputs
        if kegg_drugs:
            logger.info(f"   📋 KEGG sample (first 3):")
            for i, drug in enumerate(kegg_drugs[:3]):
                logger.info(f"      {i+1}. ID={drug.drug_id}, Name={drug.name}, Gene={getattr(drug, 'gene', 'N/A')}")

        if chembl_drugs:
            logger.info(f"   📋 ChEMBL sample (first 3):")
            for i, comp in enumerate(chembl_drugs[:3]):
                logger.info(f"      {i+1}. ID={comp.chembl_id}, Name={comp.name}")

        try:
            merged_drugs = []
            seen_ids = set()
            seen_names = set()

            # Add all KEGG drugs first
            logger.info(f"\n   🔄 Step 1: Adding KEGG drugs...")
            kegg_count = 0
            for kegg_drug in kegg_drugs:
                drug_id = kegg_drug.drug_id.upper()
                drug_name = kegg_drug.name.upper() if kegg_drug.name else ""

                if drug_id not in seen_ids:
                    merged_drugs.append(kegg_drug)
                    seen_ids.add(drug_id)
                    if drug_name:
                        seen_names.add(drug_name)
                    kegg_count += 1

            logger.info(f"   ✅ Added {kegg_count} KEGG drugs (after deduplication)")

            # Add ChEMBL compounds not already present
            logger.info(f"\n   🔄 Step 2: Adding ChEMBL compounds...")
            chembl_count = 0
            skipped_count = 0
            for chembl_compound in chembl_drugs:
                chembl_id = chembl_compound.chembl_id.upper()
                chembl_name = chembl_compound.name.upper() if chembl_compound.name else ""

                # Skip if already seen by ID or name
                if chembl_id in seen_ids or (chembl_name and chembl_name in seen_names):
                    skipped_count += 1
                    continue

                # Convert Compound to DrugInfo
                try:
                    # CRITICAL FIX: Extract bioactivity_nm from external_refs if stored
                    bioactivity_nm = None
                    if chembl_compound.external_refs and 'median_ic50_nm' in chembl_compound.external_refs:
                        try:
                            bioactivity_nm = float(chembl_compound.external_refs['median_ic50_nm'])
                        except (ValueError, TypeError):
                            pass
                    
                    # Extract drug_likeness_score if available
                    drug_likeness_score = None
                    if chembl_compound.external_refs and 'drug_likeness_score' in chembl_compound.external_refs:
                        try:
                            drug_likeness_score = float(chembl_compound.external_refs['drug_likeness_score'])
                        except (ValueError, TypeError):
                            pass
                    
                    drug_info = DrugInfo(
                        drug_id=chembl_compound.chembl_id,
                        name=chembl_compound.name or chembl_compound.chembl_id,
                        indication=None,  # ChEMBL compounds may not have indication
                        mechanism=None,
                        targets=[],  # Would need separate query to get targets
                        development_status='preclinical',  # Conservative assumption
                        drug_class='small molecule',
                        approval_status='investigational',
                        bioactivity_nm=bioactivity_nm,  # CRITICAL: Populate from compound data
                        drug_likeness_score=drug_likeness_score  # CRITICAL: Populate from compound data
                    )
                    merged_drugs.append(drug_info)
                    seen_ids.add(chembl_id)
                    if chembl_name:
                        seen_names.add(chembl_name)
                    chembl_count += 1
                except Exception as e:
                    logger.warning(f"   ⚠️ Failed to convert ChEMBL compound {chembl_compound.chembl_id}: {e}")
                    continue

            logger.info(f"   ✅ Added {chembl_count} ChEMBL compounds")
            logger.info(f"   ⏭️ Skipped {skipped_count} duplicates")

            logger.info(f"\n   📊 Final merge results:")
            logger.info(f"      Total merged drugs: {len(merged_drugs)}")
            logger.info(f"      KEGG: {kegg_count}, ChEMBL: {chembl_count}")

            if merged_drugs:
                logger.info(f"   📋 Sample merged drugs:")
                for i, drug in enumerate(merged_drugs[:3]):
                    logger.info(f"      {i+1}. {drug.name} ({drug.drug_id})")

            return merged_drugs

        except Exception as e:
            logger.error(f"❌ Failed to merge KEGG and ChEMBL drugs: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return kegg_drugs  # Return KEGG drugs as fallback

    def convert_bioactivity_units(
        self,
        value: float,
        from_unit: str,
        to_unit: str = "nM"
    ) -> float:
        """
        Convert bioactivity units (M, mM, uM, nM, pM).

        Args:
            value: Activity value
            from_unit: Source unit (M, mM, uM, nM, pM)
            to_unit: Target unit (default: nM)

        Returns:
            float: Converted value

        Example:
            >>> value_nm = convert_bioactivity_units(1.5, "uM", "nM")
            >>> # Returns 1500.0
        """
        # Conversion factors to nM
        to_nm = {
            'M': 1e9,
            'mM': 1e6,
            'uM': 1e3,
            'µM': 1e3,  # Alternative unicode
            'nM': 1.0,
            'pM': 1e-3,
            'fM': 1e-6
        }

        # Normalize unit strings
        from_unit = from_unit.strip()
        to_unit = to_unit.strip()

        if from_unit not in to_nm or to_unit not in to_nm:
            # Handle non-concentration units (e.g., %) gracefully without warning
            if from_unit == '%' or to_unit == '%':
                # Percentage values are not concentration units - cannot convert
                logger.debug(f"Skipping unit conversion for non-concentration unit: {from_unit}")
                return value
            # Only warn for truly unexpected units
            logger.debug(f"Unknown units: {from_unit} → {to_unit}, returning original value")
            return value

        # Convert to nM, then to target unit
        value_nm = value * to_nm[from_unit]
        result = value_nm / to_nm[to_unit]

        return result

    def calculate_compound_confidence(
        self,
        compound: Dict[str, Any],
        source: str = "chembl"
    ) -> float:
        """
        Calculate confidence score for compound data.

        Factors:
            - Data completeness (structure, properties)
            - Source reliability
            - Bioactivity data availability

        Args:
            compound: Raw compound data
            source: Source database

        Returns:
            float: Confidence score (0-1)
        """
        confidence = 0.5  # Base confidence

        # Structure data adds confidence
        structures = compound.get('molecule_structures', {})
        if structures.get('canonical_smiles'):
            confidence += 0.15
        if structures.get('standard_inchi_key'):
            confidence += 0.10

        # Property data adds confidence
        props = compound.get('molecule_properties', {})
        if props.get('molecular_weight'):
            confidence += 0.05
        if props.get('alogp') is not None:
            confidence += 0.05
        if props.get('hba') is not None and props.get('hbd') is not None:
            confidence += 0.05

        # Source reliability
        if source == 'chembl':
            confidence += 0.10  # ChEMBL is highly curated

        return min(confidence, 1.0)

    def aggregate_bioactivities(
        self,
        bioactivities: List[Bioactivity],
        group_by: str = "target"
    ) -> Dict[str, Any]:
        """
        Aggregate bioactivity measurements by target/compound/assay.

        Args:
            bioactivities: List of Bioactivity objects
            group_by: Grouping strategy (target, compound, assay)

        Returns:
            Dict: Aggregated bioactivity statistics

        Example:
            >>> aggregated = aggregate_bioactivities(bioactivities, "target")
            >>> print(aggregated['median_ic50'])
        """
        import statistics

        if not bioactivities:
            return {
                'median_ic50': None,
                'median_ki': None,
                'median_ec50': None,
                'activity_count': 0,
                'data_quality': 'low'
            }

        # Group by activity type
        ic50_values = []
        ki_values = []
        ec50_values = []
        kd_values = []

        for activity in bioactivities:
            if not activity.activity_value:
                continue

            activity_type = activity.activity_type.upper()
            if 'IC50' in activity_type:
                ic50_values.append(activity.activity_value)
            elif 'KI' in activity_type:
                ki_values.append(activity.activity_value)
            elif 'EC50' in activity_type:
                ec50_values.append(activity.activity_value)
            elif 'KD' in activity_type:
                kd_values.append(activity.activity_value)

        # Calculate medians
        median_ic50 = statistics.median(ic50_values) if ic50_values else None
        median_ki = statistics.median(ki_values) if ki_values else None
        median_ec50 = statistics.median(ec50_values) if ec50_values else None
        median_kd = statistics.median(kd_values) if kd_values else None

        # Assess data quality
        total_activities = len(bioactivities)
        if total_activities > 5:
            data_quality = 'high'
        elif total_activities >= 2:
            data_quality = 'medium'
        else:
            data_quality = 'low'

        return {
            'median_ic50': median_ic50,
            'median_ki': median_ki,
            'median_ec50': median_ec50,
            'median_kd': median_kd,
            'activity_count': total_activities,
            'ic50_count': len(ic50_values),
            'ki_count': len(ki_values),
            'ec50_count': len(ec50_values),
            'kd_count': len(kd_values),
            'data_quality': data_quality
        }

    def assess_drug_likeness_comprehensive(
        self,
        compound: Compound
    ) -> DrugLikenessAssessment:
        """
        Comprehensive drug-likeness assessment using multiple rules.

        Rules applied:
            - Lipinski Rule of Five
            - Veber's rules
            - Pfizer 3/75 rule

        Args:
            compound: Compound object

        Returns:
            DrugLikenessAssessment: Complete drug-likeness assessment
        """
        # Lipinski Rule of Five
        lipinski_violations = 0
        detailed_violations = {}

        if compound.molecular_weight and compound.molecular_weight > 500:
            lipinski_violations += 1
            detailed_violations['mw'] = True

        if compound.alogp and compound.alogp > 5:
            lipinski_violations += 1
            detailed_violations['logp'] = True

        if compound.hbd and compound.hbd > 5:
            lipinski_violations += 1
            detailed_violations['hbd'] = True

        if compound.hba and compound.hba > 10:
            lipinski_violations += 1
            detailed_violations['hba'] = True

        lipinski_compliant = lipinski_violations <= 1

        # Veber's rules
        veber_violations = {}
        veber_compliant = True

        if compound.psa and compound.psa > 140:
            veber_compliant = False
            veber_violations['psa'] = True

        if compound.rtb and compound.rtb > 10:
            veber_compliant = False
            veber_violations['rtb'] = True

        # Pfizer 3/75 rule (for CNS drugs)
        pfizer_compliant = True
        if compound.alogp and compound.alogp >= 3:
            pfizer_compliant = False
        if compound.psa and compound.psa >= 75:
            pfizer_compliant = False

        # Overall assessment
        if lipinski_compliant and veber_compliant:
            overall_assessment = 'drug-like'
        elif lipinski_violations <= 2:
            overall_assessment = 'lead-like'
        else:
            overall_assessment = 'non-drug-like'

        # Drug-likeness score (0-1)
        score = 1.0
        score -= (lipinski_violations * 0.2)  # -0.2 per violation
        if not veber_compliant:
            score -= 0.1

        drug_likeness_score = max(score, 0.0)

        # Generate issues and recommendations
        issues = []
        recommendations = []

        if 'mw' in detailed_violations:
            issues.append("Molecular weight > 500 Da")
            recommendations.append("Consider reducing molecular size")

        if 'logp' in detailed_violations:
            issues.append("LogP > 5 (too lipophilic)")
            recommendations.append("Add polar groups to reduce lipophilicity")

        if 'hbd' in detailed_violations:
            issues.append("Too many H-bond donors")
            recommendations.append("Reduce hydroxyl or amine groups")

        if 'hba' in detailed_violations:
            issues.append("Too many H-bond acceptors")
            recommendations.append("Reduce oxygen/nitrogen content")

        if not veber_compliant:
            issues.append("Poor oral bioavailability (Veber's rules)")
            recommendations.append("Reduce polar surface area or rotatable bonds")

        return DrugLikenessAssessment(
            compound_id=compound.chembl_id,
            lipinski_compliant=lipinski_compliant,
            ro5_violations=lipinski_violations,
            detailed_violations=detailed_violations,
            veber_compliant=veber_compliant,
            veber_violations=veber_violations,
            pfizer_compliant=pfizer_compliant,
            molecular_weight=compound.molecular_weight,
            alogp=compound.alogp,
            hbd=compound.hbd,
            hba=compound.hba,
            psa=compound.psa,
            rtb=compound.rtb,
            overall_assessment=overall_assessment,
            drug_likeness_score=drug_likeness_score,
            issues=issues,
            recommendations=recommendations
        )

    def _assess_bioactivity_confidence(self, raw: Dict[str, Any]) -> float:
        """
        Assess confidence of a bioactivity measurement.

        Factors:
            - Assay type (B=Binding is more reliable than F=Functional)
            - Standard relation (= is better than <, >, ~)
            - Data validity
        """
        confidence = 0.7  # Base confidence

        # Assay type bonus
        assay_type = raw.get('assay_type', '')
        if assay_type == 'B':  # Binding assay
            confidence += 0.1
        elif assay_type == 'F':  # Functional assay
            confidence += 0.05

        # Relation bonus
        relation = raw.get('standard_relation', '=')
        if relation == '=':
            confidence += 0.1
        elif relation in ['<', '>']:
            confidence -= 0.1

        # Data validity
        validity = raw.get('data_validity_comment', '') or ''
        if validity and 'outside typical range' in validity.lower():
            confidence -= 0.2
        elif validity and 'potential' in validity.lower() and 'error' in validity.lower():
            confidence -= 0.15

        return max(min(confidence, 1.0), 0.0)
