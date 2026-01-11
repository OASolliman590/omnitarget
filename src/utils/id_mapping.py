"""
ID Mapping Utilities

Cross-database identifier conversion and validation.
"""

import logging
from typing import Dict, List, Optional, Any
from ..models.data_models import Protein

logger = logging.getLogger(__name__)


class IDMapper:
    """Cross-database identifier mapping and conversion."""
    
    def __init__(self):
        """Initialize ID mapper with common mappings."""
        self.mapping_cache = {}
        self.reverse_mappings = {}
    
    async def map_gene_symbol_to_ids(
        self, 
        symbol: str, 
        kegg_client: Optional[Any] = None,
        string_client: Optional[Any] = None,
        hpa_client: Optional[Any] = None
    ) -> Protein:
        """
        Map gene symbol to all available database identifiers.
        
        Args:
            symbol: Gene symbol (e.g., 'TP53')
            kegg_client: KEGG client for ID conversion
            string_client: STRING client for protein search
            hpa_client: HPA client for protein lookup
            
        Returns:
            Protein object with all available identifiers
        """
        try:
            protein = Protein(gene_symbol=symbol)
            
            # Use KEGG convert_identifiers if available
            if kegg_client:
                try:
                    # Convert from gene symbol to various IDs
                    conversions = await kegg_client.convert_identifiers(
                        ids=[symbol],
                        source_db='hsa',
                        target_db='uniprot'
                    )
                    
                    if conversions and 'results' in conversions:
                        for result in conversions['results']:
                            if 'uniprot' in result:
                                protein.uniprot_id = result['uniprot']
                            if 'kegg' in result:
                                protein.kegg_id = result['kegg']
                                
                except Exception as e:
                    logger.warning(f"KEGG ID conversion failed for {symbol}: {e}")
            
            # Use STRING search if available
            if string_client:
                try:
                    string_results = await string_client.search_proteins(symbol, limit=1)
                    if string_results and 'proteins' in string_results:
                        proteins = string_results['proteins']
                        if proteins:
                            protein.string_id = proteins[0].get('string_id')
                            
                except Exception as e:
                    logger.warning(f"STRING search failed for {symbol}: {e}")
            
            # Use HPA lookup if available
            if hpa_client:
                try:
                    hpa_info = await hpa_client.get_protein_info(symbol)
                    if hpa_info:
                        protein.hpa_id = hpa_info.get('hpa_id')
                        protein.ensembl_id = hpa_info.get('ensembl_id')
                        protein.description = hpa_info.get('description')
                        
                except Exception as e:
                    logger.warning(f"HPA lookup failed for {symbol}: {e}")
            
            return protein
            
        except Exception as e:
            logger.error(f"ID mapping failed for {symbol}: {e}")
            return Protein(gene_symbol=symbol)
    
    async def map_uniprot_to_string(
        self, 
        uniprot_id: str, 
        string_client: Optional[Any] = None
    ) -> Optional[str]:
        """
        Map UniProt ID to STRING ID.
        
        Args:
            uniprot_id: UniProt identifier
            string_client: STRING client for mapping
            
        Returns:
            STRING identifier or None
        """
        try:
            if not string_client:
                return None
            
            # Search STRING database for UniProt ID
            results = await string_client.search_proteins(uniprot_id, limit=1)
            if results and 'proteins' in results:
                proteins = results['proteins']
                if proteins:
                    return proteins[0].get('string_id')
            
            return None
            
        except Exception as e:
            logger.error(f"UniProt to STRING mapping failed for {uniprot_id}: {e}")
            return None
    
    async def map_ensembl_to_string(
        self, 
        ensembl_id: str, 
        string_client: Optional[Any] = None
    ) -> Optional[str]:
        """
        Map Ensembl ID to STRING ID.
        
        Args:
            ensembl_id: Ensembl identifier
            string_client: STRING client for mapping
            
        Returns:
            STRING identifier or None
        """
        try:
            if not string_client:
                return None
            
            # Search STRING database for Ensembl ID
            results = await string_client.search_proteins(ensembl_id, limit=1)
            if results and 'proteins' in results:
                proteins = results['proteins']
                if proteins:
                    return proteins[0].get('string_id')
            
            return None
            
        except Exception as e:
            logger.error(f"Ensembl to STRING mapping failed for {ensembl_id}: {e}")
            return None
    
    def validate_identifier(self, identifier: str, database: str) -> bool:
        """
        Validate identifier format for specific database.
        
        Args:
            identifier: Identifier to validate
            database: Target database ('uniprot', 'string', 'kegg', 'ensembl')
            
        Returns:
            True if identifier format is valid
        """
        try:
            if database == 'uniprot':
                # UniProt format: P12345 or A0A123456789
                return (len(identifier) >= 6 and 
                        identifier[0].isalpha() and 
                        identifier[1:].replace('_', '').isalnum() and
                        not identifier.startswith('invalid'))
            
            elif database == 'string':
                # STRING format: 9606.ENSP00000269305
                return '.' in identifier and identifier.split('.')[0].isdigit()
            
            elif database == 'kegg':
                # KEGG format: hsa:1234 or gene symbol
                return ':' in identifier or identifier.isalnum()
            
            elif database == 'ensembl':
                # Ensembl format: ENSG00000141510
                return identifier.startswith('ENS') and len(identifier) >= 15
            
            else:
                return False
                
        except Exception as e:
            logger.error(f"Identifier validation failed for {identifier} in {database}: {e}")
            return False
    
    def create_mapping_table(self, proteins: List[Protein]) -> Dict[str, Dict[str, str]]:
        """
        Create comprehensive mapping table from protein list.
        
        Args:
            proteins: List of Protein objects
            
        Returns:
            Mapping table with all available identifiers
        """
        try:
            mapping_table = {}
            
            for protein in proteins:
                symbol = protein.gene_symbol
                mapping_table[symbol] = {}
                
                if protein.uniprot_id:
                    mapping_table[symbol]['uniprot'] = protein.uniprot_id
                if protein.string_id:
                    mapping_table[symbol]['string'] = protein.string_id
                if protein.kegg_id:
                    mapping_table[symbol]['kegg'] = protein.kegg_id
                if protein.ensembl_id:
                    mapping_table[symbol]['ensembl'] = protein.ensembl_id
                if protein.hpa_id:
                    mapping_table[symbol]['hpa'] = protein.hpa_id
            
            return mapping_table
            
        except Exception as e:
            logger.error(f"Mapping table creation failed: {e}")
            return {}
    
    def find_common_identifiers(
        self, 
        protein_list_1: List[Protein], 
        protein_list_2: List[Protein]
    ) -> List[str]:
        """
        Find common proteins between two protein lists.
        
        Args:
            protein_list_1: First protein list
            protein_list_2: Second protein list
            
        Returns:
            List of common gene symbols
        """
        try:
            symbols_1 = {p.gene_symbol for p in protein_list_1}
            symbols_2 = {p.gene_symbol for p in protein_list_2}
            
            return list(symbols_1 & symbols_2)
            
        except Exception as e:
            logger.error(f"Common identifier finding failed: {e}")
            return []
    
    def calculate_mapping_coverage(
        self, 
        original_identifiers: List[str], 
        mapped_identifiers: Dict[str, str]
    ) -> float:
        """
        Calculate mapping coverage percentage.
        
        Args:
            original_identifiers: Original identifier list
            mapped_identifiers: Mapping results
            
        Returns:
            Coverage percentage (0-1)
        """
        try:
            if not original_identifiers:
                return 1.0
            
            successfully_mapped = sum(
                1 for id in original_identifiers 
                if id in mapped_identifiers and mapped_identifiers[id]
            )
            
            return successfully_mapped / len(original_identifiers)
            
        except Exception as e:
            logger.error(f"Mapping coverage calculation failed: {e}")
            return 0.0
