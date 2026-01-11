"""
HPA (Human Protein Atlas) Parsing Utilities

Shared helpers for normalizing HPA MCP responses (list/dict formats)
and categorizing expression levels.
"""

from typing import Dict, Any, Tuple, Iterator
import logging

logger = logging.getLogger(__name__)


def _first_dict(obj: Any) -> Dict[str, Any]:
    """
    Extract first dict from list or return dict as-is.
    
    Args:
        obj: List of dicts, single dict, or other
        
    Returns:
        First dict or empty dict if not found
    """
    if isinstance(obj, list) and obj:
        item = obj[0]
        if isinstance(item, dict):
            return item
    elif isinstance(obj, dict):
        return obj
    return {}


def _iter_expr_items(records: Any) -> Iterator[Tuple[str, float]]:
    """
    Iterate over HPA expression records, yielding (tissue, nTPM) tuples.

    Supports both list and dict formats from HPA MCP.

    Args:
        records: HPA expression response (list of dicts or single dict)

    Yields:
        (tissue_name, nTPM_value) tuples
    """
    if isinstance(records, list):
        for item in records:
            if isinstance(item, dict):
                for key, value in item.items():
                    if key.startswith('Tissue RNA - ') and '[nTPM]' in key:
                        tissue = key.replace('Tissue RNA - ', '').replace(' [nTPM]', '')
                        try:
                            ntpms = float(value) if value else 0.0
                            yield (tissue, ntpms)
                        except (ValueError, TypeError):
                            continue
    elif isinstance(records, dict):
        # Check for 'expression' key or direct tissue keys
        if 'expression' in records:
            expr_dict = records['expression']
            if isinstance(expr_dict, dict):
                for tissue, level in expr_dict.items():
                    try:
                        if isinstance(level, (int, float)):
                            yield (tissue, float(level))
                        elif isinstance(level, str):
                            ntpms = float(level) if level else 0.0
                            yield (tissue, ntpms)
                    except (ValueError, TypeError):
                        continue
        else:
            # Direct tissue keys
            for key, value in records.items():
                if key.startswith('Tissue RNA - ') and '[nTPM]' in key:
                    tissue = key.replace('Tissue RNA - ', '').replace(' [nTPM]', '')
                    try:
                        ntpms = float(value) if value else 0.0
                        yield (tissue, ntpms)
                    except (ValueError, TypeError):
                        continue


def get_gene_expression(records: Any, target_gene: str) -> Iterator[Tuple[str, float]]:
    """
    Extract expression data for a specific gene from HPA response.

    The HPA get_tissue_expression returns a list of gene records.
    This function finds the record matching the target gene and
    yields its tissue expression data.

    Args:
        records: HPA expression response (list of dicts)
        target_gene: Gene symbol to extract (e.g., 'AXL')

    Yields:
        (tissue_name, nTPM_value) tuples for the target gene
    """
    if not isinstance(records, list):
        # If not a list, try the original function
        yield from _iter_expr_items(records)
        return

    # Find the gene record that matches the target gene
    for item in records:
        if isinstance(item, dict):
            # Check if this is the target gene
            gene_name = item.get('Gene') or item.get('gene')
            if gene_name and gene_name.upper() == target_gene.upper():
                # Extract tissue expression data from this gene
                for key, value in item.items():
                    if key.startswith('Tissue RNA - ') and '[nTPM]' in key:
                        tissue = key.replace('Tissue RNA - ', '').replace(' [nTPM]', '')
                        try:
                            ntpms = float(value) if value else 0.0
                            yield (tissue, ntpms)
                        except (ValueError, TypeError):
                            continue
                # Found the gene, stop searching
                return

    # If gene not found, yield nothing (or could yield from all items as fallback)
    # Downgraded to DEBUG - this is expected for many genes (HPA doesn't have all genes)
    logger.debug(f"Gene {target_gene} not found in HPA response (expected for some genes)")


def categorize_expression(ntpm: float) -> str:
    """
    Categorize expression level from nTPM value.
    
    Uses same thresholds as S1:
    - >= 10.0: High
    - >= 3.0: Medium
    - > 0: Low
    - 0: Not detected
    
    Args:
        ntpm: Normalized TPM value
        
    Returns:
        'Not detected', 'Low', 'Medium', or 'High'
    """
    if ntpm >= 10.0:
        return 'High'
    elif ntpm >= 3.0:
        return 'Medium'
    elif ntpm > 0:
        return 'Low'
    else:
        return 'Not detected'


def parse_pathology_data(response: Any) -> Dict[str, Any]:
    """
    Parse HPA pathology data response.
    
    Handles both list and dict formats, extracts prognostic markers.
    
    Args:
        response: HPA get_pathology_data response
        
    Returns:
        Normalized dict with markers, prognostic data, etc.
    """
    if isinstance(response, list):
        # List of pathology records
        markers = []
        for item in response:
            if isinstance(item, dict):
                # Extract marker data
                marker = {
                    'gene': item.get('Gene', item.get('gene', '')),
                    'prognostic_summary': item.get('Prognostic summary', item.get('prognostic_summary', '')),
                    'best_prognosis': item.get('Best prognosis', item.get('best_prognosis', '')),
                    'p_value': item.get('p_value', item.get('p-value', None)),
                    'significance': item.get('Significance', item.get('significance', ''))
                }
                if marker['gene']:
                    markers.append(marker)
        return {'markers': markers}
    elif isinstance(response, dict):
        # Dict format
        if 'markers' in response:
            return response
        else:
            # Wrap single marker
            return {'markers': [response] if response else []}
    return {'markers': []}


async def get_tissue_expression_with_fallback(hpa_client, gene: str):
    """
    Get tissue expression with automatic fallback for large genes.
    
    Delegates to the HPA client which now handles chunk limits internally.
    Kept for backward compatibility.

    Args:
        hpa_client: HPA MCP client instance
        gene: Gene symbol (e.g., 'AR', 'AXL')

    Returns:
        Expression data dict
    """
    return await hpa_client.get_tissue_expression(gene)
