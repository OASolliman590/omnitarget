"""
Scenario-level metric helpers.

Provides small utilities to summarize network topology, expression coverage,
and pathway composition so each scenario can report consistent metrics.
"""

from __future__ import annotations

from collections import Counter
from statistics import median
from typing import Any, Dict, Iterable, List, Optional, Sequence

import networkx as nx


def _get_attr(item: Any, attr: str, default: Any = None) -> Any:
    """Safely fetch attribute or dict key."""
    if item is None:
        return default
    if isinstance(item, dict):
        return item.get(attr, default)
    return getattr(item, attr, default)


def summarize_network(graph: Optional[nx.Graph], max_hubs: int = 10) -> Dict[str, Any]:
    """Calculate basic network topology metrics."""
    summary = {
        'node_count': 0,
        'edge_count': 0,
        'density': 0.0,
        'average_degree': 0.0,
        'giant_component_ratio': 0.0,
        'top_hubs': [],
    }

    if not isinstance(graph, nx.Graph):
        return summary

    node_count = graph.number_of_nodes()
    edge_count = graph.number_of_edges()
    summary['node_count'] = node_count
    summary['edge_count'] = edge_count

    if node_count == 0:
        return summary

    summary['density'] = float(nx.density(graph)) if node_count > 1 else 0.0
    degrees = dict(graph.degree())
    summary['average_degree'] = float(sum(degrees.values()) / node_count)

    # Largest connected component ratio (handles small graphs gracefully)
    if node_count > 1:
        try:
            components = list(nx.connected_components(graph))
            if components:
                largest = max(len(component) for component in components)
                summary['giant_component_ratio'] = largest / node_count
        except nx.NetworkXException:
            summary['giant_component_ratio'] = 1.0

    # Centrality metrics (limit to modest graphs to avoid expensive computations)
    centrality_ready = node_count <= 500
    betweenness = nx.betweenness_centrality(graph) if centrality_ready and node_count > 1 else {}
    closeness = nx.closeness_centrality(graph) if centrality_ready and node_count > 1 else {}

    top_nodes = sorted(graph.nodes(), key=lambda n: degrees.get(n, 0), reverse=True)[:max_hubs]
    top_hubs = []
    for node in top_nodes:
        top_hubs.append({
            'gene': node,
            'degree': degrees.get(node, 0),
            'betweenness': round(betweenness.get(node, 0.0), 6),
            'closeness': round(closeness.get(node, 0.0), 6),
        })

    summary['top_hubs'] = top_hubs
    summary['hub_fraction'] = len(top_hubs) / node_count if node_count else 0.0

    return summary


def summarize_expression_profiles(
    profiles: Sequence[Any],
    total_gene_universe: Optional[int] = None,
    max_tissues: int = 5,
) -> Dict[str, Any]:
    """Summarize tissue expression coverage and derive a scoring signal."""
    summary = {
        'profile_count': 0,
        'unique_genes': 0,
        'unique_tissues': 0,
        'coverage': 0.0,
        'high_expression_fraction': 0.0,
        'expression_score': 0.0,
        'top_tissues': [],
        'level_distribution': {},
    }

    if not profiles:
        return summary

    level_weights = {
        'High': 1.0,
        'Medium': 0.7,
        'Low': 0.4,
        'Not detected': 0.1,
    }

    genes = set()
    tissues = Counter()
    level_counter = Counter()
    intensity_scores: List[float] = []

    for profile in profiles:
        gene = _get_attr(profile, 'gene')
        tissue = _get_attr(profile, 'tissue', 'Unknown')
        level = _get_attr(profile, 'expression_level', 'Not detected')

        if gene:
            genes.add(gene)
        tissues[tissue] += 1
        level_counter[level] += 1
        intensity_scores.append(level_weights.get(level, 0.2))

    profile_count = len(profiles)
    unique_genes = len(genes)
    unique_tissues = len(tissues)
    coverage = 0.0
    if total_gene_universe:
        coverage = min(1.0, unique_genes / max(total_gene_universe, 1))
    elif unique_genes:
        coverage = 1.0

    high_expression_fraction = level_counter.get('High', 0) / profile_count if profile_count else 0.0
    expression_intensity = sum(intensity_scores) / profile_count if profile_count else 0.0
    expression_score = min(1.0, expression_intensity * (0.5 + 0.5 * coverage))

    summary.update({
        'profile_count': profile_count,
        'unique_genes': unique_genes,
        'unique_tissues': unique_tissues,
        'coverage': coverage,
        'high_expression_fraction': round(high_expression_fraction, 3),
        'expression_score': round(expression_score, 3),
        'level_distribution': dict(level_counter),
        'top_tissues': [
            {'tissue': tissue, 'count': count}
            for tissue, count in tissues.most_common(max_tissues)
        ],
    })

    return summary


def summarize_pathways(
    pathways: Sequence[Any],
    coverage: Optional[float] = None,
    max_pathways: int = 5,
) -> Dict[str, Any]:
    """Summarize pathway composition."""
    summary = {
        'pathway_count': 0,
        'average_gene_count': 0.0,
        'median_gene_count': 0.0,
        'coverage': coverage or 0.0,
        'top_pathways': [],
    }

    if not pathways:
        return summary

    counts: List[int] = []
    pathway_entries: List[Dict[str, Any]] = []
    for pathway in pathways:
        genes = _get_attr(pathway, 'genes', []) or []
        gene_count = len(genes)
        counts.append(gene_count)
        pathway_entries.append({
            'id': _get_attr(pathway, 'id') or _get_attr(pathway, 'stId'),
            'name': _get_attr(pathway, 'name', _get_attr(pathway, 'id', 'Unknown')),
            'gene_count': gene_count,
        })

    summary['pathway_count'] = len(pathways)
    if counts:
        summary['average_gene_count'] = sum(counts) / len(counts)
        summary['median_gene_count'] = median(counts)
    if summary['coverage'] == 0.0:
        summary['coverage'] = min(1.0, summary['pathway_count'] / 25.0)

    summary['top_pathways'] = sorted(
        pathway_entries,
        key=lambda p: p['gene_count'],
        reverse=True
    )[:max_pathways]

    return summary


def summarize_markers(markers: Sequence[Any], max_markers: int = 10) -> Dict[str, Any]:
    """Summarize prognostic marker distribution."""
    summary = {
        'marker_count': 0,
        'favorable': 0,
        'unfavorable': 0,
        'top_markers': [],
    }

    if not markers:
        return summary

    entries: List[Dict[str, Any]] = []
    for marker in markers:
        gene = _get_attr(marker, 'gene')
        value = (_get_attr(marker, 'prognostic_value', 'unknown') or 'unknown').lower()
        confidence = float(_get_attr(marker, 'confidence', 0.0) or 0.0)
        summary[value] = summary.get(value, 0) + 1
        if gene:
            entries.append({
                'gene': gene,
                'prognostic_value': value,
                'confidence': confidence,
            })

    summary['marker_count'] = len(markers)
    summary['top_markers'] = sorted(
        entries,
        key=lambda item: item['confidence'],
        reverse=True
    )[:max_markers]

    return summary
