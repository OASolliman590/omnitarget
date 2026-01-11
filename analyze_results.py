#!/usr/bin/env python3
"""
Comprehensive Scenario Results Analysis

Analyzes the JSON results file and provides detailed metrics for each scenario.
"""

import json
from pathlib import Path
from typing import Any, Dict, List

def safe_get(data: Any, *keys, default=None):
    """Safely get nested dictionary values."""
    try:
        result = data
        for key in keys:
            if isinstance(result, dict):
                result = result.get(key)
            elif isinstance(result, list):
                result = result[key] if key < len(result) else None
            else:
                return default
            if result is None:
                return default
        return result
    except:
        return default

def count_items(data: Any) -> int:
    """Count items in data structure."""
    if isinstance(data, list):
        return len(data)
    elif isinstance(data, dict):
        return len(data)
    elif isinstance(data, str):
        return 1
    return 0

def analyze_scenario_1(data: Dict[str, Any]) -> str:
    """Analyze Scenario 1: Disease Network"""
    output = []
    output.append("📈 DISEASE NETWORK METRICS:")
    
    disease = safe_get(data, 'disease', default={})
    if disease:
        output.append(f"  Disease: {disease.get('name', 'N/A')} ({disease.get('id', 'N/A')})")
        output.append(f"  Source: {disease.get('source_db', 'N/A')}")
        output.append(f"  Confidence: {disease.get('confidence', 0):.2f}")
    
    pathways = safe_get(data, 'pathways', default=[])
    if pathways:
        output.append(f"  Pathways: {len(pathways)}")
        for p in pathways[:5]:
            gene_count = len(p.get('genes', [])) if isinstance(p, dict) else 0
            name = p.get('name', p.get('id', 'N/A')) if isinstance(p, dict) else str(p)
            output.append(f"    - {name[:60]}: {gene_count} genes")
    
    nodes = safe_get(data, 'network_nodes', default=[])
    edges = safe_get(data, 'network_edges', default=[])
    output.append(f"  Network Nodes: {count_items(nodes)}")
    output.append(f"  Network Edges: {count_items(edges)}")
    if nodes and edges:
        density = count_items(edges) / count_items(nodes) if count_items(nodes) > 0 else 0
        output.append(f"  Edge Density: {density:.2f} edges/node")
    network_summary = data.get('network_summary')
    if network_summary:
        output.append(
            f"  Network Density: {network_summary.get('density', 0.0):.3f} | "
            f"Avg Degree: {network_summary.get('average_degree', 0.0):.2f}"
        )
        hubs = network_summary.get('top_hubs', [])[:3]
        if hubs:
            hub_str = ", ".join(f"{hub['gene']} (deg {hub['degree']})" for hub in hubs)
            output.append(f"  Top Hubs: {hub_str}")
    
    expr = safe_get(data, 'expression_profiles', default=[])
    output.append(f"  Expression Profiles: {count_items(expr)}")
    if expr and isinstance(expr, list):
        tissues = {}
        for ep in expr[:100]:  # Sample first 100
            if isinstance(ep, dict):
                tissue = ep.get('tissue', 'Unknown')
                tissues[tissue] = tissues.get(tissue, 0) + 1
        output.append(f"  Unique Tissues: {len(tissues)}")
        if tissues:
            top_tissues = sorted(tissues.items(), key=lambda x: x[1], reverse=True)[:5]
            output.append(f"  Top Tissues: {top_tissues}")
    expression_summary = data.get('expression_summary')
    if expression_summary:
        output.append(
            f"  Expression Score: {expression_summary.get('expression_score', 0.0):.2f} "
            f"(coverage={expression_summary.get('coverage', 0.0):.2f})"
        )
    pathway_summary = data.get('pathway_summary')
    if pathway_summary:
        output.append(
            f"  Pathway Coverage: {pathway_summary.get('coverage', 0.0):.2f} | "
            f"Median genes/pathway: {pathway_summary.get('median_gene_count', 0.0):.1f}"
        )
    
    markers = safe_get(data, 'cancer_markers', default=[])
    output.append(f"  Cancer Markers: {count_items(markers)}")
    
    validation = safe_get(data, 'validation_score', default=0)
    output.append(f"  Validation Score: {validation:.3f}")
    
    return "\n".join(output)

def analyze_scenario_2(data: Dict[str, Any]) -> str:
    """Analyze Scenario 2: Target Analysis"""
    output = []
    output.append("🎯 TARGET ANALYSIS METRICS:")
    
    target = safe_get(data, 'target', default={})
    if target:
        output.append(f"  Target: {target.get('gene_symbol', 'N/A')} ({target.get('uniprot_id', 'N/A')})")
        druggability = safe_get(data, 'druggability_score', default=target.get('druggability_score', 0))
        output.append(f"  Druggability Score: {druggability:.3f}")
    
    pathways = safe_get(data, 'pathways', default=[])
    output.append(f"  Pathways: {count_items(pathways)}")
    
    interactions = safe_get(data, 'interactions', default=[])
    output.append(f"  Protein Interactions: {count_items(interactions)}")
    
    expr = safe_get(data, 'expression_profiles', default=[])
    output.append(f"  Expression Profiles: {count_items(expr)}")
    if expr and isinstance(expr, list):
        tissues = {}
        for ep in expr[:100]:
            if isinstance(ep, dict):
                tissue = ep.get('tissue', 'Unknown')
                tissues[tissue] = tissues.get(tissue, 0) + 1
        output.append(f"  Unique Tissues: {len(tissues)}")
        if tissues:
            top_tissues = sorted(tissues.items(), key=lambda x: x[1], reverse=True)[:5]
            output.append(f"  Top Tissues: {top_tissues}")
    
    drugs = safe_get(data, 'known_drugs', default=[])
    output.append(f"  Known Drugs: {count_items(drugs)}")
    if drugs and isinstance(drugs, list):
        for drug in drugs[:3]:
            if isinstance(drug, dict):
                drug_id = drug.get('drug_id', drug.get('name', 'N/A'))
                output.append(f"    - {drug_id}")

    network_summary = data.get('network_summary')
    if network_summary:
        output.append(
            f"  Target Network: nodes={network_summary.get('node_count', 0)}, "
            f"density={network_summary.get('density', 0.0):.3f}"
        )
    expression_summary = data.get('expression_summary')
    if expression_summary:
        output.append(
            f"  Expression Score: {expression_summary.get('expression_score', 0.0):.2f} "
            f"(coverage={expression_summary.get('coverage', 0.0):.2f})"
        )
    priority = data.get('prioritization_summary')
    if priority:
        output.append(
            f"  Composite Priority: {priority.get('composite_priority', 0.0):.2f} "
            f"[network={priority.get('network_score', 0.0):.2f}, "
            f"expression={priority.get('expression_score', 0.0):.2f}, "
            f"druggability={priority.get('druggability_score', 0.0):.2f}]"
        )
    
    return "\n".join(output)

def analyze_scenario_3(data: Dict[str, Any]) -> str:
    """Analyze Scenario 3: Cancer Analysis"""
    output = []
    output.append("🔬 CANCER ANALYSIS METRICS:")
    
    cancer_type = safe_get(data, 'cancer_type', default='N/A')
    output.append(f"  Cancer Type: {cancer_type}")
    
    nodes = safe_get(data, 'network_nodes', default=[])
    edges = safe_get(data, 'network_edges', default=[])
    output.append(f"  Network Nodes: {count_items(nodes)}")
    output.append(f"  Network Edges: {count_items(edges)}")
    if nodes and edges:
        density = count_items(edges) / count_items(nodes) if count_items(nodes) > 0 else 0
        output.append(f"  Edge Density: {density:.2f} edges/node")
    
    targets = safe_get(data, 'prioritized_targets', default=[])
    output.append(f"  Prioritized Targets: {count_items(targets)}")
    if targets and isinstance(targets, list):
        for t in targets[:5]:
            if isinstance(t, dict):
                gene = t.get('gene_symbol', 'N/A')
                score = t.get('priority_score', t.get('score', 0))
                output.append(f"    - {gene}: score={score:.3f}")
    
    markers = safe_get(data, 'prognostic_markers', default=[])
    output.append(f"  Prognostic Markers: {count_items(markers)}")
    if markers and isinstance(markers, list):
        for m in markers[:5]:
            if isinstance(m, dict):
                gene = m.get('gene_symbol', m.get('gene', 'N/A'))
                output.append(f"    - {gene}")
    
    dysreg = safe_get(data, 'expression_dysregulation', default=[])
    output.append(f"  Dysregulation Patterns: {count_items(dysreg)}")
    network_summary = data.get('network_summary')
    if network_summary:
        output.append(
            f"  Network Density: {network_summary.get('density', 0.0):.3f} | "
            f"Giant component={network_summary.get('giant_component_ratio', 0.0):.2f}"
        )
    expression_summary = data.get('expression_summary')
    if expression_summary:
        output.append(
            f"  Expression Score: {expression_summary.get('expression_score', 0.0):.2f} "
            f"(coverage={expression_summary.get('coverage', 0.0):.2f})"
        )
    marker_summary = data.get('marker_summary')
    if marker_summary:
        output.append(
            f"  Marker Distribution: favorable={marker_summary.get('favorable', 0)}, "
            f"unfavorable={marker_summary.get('unfavorable', 0)}"
        )
    
    return "\n".join(output)

def analyze_scenario_4(data: Dict[str, Any]) -> str:
    """Analyze Scenario 4: MRA Simulation"""
    output = []
    output.append("🧮 MRA SIMULATION METRICS:")
    
    targets = safe_get(data, 'targets', default=[])
    output.append(f"  Simulated Targets: {count_items(targets)}")
    if targets and isinstance(targets, list):
        for t in targets[:5]:
            if isinstance(t, dict):
                gene = t.get('gene_symbol', t.get('gene', 'N/A'))
                results = t.get('simulation_results', [])
                result_count = count_items(results)
                output.append(f"    - {gene}: {result_count} simulation results")
            elif isinstance(t, str):
                output.append(f"    - {t}")
    
    individual = safe_get(data, 'individual_results', default=[])
    output.append(f"  Individual Results: {count_items(individual)}")
    
    combined = safe_get(data, 'combined_effects', default={})
    if combined:
        output.append(f"  Combined Effects: {len(combined) if isinstance(combined, dict) else 'N/A'}")
    
    synergy = safe_get(data, 'synergy_analysis', default={})
    if synergy:
        output.append(f"  Synergy Analysis: {len(synergy) if isinstance(synergy, dict) else 'N/A'}")
    
    return "\n".join(output)

def analyze_scenario_5(data: Dict[str, Any]) -> str:
    """Analyze Scenario 5: Pathway Comparison"""
    output = []
    output.append("🔄 PATHWAY COMPARISON METRICS:")
    
    query = safe_get(data, 'pathway_query', default='N/A')
    output.append(f"  Query: {query}")
    
    kegg = safe_get(data, 'kegg_pathways', default=[])
    output.append(f"  KEGG Pathways: {count_items(kegg)}")
    if kegg and isinstance(kegg, list):
        total_kegg_genes = sum(len(p.get('genes', [])) for p in kegg if isinstance(p, dict))
        output.append(f"    Total KEGG genes: {total_kegg_genes}")
        for p in kegg[:3]:
            if isinstance(p, dict):
                path_id = p.get('id', 'N/A')
                gene_count = len(p.get('genes', []))
                output.append(f"    - {path_id}: {gene_count} genes")
    
    reactome = safe_get(data, 'reactome_pathways', default=[])
    output.append(f"  Reactome Pathways: {count_items(reactome)}")
    if reactome and isinstance(reactome, list):
        total_reactome_genes = sum(len(p.get('genes', [])) for p in reactome if isinstance(p, dict))
        output.append(f"    Total Reactome genes: {total_reactome_genes}")
        for p in reactome[:3]:
            if isinstance(p, dict):
                path_id = p.get('id', 'N/A')
                gene_count = len(p.get('genes', []))
                output.append(f"    - {path_id}: {gene_count} genes")
    
    overlap = safe_get(data, 'gene_overlap', default={})
    if overlap:
        jaccard = overlap.get('jaccard_similarity', 0)
        output.append(f"  Gene Overlap Jaccard: {jaccard:.3f}")
        shared_genes = overlap.get('common_genes', [])
        kegg_unique = overlap.get('kegg_unique_genes', [])
        reactome_unique = overlap.get('reactome_unique_genes', [])
        if isinstance(shared_genes, list):
            output.append(f"    Shared Genes: {len(shared_genes)}")
        else:
            output.append(f"    Shared Genes: {shared_genes}")
        output.append(f"    KEGG Unique: {len(kegg_unique) if isinstance(kegg_unique, list) else kegg_unique}")
        output.append(f"    Reactome Unique: {len(reactome_unique) if isinstance(reactome_unique, list) else reactome_unique}")

    overlap_detail = safe_get(data, 'pathway_overlap', default={})
    if overlap_detail:
        modules = overlap_detail.get('functional_modules', [])
        if modules:
            module_summary = ", ".join(f"{m['module']} ({len(m.get('genes', []))} genes)" for m in modules[:3])
            output.append(f"  Crosstalk Modules: {module_summary}")
        resistance = overlap_detail.get('resistance_mechanisms', [])
        if resistance:
            resistance_summary = ", ".join(r['name'] for r in resistance[:3])
            output.append(f"  Resistance Mechanisms: {resistance_summary}")
    
    consensus = safe_get(data, 'consensus_pathways', default=[])
    output.append(f"  Consensus Pathways: {count_items(consensus)}")
    
    return "\n".join(output)

def analyze_scenario_6(data: Dict[str, Any]) -> str:
    """Analyze Scenario 6: Drug Repurposing"""
    output = []
    output.append("💊 DRUG REPURPOSING METRICS:")
    
    query = safe_get(data, 'disease_query', default='N/A')
    output.append(f"  Disease Query: {query}")
    
    pathways = safe_get(data, 'disease_pathways', default=[])
    if pathways:
        reactome = [p for p in pathways if isinstance(p, dict) and p.get('source_db') == 'reactome']
        kegg = [p for p in pathways if isinstance(p, dict) and p.get('source_db') == 'kegg']
        output.append(f"  Disease Pathways: {len(pathways)}")
        output.append(f"    - Reactome: {len(reactome)} pathways")
        output.append(f"    - KEGG: {len(kegg)} pathways")
        if reactome:
            total_reactome = sum(len(p.get('genes', [])) for p in reactome)
            output.append(f"    - Reactome genes: {total_reactome} total")
        if kegg:
            total_kegg = sum(len(p.get('genes', [])) for p in kegg)
            output.append(f"    - KEGG genes: {total_kegg} total")
    
    drugs = safe_get(data, 'candidate_drugs', default=[])
    output.append(f"  Candidate Drugs: {count_items(drugs)}")
    if drugs and isinstance(drugs, list):
        for d in drugs[:5]:
            if isinstance(d, dict):
                drug_id = d.get('drug_id', d.get('drug_name', 'N/A'))
                score = d.get('repurposing_score', d.get('score', 0))
                output.append(f"    - {drug_id}: score={score:.3f}")
    
    scores = safe_get(data, 'repurposing_scores', default=[])
    output.append(f"  Repurposing Scores: {count_items(scores)}")
    
    validation = safe_get(data, 'network_validation', default={})
    if validation:
        output.append(f"  Network Validation: {list(validation.keys()) if isinstance(validation, dict) else 'N/A'}")
    
    return "\n".join(output)

def main():
    """Main analysis function"""
    results_file = Path('results/comprehensive_axl_breast_cancer_analysis.json')
    
    if not results_file.exists():
        print(f"❌ Results file not found: {results_file}")
        return
    
    print("=" * 80)
    print("📊 COMPREHENSIVE SCENARIO RESULTS REVIEW")
    print("=" * 80)
    print()
    
    with open(results_file, 'r') as f:
        data = json.load(f)
    
    scenarios = {
        1: ("Disease Network", analyze_scenario_1),
        2: ("Target Analysis", analyze_scenario_2),
        3: ("Cancer Analysis", analyze_scenario_3),
        4: ("MRA Simulation", analyze_scenario_4),
        5: ("Pathway Comparison", analyze_scenario_5),
        6: ("Drug Repurposing", analyze_scenario_6)
    }
    
    for scenario_id, (scenario_name, analyzer) in scenarios.items():
        result = [r for r in data['results'] if r.get('scenario_id') == scenario_id]
        
        if not result:
            print(f"\n{'='*80}")
            print(f"SCENARIO {scenario_id}: {scenario_name.upper()} - NOT FOUND")
            print(f"{'='*80}")
            continue
        
        result = result[0]
        print(f"\n{'='*80}")
        print(f"SCENARIO {scenario_id}: {scenario_name.upper()}")
        print(f"{'='*80}")
        print(f"Status: {result.get('status', 'unknown').upper()}")
        print(f"Execution Time: {result.get('execution_time', 'N/A')}")
        print()
        
        scenario_data = result.get('data', {})
        data_keys = list(scenario_data.keys())
        print(f"Available Data Keys ({len(data_keys)}): {', '.join(data_keys)}")
        print()
        
        # Run scenario-specific analysis
        try:
            analysis = analyzer(scenario_data)
            print(analysis)
        except Exception as e:
            print(f"  ⚠️  Analysis error: {e}")
        
        print()
    
    print("=" * 80)
    print("✅ REVIEW COMPLETE")
    print("=" * 80)

if __name__ == "__main__":
    main()





