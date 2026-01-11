#!/usr/bin/env python3
"""
Generate Week 1 Critical Validation Plots

Creates validation visualizations for all 6 scenarios based on fresh results.
Focuses on validating fixes and showing key scientific findings.

Usage:
    python generate_validation_plots.py <path_to_results.json>
"""

import json
import sys
from pathlib import Path
from typing import Dict, Any, List
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import numpy as np
from collections import Counter

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10

def load_results(filepath: str) -> Dict[str, Any]:
    """Load results JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)

def plot_s3_hub_network(data: Dict[str, Any], output_dir: Path):
    """S3: Hub protein network visualization."""
    s3 = next((s for s in data['results'] if s['scenario_id'] == 3), None)
    if not s3:
        print("⚠️  Scenario 3 not found")
        return
    
    s3_data = s3['data']
    nodes = s3_data.get('network_nodes', [])
    edges = s3_data.get('network_edges', [])
    
    print(f"\n📊 S3 Network: {len(nodes)} nodes, {len(edges)} edges")
    
    # Extract centrality measures
    centrality_data = []
    for node in nodes[:50]:  # Top 50 for readability
        if isinstance(node, dict):
            centrality = node.get('centrality_measures', {})
            gene = node.get('gene_symbol', node.get('id', 'Unknown'))
            degree = centrality.get('degree_centrality', 0.0) if isinstance(centrality, dict) else 0.0
            betweenness = centrality.get('betweenness_centrality', 0.0) if isinstance(centrality, dict) else 0.0
            centrality_data.append({
                'gene': gene,
                'degree': degree,
                'betweenness': betweenness
            })
    
    if not centrality_data:
        print("⚠️  No centrality data found")
        return
    
    # Sort by degree
    centrality_data.sort(key=lambda x: x['degree'], reverse=True)
    top_20 = centrality_data[:20]
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1: Degree centrality
    genes = [d['gene'] for d in top_20]
    degrees = [d['degree'] for d in top_20]
    
    ax1.barh(range(len(genes)), degrees, color='steelblue', alpha=0.7)
    ax1.set_yticks(range(len(genes)))
    ax1.set_yticklabels(genes)
    ax1.set_xlabel('Degree Centrality', fontsize=12, fontweight='bold')
    ax1.set_title('S3: Top 20 Hub Proteins by Degree Centrality', fontsize=14, fontweight='bold')
    ax1.grid(axis='x', alpha=0.3)
    ax1.invert_yaxis()
    
    # Plot 2: Betweenness centrality
    betweenness = [d['betweenness'] for d in top_20]
    ax2.barh(range(len(genes)), betweenness, color='coral', alpha=0.7)
    ax2.set_yticks(range(len(genes)))
    ax2.set_yticklabels(genes)
    ax2.set_xlabel('Betweenness Centrality', fontsize=12, fontweight='bold')
    ax2.set_title('S3: Top 20 Hub Proteins by Betweenness Centrality', fontsize=14, fontweight='bold')
    ax2.grid(axis='x', alpha=0.3)
    ax2.invert_yaxis()
    
    plt.tight_layout()
    output_path = output_dir / 's3_hub_proteins.png'
    plt.savefig(output_path, bbox_inches='tight')
    print(f"✅ Saved: {output_path}")
    plt.close()

def plot_s1_disease_network(data: Dict[str, Any], output_dir: Path):
    """S1: Disease network with centrality."""
    s1 = next((s for s in data['results'] if s['scenario_id'] == 1), None)
    if not s1:
        print("⚠️  Scenario 1 not found")
        return
    
    s1_data = s1['data']
    nodes = s1_data.get('network_nodes', [])
    pathways = s1_data.get('pathways', [])
    
    print(f"\n📊 S1 Network: {len(nodes)} nodes, {len(pathways)} pathways")
    
    # Extract pathway distribution
    pathway_counts = Counter()
    for pathway in pathways:
        if isinstance(pathway, dict):
            name = pathway.get('name', 'Unknown')
            pathway_counts[name] += 1
    
    if not pathway_counts:
        print("⚠️  No pathway data found")
        return
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1: Pathway distribution
    top_pathways = pathway_counts.most_common(10)
    pathway_names = [p[0][:40] + '...' if len(p[0]) > 40 else p[0] for p in top_pathways]
    pathway_values = [p[1] for p in top_pathways]
    
    ax1.barh(range(len(pathway_names)), pathway_values, color='mediumseagreen', alpha=0.7)
    ax1.set_yticks(range(len(pathway_names)))
    ax1.set_yticklabels(pathway_names)
    ax1.set_xlabel('Gene Count', fontsize=12, fontweight='bold')
    ax1.set_title('S1: Top 10 Pathways by Gene Count', fontsize=14, fontweight='bold')
    ax1.grid(axis='x', alpha=0.3)
    ax1.invert_yaxis()
    
    # Plot 2: Network summary
    centrality_data = []
    for node in nodes[:30]:
        if isinstance(node, dict):
            centrality = node.get('centrality_measures', {})
            gene = node.get('gene_symbol', node.get('id', 'Unknown'))
            degree = centrality.get('degree_centrality', 0.0) if isinstance(centrality, dict) else 0.0
            if degree > 0:
                centrality_data.append({'gene': gene, 'degree': degree})
    
    if centrality_data:
        centrality_data.sort(key=lambda x: x['degree'], reverse=True)
        top_15 = centrality_data[:15]
        genes = [d['gene'] for d in top_15]
        degrees = [d['degree'] for d in top_15]
        
        ax2.barh(range(len(genes)), degrees, color='steelblue', alpha=0.7)
        ax2.set_yticks(range(len(genes)))
        ax2.set_yticklabels(genes)
        ax2.set_xlabel('Degree Centrality', fontsize=12, fontweight='bold')
        ax2.set_title('S1: Top 15 Central Nodes', fontsize=14, fontweight='bold')
        ax2.grid(axis='x', alpha=0.3)
        ax2.invert_yaxis()
    else:
        ax2.text(0.5, 0.5, 'No centrality data available', 
                ha='center', va='center', transform=ax2.transAxes, fontsize=12)
        ax2.set_title('S1: Centrality Data', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    output_path = output_dir / 's1_disease_network.png'
    plt.savefig(output_path, bbox_inches='tight')
    print(f"✅ Saved: {output_path}")
    plt.close()

def plot_s4_individual_results(data: Dict[str, Any], output_dir: Path):
    """S4: Per-target panels with biological context."""
    s4 = next((s for s in data['results'] if s['scenario_id'] == 4), None)
    if not s4:
        print("⚠️  Scenario 4 not found")
        return
    
    s4_data = s4['data']
    individual_results = s4_data.get('individual_results', [])
    
    print(f"\n📊 S4 Individual Results: {len(individual_results)} targets")
    
    if not individual_results:
        print("⚠️  No individual results found")
        return
    
    # Extract target data
    target_data = []
    for result in individual_results:
        if isinstance(result, dict):
            target = result.get('target', 'Unknown')
            centrality = result.get('centrality', 0.0)
            has_bio = 'biological_context' in result
            has_drugs = 'drug_annotations' in result
            
            # Get perturbation effect if available
            perturbation = result.get('perturbation_effect', {})
            if isinstance(perturbation, dict):
                effect = perturbation.get('overall_effect', 0.0)
            else:
                effect = 0.0
            
            target_data.append({
                'target': target,
                'centrality': centrality,
                'has_bio': has_bio,
                'has_drugs': has_drugs,
                'effect': effect
            })
    
    if not target_data:
        print("⚠️  No valid target data found")
        return
    
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('S4: AXL Inhibition MRA Simulation - Target Analysis', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    # Plot 1: Centrality
    targets = [d['target'] for d in target_data]
    centralities = [d['centrality'] for d in target_data]
    ax1 = axes[0, 0]
    ax1.barh(targets, centralities, color='steelblue', alpha=0.7)
    ax1.set_xlabel('Centrality', fontsize=11, fontweight='bold')
    ax1.set_title('Target Centrality in Network', fontsize=12, fontweight='bold')
    ax1.grid(axis='x', alpha=0.3)
    ax1.invert_yaxis()
    
    # Plot 2: Biological context presence
    ax2 = axes[0, 1]
    has_bio_count = sum(1 for d in target_data if d['has_bio'])
    has_drugs_count = sum(1 for d in target_data if d['has_drugs'])
    categories = ['Biological\nContext', 'Drug\nAnnotations']
    counts = [has_bio_count, has_drugs_count]
    colors = ['mediumseagreen', 'coral']
    ax2.bar(categories, counts, color=colors, alpha=0.7)
    ax2.set_ylabel('Number of Targets', fontsize=11, fontweight='bold')
    ax2.set_title('Data Completeness per Target', fontsize=12, fontweight='bold')
    ax2.set_ylim(0, len(target_data) + 1)
    ax2.grid(axis='y', alpha=0.3)
    
    # Plot 3: Perturbation effects (if available)
    ax3 = axes[1, 0]
    effects = [d['effect'] for d in target_data if d['effect'] != 0.0]
    if effects:
        ax3.barh(targets[:len(effects)], effects, color='crimson', alpha=0.7)
        ax3.set_xlabel('Perturbation Effect', fontsize=11, fontweight='bold')
        ax3.set_title('Simulated Perturbation Effects', fontsize=12, fontweight='bold')
        ax3.grid(axis='x', alpha=0.3)
        ax3.invert_yaxis()
    else:
        ax3.text(0.5, 0.5, 'No perturbation effect data', 
                ha='center', va='center', transform=ax3.transAxes, fontsize=11)
        ax3.set_title('Perturbation Effects', fontsize=12, fontweight='bold')
    
    # Plot 4: Summary statistics
    ax4 = axes[1, 1]
    ax4.axis('off')
    summary_text = f"""
    Summary Statistics:
    
    Total Targets: {len(target_data)}
    Targets with Biological Context: {has_bio_count}/{len(target_data)} ({100*has_bio_count/len(target_data):.0f}%)
    Targets with Drug Annotations: {has_drugs_count}/{len(target_data)} ({100*has_drugs_count/len(target_data):.0f}%)
    
    Average Centrality: {np.mean(centralities):.4f}
    Max Centrality: {max(centralities):.4f}
    Min Centrality: {min(centralities):.4f}
    
    Validation Score: {s4_data.get('validation_score', 0.0):.3f}
    """
    ax4.text(0.1, 0.5, summary_text, fontsize=11, family='monospace',
            verticalalignment='center', transform=ax4.transAxes)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    output_path = output_dir / 's4_individual_results.png'
    plt.savefig(output_path, bbox_inches='tight')
    print(f"✅ Saved: {output_path}")
    plt.close()

def plot_s5_pathway_comparison(data: Dict[str, Any], output_dir: Path):
    """S5: Pathway comparison (human-only)."""
    s5 = next((s for s in data['results'] if s['scenario_id'] == 5), None)
    if not s5:
        print("⚠️  Scenario 5 not found")
        return
    
    s5_data = s5['data']
    reactome_pathways = s5_data.get('reactome_pathways', [])
    kegg_pathways = s5_data.get('kegg_pathways', [])
    
    print(f"\n📊 S5 Pathways: {len(reactome_pathways)} Reactome, {len(kegg_pathways)} KEGG")
    
    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('S5: AXL Pathway Comparison (Human-Only Filter Applied)', 
                 fontsize=16, fontweight='bold', y=1.02)
    
    # Plot 1: Reactome pathways
    ax1 = axes[0]
    if reactome_pathways:
        pathway_names = []
        gene_counts = []
        for pathway in reactome_pathways[:10]:
            if isinstance(pathway, dict):
                name = pathway.get('name', 'Unknown')
                genes = pathway.get('genes', [])
                pathway_names.append(name[:50] + '...' if len(name) > 50 else name)
                gene_counts.append(len(genes) if isinstance(genes, list) else 0)
        
        if pathway_names:
            ax1.barh(range(len(pathway_names)), gene_counts, color='steelblue', alpha=0.7)
            ax1.set_yticks(range(len(pathway_names)))
            ax1.set_yticklabels(pathway_names)
            ax1.set_xlabel('Gene Count', fontsize=11, fontweight='bold')
            ax1.set_title(f'Reactome Pathways ({len(reactome_pathways)} total)', 
                         fontsize=12, fontweight='bold')
            ax1.grid(axis='x', alpha=0.3)
            ax1.invert_yaxis()
        else:
            ax1.text(0.5, 0.5, 'No Reactome pathway data', 
                    ha='center', va='center', transform=ax1.transAxes, fontsize=11)
    else:
        ax1.text(0.5, 0.5, 'No Reactome pathways found', 
                ha='center', va='center', transform=ax1.transAxes, fontsize=11)
    
    # Plot 2: KEGG pathways
    ax2 = axes[1]
    if kegg_pathways:
        pathway_names = []
        gene_counts = []
        for pathway in kegg_pathways[:10]:
            if isinstance(pathway, dict):
                name = pathway.get('name', 'Unknown')
                genes = pathway.get('genes', [])
                pathway_names.append(name[:50] + '...' if len(name) > 50 else name)
                gene_counts.append(len(genes) if isinstance(genes, list) else 0)
        
        if pathway_names:
            ax2.barh(range(len(pathway_names)), gene_counts, color='coral', alpha=0.7)
            ax2.set_yticks(range(len(pathway_names)))
            ax2.set_yticklabels(pathway_names)
            ax2.set_xlabel('Gene Count', fontsize=11, fontweight='bold')
            ax2.set_title(f'KEGG Pathways ({len(kegg_pathways)} total)', 
                         fontsize=12, fontweight='bold')
            ax2.grid(axis='x', alpha=0.3)
            ax2.invert_yaxis()
        else:
            ax2.text(0.5, 0.5, 'No KEGG pathway data', 
                    ha='center', va='center', transform=ax2.transAxes, fontsize=11)
    else:
        ax2.text(0.5, 0.5, 'No KEGG pathways found', 
                ha='center', va='center', transform=ax2.transAxes, fontsize=11)
    
    plt.tight_layout()
    output_path = output_dir / 's5_pathway_comparison.png'
    plt.savefig(output_path, bbox_inches='tight')
    print(f"✅ Saved: {output_path}")
    plt.close()

def plot_s6_diagnostics(data: Dict[str, Any], output_dir: Path):
    """S6: Diagnostic plots (since no candidate drugs found)."""
    s6 = next((s for s in data['results'] if s['scenario_id'] == 6), None)
    if not s6:
        print("⚠️  Scenario 6 not found")
        return
    
    s6_data = s6['data']
    candidate_drugs = s6_data.get('candidate_drugs', [])
    network_validation = s6_data.get('network_validation', {})
    disease_pathways = s6_data.get('disease_pathways', [])
    
    print(f"\n📊 S6 Diagnostics:")
    print(f"   Candidate drugs: {len(candidate_drugs)}")
    print(f"   Network nodes: {network_validation.get('network_nodes', 0)}")
    print(f"   Disease pathways: {len(disease_pathways)}")
    
    # Create diagnostic figure
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('S6: Drug Repurposing Diagnostics', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    # Plot 1: Network validation metrics
    ax1 = axes[0, 0]
    metrics = {
        'Nodes': network_validation.get('network_nodes', 0),
        'Edges': network_validation.get('network_edges', 0),
        'Density': network_validation.get('network_density', 0.0) * 100,
        'Target Coverage': network_validation.get('target_coverage', 0.0) * 100,
        'Pathway Coverage': network_validation.get('pathway_coverage', 0.0) * 100
    }
    
    if any(v > 0 for v in metrics.values()):
        bars = ax1.bar(range(len(metrics)), list(metrics.values()), 
                      color=['steelblue', 'coral', 'mediumseagreen', 'gold', 'purple'], 
                      alpha=0.7)
        ax1.set_xticks(range(len(metrics)))
        ax1.set_xticklabels(list(metrics.keys()), rotation=45, ha='right')
        ax1.set_ylabel('Value', fontsize=11, fontweight='bold')
        ax1.set_title('Network Validation Metrics', fontsize=12, fontweight='bold')
        ax1.grid(axis='y', alpha=0.3)
        
        # Add value labels
        for i, (bar, val) in enumerate(zip(bars, metrics.values())):
            if val > 0:
                ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(metrics.values())*0.02,
                        f'{val:.1f}', ha='center', va='bottom', fontsize=9)
    else:
        ax1.text(0.5, 0.5, '⚠️ Network metrics are zero\n(Network construction may have failed)', 
                ha='center', va='center', transform=ax1.transAxes, fontsize=11,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        ax1.set_title('Network Validation Metrics', fontsize=12, fontweight='bold')
    
    # Plot 2: Candidate drugs status
    ax2 = axes[0, 1]
    if len(candidate_drugs) > 0:
        scores = [d.get('repurposing_score', 0.0) for d in candidate_drugs 
                 if isinstance(d, dict)]
        if scores:
            ax2.hist(scores, bins=30, edgecolor='black', alpha=0.7, color='steelblue')
            ax2.set_xlabel('Repurposing Score', fontsize=11, fontweight='bold')
            ax2.set_ylabel('Count', fontsize=11, fontweight='bold')
            ax2.set_title(f'Score Distribution ({len(scores)} drugs)', 
                         fontsize=12, fontweight='bold')
            ax2.grid(axis='y', alpha=0.3)
        else:
            ax2.text(0.5, 0.5, 'No score data available', 
                    ha='center', va='center', transform=ax2.transAxes, fontsize=11)
            ax2.set_title('Score Distribution', fontsize=12, fontweight='bold')
    else:
        ax2.text(0.5, 0.5, '⚠️ No candidate drugs found\n(Multi-tier filter may be too strict)', 
                ha='center', va='center', transform=ax2.transAxes, fontsize=11,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        ax2.set_title('Candidate Drugs Status', fontsize=12, fontweight='bold')
    
    # Plot 3: Disease pathways
    ax3 = axes[1, 0]
    if disease_pathways:
        pathway_sources = Counter()
        for pathway in disease_pathways:
            if isinstance(pathway, dict):
                source = pathway.get('source_db', 'unknown')
                pathway_sources[source] += 1
        
        if pathway_sources:
            sources = list(pathway_sources.keys())
            counts = [pathway_sources[s] for s in sources]
            colors = ['steelblue' if s == 'kegg' else 'coral' for s in sources]
            ax3.bar(sources, counts, color=colors, alpha=0.7)
            ax3.set_ylabel('Pathway Count', fontsize=11, fontweight='bold')
            ax3.set_title(f'Disease Pathways by Source ({len(disease_pathways)} total)', 
                         fontsize=12, fontweight='bold')
            ax3.grid(axis='y', alpha=0.3)
        else:
            ax3.text(0.5, 0.5, 'No pathway source data', 
                    ha='center', va='center', transform=ax3.transAxes, fontsize=11)
            ax3.set_title('Disease Pathways', fontsize=12, fontweight='bold')
    else:
        ax3.text(0.5, 0.5, 'No disease pathways found', 
                ha='center', va='center', transform=ax3.transAxes, fontsize=11)
        ax3.set_title('Disease Pathways', fontsize=12, fontweight='bold')
    
    # Plot 4: Summary
    ax4 = axes[1, 1]
    ax4.axis('off')
    summary_text = f"""
    S6 Diagnostic Summary:
    
    Candidate Drugs: {len(candidate_drugs)}
    Network Nodes: {network_validation.get('network_nodes', 0)}
    Network Edges: {network_validation.get('network_edges', 0)}
    Disease Pathways: {len(disease_pathways)}
    
    Validation Score: {s6_data.get('validation_score', 0.0):.3f}
    
    Issues Identified:
    • No candidate drugs (filter too strict?)
    • Network construction returned 0 nodes
    • Need to investigate Step 4 network build
    """
    ax4.text(0.1, 0.5, summary_text, fontsize=10, family='monospace',
            verticalalignment='center', transform=ax4.transAxes,
            bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.3))
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    output_path = output_dir / 's6_diagnostics.png'
    plt.savefig(output_path, bbox_inches='tight')
    print(f"✅ Saved: {output_path}")
    plt.close()

def main():
    if len(sys.argv) < 2:
        # Try to find latest results file
        results_dir = Path('results')
        if results_dir.exists():
            pattern = 'axl_breast_cancer_all_6_scenarios_*.json'
            results_files = list(results_dir.glob(pattern))
            if results_files:
                latest_file = max(results_files, key=lambda p: p.stat().st_mtime)
                print(f"📁 Using latest results file: {latest_file.name}")
            else:
                print("❌ No results files found in results/ directory")
                print(f"\nUsage: {sys.argv[0]} <path_to_results.json>")
                sys.exit(1)
        else:
            print("❌ Results directory not found")
            print(f"\nUsage: {sys.argv[0]} <path_to_results.json>")
            sys.exit(1)
    else:
        latest_file = Path(sys.argv[1])
    
    if not latest_file.exists():
        print(f"❌ Results file not found: {latest_file}")
        sys.exit(1)
    
    # Create output directory
    output_dir = Path('validation_plots')
    output_dir.mkdir(exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"GENERATING WEEK 1 VALIDATION PLOTS")
    print(f"{'='*60}")
    print(f"\nLoading: {latest_file}")
    
    # Load results
    data = load_results(str(latest_file))
    
    print(f"\n📊 Generating visualizations...")
    
    # Generate plots for each scenario
    plot_s1_disease_network(data, output_dir)
    plot_s3_hub_network(data, output_dir)
    plot_s4_individual_results(data, output_dir)
    plot_s5_pathway_comparison(data, output_dir)
    plot_s6_diagnostics(data, output_dir)
    
    print(f"\n{'='*60}")
    print(f"✅ VALIDATION PLOTS GENERATED")
    print(f"{'='*60}")
    print(f"\nOutput directory: {output_dir.absolute()}")
    print(f"\nGenerated plots:")
    print(f"  - s1_disease_network.png")
    print(f"  - s3_hub_proteins.png")
    print(f"  - s4_individual_results.png")
    print(f"  - s5_pathway_comparison.png")
    print(f"  - s6_diagnostics.png")
    print(f"\n🎨 Ready for review!")

if __name__ == '__main__':
    main()





