#!/usr/bin/env python3
"""
Network Pharmacology Analysis Pipeline

This script demonstrates the complete workflow for analyzing gene interaction
networks and drug-target relationships for network pharmacology studies.

Usage:
    python main.py --input results.json --output figures/
    python main.py --gene AXL --expand 2 --interactive
"""

import argparse
import json
import logging
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.data_retrieval import (
    load_network_from_json,
    extract_gene_interactions,
    NetworkPharmacologyRetriever
)
from src.visualization import NetworkVisualizer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Network Pharmacology Analysis Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze from pre-computed results
  python main.py --input ../results/axl_breast_cancer_results.json --output figures/
  
  # Generate interactive visualizations
  python main.py --input results.json --output figures/ --interactive
  
  # Quick analysis with sample data
  python main.py --sample
        """
    )
    
    parser.add_argument(
        '--input', '-i',
        type=str,
        help='Path to input JSON file with pipeline results'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        default='figures',
        help='Output directory for generated figures (default: figures/)'
    )
    
    parser.add_argument(
        '--interactive',
        action='store_true',
        help='Generate interactive HTML visualizations'
    )
    
    parser.add_argument(
        '--sample',
        action='store_true',
        help='Run with sample/demo data'
    )
    
    parser.add_argument(
        '--style',
        type=str,
        default='publication',
        choices=['publication', 'presentation', 'dark'],
        help='Visual style for figures'
    )
    
    parser.add_argument(
        '--dpi',
        type=int,
        default=300,
        help='DPI for static images (default: 300)'
    )
    
    args = parser.parse_args()
    
    # Validate input
    if not args.sample and not args.input:
        parser.error("Either --input or --sample must be provided")
    
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Initialize visualizer
    visualizer = NetworkVisualizer(style=args.style, dpi=args.dpi)
    
    if args.sample:
        # Run with sample data
        logger.info("Running with sample data...")
        network_data, drugs = generate_sample_data()
    else:
        # Load from file
        logger.info(f"Loading data from {args.input}...")
        try:
            raw_data = load_network_from_json(args.input)
            
            # Extract gene interactions
            network_data = extract_gene_interactions(raw_data['mra_results'])
            drugs = raw_data['drug_candidates']
            
            logger.info(f"Loaded {network_data['summary']['total_nodes']} genes, "
                       f"{network_data['summary']['total_edges']} interactions, "
                       f"{len(drugs)} drug candidates")
            
        except Exception as e:
            logger.error(f"Failed to load data: {e}")
            sys.exit(1)
    
    # Generate visualizations
    logger.info("Generating gene interaction network...")
    gene_network_file = visualizer.visualize_gene_network(
        network_data,
        output_path,
        show_all_labels=True,
        interactive=args.interactive,
        title="Comprehensive Gene Interaction Network"
    )
    logger.info(f"Saved: {gene_network_file}")
    
    # Extract gene list for drug-gene network
    genes = list(network_data['nodes'].keys())[:20]
    
    logger.info("Generating drug-gene targeting network...")
    drug_network_file = visualizer.visualize_drug_gene_network(
        genes,
        drugs[:50],  # Top 50 drugs
        output_path,
        show_all_labels=True,
        interactive=args.interactive,
        title="Drug-Gene Targeting Network"
    )
    logger.info(f"Saved: {drug_network_file}")
    
    # Summary
    print("\n" + "=" * 60)
    print("NETWORK PHARMACOLOGY ANALYSIS COMPLETE")
    print("=" * 60)
    print(f"\nGenerated files in {output_path}/:")
    print(f"  - {gene_network_file.name}")
    print(f"  - {drug_network_file.name}")
    print(f"\nNetwork Statistics:")
    print(f"  - Total genes: {network_data['summary']['total_nodes']}")
    print(f"  - Total interactions: {network_data['summary']['total_edges']}")
    print(f"  - Primary targets: {network_data['summary']['targets']}")
    print(f"  - Drug candidates analyzed: {len(drugs)}")
    print("=" * 60)


def generate_sample_data():
    """Generate sample data for demonstration."""
    # Sample network data
    network_data = {
        'nodes': {
            'AXL': {'type': 'target', 'effect': 1.0},
            'AKT1': {'type': 'target', 'effect': 0.9},
            'GRB2': {'type': 'direct', 'effect': 0.7},
            'SRC': {'type': 'direct', 'effect': 0.65},
            'JAK2': {'type': 'direct', 'effect': 0.6},
            'MAPK1': {'type': 'downstream', 'effect': 0.5},
            'PIK3CA': {'type': 'downstream', 'effect': 0.45},
            'STAT3': {'type': 'feedback', 'effect': 0.4},
        },
        'edges': [
            {'source': 'AXL', 'target': 'GRB2', 'type': 'direct'},
            {'source': 'AXL', 'target': 'SRC', 'type': 'direct'},
            {'source': 'AXL', 'target': 'JAK2', 'type': 'direct'},
            {'source': 'AKT1', 'target': 'MAPK1', 'type': 'direct'},
            {'source': 'GRB2', 'target': 'PIK3CA', 'type': 'downstream'},
            {'source': 'JAK2', 'target': 'STAT3', 'type': 'downstream'},
            {'source': 'STAT3', 'target': 'AXL', 'type': 'feedback'},
        ],
        'summary': {
            'total_nodes': 8,
            'total_edges': 7,
            'targets': 2,
            'direct': 3,
            'downstream': 3
        }
    }
    
    # Sample drugs
    drugs = [
        {'drug_name': 'BMS-777607', 'target_protein': 'AXL', 'repurposing_score': 0.85},
        {'drug_name': 'Cabozantinib', 'target_protein': 'AXL', 'repurposing_score': 0.82},
        {'drug_name': 'Gilteritinib', 'target_protein': 'AXL', 'repurposing_score': 0.78},
        {'drug_name': 'Bemcentinib', 'target_protein': 'AXL', 'repurposing_score': 0.75},
        {'drug_name': 'MK-2206', 'target_protein': 'AKT1', 'repurposing_score': 0.72},
        {'drug_name': 'Ipatasertib', 'target_protein': 'AKT1', 'repurposing_score': 0.70},
        {'drug_name': 'Dasatinib', 'target_protein': 'SRC', 'repurposing_score': 0.68},
        {'drug_name': 'Ruxolitinib', 'target_protein': 'JAK2', 'repurposing_score': 0.65},
    ]
    
    return network_data, drugs


if __name__ == '__main__':
    main()
