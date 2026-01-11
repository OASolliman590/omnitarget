"""
Simulation Engine Example

Demonstrates simplified perturbation simulation and MRA analysis.
"""

import asyncio
import json
import networkx as nx
from pathlib import Path
import sys

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

from core.simulation.simple_simulator import SimplePerturbationSimulator
from core.simulation.mra_simulator import MRASimulator
from core.simulation.feedback_analyzer import FeedbackAnalyzer


def create_sample_network():
    """Create a sample protein interaction network."""
    G = nx.Graph()
    
    # Add nodes (proteins)
    nodes = [
        'TP53', 'MDM2', 'BRCA1', 'BRCA2', 'ATM', 'ATR', 
        'CHEK1', 'CHEK2', 'CDKN1A', 'RB1', 'MYC', 'CCND1'
    ]
    G.add_nodes_from(nodes)
    
    # Add edges (interactions) with weights
    edges = [
        ('TP53', 'MDM2', 0.9),
        ('TP53', 'BRCA1', 0.8),
        ('TP53', 'ATM', 0.7),
        ('TP53', 'CDKN1A', 0.8),
        ('MDM2', 'BRCA1', 0.6),
        ('BRCA1', 'BRCA2', 0.9),
        ('ATM', 'ATR', 0.5),
        ('CHEK1', 'TP53', 0.7),
        ('CHEK2', 'TP53', 0.6),
        ('CDKN1A', 'RB1', 0.6),
        ('RB1', 'TP53', 0.5),
        ('MYC', 'TP53', 0.4),
        ('CCND1', 'RB1', 0.7)
    ]
    
    for source, target, weight in edges:
        G.add_edge(source, target, weight=weight)
    
    return G


def create_sample_mcp_data():
    """Create sample MCP data for simulation."""
    return {
        'kegg_pathways': [
            {
                'id': 'hsa05224',
                'name': 'Breast cancer',
                'genes': ['TP53', 'BRCA1', 'BRCA2']
            },
            {
                'id': 'hsa04110',
                'name': 'Cell cycle',
                'genes': ['TP53', 'MDM2', 'CHEK1', 'CHEK2', 'CDKN1A', 'RB1', 'MYC', 'CCND1']
            }
        ],
        'reactome_pathways': [
            {
                'id': 'R-HSA-73864',
                'name': 'DNA Repair',
                'genes': ['TP53', 'BRCA1', 'BRCA2', 'ATM', 'ATR']
            }
        ],
        'string_interactions': [
            {
                'protein_a': 'TP53',
                'protein_b': 'MDM2',
                'combined_score': 0.9
            },
            {
                'protein_a': 'TP53',
                'protein_b': 'BRCA1',
                'combined_score': 0.8
            },
            {
                'protein_a': 'BRCA1',
                'protein_b': 'BRCA2',
                'combined_score': 0.9
            },
            {
                'protein_a': 'RB1',
                'protein_b': 'TP53',
                'combined_score': 0.5
            }
        ],
        'hpa_expression': [
            {
                'gene': 'TP53',
                'tissue': 'breast',
                'expression_level': 'High'
            },
            {
                'gene': 'BRCA1',
                'tissue': 'breast',
                'expression_level': 'Medium'
            },
            {
                'gene': 'MDM2',
                'tissue': 'breast',
                'expression_level': 'High'
            }
        ]
    }


async def run_simple_simulation():
    """Run simplified perturbation simulation."""
    print("🔬 Running Simplified Perturbation Simulation")
    print("=" * 50)
    
    # Create network and data
    network = create_sample_network()
    mcp_data = create_sample_mcp_data()
    
    # Create simulator
    simulator = SimplePerturbationSimulator(network, mcp_data)
    
    print(f"📊 Network Info:")
    info = simulator.get_network_info()
    for key, value in info.items():
        print(f"  {key}: {value}")
    
    # Run simulation
    print(f"\n🎯 Simulating TP53 inhibition...")
    result = await simulator.simulate_perturbation(
        target_node='TP53',
        perturbation_strength=0.9,
        mode='inhibit'
    )
    
    print(f"\n📈 Simulation Results:")
    print(f"  Target: {result.target_node}")
    print(f"  Mode: {result.mode}")
    print(f"  Affected nodes: {len(result.affected_nodes)}")
    print(f"  Direct targets: {len(result.direct_targets)}")
    print(f"  Downstream: {len(result.downstream)}")
    print(f"  Execution time: {result.execution_time:.3f}s")
    
    print(f"\n🔍 Top Affected Nodes:")
    sorted_effects = sorted(
        result.affected_nodes.items(), 
        key=lambda x: abs(x[1]), 
        reverse=True
    )[:5]
    
    for node, effect in sorted_effects:
        confidence = result.confidence_scores.get(node, 0.0)
        print(f"  {node}: {effect:.3f} (confidence: {confidence:.3f})")
    
    print(f"\n📊 Network Impact:")
    for metric, value in result.network_impact.items():
        print(f"  {metric}: {value}")
    
    return result


async def run_mra_simulation():
    """Run MRA simulation."""
    print("\n🧮 Running MRA Simulation")
    print("=" * 50)
    
    # Create network and data
    network = create_sample_network()
    mcp_data = create_sample_mcp_data()
    
    # Create MRA simulator
    mra_simulator = MRASimulator(network, mcp_data)
    
    # Run MRA simulation
    print(f"🎯 Running MRA analysis for TP53 inhibition...")
    result = await mra_simulator.simulate_perturbation(
        target_node='TP53',
        perturbation_type='inhibit',
        perturbation_strength=0.9,
        tissue_context='breast'
    )
    
    print(f"\n📈 MRA Results:")
    print(f"  Target: {result.target_node}")
    print(f"  Mode: {result.mode}")
    print(f"  Steady state nodes: {len(result.steady_state)}")
    print(f"  Execution time: {result.execution_time:.3f}s")
    
    print(f"\n🔬 Convergence Info:")
    for key, value in result.convergence_info.items():
        if key != 'convergence_history':
            print(f"  {key}: {value}")
    
    print(f"\n🔍 Top Steady State Effects:")
    sorted_steady_state = sorted(
        result.steady_state.items(), 
        key=lambda x: abs(x[1]), 
        reverse=True
    )[:5]
    
    for node, effect in sorted_steady_state:
        print(f"  {node}: {effect:.3f}")
    
    print(f"\n🔄 Feedback Loops:")
    for i, loop in enumerate(result.feedback_loops):
        print(f"  Loop {i+1}: {loop.nodes} (type: {loop.loop_type}, strength: {loop.strength:.3f})")
    
    print(f"\n📊 Upstream/Downstream Classification:")
    print(f"  Upstream: {len(result.upstream_classification)} nodes")
    print(f"  Downstream: {len(result.downstream_classification)} nodes")
    
    return result


async def run_feedback_analysis():
    """Run feedback loop analysis."""
    print("\n🔄 Running Feedback Loop Analysis")
    print("=" * 50)
    
    # Create network
    network = create_sample_network()
    
    # Create feedback analyzer
    analyzer = FeedbackAnalyzer(network)
    
    # Detect feedback loops
    print(f"🔍 Detecting feedback loops for TP53...")
    feedback_loops = analyzer.detect_feedback_loops('TP53', max_length=5)
    
    print(f"\n📊 Found {len(feedback_loops)} feedback loops:")
    
    for i, loop in enumerate(feedback_loops):
        print(f"\n  Loop {i+1}:")
        print(f"    Nodes: {loop.nodes}")
        print(f"    Type: {loop.loop_type}")
        print(f"    Strength: {loop.strength:.3f}")
        print(f"    Pathway context: {loop.pathway_context}")
        print(f"    Biological function: {loop.biological_function}")
        
        # Analyze loop impact
        impact = analyzer.analyze_loop_impact(loop, 0.9)
        print(f"    Impact analysis:")
        for key, value in impact.items():
            print(f"      {key}: {value}")
    
    # Get network summary
    print(f"\n📈 Network Feedback Summary:")
    summary = analyzer.get_network_feedback_summary()
    for key, value in summary.items():
        print(f"  {key}: {value}")
    
    return feedback_loops


async def compare_simulators():
    """Compare simple vs MRA simulation results."""
    print("\n⚖️ Comparing Simple vs MRA Simulation")
    print("=" * 50)
    
    # Create network and data
    network = create_sample_network()
    mcp_data = create_sample_mcp_data()
    
    # Create both simulators
    simple_simulator = SimplePerturbationSimulator(network, mcp_data)
    mra_simulator = MRASimulator(network, mcp_data)
    
    # Run both simulations
    print("🔬 Running simple simulation...")
    simple_result = await simple_simulator.simulate_perturbation(
        target_node='TP53',
        perturbation_strength=0.9,
        mode='inhibit'
    )
    
    print("🧮 Running MRA simulation...")
    mra_result = await mra_simulator.simulate_perturbation(
        target_node='TP53',
        perturbation_type='inhibit',
        perturbation_strength=0.9
    )
    
    # Compare results
    print(f"\n📊 Comparison Results:")
    print(f"  Simple - Affected nodes: {len(simple_result.affected_nodes)}")
    print(f"  MRA - Steady state nodes: {len(mra_result.steady_state)}")
    print(f"  Simple - Execution time: {simple_result.execution_time:.3f}s")
    print(f"  MRA - Execution time: {mra_result.execution_time:.3f}s")
    
    # Find common nodes
    simple_nodes = set(simple_result.affected_nodes.keys())
    mra_nodes = set(mra_result.steady_state.keys())
    common_nodes = simple_nodes & mra_nodes
    
    print(f"\n🔍 Common affected nodes: {len(common_nodes)}")
    for node in sorted(common_nodes):
        simple_effect = simple_result.affected_nodes[node]
        mra_effect = mra_result.steady_state[node]
        print(f"  {node}: Simple={simple_effect:.3f}, MRA={mra_effect:.3f}")
    
    # Calculate correlation (simplified)
    if common_nodes:
        simple_values = [simple_result.affected_nodes[node] for node in common_nodes]
        mra_values = [mra_result.steady_state[node] for node in common_nodes]
        
        correlation = np.corrcoef(simple_values, mra_values)[0, 1]
        print(f"\n📈 Correlation between methods: {correlation:.3f}")
    
    return simple_result, mra_result


async def main():
    """Main example function."""
    print("🧬 OmniTarget Simulation Engine Example")
    print("=" * 60)
    
    try:
        # Run simple simulation
        simple_result = await run_simple_simulation()
        
        # Run MRA simulation
        mra_result = await run_mra_simulation()
        
        # Run feedback analysis
        feedback_loops = await run_feedback_analysis()
        
        # Compare simulators
        simple_result2, mra_result2 = await compare_simulators()
        
        print("\n🎉 Simulation Engine Example Completed Successfully!")
        print("\n📊 Summary:")
        print(f"  Simple simulation: {len(simple_result.affected_nodes)} affected nodes")
        print(f"  MRA simulation: {len(mra_result.steady_state)} steady state nodes")
        print(f"  Feedback loops detected: {len(feedback_loops)}")
        print(f"  MRA convergence: {mra_result.convergence_info['converged']}")
        
    except Exception as e:
        print(f"❌ Error during simulation: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
