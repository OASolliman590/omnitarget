#!/usr/bin/env python3
"""
Test script to debug STRING interaction network response
"""
import asyncio
import json
import logging
from src.core.mcp_client_manager import MCPClientManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def test_string_confidence():
    """Test STRING get_interaction_network to see actual response structure"""

    print("=" * 80)
    print("STRING INTERACTION NETWORK DEBUG TEST")
    print("=" * 80)

    # Test genes
    test_genes = ['AXL', 'BRCA1', 'TP53']

    for gene_set in [test_genes[:2], test_genes]:
        print(f"\n{'-' * 80}")
        print(f"Testing genes: {gene_set}")
        print(f"{'-' * 80}")

        try:
            # Initialize MCP manager
            manager = MCPClientManager('config/mcp_servers.json')

            async with manager.session() as session:
                # Call STRING get_interaction_network
                print(f"\n📡 Calling STRING.get_interaction_network()...")
                print(f"   protein_ids: {gene_set}")
                print(f"   species: 9606")
                print(f"   required_score: 400")
                print(f"   add_nodes: 20")

                result = await session.string.get_interaction_network(
                    protein_ids=gene_set,
                    species="9606",
                    required_score=400,
                    add_nodes=20
                )

                print(f"\n✅ STRING returned response")
                print(f"   Type: {type(result)}")

                if result is None:
                    print(f"   ⚠️ Response is None")
                elif isinstance(result, dict):
                    print(f"   Keys: {list(result.keys())}")
                    print(f"   Size: {len(result)} items")

                    # Check nodes
                    if 'nodes' in result:
                        nodes = result['nodes']
                        print(f"\n   📊 NODES: {len(nodes)}")
                        if nodes:
                            print(f"      First node keys: {list(nodes[0].keys())}")
                            for key, value in list(nodes[0].items())[:5]:
                                print(f"         {key}: {value}")

                    # Check edges
                    if 'edges' in result:
                        edges = result['edges']
                        print(f"\n   📊 EDGES: {len(edges)}")

                        # Check for confidence-related fields in edges
                        if edges:
                            edge_keys = set()
                            confidence_keys = ['score', 'combined_score', 'confidence', 'weight', 'prob', 'p_value']
                            found_confidence_keys = []

                            for edge in edges[:3]:  # Check first 3 edges
                                edge_keys.update(edge.keys())
                                for ck in confidence_keys:
                                    if ck in edge:
                                        found_confidence_keys.append(ck)
                                        print(f"\n   🔍 Found confidence field '{ck}': {edge[ck]}")

                            print(f"\n   📋 All edge keys: {sorted(list(edge_keys))}")
                            print(f"   🎯 Confidence fields found: {sorted(list(set(found_confidence_keys)))}")

                            # Calculate confidence stats
                            confidence_values = []
                            for edge in edges:
                                for ck in confidence_keys:
                                    if ck in edge and edge[ck] is not None:
                                        try:
                                            val = float(edge[ck])
                                            confidence_values.append(val)
                                        except:
                                            pass

                            if confidence_values:
                                print(f"\n   📈 Confidence statistics:")
                                print(f"      Count: {len(confidence_values)}")
                                print(f"      Min: {min(confidence_values):.3f}")
                                print(f"      Max: {max(confidence_values):.3f}")
                                print(f"      Mean: {sum(confidence_values)/len(confidence_values):.3f}")
                            else:
                                print(f"\n   ⚠️ No confidence values found!")

                elif isinstance(result, list):
                    print(f"   List length: {len(result)}")
                    if len(result) > 0:
                        print(f"   First item type: {type(result[0])}")
                        if isinstance(result[0], dict):
                            print(f"   First item keys: {list(result[0].keys())}")

                else:
                    print(f"   Value: {result}")

                # Save full response to file
                output_file = f"/Users/omara.soliman/Desktop/Projects /My Projects/15-OmniTarget_/string_response_{'_'.join(gene_set).lower()}.json"
                with open(output_file, 'w') as f:
                    json.dump(result, f, indent=2, default=str)
                print(f"\n💾 Full response saved to: {output_file}")

        except Exception as e:
            print(f"\n❌ Error: {e}")
            import traceback
            traceback.print_exc()

        # Small delay between requests
        await asyncio.sleep(1)

    print(f"\n{'=' * 80}")
    print("TEST COMPLETE")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    asyncio.run(test_string_confidence())
