#!/usr/bin/env python3
"""
Test script to debug HPA tissue expression response
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


async def test_hpa_tissue_expression():
    """Test HPA get_tissue_expression to see actual response structure"""

    print("=" * 80)
    print("HPA TISSUE EXPRESSION DEBUG TEST")
    print("=" * 80)

    # Test genes
    test_genes = ['AXL', 'BRCA1', 'TP53']
    tissue_context = 'breast'

    for gene in test_genes:
        print(f"\n{'-' * 80}")
        print(f"Testing gene: {gene}")
        print(f"Target tissue context: {tissue_context}")
        print(f"{'-' * 80}")

        try:
            # Initialize MCP manager
            manager = MCPClientManager('config/mcp_servers.json')

            async with manager.session() as session:
                # Call HPA get_tissue_expression
                print(f"\n📡 Calling HPA.get_tissue_expression(gene='{gene}')...")
                print(f"   (Will filter for tissue_context='{tissue_context}' on client side)")

                result = await session.hpa.get_tissue_expression(gene)

                print(f"\n✅ HPA returned response")
                print(f"   Type: {type(result)}")

                if result is None:
                    print(f"   ⚠️ Response is None")
                elif isinstance(result, dict):
                    print(f"   Keys: {list(result.keys())}")
                    print(f"   Size: {len(result)} items")

                    # Log each key-value pair
                    for key, value in list(result.items())[:10]:  # First 10 keys
                        print(f"\n   Key: '{key}'")
                        print(f"   Value type: {type(value)}")
                        if isinstance(value, (dict, list)) and len(str(value)) < 500:
                            print(f"   Value: {value}")
                        elif isinstance(value, str) and len(value) < 200:
                            print(f"   Value: {value}")
                        else:
                            print(f"   Value: {str(value)[:100]}...")

                elif isinstance(result, list):
                    print(f"   List length: {len(result)}")
                    if len(result) > 0:
                        print(f"   First item type: {type(result[0])}")
                        if isinstance(result[0], dict):
                            print(f"   First item keys: {list(result[0].keys())}")
                            for key, value in list(result[0].items())[:5]:
                                print(f"      {key}: {value}")

                else:
                    print(f"   Value: {result}")

                # Save full response to file
                output_file = f"/Users/omara.soliman/Desktop/Projects /My Projects/15-OmniTarget_/hpa_response_{gene}.json"
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
    asyncio.run(test_hpa_tissue_expression())
