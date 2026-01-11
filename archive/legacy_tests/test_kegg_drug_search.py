#!/usr/bin/env python3
"""
Test Circuit: KEGG Drug Search Approaches

Tests different approaches to finding drugs by gene target:
1. search_drugs with gene ID (should fail - 400 error)
2. search_drugs with drug name (should work)
3. find_related_entries gene → drug (should work)
"""

import asyncio
import json
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.core.mcp_client_manager import MCPClientManager


async def test_kegg_drug_search():
    """Test KEGG drug search approaches."""
    print("=" * 80)
    print("Test Circuit: KEGG Drug Search Approaches")
    print("=" * 80)
    print()
    
    # Create MCP manager
    config_path = str(Path(__file__).parent / "config" / "mcp_servers.json")
    manager = MCPClientManager(config_path)
    
    test_gene_id = "hsa:91464"  # AXL
    test_drug_name = "aspirin"  # Known drug name
    
    try:
        # Start all servers
        print("🚀 Starting MCP servers...")
        await manager.start_all()
        print("✅ MCP servers started")
        print()
        
        # Test 1: search_drugs with gene ID (current incorrect approach)
        print(f"1. Testing search_drugs with gene ID: '{test_gene_id}'")
        print("-" * 80)
        try:
            result = await manager.kegg.search_drugs(test_gene_id, limit=10)
            print(f"⚠️  Unexpected success - this should fail with 400")
            print(f"   Response: {result}")
        except Exception as e:
            error_msg = str(e)
            if '400' in error_msg or 'status code 400' in error_msg:
                print(f"✅ Expected 400 error:")
                print(f"   Error: {error_msg[:200]}")
                print(f"   Reason: search_drugs expects drug name, not gene ID")
            else:
                print(f"❌ Unexpected error: {e}")
        print()
        
        # Test 2: search_drugs with drug name (correct usage)
        print(f"2. Testing search_drugs with drug name: '{test_drug_name}'")
        print("-" * 80)
        try:
            result = await manager.kegg.search_drugs(test_drug_name, limit=10)
            
            print(f"✅ Response received")
            print(f"   Response type: {type(result).__name__}")
            
            # Save full response
            output_file = Path(__file__).parent / "kegg_search_drugs_result.json"
            with open(output_file, 'w') as f:
                json.dump(result, f, indent=2, default=str)
            print(f"   Full response saved to: {output_file}")
            print()
            
            # Analyze structure
            if isinstance(result, dict):
                print("   Response keys:")
                for key in result.keys():
                    print(f"     - {key}")
                print()
                
                drugs = result.get('drugs', [])
                if not drugs:
                    drugs = result.get('results', [])
                if not drugs:
                    drugs = result.get('entries', [])
                
                if drugs:
                    print(f"   Found {len(drugs)} drugs")
                    print("   Sample drugs (first 3):")
                    for i, drug in enumerate(drugs[:3]):
                        if isinstance(drug, dict):
                            print(f"     {i+1}. ID: {drug.get('id', drug.get('drug_id', 'N/A'))}")
                            print(f"        Name: {drug.get('name', drug.get('title', 'N/A'))}")
                        else:
                            print(f"     {i+1}. {drug}")
        except Exception as e:
            print(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()
        print()
        
        # Test 3: find_related_entries gene → drug (correct approach)
        print(f"3. Testing find_related_entries (gene → drug): '{test_gene_id}'")
        print("-" * 80)
        try:
            result = await manager.kegg.find_related_entries(
                source_entries=[test_gene_id],
                source_db="gene",
                target_db="drug"
            )
            
            print(f"✅ Response received")
            print(f"   Response type: {type(result).__name__}")
            
            # Save full response
            output_file = Path(__file__).parent / "kegg_gene_to_drug_result.json"
            with open(output_file, 'w') as f:
                json.dump(result, f, indent=2, default=str)
            print(f"   Full response saved to: {output_file}")
            print()
            
            # Analyze structure
            if isinstance(result, dict):
                print("   Response keys:")
                for key in result.keys():
                    print(f"     - {key}")
                print()
                
                # Check link_count
                link_count = result.get('link_count', 'N/A')
                print(f"   link_count: {link_count}")
                
                # Check links
                if 'links' in result:
                    links = result['links']
                    if isinstance(links, dict):
                        gene_links = links.get(test_gene_id, [])
                        print(f"   Drugs for {test_gene_id}: {len(gene_links) if isinstance(gene_links, list) else 'N/A'}")
                        if isinstance(gene_links, list) and len(gene_links) > 0:
                            print(f"   Sample drugs: {gene_links[:5]}")
        except Exception as e:
            print(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()
        print()
        
        print("=" * 80)
        print("✅ Test completed!")
        print("=" * 80)
        print()
        print("Recommendation:")
        print("  Use find_related_entries(source_db='gene', target_db='drug')")
        print("  instead of search_drugs() for finding drugs by gene target")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        try:
            await manager.stop_all()
            print("\n🛑 MCP servers stopped")
        except:
            pass


if __name__ == "__main__":
    success = asyncio.run(test_kegg_drug_search())
    sys.exit(0 if success else 1)




