#!/usr/bin/env python3
"""
HPA UniProt Debug Script
Purpose: Debug HPA UniProt extraction to see why we get 0 UniProt IDs
Created: 2025-10-27
"""

import asyncio
import json
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.core.mcp_client_manager import MCPClientManager

async def debug_hpa_uniprot():
    """Debug HPA UniProt extraction"""
    print("🔍 HPA UniProt Extraction Debug")
    print("=" * 50)
    
    try:
        # Use the same MCP manager as the pipeline
        mcp_manager = MCPClientManager("config/mcp_servers.json")
        
        # Start all MCP servers
        print("Starting MCP servers...")
        await mcp_manager.start_all()
        print("✅ MCP servers started")
        
        # Test genes from our network
        test_genes = ['BRCA1', 'TP53', 'EGFR', 'AXL', 'MYC', 'ALK', 'PTEN']
        
        print(f"\nTesting HPA UniProt extraction for {len(test_genes)} genes:")
        print("-" * 60)
        
        for gene in test_genes:
            print(f"\nTesting {gene}:")
            try:
                protein_info = await mcp_manager.hpa.get_protein_info(gene)
                print(f"  Raw response type: {type(protein_info)}")
                
                if isinstance(protein_info, list):
                    print(f"  Response is list with {len(protein_info)} items")
                    if protein_info:
                        first_item = protein_info[0]
                        print(f"  First item type: {type(first_item)}")
                        print(f"  First item keys: {list(first_item.keys()) if isinstance(first_item, dict) else 'Not a dict'}")
                        
                        if isinstance(first_item, dict):
                            uniprot_data = first_item.get('Uniprot', [])
                            print(f"  Uniprot data: {uniprot_data} (type: {type(uniprot_data)})")
                            
                            if uniprot_data:
                                uniprot_id = uniprot_data[0] if uniprot_data else None
                                print(f"  Extracted UniProt ID: {uniprot_id}")
                            else:
                                print(f"  No UniProt data found")
                        else:
                            print(f"  First item is not a dict: {first_item}")
                elif isinstance(protein_info, dict):
                    print(f"  Response is dict with keys: {list(protein_info.keys())}")
                    uniprot_data = protein_info.get('Uniprot', [])
                    print(f"  Uniprot data: {uniprot_data} (type: {type(uniprot_data)})")
                    
                    if uniprot_data:
                        uniprot_id = uniprot_data[0] if uniprot_data else None
                        print(f"  Extracted UniProt ID: {uniprot_id}")
                    else:
                        print(f"  No UniProt data found")
                else:
                    print(f"  Unexpected response type: {type(protein_info)}")
                    print(f"  Response: {protein_info}")
                    
            except Exception as e:
                print(f"  ❌ Error: {e}")
        
        # Test our current extraction logic
        print(f"\n" + "="*60)
        print("Testing current extraction logic:")
        print("="*60)
        
        for gene in test_genes[:3]:  # Test first 3
            print(f"\nTesting extraction logic for {gene}:")
            try:
                protein_info = await mcp_manager.hpa.get_protein_info(gene)
                
                # Current logic from our code
                uniprot_data = protein_info.get('Uniprot', [])
                uniprot_id = uniprot_data[0] if uniprot_data else None
                
                print(f"  protein_info.get('Uniprot', []): {uniprot_data}")
                print(f"  uniprot_data[0] if uniprot_data else None: {uniprot_id}")
                
                # Check if it's actually a list
                if isinstance(protein_info, list) and protein_info:
                    first_item = protein_info[0]
                    if isinstance(first_item, dict):
                        uniprot_data = first_item.get('Uniprot', [])
                        uniprot_id = uniprot_data[0] if uniprot_data else None
                        print(f"  Fixed logic (list[0]): {uniprot_id}")
                
            except Exception as e:
                print(f"  ❌ Error: {e}")
        
        # Stop MCP servers
        print(f"\nStopping MCP servers...")
        await mcp_manager.stop_all()
        print("✅ MCP servers stopped")
        
    except Exception as e:
        print(f"❌ Debug failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(debug_hpa_uniprot())
