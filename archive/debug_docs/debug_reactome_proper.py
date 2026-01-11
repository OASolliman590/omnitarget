#!/usr/bin/env python3
"""
Proper Reactome Gene Extraction Debug Script
Purpose: Debug Reactome gene extraction using the same MCP manager as the pipeline
Created: 2025-10-27
"""

import asyncio
import json
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.core.mcp_client_manager import MCPClientManager

async def debug_reactome_proper():
    """Debug Reactome gene extraction using proper MCP manager"""
    print("🔍 Proper Reactome Gene Extraction Debug")
    print("=" * 50)
    
    try:
        # Use the same MCP manager as the pipeline
        mcp_manager = MCPClientManager("config/mcp_servers.json")
        
        # Start all MCP servers (same as pipeline)
        print("Starting MCP servers...")
        await mcp_manager.start_all()
        print("✅ MCP servers started")
        
        # Test Reactome
        print("\nTesting Reactome...")
        search_result = await mcp_manager.reactome.find_pathways_by_disease("breast cancer")
        print(f"✅ Found {len(search_result.get('pathways', []))} pathways")
        
        if search_result.get('pathways'):
            sample_pathway = search_result['pathways'][0]
            print(f"Sample pathway: {sample_pathway}")
            pathway_id = sample_pathway.get('id') or sample_pathway.get('stId')
            print(f"Pathway ID: {pathway_id}")
            
            # Get pathway details
            print(f"\nGetting details for {pathway_id}...")
            details = await mcp_manager.reactome.get_pathway_details(pathway_id)
            print(f"✅ Details retrieved")
            print(f"Response keys: {list(details.keys())}")
            
            # Check entities
            entities = details.get('entities', [])
            print(f"\nEntities count: {len(entities)}")
            
            if entities:
                print(f"Sample entity: {entities[0]}")
                print(f"Entity keys: {list(entities[0].keys())}")
                
                # Extract gene names
                gene_names = []
                for i, entity in enumerate(entities[:10]):
                    gene_name = (entity.get('geneName') or 
                               entity.get('displayName') or 
                               entity.get('name'))
                    if gene_name and not gene_name.startswith('R-'):
                        gene_symbol = gene_name.split()[0].split('[')[0].split('(')[0]
                        if len(gene_symbol) > 1:
                            gene_names.append(gene_symbol)
                
                print(f"Extracted genes from entities: {gene_names}")
            else:
                print("❌ No entities found")
                
            # Check hasEvent
            has_event = details.get('hasEvent', [])
            print(f"\nHasEvent count: {len(has_event)}")
            
            if has_event and has_event[0].get('participants'):
                participants = has_event[0]['participants']
                print(f"Participants count: {len(participants)}")
                
                # Extract gene names from participants
                participant_genes = []
                for p in participants[:10]:
                    gene_name = (p.get('geneName') or 
                               p.get('displayName') or 
                               p.get('name'))
                    if gene_name and not gene_name.startswith('R-'):
                        gene_symbol = gene_name.split()[0].split('[')[0].split('(')[0]
                        if len(gene_symbol) > 1:
                            participant_genes.append(gene_symbol)
                
                print(f"Extracted genes from participants: {participant_genes}")
            else:
                print("❌ No participants found")
                
            # Try backup method
            print(f"\nTrying backup method...")
            try:
                participants_result = await mcp_manager.reactome.get_pathway_participants(pathway_id)
                participants_list = participants_result.get('participants', [])
                print(f"Backup participants count: {len(participants_list)}")
                
                if participants_list:
                    backup_genes = []
                    for p in participants_list[:10]:
                        gene_name = (p.get('displayName') or 
                                   p.get('name') or
                                   p.get('geneName'))
                        if gene_name and not gene_name.startswith('R-'):
                            gene_symbol = gene_name.split()[0].split('[')[0].split('(')[0]
                            if len(gene_symbol) > 1:
                                backup_genes.append(gene_symbol)
                    
                    print(f"Extracted genes from backup: {backup_genes}")
                else:
                    print("❌ No participants in backup method")
                    
            except Exception as e:
                print(f"❌ Backup method failed: {e}")
                
        # Test HPA
        print(f"\nTesting HPA...")
        try:
            protein_info = await mcp_manager.hpa.get_protein_info('BRCA1')
            print(f"✅ HPA working: {protein_info}")
        except Exception as e:
            print(f"❌ HPA failed: {e}")
            
        # Stop MCP servers
        print(f"\nStopping MCP servers...")
        await mcp_manager.stop_all()
        print("✅ MCP servers stopped")
        
    except Exception as e:
        print(f"❌ Debug failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(debug_reactome_proper())
