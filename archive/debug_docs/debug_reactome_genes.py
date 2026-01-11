#!/usr/bin/env python3
"""
Reactome Gene Extraction Debug Script
Purpose: Debug why Reactome pathways have 0 genes
Created: 2025-10-27
"""

import asyncio
import json
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from mcp_clients.reactome_client import ReactomeClient

async def debug_reactome_gene_extraction():
    """Debug Reactome gene extraction step by step"""
    print("🔍 Reactome Gene Extraction Debug")
    print("=" * 50)
    
    try:
        client = ReactomeClient("config/mcp_servers.json")
        
        # Step 1: Search for pathways
        print("\nStep 1: Search for breast cancer pathways")
        search_result = await client.find_pathways_by_disease("breast cancer")
        print(f"✅ Found {len(search_result.get('pathways', []))} pathways")
        
        if search_result.get('pathways'):
            sample_pathway = search_result['pathways'][0]
            print(f"Sample pathway: {sample_pathway}")
            pathway_id = sample_pathway.get('id') or sample_pathway.get('stId')
            print(f"Pathway ID: {pathway_id}")
            
            # Step 2: Get pathway details
            print(f"\nStep 2: Get details for {pathway_id}")
            details = await client.get_pathway_details(pathway_id)
            print(f"✅ Details retrieved")
            print(f"Response keys: {list(details.keys())}")
            
            # Step 3: Check entities
            print(f"\nStep 3: Analyze entities")
            entities = details.get('entities', [])
            print(f"Entities count: {len(entities)}")
            
            if entities:
                print(f"Sample entity: {entities[0]}")
                print(f"Entity keys: {list(entities[0].keys())}")
                
                # Extract gene names from entities
                gene_names = []
                for i, entity in enumerate(entities[:10]):  # Check first 10
                    print(f"\nEntity {i}:")
                    print(f"  Raw entity: {entity}")
                    
                    gene_name = (entity.get('geneName') or 
                               entity.get('displayName') or 
                               entity.get('name'))
                    print(f"  Gene name: {gene_name}")
                    
                    if gene_name and not gene_name.startswith('R-'):
                        gene_symbol = gene_name.split()[0].split('[')[0].split('(')[0]
                        print(f"  Gene symbol: {gene_symbol}")
                        if len(gene_symbol) > 1:
                            gene_names.append(gene_symbol)
                            print(f"  ✅ Added: {gene_symbol}")
                        else:
                            print(f"  ❌ Too short: {gene_symbol}")
                    else:
                        print(f"  ❌ Invalid: {gene_name}")
                
                print(f"\nExtracted genes: {gene_names}")
            else:
                print("❌ No entities found")
                
            # Step 4: Check hasEvent
            print(f"\nStep 4: Analyze hasEvent")
            has_event = details.get('hasEvent', [])
            print(f"HasEvent count: {len(has_event)}")
            
            if has_event:
                print(f"Sample event: {has_event[0]}")
                print(f"Event keys: {list(has_event[0].keys())}")
                
                # Check participants
                participants = has_event[0].get('participants', [])
                print(f"Participants count: {len(participants)}")
                
                if participants:
                    print(f"Sample participant: {participants[0]}")
                    print(f"Participant keys: {list(participants[0].keys())}")
                    
                    # Extract gene names from participants
                    participant_genes = []
                    for i, p in enumerate(participants[:5]):
                        print(f"\nParticipant {i}:")
                        print(f"  Raw participant: {p}")
                        
                        gene_name = (p.get('geneName') or 
                                   p.get('displayName') or 
                                   p.get('name'))
                        print(f"  Gene name: {gene_name}")
                        
                        if gene_name and not gene_name.startswith('R-'):
                            gene_symbol = gene_name.split()[0].split('[')[0].split('(')[0]
                            print(f"  Gene symbol: {gene_symbol}")
                            if len(gene_symbol) > 1:
                                participant_genes.append(gene_symbol)
                                print(f"  ✅ Added: {gene_symbol}")
                            else:
                                print(f"  ❌ Too short: {gene_symbol}")
                        else:
                            print(f"  ❌ Invalid: {gene_name}")
                    
                    print(f"\nParticipant genes: {participant_genes}")
                else:
                    print("❌ No participants found")
            else:
                print("❌ No hasEvent found")
                
            # Step 5: Try backup method - get_pathway_participants
            print(f"\nStep 5: Try backup method - get_pathway_participants")
            try:
                participants_result = await client.get_pathway_participants(pathway_id)
                print(f"✅ Participants retrieved")
                print(f"Response keys: {list(participants_result.keys())}")
                
                participants_list = participants_result.get('participants', [])
                print(f"Participants count: {len(participants_list)}")
                
                if participants_list:
                    print(f"Sample participant: {participants_list[0]}")
                    print(f"Participant keys: {list(participants_list[0].keys())}")
                    
                    # Extract gene names
                    backup_genes = []
                    for i, p in enumerate(participants_list[:5]):
                        print(f"\nBackup Participant {i}:")
                        print(f"  Raw participant: {p}")
                        
                        gene_name = (p.get('displayName') or 
                                   p.get('name') or
                                   p.get('geneName'))
                        print(f"  Gene name: {gene_name}")
                        
                        if gene_name and not gene_name.startswith('R-'):
                            gene_symbol = gene_name.split()[0].split('[')[0].split('(')[0]
                            print(f"  Gene symbol: {gene_symbol}")
                            if len(gene_symbol) > 1:
                                backup_genes.append(gene_symbol)
                                print(f"  ✅ Added: {gene_symbol}")
                            else:
                                print(f"  ❌ Too short: {gene_symbol}")
                        else:
                            print(f"  ❌ Invalid: {gene_name}")
                    
                    print(f"\nBackup genes: {backup_genes}")
                else:
                    print("❌ No participants in backup method")
                    
            except Exception as e:
                print(f"❌ Backup method failed: {e}")
                
        else:
            print("❌ No pathways found")
            
    except Exception as e:
        print(f"❌ Debug failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(debug_reactome_gene_extraction())
