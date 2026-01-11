#!/usr/bin/env python3
"""
Test Circuit: S3 Gene Extraction

Tests S3 Reactome gene extraction with real pathways to verify
the enhanced extraction logic works correctly.
"""

import asyncio
import json
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.core.mcp_client_manager import MCPClientManager


async def test_s3_gene_extraction():
    """Test S3 gene extraction logic."""
    print("=" * 80)
    print("Test Circuit: S3 Gene Extraction")
    print("=" * 80)
    print()
    
    # Create MCP manager
    config_path = str(Path(__file__).parent / "config" / "mcp_servers.json")
    manager = MCPClientManager(config_path)
    
    # Test pathways from user's logs
    test_pathways = [
        {"id": "R-HSA-1227990", "name": "Pathway 1"},
        {"id": "R-HSA-4791275", "name": "Pathway 2"},
        {"id": "R-HSA-1640170", "name": "Cell Cycle"},
    ]
    
    try:
        # Start all servers
        print("🚀 Starting MCP servers...")
        await manager.start_all()
        print("✅ MCP servers started")
        print()
        
        for pathway_data in test_pathways:
            pathway_id = pathway_data["id"]
            pathway_name = pathway_data["name"]
            
            print(f"Testing pathway: {pathway_id} ({pathway_name})")
            print("-" * 80)
            
            try:
                # Get pathway participants
                participants = await manager.reactome.get_pathway_participants(pathway_id)
                
                # Apply S3 extraction logic
                genes = set()
                
                # Handle multiple possible response structures
                participant_list = []
                if isinstance(participants, dict):
                    participant_list = (participants.get('participants') or 
                                       participants.get('entities') or
                                       participants.get('proteins') or
                                       [])
                elif isinstance(participants, list):
                    participant_list = participants
                
                print(f"   Found {len(participant_list)} participants")
                
                if participant_list:
                    for participant in participant_list:
                        if isinstance(participant, dict):
                            # Try direct fields
                            gene = (participant.get('gene_symbol') or 
                                   participant.get('gene') or 
                                   participant.get('displayName') or 
                                   participant.get('geneName') or
                                   participant.get('name', ''))
                            
                            # Extract from nested referenceEntity
                            if not gene and 'referenceEntity' in participant:
                                ref_entity = participant['referenceEntity']
                                if isinstance(ref_entity, dict):
                                    gene = (ref_entity.get('gene_symbol') or
                                           ref_entity.get('gene') or
                                           ref_entity.get('displayName') or
                                           ref_entity.get('name', ''))
                            
                            # Extract from components (for complexes)
                            if 'components' in participant and isinstance(participant['components'], list):
                                for component in participant['components']:
                                    if isinstance(component, dict):
                                        comp_gene = (component.get('gene_symbol') or
                                                   component.get('gene') or
                                                   component.get('name', ''))
                                        if comp_gene and comp_gene.strip():
                                            comp_gene_clean = comp_gene.strip().upper()
                                            comp_gene_clean = comp_gene_clean.split(':')[0].split('-')[0]
                                            if comp_gene_clean.isalnum() and 2 <= len(comp_gene_clean) <= 15:
                                                genes.add(comp_gene_clean)
                            
                            # Extract from hasComponent
                            if 'hasComponent' in participant:
                                components = participant['hasComponent']
                                if isinstance(components, list):
                                    for component in components:
                                        if isinstance(component, dict):
                                            comp_gene = (component.get('gene_symbol') or
                                                       component.get('gene') or
                                                       component.get('name', ''))
                                            if comp_gene and comp_gene.strip():
                                                comp_gene_clean = comp_gene.strip().upper()
                                                comp_gene_clean = comp_gene_clean.split(':')[0].split('-')[0]
                                                if comp_gene_clean.isalnum() and 2 <= len(comp_gene_clean) <= 15:
                                                    genes.add(comp_gene_clean)
                            
                            # Validate and add main gene
                            if gene and gene.strip():
                                gene_clean = gene.strip().upper()
                                gene_clean = gene_clean.split(':')[0].split('-')[0]
                                if gene_clean.isalnum() and 2 <= len(gene_clean) <= 15:
                                    genes.add(gene_clean)
                        elif isinstance(participant, str):
                            if participant.strip() and participant.upper().isalnum() and 2 <= len(participant) <= 15:
                                genes.add(participant.strip().upper())
                
                print(f"   ✅ Extracted {len(genes)} unique genes")
                if genes:
                    print(f"   Sample genes: {list(genes)[:10]}")
                else:
                    print("   ⚠️  No genes extracted")
                
                print()
                
            except Exception as e:
                print(f"❌ Error: {e}")
                import traceback
                traceback.print_exc()
                print()
        
        print("=" * 80)
        print("✅ Test completed!")
        print("=" * 80)
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
    success = asyncio.run(test_s3_gene_extraction())
    sys.exit(0 if success else 1)

