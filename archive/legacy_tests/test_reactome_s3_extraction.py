#!/usr/bin/env python3
"""
Test Circuit: Reactome S3 Gene Extraction

Tests Reactome get_pathway_participants and get_pathway_details
to understand why S3 returns 0 genes while S1 works.
"""

import asyncio
import json
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.core.mcp_client_manager import MCPClientManager


async def test_reactome_s3_extraction():
    """Test Reactome pathway gene extraction methods."""
    print("=" * 80)
    print("Test Circuit: Reactome S3 Gene Extraction")
    print("=" * 80)
    print()
    
    # Create MCP manager
    config_path = str(Path(__file__).parent / "config" / "mcp_servers.json")
    manager = MCPClientManager(config_path)
    
    # Test pathway from user's logs
    test_pathway = "R-HSA-1227990"  # One that returned 0 genes in S3
    
    try:
        # Start all servers
        print("🚀 Starting MCP servers...")
        await manager.start_all()
        print("✅ MCP servers started")
        print()
        
        print(f"Testing pathway: {test_pathway}")
        print("-" * 80)
        
        # Test 1: get_pathway_participants
        print("\n1. Testing get_pathway_participants:")
        print("-" * 80)
        try:
            participants_result = await manager.reactome.get_pathway_participants(test_pathway)
            
            print(f"✅ Response received")
            print(f"   Response type: {type(participants_result).__name__}")
            
            # Save full response
            output_file = Path(__file__).parent / f"reactome_participants_{test_pathway}.json"
            with open(output_file, 'w') as f:
                json.dump(participants_result, f, indent=2, default=str)
            print(f"   Full response saved to: {output_file}")
            print()
            
            # Analyze structure
            if isinstance(participants_result, dict):
                print("   Response keys:")
                for key in list(participants_result.keys())[:10]:
                    print(f"     - {key}")
                print()
                
                # Check for participants list
                participants = participants_result.get('participants', [])
                if not participants:
                    participants = participants_result.get('entities', [])
                if not participants:
                    participants = participants_result.get('results', [])
                
                if participants:
                    print(f"   Found {len(participants)} participants")
                    print()
                    print("   Sample participants (first 5):")
                    for i, participant in enumerate(participants[:5]):
                        if isinstance(participant, dict):
                            print(f"     {i+1}. Type: {participant.get('type', 'N/A')}")
                            print(f"        Name: {participant.get('name', participant.get('displayName', 'N/A'))}")
                            print(f"        Keys: {list(participant.keys())[:5]}")
                        else:
                            print(f"     {i+1}. {participant}")
                    print()
                    
                    # Test S3 extraction logic
                    print("   Testing S3 extraction logic:")
                    genes = set()
                    for participant in participants:
                        if isinstance(participant, dict):
                            part_type = participant.get('type', '').lower()
                            if part_type not in ['pathway', 'reaction', 'event']:
                                part_gene = (participant.get('gene_symbol') or
                                           participant.get('gene') or
                                           participant.get('displayName') or
                                           participant.get('name', ''))
                                if part_gene and part_gene.strip():
                                    gene_clean = part_gene.strip().upper()
                                    gene_clean = gene_clean.split(':')[0].split('-')[0]
                                    if gene_clean.isalnum() and 2 <= len(gene_clean) <= 15:
                                        genes.add(gene_clean)
                    
                    print(f"     Extracted genes: {len(genes)}")
                    if genes:
                        print(f"     Gene symbols: {list(genes)[:10]}")
                    else:
                        print(f"     ⚠️  No genes extracted from participants")
                else:
                    print("   ⚠️  No participants found in response")
            elif isinstance(participants_result, list):
                print(f"   Response is list with {len(participants_result)} items")
                if len(participants_result) > 0:
                    print(f"   First item type: {type(participants_result[0]).__name__}")
        except Exception as e:
            print(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()
        
        # Test 2: get_pathway_details (fallback)
        print("\n2. Testing get_pathway_details (fallback):")
        print("-" * 80)
        try:
            details_result = await manager.reactome.get_pathway_details(test_pathway)
            
            print(f"✅ Response received")
            print(f"   Response type: {type(details_result).__name__}")
            
            # Save full response
            output_file = Path(__file__).parent / f"reactome_details_{test_pathway}.json"
            with open(output_file, 'w') as f:
                json.dump(details_result, f, indent=2, default=str)
            print(f"   Full response saved to: {output_file}")
            print()
            
            # Analyze structure
            if isinstance(details_result, dict):
                print("   Response keys:")
                for key in list(details_result.keys())[:15]:
                    print(f"     - {key}")
                print()
                
                # Check for entities
                entities = details_result.get('entities', [])
                if entities:
                    print(f"   Found {len(entities)} entities")
                    print()
                    print("   Sample entities (first 5):")
                    for i, entity in enumerate(entities[:5]):
                        if isinstance(entity, dict):
                            entity_type = entity.get('type', 'N/A')
                            print(f"     {i+1}. Type: {entity_type}")
                            
                            # Check for gene_symbol
                            gene_symbol = (entity.get('gene_symbol') or
                                         entity.get('gene') or
                                         entity.get('displayName') or
                                         entity.get('name', ''))
                            print(f"        Gene symbol: {gene_symbol}")
                            
                            # Check for referenceEntity
                            if 'referenceEntity' in entity:
                                ref_entity = entity['referenceEntity']
                                if isinstance(ref_entity, dict):
                                    ref_gene = (ref_entity.get('gene_symbol') or
                                              ref_entity.get('gene') or
                                              ref_entity.get('name', ''))
                                    print(f"        ReferenceEntity gene: {ref_gene}")
                            
                            print(f"        Keys: {list(entity.keys())[:5]}")
                    print()
                    
                    # Test S3 fallback extraction logic
                    print("   Testing S3 fallback extraction logic:")
                    genes = set()
                    for entity in entities:
                        if isinstance(entity, dict):
                            entity_type = entity.get('type', '').lower()
                            if entity_type not in ['pathway', 'reaction', 'event']:
                                entity_gene = (entity.get('gene_symbol') or
                                            entity.get('gene') or
                                            entity.get('displayName') or
                                            entity.get('name', ''))
                                
                                # Extract from referenceEntity
                                if not entity_gene and 'referenceEntity' in entity:
                                    ref_entity = entity['referenceEntity']
                                    if isinstance(ref_entity, dict):
                                        entity_gene = (ref_entity.get('gene_symbol') or
                                                      ref_entity.get('gene') or
                                                      ref_entity.get('name', ''))
                                
                                if entity_gene and entity_gene.strip():
                                    gene_clean = entity_gene.strip().upper()
                                    gene_clean = gene_clean.split(':')[0].split('-')[0]
                                    if gene_clean.isalnum() and 2 <= len(gene_clean) <= 15:
                                        genes.add(gene_clean)
                    
                    print(f"     Extracted genes: {len(genes)}")
                    if genes:
                        print(f"     Gene symbols: {list(genes)[:10]}")
                    else:
                        print(f"     ⚠️  No genes extracted from entities")
                
                # Check for hasEvent
                has_event = details_result.get('hasEvent', [])
                if has_event:
                    print(f"\n   Found {len(has_event)} events")
                    print("   Checking events for participants...")
                    event_genes = set()
                    for event in has_event[:5]:
                        if isinstance(event, dict) and event.get('participants'):
                            for participant in event['participants']:
                                if isinstance(participant, dict):
                                    part_type = participant.get('type', '').lower()
                                    if part_type not in ['pathway', 'reaction', 'event']:
                                        part_gene = (participant.get('gene_symbol') or
                                                   participant.get('gene') or
                                                   participant.get('name', ''))
                                        if part_gene and part_gene.strip():
                                            gene_clean = part_gene.strip().upper()
                                            gene_clean = gene_clean.split(':')[0].split('-')[0]
                                            if gene_clean.isalnum() and 2 <= len(gene_clean) <= 15:
                                                event_genes.add(gene_clean)
                    if event_genes:
                        print(f"     Found {len(event_genes)} genes in events")
                        print(f"     Gene symbols: {list(event_genes)[:10]}")
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
    success = asyncio.run(test_reactome_s3_extraction())
    sys.exit(0 if success else 1)




