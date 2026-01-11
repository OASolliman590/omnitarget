#!/usr/bin/env python3
"""
Test Circuit: Reactome get_pathway_participants Response Structure

Tests Reactome get_pathway_participants to capture actual response structure
and validate against documentation expectations.
"""

import asyncio
import json
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.core.mcp_client_manager import MCPClientManager


async def test_reactome_participants_structure():
    """Test Reactome get_pathway_participants response structure."""
    print("=" * 80)
    print("Test Circuit: Reactome get_pathway_participants Response Structure")
    print("=" * 80)
    print()
    
    # Create MCP manager
    config_path = str(Path(__file__).parent / "config" / "mcp_servers.json")
    manager = MCPClientManager(config_path)
    
    # Test pathways from user's logs (that returned 0 genes)
    test_pathways = [
        "R-HSA-1227990",
        "R-HSA-4791275",
        "R-HSA-9842640",
        "R-HSA-1640170",  # Cell Cycle (should have genes)
    ]
    
    try:
        # Start all servers
        print("🚀 Starting MCP servers...")
        await manager.start_all()
        print("✅ MCP servers started")
        print()
        
        for pathway_id in test_pathways:
            print(f"Testing pathway: {pathway_id}")
            print("-" * 80)
            
            try:
                # Call get_pathway_participants
                result = await manager.reactome.get_pathway_participants(pathway_id)
                
                print(f"✅ Response received")
                print(f"   Response type: {type(result).__name__}")
                
                # Detailed structure analysis
                if isinstance(result, dict):
                    print(f"   Response keys: {list(result.keys())}")
                    print()
                    print("   Key analysis:")
                    for key in result.keys():
                        value = result[key]
                        print(f"     '{key}': type={type(value).__name__}, "
                              f"is_list={isinstance(value, list)}, "
                              f"is_dict={isinstance(value, dict)}")
                        if isinstance(value, list):
                            print(f"       List length: {len(value)}")
                            if len(value) > 0:
                                first_item = value[0]
                                print(f"       First item type: {type(first_item).__name__}")
                                if isinstance(first_item, dict):
                                    print(f"       First item keys: {list(first_item.keys())[:10]}")
                                    # Check for gene-related fields
                                    gene_fields = ['gene_symbol', 'gene', 'displayName', 'geneName', 'name', 'referenceEntity']
                                    found_fields = [f for f in gene_fields if f in first_item]
                                    if found_fields:
                                        print(f"       Gene fields found: {found_fields}")
                                        if 'referenceEntity' in first_item:
                                            ref_entity = first_item['referenceEntity']
                                            if isinstance(ref_entity, dict):
                                                print(f"       referenceEntity keys: {list(ref_entity.keys())[:5]}")
                                                ref_gene_fields = [f for f in gene_fields if f in ref_entity]
                                                if ref_gene_fields:
                                                    print(f"       referenceEntity gene fields: {ref_gene_fields}")
                
                elif isinstance(result, list):
                    print(f"   List length: {len(result)}")
                    if len(result) > 0:
                        print(f"   First item type: {type(result[0]).__name__}")
                        if isinstance(result[0], dict):
                            print(f"   First item keys: {list(result[0].keys())[:10]}")
                
                # Test extraction logic
                print()
                print("   Testing extraction logic:")
                participant_list = []
                if isinstance(result, dict):
                    participant_list = (result.get('participants') or 
                                      result.get('entities') or
                                      result.get('proteins') or
                                      [])
                elif isinstance(result, list):
                    participant_list = result
                
                print(f"   Found {len(participant_list)} participants")
                
                # Extract genes
                genes = set()
                if participant_list:
                    for idx, participant in enumerate(participant_list[:3]):  # Test first 3
                        if isinstance(participant, dict):
                            gene = (participant.get('gene_symbol') or 
                                   participant.get('gene') or 
                                   participant.get('displayName') or
                                   participant.get('name', ''))
                            
                            if not gene and 'referenceEntity' in participant:
                                ref_entity = participant['referenceEntity']
                                if isinstance(ref_entity, dict):
                                    gene = (ref_entity.get('gene_symbol') or
                                           ref_entity.get('gene') or
                                           ref_entity.get('name', ''))
                            
                            if gene:
                                genes.add(gene.strip().upper())
                                print(f"     Participant {idx}: extracted '{gene}'")
                
                print(f"   Extracted {len(genes)} unique genes from first 3 participants")
                if genes:
                    print(f"   Sample genes: {list(genes)[:5]}")
                else:
                    print("   ⚠️  No genes extracted from participants")
                
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
    success = asyncio.run(test_reactome_participants_structure())
    sys.exit(0 if success else 1)

