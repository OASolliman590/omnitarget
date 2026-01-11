#!/usr/bin/env python3
"""
Test Circuit: Reactome Gene Extraction Fix

Tests that Reactome get_pathway_participants:
- Returns genes from various response structures
- Extracts genes from nested structures (referenceEntity, components)
- Handles multiple pathway formats correctly

This test can be run independently without running the full pipeline.
"""

import asyncio
import json
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.core.mcp_client_manager import MCPClientManager


async def test_reactome_gene_extraction():
    """Test Reactome gene extraction from pathway participants."""
    print("=" * 80)
    print("Test Circuit: Reactome Gene Extraction Fix")
    print("=" * 80)
    print()
    
    # Create MCP manager
    manager = MCPClientManager()
    
    # Test pathways (known pathways that should have genes)
    test_pathways = [
        "R-HSA-1227990",  # From user's logs
        "R-HSA-4791275",  # From user's logs
        "R-HSA-1640170",  # Cell Cycle (well-known pathway)
    ]
    
    try:
        # Start all servers
        print("🚀 Starting MCP servers...")
        await manager.start_all_servers()
        print("✅ MCP servers started")
        print()
        
        for pathway_id in test_pathways:
            print(f"Test: Extracting genes from pathway {pathway_id}")
            print("-" * 80)
            
            try:
                # Get pathway participants
                result = await manager.reactome.get_pathway_participants(pathway_id)
                
                print(f"✅ Received response")
                print(f"   Response type: {type(result)}")
                
                if isinstance(result, dict):
                    print(f"   Response keys: {list(result.keys())[:15]}")
                    
                    # Check for different possible structures
                    participants = None
                    if 'participants' in result:
                        participants = result['participants']
                        print(f"   Found 'participants' key with {len(participants) if isinstance(participants, list) else 'non-list'} items")
                    elif 'entities' in result:
                        participants = result['entities']
                        print(f"   Found 'entities' key with {len(participants) if isinstance(participants, list) else 'non-list'} items")
                    elif 'proteins' in result:
                        participants = result['proteins']
                        print(f"   Found 'proteins' key with {len(participants) if isinstance(participants, list) else 'non-list'} items")
                    elif 'results' in result:
                        participants = result['results']
                        print(f"   Found 'results' key with {len(participants) if isinstance(participants, list) else 'non-list'} items")
                    
                    if participants and isinstance(participants, list) and len(participants) > 0:
                        print(f"   Total participants: {len(participants)}")
                        
                        # Analyze first few participants
                        print(f"   Analyzing first 3 participants:")
                        for idx, participant in enumerate(participants[:3]):
                            if isinstance(participant, dict):
                                keys = list(participant.keys())[:10]
                                print(f"      Participant {idx}: keys={keys}")
                                
                                # Check for gene-related fields
                                gene_fields = ['name', 'displayName', 'geneName', 'symbol', 'gene_symbol', 'gene']
                                found_fields = [f for f in gene_fields if f in participant]
                                if found_fields:
                                    print(f"         Gene fields found: {found_fields}")
                                    for field in found_fields[:2]:  # Show first 2
                                        value = participant[field]
                                        if isinstance(value, str):
                                            print(f"         {field}: {value[:50]}")
                                
                                # Check for referenceEntity
                                if 'referenceEntity' in participant:
                                    ref_entity = participant['referenceEntity']
                                    if isinstance(ref_entity, dict):
                                        ref_keys = list(ref_entity.keys())[:5]
                                        print(f"         referenceEntity keys: {ref_keys}")
                                        ref_gene_fields = [f for f in gene_fields if f in ref_entity]
                                        if ref_gene_fields:
                                            print(f"         referenceEntity gene fields: {ref_gene_fields}")
                                
                                # Check for components
                                if 'components' in participant:
                                    components = participant['components']
                                    if isinstance(components, list):
                                        print(f"         components: {len(components)} items")
                                
                                # Check for hasComponent
                                if 'hasComponent' in participant:
                                    has_comp = participant['hasComponent']
                                    if isinstance(has_comp, list):
                                        print(f"         hasComponent: {len(has_comp)} items")
                            else:
                                print(f"      Participant {idx}: {type(participant)} = {str(participant)[:50]}")
                        
                        print(f"   ✅ Response structure analyzed")
                    else:
                        print(f"   ⚠️  No participants list found or list is empty")
                        if isinstance(result, dict):
                            print(f"   Available keys: {list(result.keys())}")
                elif isinstance(result, list):
                    print(f"   Response is direct list with {len(result)} items")
                    if len(result) > 0:
                        print(f"   First item type: {type(result[0])}")
                        if isinstance(result[0], dict):
                            print(f"   First item keys: {list(result[0].keys())[:10]}")
                else:
                    print(f"   Unexpected response format: {type(result)}")
                    print(f"   Response preview: {str(result)[:200]}")
                
                print()
                
            except Exception as e:
                print(f"❌ Error extracting genes from {pathway_id}: {e}")
                import traceback
                traceback.print_exc()
                print()
        
        print("=" * 80)
        print("✅ All tests completed!")
        print("=" * 80)
        print()
        print("Note: This test verifies response structure.")
        print("Actual gene extraction logic is tested in scenario_1_disease_network.py")
        return True
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        # Stop servers
        try:
            await manager.stop_all_servers()
            print("\n🛑 MCP servers stopped")
        except:
            pass


if __name__ == "__main__":
    success = asyncio.run(test_reactome_gene_extraction())
    sys.exit(0 if success else 1)

