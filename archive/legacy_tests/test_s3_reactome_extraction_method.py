#!/usr/bin/env python3
"""
Test Circuit: S3 Reactome Gene Extraction Method

Tests S3's _extract_reactome_genes method with a known pathway ID
to capture the raw response structure and understand why 0 genes are extracted.
"""

import asyncio
import json
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.core.mcp_client_manager import MCPClientManager
from src.scenarios.scenario_3_cancer_analysis import CancerAnalysisScenario


async def test_s3_extraction_method():
    """Test S3's _extract_reactome_genes method."""
    print("=" * 80)
    print("Test Circuit: S3 Reactome Gene Extraction Method")
    print("=" * 80)
    print()
    
    # Create MCP manager
    config_path = str(Path(__file__).parent / "config" / "mcp_servers.json")
    manager = MCPClientManager(config_path)
    
    # Create S3 scenario instance
    scenario = CancerAnalysisScenario(manager)
    
    # Test pathway from user's logs - one that returned 0 genes
    test_pathway = "R-HSA-1227990"
    
    try:
        # Start all servers
        print("🚀 Starting MCP servers...")
        await manager.start_all()
        print("✅ MCP servers started")
        print()
        
        print(f"Testing S3 _extract_reactome_genes with pathway: {test_pathway}")
        print("-" * 80)
        
        # Test S3's extraction logic by simulating what _phase2_cancer_pathway_discovery does
        print("\n📞 Testing S3 extraction logic (simulating _phase2_cancer_pathway_discovery)...")
        
        # Step 1: Get pathway participants (S3's primary method)
        genes = set()
        try:
            participants = await manager.reactome.get_pathway_participants(test_pathway)
            
            # Handle multiple response structures (S3-style)
            participant_list = []
            if isinstance(participants, dict):
                participant_list = (participants.get('participants') or 
                                   participants.get('entities') or
                                   participants.get('proteins') or
                                   [])
            elif isinstance(participants, list):
                participant_list = participants
            
            # Extract genes using S3's robust extraction
            if participant_list:
                for participant in participant_list:
                    if isinstance(participant, dict):
                        # Filter out pathways/reactions
                        participant_type = participant.get('type', '').lower()
                        if participant_type in ['pathway', 'reaction', 'event']:
                            continue
                        
                        # Use S3's extraction method
                        gene_names = scenario._extract_gene_names_from_entity_s3(participant)
                        genes.update(gene_names)
            
            # Fallback: get_pathway_details if participants empty
            if not genes:
                print("   No genes from participants, trying get_pathway_details fallback...")
                pathway_details = await manager.reactome.get_pathway_details(test_pathway)
                
                if pathway_details:
                    # Extract from entities (if present)
                    if pathway_details.get('entities'):
                        for entity in pathway_details['entities']:
                            entity_type = entity.get('type', '').lower()
                            if entity_type not in ['pathway', 'reaction', 'event']:
                                gene_names = scenario._extract_gene_names_from_entity_s3(entity)
                                genes.update(gene_names)
                    
                    # CRITICAL FIX: Extract from participants (get_pathway_details format)
                    # Participants have refEntities array with actual protein info
                    if pathway_details.get('participants'):
                        print(f"   Found {len(pathway_details['participants'])} participants in details")
                        for participant in pathway_details['participants']:
                            if isinstance(participant, dict):
                                # Extract from refEntities array (Reactome details format)
                                ref_entities = participant.get('refEntities', [])
                                if isinstance(ref_entities, list) and ref_entities:
                                    print(f"   Processing participant with {len(ref_entities)} refEntities")
                                    for ref_entity in ref_entities:
                                        if isinstance(ref_entity, dict):
                                            # Extract from displayName (e.g., "ERBB2" from "UniProt:O14511 NRG2")
                                            display_name = ref_entity.get('displayName', '')
                                            if display_name:
                                                # Parse gene symbol from displayName
                                                parts = display_name.split()
                                                for p in parts:
                                                    # Remove UniProt: prefix and extract gene
                                                    clean = p.split(':')[-1].split('-')[0].split('(')[0].split(')')[0].strip()
                                                    if clean and clean.isupper() and 2 <= len(clean) <= 15 and clean.isalnum():
                                                        genes.add(clean)
                                                        break
                                
                                # Also extract from participant's displayName directly
                                display_name = participant.get('displayName', '')
                                if display_name:
                                    # Try to extract gene from displayName (e.g., "p-6Y-ERBB2")
                                    parts = display_name.split()
                                    for p in parts:
                                        # Remove prefixes like p-, p-6Y-, etc.
                                        clean = p.split(':')[0].split('-')[-1].split('(')[0].split(')')[0].strip()
                                        if clean and clean.isupper() and 2 <= len(clean) <= 15 and clean.isalnum():
                                            genes.add(clean)
                                            break
                    
                    if pathway_details.get('hasEvent'):
                        for event in pathway_details['hasEvent']:
                            if event.get('participants'):
                                for participant in event['participants']:
                                    gene_names = scenario._extract_gene_names_from_entity_s3(participant)
                                    genes.update(gene_names)
            
            # Filter and validate
            filtered_genes = scenario._filter_valid_gene_symbols_s3(genes)
            genes = filtered_genes
        except Exception as e:
            print(f"   ❌ Error during extraction: {e}")
            import traceback
            traceback.print_exc()
            genes = []
        
        print(f"\n📊 Results:")
        print(f"   Genes extracted: {len(genes)}")
        if genes:
            print(f"   Gene symbols: {genes[:10]}")  # Show first 10
        else:
            print("   ⚠️  No genes extracted!")
        
        # Also test the raw Reactome response to understand structure
        print(f"\n📞 Testing raw Reactome responses for comparison...")
        
        # Test get_pathway_participants
        print("\n1. get_pathway_participants:")
        try:
            participants = await manager.reactome.get_pathway_participants(test_pathway)
            print(f"   ✅ Response type: {type(participants)}")
            if isinstance(participants, dict):
                print(f"   ✅ Response keys: {list(participants.keys())[:10]}")
                # Save to file for inspection
                with open("test_s3_participants_response.json", "w") as f:
                    json.dump(participants, f, indent=2, default=str)
                print(f"   💾 Saved to: test_s3_participants_response.json")
        except Exception as e:
            print(f"   ❌ Error: {e}")
        
        # Test get_pathway_details
        print("\n2. get_pathway_details:")
        try:
            details = await manager.reactome.get_pathway_details(test_pathway)
            print(f"   ✅ Response type: {type(details)}")
            if isinstance(details, dict):
                print(f"   ✅ Response keys: {list(details.keys())[:10]}")
                # Save to file for inspection
                with open("test_s3_details_response.json", "w") as f:
                    json.dump(details, f, indent=2, default=str)
                print(f"   💾 Saved to: test_s3_details_response.json")
        except Exception as e:
            print(f"   ❌ Error: {e}")
        
        # Summary
        print("\n" + "=" * 80)
        print("Summary")
        print("=" * 80)
        print(f"Pathway: {test_pathway}")
        print(f"Genes extracted by S3 method: {len(genes)}")
        if len(genes) == 0:
            print("⚠️  ISSUE: S3 extraction returned 0 genes")
            print("   Check saved JSON files to understand response structure")
        else:
            print("✅ S3 extraction working correctly")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Stop all servers
        print("\n🛑 Stopping MCP servers...")
        await manager.stop_all()
        print("✅ MCP servers stopped")


if __name__ == "__main__":
    asyncio.run(test_s3_extraction_method())

