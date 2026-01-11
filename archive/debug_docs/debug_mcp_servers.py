#!/usr/bin/env python3
"""
MCP Server Debugging Script
Purpose: Test and debug Reactome and HPA MCP servers to fix S1 issues
Created: 2025-10-27
"""

import asyncio
import json
import sys
from pathlib import Path
from typing import Dict, Any, List

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from mcp_clients.reactome_client import ReactomeClient
from mcp_clients.hpa_client import HPAClient
from mcp_clients.kegg_client import KEGGClient

class MCPServerDebugger:
    def __init__(self):
        self.config_path = "config/mcp_servers.json"
        self.results = {}
        
    async def debug_all_servers(self):
        """Debug all MCP servers systematically"""
        print("🔍 MCP Server Debugging Report")
        print("=" * 80)
        
        # Test Reactome
        await self.debug_reactome()
        
        # Test HPA
        await self.debug_hpa()
        
        # Test KEGG (for comparison)
        await self.debug_kegg()
        
        # Generate summary
        self.generate_summary()
        
    async def debug_reactome(self):
        """Debug Reactome MCP server"""
        print("\n🧬 REACTOME MCP SERVER DEBUG")
        print("-" * 50)
        
        try:
            client = ReactomeClient(self.config_path)
            
            # Test 1: Basic pathway search
            print("Test 1: Pathway search for 'breast cancer'")
            try:
                search_result = await client.find_pathways_by_disease("breast cancer")
                print(f"✅ Search successful: {len(search_result.get('pathways', []))} pathways found")
                
                if search_result.get('pathways'):
                    sample_pathway = search_result['pathways'][0]
                    print(f"   Sample pathway: {sample_pathway}")
                    pathway_id = sample_pathway.get('id') or sample_pathway.get('stId')
                    print(f"   Pathway ID: {pathway_id}")
                    
                    # Test 2: Get pathway details
                    if pathway_id:
                        print(f"\nTest 2: Get details for pathway {pathway_id}")
                        try:
                            details = await client.get_pathway_details(pathway_id)
                            print(f"✅ Details retrieved successfully")
                            print(f"   Response keys: {list(details.keys())}")
                            
                            # Check entities
                            entities = details.get('entities', [])
                            print(f"   Entities count: {len(entities)}")
                            if entities:
                                print(f"   Sample entity: {entities[0]}")
                                print(f"   Entity keys: {list(entities[0].keys())}")
                                
                                # Look for gene names
                                gene_names = []
                                for entity in entities[:5]:  # Check first 5
                                    gene_name = (entity.get('geneName') or 
                                               entity.get('displayName') or 
                                               entity.get('name'))
                                    if gene_name and not gene_name.startswith('R-'):
                                        gene_names.append(gene_name)
                                
                                print(f"   Gene names found: {gene_names}")
                            else:
                                print("   ❌ No entities found")
                            
                            # Check hasEvent
                            has_event = details.get('hasEvent', [])
                            print(f"   HasEvent count: {len(has_event)}")
                            if has_event:
                                print(f"   Sample event: {has_event[0]}")
                                print(f"   Event keys: {list(has_event[0].keys())}")
                                
                                # Look for participants
                                participants = has_event[0].get('participants', [])
                                print(f"   Participants count: {len(participants)}")
                                if participants:
                                    print(f"   Sample participant: {participants[0]}")
                            else:
                                print("   ❌ No hasEvent found")
                                
                        except Exception as e:
                            print(f"❌ Details failed: {e}")
                            
                        # Test 3: Get pathway participants (backup method)
                        print(f"\nTest 3: Get participants for pathway {pathway_id}")
                        try:
                            participants = await client.get_pathway_participants(pathway_id)
                            print(f"✅ Participants retrieved successfully")
                            print(f"   Response keys: {list(participants.keys())}")
                            
                            participants_list = participants.get('participants', [])
                            print(f"   Participants count: {len(participants_list)}")
                            if participants_list:
                                print(f"   Sample participant: {participants_list[0]}")
                                print(f"   Participant keys: {list(participants_list[0].keys())}")
                                
                                # Look for gene names
                                gene_names = []
                                for p in participants_list[:5]:
                                    gene_name = (p.get('displayName') or 
                                               p.get('name') or
                                               p.get('geneName'))
                                    if gene_name and not gene_name.startswith('R-'):
                                        gene_names.append(gene_name)
                                
                                print(f"   Gene names found: {gene_names}")
                            else:
                                print("   ❌ No participants found")
                                
                        except Exception as e:
                            print(f"❌ Participants failed: {e}")
                            
                    else:
                        print("❌ No pathway ID found in search result")
                        
                else:
                    print("❌ No pathways found in search")
                    
            except Exception as e:
                print(f"❌ Search failed: {e}")
                
            # Test 4: Try with known pathway IDs
            print(f"\nTest 4: Known pathway IDs")
            known_pathways = [
                'R-HSA-1227990',  # Signaling by ERBB2 in Cancer
                'R-HSA-4791275',  # Signaling by WNT in cancer
                'R-HSA-1640170'   # Cell Cycle
            ]
            
            for pathway_id in known_pathways:
                print(f"   Testing {pathway_id}...")
                try:
                    details = await client.get_pathway_details(pathway_id)
                    entities = details.get('entities', [])
                    print(f"   ✅ {pathway_id}: {len(entities)} entities")
                    
                    if entities:
                        # Extract gene names
                        gene_names = []
                        for entity in entities[:10]:
                            gene_name = (entity.get('geneName') or 
                                       entity.get('displayName') or 
                                       entity.get('name'))
                            if gene_name and not gene_name.startswith('R-') and len(gene_name) > 1:
                                gene_symbol = gene_name.split()[0].split('[')[0].split('(')[0]
                                if len(gene_symbol) > 1:
                                    gene_names.append(gene_symbol)
                        
                        print(f"   Genes: {gene_names[:10]}")
                        
                except Exception as e:
                    print(f"   ❌ {pathway_id}: {e}")
                    
            self.results['reactome'] = {
                'search_works': 'search_result' in locals(),
                'details_works': 'details' in locals() if 'search_result' in locals() else False,
                'participants_works': 'participants' in locals() if 'search_result' in locals() else False,
                'entities_found': len(entities) if 'entities' in locals() else 0,
                'genes_extracted': len(gene_names) if 'gene_names' in locals() else 0
            }
            
        except Exception as e:
            print(f"❌ Reactome client initialization failed: {e}")
            self.results['reactome'] = {'error': str(e)}
            
    async def debug_hpa(self):
        """Debug HPA MCP server"""
        print("\n🧬 HPA MCP SERVER DEBUG")
        print("-" * 50)
        
        try:
            client = HPAClient(self.config_path)
            
            # Test genes
            test_genes = ['BRCA1', 'TP53', 'EGFR', 'AXL', 'MYC']
            
            print("Test 1: Protein info for known genes")
            successful_genes = []
            
            for gene in test_genes:
                print(f"   Testing {gene}...")
                try:
                    protein_info = await client.get_protein_info(gene)
                    print(f"   ✅ {gene}: {protein_info}")
                    
                    if protein_info.get('uniprot'):
                        successful_genes.append(gene)
                        print(f"      UniProt: {protein_info['uniprot']}")
                    else:
                        print(f"      No UniProt ID found")
                        
                except Exception as e:
                    print(f"   ❌ {gene}: {e}")
                    
            print(f"\nSuccessful genes: {successful_genes}")
            
            # Test 2: Expression data
            print(f"\nTest 2: Expression data for BRCA1")
            try:
                expression = await client.get_tissue_expression('BRCA1')
                print(f"✅ Expression data: {expression}")
            except Exception as e:
                print(f"❌ Expression failed: {e}")
                
            # Test 3: Pathology data
            print(f"\nTest 3: Pathology data for breast cancer")
            try:
                pathology = await client.get_pathology_data('breast cancer')
                print(f"✅ Pathology data: {pathology}")
            except Exception as e:
                print(f"❌ Pathology failed: {e}")
                
            self.results['hpa'] = {
                'protein_info_works': len(successful_genes) > 0,
                'successful_genes': successful_genes,
                'expression_works': 'expression' in locals(),
                'pathology_works': 'pathology' in locals()
            }
            
        except Exception as e:
            print(f"❌ HPA client initialization failed: {e}")
            self.results['hpa'] = {'error': str(e)}
            
    async def debug_kegg(self):
        """Debug KEGG MCP server (for comparison)"""
        print("\n🧬 KEGG MCP SERVER DEBUG")
        print("-" * 50)
        
        try:
            client = KEGGClient(self.config_path)
            
            # Test pathway search
            print("Test 1: Pathway search for 'breast cancer'")
            try:
                search_result = await client.search_pathways('breast cancer', limit=5)
                print(f"✅ Search successful: {len(search_result.get('pathways', []))} pathways found")
                
                if search_result.get('pathways'):
                    sample_pathway = search_result['pathways'][0]
                    print(f"   Sample pathway: {sample_pathway}")
                    
                    # Test pathway genes
                    pathway_id = sample_pathway.get('id') or sample_pathway.get('entry_id')
                    if pathway_id:
                        print(f"\nTest 2: Get genes for pathway {pathway_id}")
                        try:
                            genes_result = await client.get_pathway_genes(pathway_id)
                            print(f"✅ Genes retrieved: {genes_result}")
                            
                            genes = genes_result.get('genes', [])
                            print(f"   Genes count: {len(genes)}")
                            if genes:
                                print(f"   Sample genes: {genes[:5]}")
                                
                        except Exception as e:
                            print(f"❌ Genes failed: {e}")
                            
            except Exception as e:
                print(f"❌ KEGG search failed: {e}")
                
            self.results['kegg'] = {
                'search_works': 'search_result' in locals(),
                'genes_works': 'genes_result' in locals() if 'search_result' in locals() else False
            }
            
        except Exception as e:
            print(f"❌ KEGG client initialization failed: {e}")
            self.results['kegg'] = {'error': str(e)}
            
    def generate_summary(self):
        """Generate debugging summary"""
        print("\n" + "=" * 80)
        print("🔍 MCP SERVER DEBUGGING SUMMARY")
        print("=" * 80)
        
        for server, result in self.results.items():
            print(f"\n{server.upper()}:")
            if 'error' in result:
                print(f"  ❌ Initialization failed: {result['error']}")
            else:
                for key, value in result.items():
                    status = "✅" if value else "❌"
                    print(f"  {status} {key}: {value}")
                    
        print(f"\n🎯 RECOMMENDATIONS:")
        
        if self.results.get('reactome', {}).get('entities_found', 0) == 0:
            print("  - Reactome: Entities not found - check API response format")
            print("  - Try alternative pathway IDs or different search terms")
            
        if self.results.get('hpa', {}).get('successful_genes', []) == []:
            print("  - HPA: No successful gene lookups - check gene symbol format")
            print("  - Try with/without organism prefixes")
            
        if self.results.get('kegg', {}).get('genes_works', False):
            print("  - KEGG: Working - use as fallback for pathway genes")
            
        print(f"\n📝 Next steps:")
        print("  1. Fix Reactome gene extraction based on debug results")
        print("  2. Fix HPA UniProt lookup based on debug results") 
        print("  3. Re-test S1 with fixed MCP servers")
        print("  4. Achieve 5/5 validation success")

async def main():
    """Main debugging function"""
    debugger = MCPServerDebugger()
    await debugger.debug_all_servers()

if __name__ == "__main__":
    asyncio.run(main())
