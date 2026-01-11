#!/usr/bin/env python3
"""
Quick test script for Phase 1 (P1) enhancements.
Tests UniProt function and HPA subcellular location enrichment.
"""

import asyncio
import sys
import logging
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

from src.core.mcp_client_manager import MCPClientManager
from src.models.data_models import NetworkNode

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def test_p1_enhancements():
    """Test Phase 1 enhancements on a small set of genes."""
    
    config_path = "config/mcp_servers.json"
    test_genes = ["BRCA1", "TP53", "EGFR"]
    
    logger.info("=" * 60)
    logger.info("Phase 1 (P1) Enhancement Test")
    logger.info("=" * 60)
    
    # Initialize MCP manager
    manager = MCPClientManager(config_path)
    
    async with manager.session():
        logger.info(f"✅ MCP servers started")
        logger.info(f"   - UniProt available: {manager.uniprot is not None}")
        
        results = []
        
        for gene in test_genes:
            logger.info(f"\n🧬 Testing {gene}:")
            
            # Get UniProt ID from HPA
            uniprot_id = None
            try:
                protein_info = await manager.hpa.get_protein_info(gene)
                if protein_info and isinstance(protein_info, list) and protein_info:
                    first_protein = protein_info[0]
                    uniprot_data = first_protein.get('Uniprot', [])
                    uniprot_id = uniprot_data[0] if uniprot_data else None
                    logger.info(f"   ✅ UniProt ID: {uniprot_id}")
            except Exception as e:
                logger.warning(f"   ❌ UniProt ID failed: {e}")
            
            # P1 Enhancement: Get function from UniProt
            function = None
            if uniprot_id and manager.uniprot:
                try:
                    uniprot_info = await manager.uniprot.get_protein_info(uniprot_id)
                    if uniprot_info:
                        function = uniprot_info.get('function') or uniprot_info.get('description')
                        if function:
                            logger.info(f"   ✅ Function (P1): {function[:100]}...")
                        else:
                            logger.warning(f"   ⚠️  Function (P1): Not found in response")
                except Exception as e:
                    logger.warning(f"   ❌ Function (P1) failed: {e}")
            elif not manager.uniprot:
                logger.info(f"   ⚠️  UniProt MCP not available, skipping function")
            
            # P1 Enhancement: Get subcellular location from HPA
            subcellular_location = []
            try:
                location_data = await manager.hpa.get_subcellular_location(gene)
                if location_data and isinstance(location_data, list):
                    for loc in location_data:
                        if isinstance(loc, dict):
                            main_loc = loc.get('Main location', loc.get('main_location'))
                            if main_loc and isinstance(main_loc, str):
                                subcellular_location.append(main_loc)
                            elif main_loc and isinstance(main_loc, list):
                                subcellular_location.extend(main_loc)
                    if subcellular_location:
                        logger.info(f"   ✅ Location (P1): {subcellular_location}")
                    else:
                        logger.warning(f"   ⚠️  Location (P1): No valid locations found")
            except Exception as e:
                logger.warning(f"   ❌ Location (P1) failed: {e}")
            
            results.append({
                'gene': gene,
                'uniprot_id': uniprot_id,
                'function': function,
                'subcellular_location': subcellular_location
            })
            
            # Small delay to avoid MCP contention
            await asyncio.sleep(0.2)
        
        # Summary
        logger.info("\n" + "=" * 60)
        logger.info("Summary:")
        logger.info("=" * 60)
        
        uniprot_count = sum(1 for r in results if r['uniprot_id'])
        function_count = sum(1 for r in results if r['function'])
        location_count = sum(1 for r in results if r['subcellular_location'])
        
        logger.info(f"UniProt IDs: {uniprot_count}/{len(results)} ({100*uniprot_count/len(results):.0f}%)")
        logger.info(f"Functions (P1): {function_count}/{len(results)} ({100*function_count/len(results):.0f}%)")
        logger.info(f"Locations (P1): {location_count}/{len(results)} ({100*location_count/len(results):.0f}%)")
        
        if function_count >= 2 and location_count >= 2:
            logger.info("\n✅ Phase 1 enhancements WORKING!")
            return True
        else:
            logger.warning("\n⚠️  Phase 1 enhancements PARTIALLY WORKING")
            return False


if __name__ == "__main__":
    try:
        success = asyncio.run(test_p1_enhancements())
        sys.exit(0 if success else 1)
    except Exception as e:
        logger.error(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

