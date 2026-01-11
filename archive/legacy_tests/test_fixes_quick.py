#!/usr/bin/env python3
"""
Quick test script for KEGG and Reactome fixes.
Tests the fixes without running the full pipeline.
"""

import asyncio
import json
import sys
from pathlib import Path

# Use the module approach
import subprocess
import os


async def test_kegg_fallback():
    """Test KEGG find_related_entries fallback with correct parameters."""
    print("\n" + "="*70)
    print("TEST 1: KEGG find_related_entries Fallback")
    print("="*70)
    
    # Use Python module to run a simple test
    test_code = """
import asyncio
import sys
sys.path.insert(0, '.')

async def test():
    from src.core.mcp_client_manager import MCPClientManager
    
    manager = MCPClientManager()
    await manager.start_all_servers()
    
    try:
        gene_id = "hsa:84839"  # AXL gene
        print(f"Testing KEGG get_gene_pathways for: {gene_id}")
        
        result = await manager.kegg.get_gene_pathways(gene_id)
        
        print(f"✅ KEGG call successful!")
        print(f"   Result type: {type(result)}")
        if isinstance(result, dict):
            print(f"   Result keys: {list(result.keys())}")
            pathways = result.get('pathways', [])
            print(f"   Pathways found: {len(pathways)}")
            print(f"   Source: {result.get('source', 'unknown')}")
            return True
        return False
    except Exception as e:
        error_msg = str(e)
        print(f"❌ KEGG test failed: {error_msg}")
        if 'Source and target databases are required' in error_msg:
            print("   ⚠️  This is the error we're fixing - parameter mismatch")
        return False
    finally:
        await manager.stop_all_servers()

result = asyncio.run(test())
sys.exit(0 if result else 1)
"""
    
    result = subprocess.run(
        [sys.executable, "-c", test_code],
        cwd=Path(__file__).parent,
        capture_output=True,
        text=True
    )
    
    print(result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr)
    
    return result.returncode == 0


async def test_reactome_gene_extraction():
    """Test Reactome gene extraction with get_pathway_participants."""
    print("\n" + "="*70)
    print("TEST 2: Reactome Gene Extraction (get_pathway_participants)")
    print("="*70)
    
    test_code = """
import asyncio
import sys
sys.path.insert(0, '.')

async def test():
    from src.core.mcp_client_manager import MCPClientManager
    
    manager = MCPClientManager()
    await manager.start_all_servers()
    
    try:
        pathway_id = "R-HSA-1227990"  # Signaling by ERBB2 in Cancer
        print(f"Testing Reactome get_pathway_participants for: {pathway_id}")
        
        participants = await manager.reactome.get_pathway_participants(pathway_id)
        
        print(f"✅ Reactome get_pathway_participants successful!")
        print(f"   Response type: {type(participants)}")
        
        if isinstance(participants, dict):
            print(f"   Response keys: {list(participants.keys())[:10]}")
            
            participant_list = []
            if participants.get('participants'):
                participant_list = participants['participants']
                print(f"   ✅ Found {len(participant_list)} participants in 'participants' key")
            elif participants.get('entities'):
                participant_list = participants['entities']
                print(f"   ✅ Found {len(participant_list)} entities in 'entities' key")
            elif isinstance(participants, list):
                participant_list = participants
                print(f"   ✅ Response is direct list with {len(participant_list)} items")
            
            if participant_list:
                print(f"   Sample participant keys: {list(participant_list[0].keys())[:10] if isinstance(participant_list[0], dict) else 'N/A'}")
                return True
        elif isinstance(participants, list):
            print(f"   ✅ Response is direct list with {len(participants)} items")
            return True
        
        return False
    except Exception as e:
        print(f"❌ Reactome gene extraction test failed: {type(e).__name__}: {e}")
        return False
    finally:
        await manager.stop_all_servers()

result = asyncio.run(test())
sys.exit(0 if result else 1)
"""
    
    result = subprocess.run(
        [sys.executable, "-c", test_code],
        cwd=Path(__file__).parent,
        capture_output=True,
        text=True
    )
    
    print(result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr)
    
    return result.returncode == 0


async def test_reactome_timeout_fallback():
    """Test Reactome timeout fallback to search_pathways."""
    print("\n" + "="*70)
    print("TEST 3: Reactome Timeout Fallback")
    print("="*70)
    
    test_code = """
import asyncio
import sys
sys.path.insert(0, '.')

async def test():
    from src.core.mcp_client_manager import MCPClientManager
    
    manager = MCPClientManager()
    await manager.start_all_servers()
    
    try:
        disease_name = "breast cancer"
        print(f"Testing Reactome find_pathways_by_disease for: '{disease_name}'")
        print("(Timeout increased to 60s, will fallback to search_pathways if timeout)")
        
        result = await manager.reactome.find_pathways_by_disease(disease_name)
        
        print(f"✅ Reactome call successful!")
        if isinstance(result, dict):
            pathways = result.get('pathways', [])
            print(f"   Pathways found: {len(pathways)}")
            if pathways:
                print(f"   Sample pathway: {pathways[0].get('name', pathways[0].get('displayName', 'N/A')) if isinstance(pathways[0], dict) else pathways[0]}")
        return True
    except Exception as e:
        error_msg = str(e)
        print(f"❌ Reactome timeout test failed: {error_msg}")
        if 'timeout' in error_msg.lower() or 'exceeded' in error_msg.lower():
            print("   ⚠️  Still timing out - may need even longer timeout or server issue")
        return False
    finally:
        await manager.stop_all_servers()

result = asyncio.run(test())
sys.exit(0 if result else 1)
"""
    
    result = subprocess.run(
        [sys.executable, "-c", test_code],
        cwd=Path(__file__).parent,
        capture_output=True,
        text=True
    )
    
    print(result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr)
    
    return result.returncode == 0


async def main():
    """Run all tests."""
    print("\n" + "="*70)
    print("QUICK FIX VALIDATION TESTS")
    print("="*70)
    print("\nTesting fixes without running full pipeline...")
    
    results = {}
    
    # Test 1: KEGG fallback
    try:
        results['kegg'] = await test_kegg_fallback()
    except Exception as e:
        print(f"\n❌ KEGG test crashed: {e}")
        results['kegg'] = False
    
    # Test 2: Reactome gene extraction
    try:
        results['reactome_genes'] = await test_reactome_gene_extraction()
    except Exception as e:
        print(f"\n❌ Reactome gene extraction test crashed: {e}")
        results['reactome_genes'] = False
    
    # Test 3: Reactome timeout
    try:
        results['reactome_timeout'] = await test_reactome_timeout_fallback()
    except Exception as e:
        print(f"\n❌ Reactome timeout test crashed: {e}")
        results['reactome_timeout'] = False
    
    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {status}: {test_name}")
    
    all_passed = all(results.values())
    
    if all_passed:
        print("\n✅ ALL TESTS PASSED - Fixes are working!")
    else:
        print("\n⚠️  SOME TESTS FAILED - Review errors above")
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
