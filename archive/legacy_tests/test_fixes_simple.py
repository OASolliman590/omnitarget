#!/usr/bin/env python3
"""
Simple test to verify KEGG and Reactome fixes by checking code changes.
"""

import re
from pathlib import Path

def test_kegg_parameter_fix():
    """Verify KEGG find_related_entries has correct parameters."""
    print("\n" + "="*70)
    print("TEST 1: KEGG find_related_entries Parameter Fix")
    print("="*70)
    
    kegg_file = Path("src/mcp_clients/kegg_client.py")
    content = kegg_file.read_text()
    
    # Check if fix is present
    if 'source_database' in content and 'target_database' in content:
        # Find the specific line
        lines = content.split('\n')
        for i, line in enumerate(lines[325:335], start=326):
            if 'find_related_entries' in line or 'source_database' in line or 'target_database' in line:
                print(f"   Line {i}: {line.strip()}")
        
        if 'source_database": "gene"' in content and 'target_database": "pathway"' in content:
            print("\n✅ KEGG parameter fix is present!")
            print("   - source_database: 'gene' ✅")
            print("   - target_database: 'pathway' ✅")
            return True
        else:
            print("\n⚠️  KEGG parameters found but may not be correct")
            return False
    else:
        print("\n❌ KEGG parameter fix NOT found")
        return False


def test_reactome_gene_extraction_fix():
    """Verify Reactome gene extraction uses get_pathway_participants first."""
    print("\n" + "="*70)
    print("TEST 2: Reactome Gene Extraction Fix")
    print("="*70)
    
    s1_file = Path("src/scenarios/scenario_1_disease_network.py")
    content = s1_file.read_text()
    
    # Check if get_pathway_participants is used first
    method_start = content.find("async def _extract_reactome_genes")
    if method_start == -1:
        print("\n❌ _extract_reactome_genes method not found")
        return False
    
    method_content = content[method_start:method_start+2000]  # First 2000 chars of method
    
    # Check order: get_pathway_participants should come before get_pathway_details
    participants_pos = method_content.find("get_pathway_participants")
    details_pos = method_content.find("get_pathway_details")
    
    if participants_pos == -1:
        print("\n❌ get_pathway_participants not found in method")
        return False
    
    if details_pos == -1:
        print("\n⚠️  get_pathway_details not found (may be removed)")
    elif participants_pos > details_pos:
        print("\n❌ get_pathway_participants comes AFTER get_pathway_details (wrong order)")
        return False
    
    # Check for diagnostic logging
    has_diagnostic = "[Reactome]" in method_content or "get_pathway_participants response" in method_content
    
    # Check for multiple structure handling
    has_structure_handling = "participants.get('participants')" in method_content or "participants.get('entities')" in method_content
    
    print("\n✅ Reactome gene extraction fix is present!")
    print(f"   - get_pathway_participants is primary: ✅")
    print(f"   - Diagnostic logging: {'✅' if has_diagnostic else '⚠️'}")
    print(f"   - Multiple structure handling: {'✅' if has_structure_handling else '⚠️'}")
    
    return True


def test_reactome_timeout_fix():
    """Verify Reactome timeout fix."""
    print("\n" + "="*70)
    print("TEST 3: Reactome Timeout Fix")
    print("="*70)
    
    reactome_file = Path("src/mcp_clients/reactome_client.py")
    content = reactome_file.read_text()
    
    # Check timeout increase
    has_timeout_60 = 'timeout=60' in content or 'timeout: int = 60' in content
    
    # Check fallback logic
    has_fallback = 'search_pathways' in content and 'timeout' in content.lower() and 'fallback' in content.lower()
    
    print(f"\n   Timeout increased to 60s: {'✅' if has_timeout_60 else '❌'}")
    print(f"   Fallback to search_pathways: {'✅' if has_fallback else '❌'}")
    
    if has_timeout_60 and has_fallback:
        print("\n✅ Reactome timeout fix is present!")
        return True
    else:
        print("\n⚠️  Reactome timeout fix may be incomplete")
        return False


def main():
    """Run all code verification tests."""
    print("\n" + "="*70)
    print("CODE VERIFICATION TESTS")
    print("="*70)
    print("\nVerifying fixes in code (not running full pipeline)...")
    
    results = {}
    results['kegg'] = test_kegg_parameter_fix()
    results['reactome_genes'] = test_reactome_gene_extraction_fix()
    results['reactome_timeout'] = test_reactome_timeout_fix()
    
    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {status}: {test_name}")
    
    all_passed = all(results.values())
    
    if all_passed:
        print("\n✅ ALL FIXES VERIFIED IN CODE!")
        print("\nNext step: Run full pipeline to test in practice:")
        print("  python -m src.cli yaml examples/yaml_configs/axl_breast_cancer_all_6_scenarios.yaml")
    else:
        print("\n⚠️  SOME FIXES NOT VERIFIED - Review code above")
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    import sys
    sys.exit(main())




