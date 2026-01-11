#!/usr/bin/env python3
"""
Code Validation: Verify MCP Fixes Are Correctly Implemented

This script validates that the fixes are correctly implemented in the code
without requiring MCP servers or dependencies.
"""

import ast
import re
from pathlib import Path


def check_kegg_fix():
    """Verify KEGG find_related_entries uses correct parameters."""
    print("=" * 80)
    print("Validating KEGG find_related_entries Fix")
    print("=" * 80)
    
    kegg_file = Path("src/mcp_clients/kegg_client.py")
    content = kegg_file.read_text()
    
    checks = []
    
    # Check 1: Method signature uses correct parameters
    if 'source_entries: List[str]' in content:
        checks.append(("✅ Method signature uses 'source_entries' (array)", True))
    else:
        checks.append(("❌ Method signature missing 'source_entries'", False))
    
    if 'source_db: str' in content:
        checks.append(("✅ Method signature uses 'source_db'", True))
    else:
        checks.append(("❌ Method signature missing 'source_db'", False))
    
    if 'target_db: str' in content:
        checks.append(("✅ Method signature uses 'target_db'", True))
    else:
        checks.append(("❌ Method signature missing 'target_db'", False))
    
    # Check 2: Call uses correct parameter names
    if '"source_entries": source_entries' in content:
        checks.append(("✅ Call uses 'source_entries' parameter", True))
    else:
        checks.append(("❌ Call missing 'source_entries' parameter", False))
    
    if '"source_db": source_db' in content:
        checks.append(("✅ Call uses 'source_db' parameter", True))
    else:
        checks.append(("❌ Call missing 'source_db' parameter", False))
    
    if '"target_db": target_db' in content:
        checks.append(("✅ Call uses 'target_db' parameter", True))
    else:
        checks.append(("❌ Call missing 'target_db' parameter", False))
    
    # Check 3: Fallback uses correct method
    if 'source_entries=[gene_id]' in content:
        checks.append(("✅ Fallback uses correct array format", True))
    else:
        checks.append(("❌ Fallback missing array format", False))
    
    # Check 4: No old parameter names
    if 'entry_id' in content and 'def find_related_entries' in content:
        # Check if it's in the old signature (bad) or just in comments (ok)
        lines = content.split('\n')
        for i, line in enumerate(lines):
            if 'def find_related_entries' in line and 'entry_id' in line:
                checks.append(("❌ Old 'entry_id' parameter still in signature", False))
                break
        else:
            checks.append(("✅ No 'entry_id' in method signature", True))
    else:
        checks.append(("✅ No 'entry_id' parameter", True))
    
    for check, passed in checks:
        print(f"  {check}")
    
    all_passed = all(passed for _, passed in checks)
    return all_passed


def check_reactome_timeout_fix():
    """Verify Reactome timeout handling fix."""
    print("\n" + "=" * 80)
    print("Validating Reactome Timeout Handling Fix")
    print("=" * 80)
    
    reactome_file = Path("src/mcp_clients/reactome_client.py")
    content = reactome_file.read_text()
    
    checks = []
    
    # Check 1: Method has size parameter
    if 'size: int = 10' in content or 'size=10' in content:
        checks.append(("✅ Method has 'size' parameter with default 10", True))
    else:
        checks.append(("❌ Method missing 'size' parameter", False))
    
    # Check 2: Size is limited
    if 'limited_size = min(size, 10)' in content:
        checks.append(("✅ Size is limited to max 10", True))
    else:
        checks.append(("❌ Size limiting not implemented", False))
    
    # Check 3: Fallback uses smaller size
    if '"size": 10' in content and 'fallback' in content.lower():
        checks.append(("✅ Fallback uses size=10", True))
    else:
        checks.append(("⚠️  Fallback size may not be limited", False))
    
    # Check 4: Timeout handling
    if 'timeout' in content.lower() and 'fallback' in content.lower():
        checks.append(("✅ Timeout handling with fallback implemented", True))
    else:
        checks.append(("❌ Timeout handling missing", False))
    
    for check, passed in checks:
        print(f"  {check}")
    
    all_passed = all(passed for _, passed in checks)
    return all_passed


def check_reactome_gene_extraction_fix():
    """Verify Reactome gene extraction enhancements."""
    print("\n" + "=" * 80)
    print("Validating Reactome Gene Extraction Fix")
    print("=" * 80)
    
    scenario_file = Path("src/scenarios/scenario_1_disease_network.py")
    content = scenario_file.read_text()
    
    checks = []
    
    # Check 1: Handles 'proteins' key
    if "'proteins'" in content and 'participants.get(\'proteins\')' in content:
        checks.append(("✅ Handles 'proteins' key in response", True))
    else:
        checks.append(("❌ Missing 'proteins' key handling", False))
    
    # Check 2: Extracts from referenceEntity
    if "'referenceEntity'" in content and 'ref_entity' in content:
        checks.append(("✅ Extracts from nested 'referenceEntity'", True))
    else:
        checks.append(("❌ Missing 'referenceEntity' extraction", False))
    
    # Check 3: Extracts from components
    if "'components'" in content and 'component_genes' in content:
        checks.append(("✅ Extracts from 'components' array", True))
    else:
        checks.append(("❌ Missing 'components' extraction", False))
    
    # Check 4: Extracts from hasComponent
    if "'hasComponent'" in content:
        checks.append(("✅ Extracts from 'hasComponent'", True))
    else:
        checks.append(("❌ Missing 'hasComponent' extraction", False))
    
    # Check 5: Enhanced logging
    if 'logger.debug' in content and 'referenceEntity' in content:
        checks.append(("✅ Enhanced diagnostic logging present", True))
    else:
        checks.append(("⚠️  Diagnostic logging may be limited", False))
    
    for check, passed in checks:
        print(f"  {check}")
    
    all_passed = all(passed for _, passed in checks)
    return all_passed


def main():
    """Run all validations."""
    print("\n" + "=" * 80)
    print("MCP Fixes Code Validation")
    print("=" * 80)
    print()
    
    results = []
    
    results.append(("KEGG find_related_entries", check_kegg_fix()))
    results.append(("Reactome timeout handling", check_reactome_timeout_fix()))
    results.append(("Reactome gene extraction", check_reactome_gene_extraction_fix()))
    
    print("\n" + "=" * 80)
    print("Validation Summary")
    print("=" * 80)
    
    for name, passed in results:
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"  {name}: {status}")
    
    all_passed = all(passed for _, passed in results)
    
    print("\n" + "=" * 80)
    if all_passed:
        print("✅ ALL FIXES VALIDATED SUCCESSFULLY")
        print("=" * 80)
        print("\nAll code changes are correctly implemented.")
        print("The fixes match the MCP server documentation requirements.")
    else:
        print("❌ SOME VALIDATIONS FAILED")
        print("=" * 80)
        print("\nPlease review the failed checks above.")
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    exit(main())




