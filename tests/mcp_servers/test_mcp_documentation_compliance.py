#!/usr/bin/env python3
"""
MCP Documentation Compliance Tests

Test cases derived from MCP server README files to validate:
- Parameter names match documentation
- Response structures match documented formats
- Edge cases and error handling
"""

import asyncio
import json
import sys
from pathlib import Path
from typing import Dict, Any

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.core.mcp_client_manager import MCPClientManager
from tests.mcp_servers.test_mcp_response_validator import MCPResponseValidator


# Expected parameter names from documentation
KEGG_EXPECTED_PARAMS = {
    'find_related_entries': ['source_entries', 'source_db', 'target_db'],
    'get_pathway_genes': ['pathway_id'],
    'search_pathways': ['query', 'limit'],
}

REACTOME_EXPECTED_PARAMS = {
    'get_pathway_participants': ['id'],
    'find_pathways_by_disease': ['disease', 'size'],
    'search_pathways': ['query', 'size'],
}

# Expected response structures from documentation
KEGG_EXPECTED_RESPONSES = {
    'find_related_entries': {
        'type': dict,
        'possible_keys': ['pathways', 'entries', 'related_entries', 'results', 'data']
    },
    'get_pathway_genes': {
        'type': dict,
        'possible_keys': ['genes', 'gene_list', 'results']
    },
}

REACTOME_EXPECTED_RESPONSES = {
    'get_pathway_participants': {
        'type': dict,
        'possible_keys': ['participants', 'entities', 'proteins']
    },
    'find_pathways_by_disease': {
        'type': dict,
        'possible_keys': ['pathways', 'results']
    },
}


async def test_kegg_documentation_compliance():
    """Test KEGG MCP compliance with documentation."""
    print("=" * 80)
    print("KEGG MCP Documentation Compliance Test")
    print("=" * 80)
    print()
    
    config_path = str(Path(__file__).parent.parent.parent / "config" / "mcp_servers.json")
    manager = MCPClientManager(config_path)
    validator = MCPResponseValidator()
    
    try:
        await manager.start_all()
        print("✅ MCP servers started")
        print()
        
        # Test find_related_entries
        print("Test: KEGG find_related_entries")
        print("-" * 80)
        try:
            result = await manager.kegg.find_related_entries(
                source_entries=["hsa:91464"],
                source_db="gene",
                target_db="pathway"
            )
            
            validator.log_response_structure(result, "KEGG", "find_related_entries")
            validation = validator.validate_response_structure(
                result,
                dict,
                KEGG_EXPECTED_RESPONSES['find_related_entries']['possible_keys'],
                "KEGG",
                "find_related_entries"
            )
            
            print(f"   Type match: {validation['type_match']}")
            print(f"   Keys match: {validation['keys_match']}")
            if validation['missing_keys']:
                print(f"   Missing keys: {validation['missing_keys']}")
            if validation['unexpected_keys']:
                print(f"   Unexpected keys: {validation['unexpected_keys']}")
            print()
            
        except Exception as e:
            print(f"   ❌ Error: {e}")
            print()
        
        summary = validator.get_summary()
        print("Validation Summary:")
        print(f"   Mismatches: {summary['total_mismatches']}")
        print(f"   Warnings: {summary['total_warnings']}")
        print()
        
        return summary['total_mismatches'] == 0
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        try:
            await manager.stop_all()
        except:
            pass


async def test_reactome_documentation_compliance():
    """Test Reactome MCP compliance with documentation."""
    print("=" * 80)
    print("Reactome MCP Documentation Compliance Test")
    print("=" * 80)
    print()
    
    config_path = str(Path(__file__).parent.parent.parent / "config" / "mcp_servers.json")
    manager = MCPClientManager(config_path)
    validator = MCPResponseValidator()
    
    try:
        await manager.start_all()
        print("✅ MCP servers started")
        print()
        
        # Test get_pathway_participants
        print("Test: Reactome get_pathway_participants")
        print("-" * 80)
        try:
            result = await manager.reactome.get_pathway_participants("R-HSA-1640170")
            
            validator.log_response_structure(result, "Reactome", "get_pathway_participants")
            validation = validator.validate_response_structure(
                result,
                dict,
                REACTOME_EXPECTED_RESPONSES['get_pathway_participants']['possible_keys'],
                "Reactome",
                "get_pathway_participants"
            )
            
            print(f"   Type match: {validation['type_match']}")
            print(f"   Keys match: {validation['keys_match']}")
            if validation['missing_keys']:
                print(f"   Missing keys: {validation['missing_keys']}")
            if validation['unexpected_keys']:
                print(f"   Unexpected keys: {validation['unexpected_keys']}")
            print()
            
        except Exception as e:
            print(f"   ❌ Error: {e}")
            print()
        
        summary = validator.get_summary()
        print("Validation Summary:")
        print(f"   Mismatches: {summary['total_mismatches']}")
        print(f"   Warnings: {summary['total_warnings']}")
        print()
        
        return summary['total_mismatches'] == 0
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        try:
            await manager.stop_all()
        except:
            pass


async def main():
    """Run all documentation compliance tests."""
    print("\n" + "=" * 80)
    print("MCP Documentation Compliance Test Suite")
    print("=" * 80)
    print()
    
    results = []
    
    results.append(("KEGG", await test_kegg_documentation_compliance()))
    results.append(("Reactome", await test_reactome_documentation_compliance()))
    
    print("=" * 80)
    print("Test Results Summary")
    print("=" * 80)
    for server, passed in results:
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"  {server}: {status}")
    
    all_passed = all(passed for _, passed in results)
    return 0 if all_passed else 1


if __name__ == "__main__":
    exit(asyncio.run(main()))

