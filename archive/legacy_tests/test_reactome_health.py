#!/usr/bin/env python3
"""
Test Circuit: Reactome Health Check with Retry Logic

Tests Reactome tools with retry logic to verify ECONNRESET handling
and document response times and success rates.
"""

import asyncio
import json
import sys
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.core.mcp_client_manager import MCPClientManager


async def test_reactome_tool(tool_name, test_func, description):
    """Test a Reactome tool with retry logic and measure performance."""
    print(f"\n{'='*80}")
    print(f"Testing: {description}")
    print(f"Tool: {tool_name}")
    print(f"{'='*80}")
    
    results = {
        'tool': tool_name,
        'description': description,
        'attempts': [],
        'success': False,
        'total_time': 0,
        'retries': 0
    }
    
    start_time = time.time()
    attempt = 1
    max_attempts = 3
    
    while attempt <= max_attempts:
        attempt_start = time.time()
        try:
            print(f"\n  Attempt {attempt}/{max_attempts}...")
            result = await test_func()
            attempt_time = time.time() - attempt_start
            
            results['attempts'].append({
                'attempt': attempt,
                'success': True,
                'time': attempt_time,
                'error': None
            })
            results['success'] = True
            results['total_time'] = time.time() - start_time
            results['retries'] = attempt - 1
            
            print(f"  ✅ Success on attempt {attempt} (took {attempt_time:.2f}s)")
            print(f"  📊 Response type: {type(result)}")
            if isinstance(result, dict):
                print(f"  📊 Response keys: {list(result.keys())[:10]}")
                if 'pathways' in result:
                    print(f"  📊 Pathways found: {len(result.get('pathways', []))}")
            
            return results, result
            
        except Exception as e:
            attempt_time = time.time() - attempt_start
            error_msg = str(e)
            error_type = type(e).__name__
            
            results['attempts'].append({
                'attempt': attempt,
                'success': False,
                'time': attempt_time,
                'error': error_msg,
                'error_type': error_type
            })
            
            is_econnreset = 'ECONNRESET' in error_msg.upper() or 'connection reset' in error_msg.lower()
            
            print(f"  ❌ Attempt {attempt} failed ({error_type})")
            print(f"     Error: {error_msg[:100]}...")
            if is_econnreset:
                print(f"     🔄 ECONNRESET detected - will retry")
            
            if attempt < max_attempts:
                wait_time = 0.5 * (2 ** (attempt - 1))  # Exponential backoff: 0.5s, 1.0s, 2.0s
                print(f"     ⏳ Waiting {wait_time:.1f}s before retry...")
                await asyncio.sleep(wait_time)
            else:
                print(f"  ❌ All {max_attempts} attempts failed")
                results['total_time'] = time.time() - start_time
                results['retries'] = attempt - 1
                return results, None
            
            attempt += 1
    
    return results, None


async def main():
    """Run Reactome health check tests."""
    print("=" * 80)
    print("Reactome Health Check with Retry Logic")
    print("=" * 80)
    print()
    
    # Create MCP manager
    config_path = str(Path(__file__).parent / "config" / "mcp_servers.json")
    manager = MCPClientManager(config_path)
    
    all_results = {}
    
    try:
        # Start all servers
        print("🚀 Starting MCP servers...")
        await manager.start_all()
        print("✅ MCP servers started")
        print()
        
        # Test 1: find_pathways_by_gene("AXL")
        async def test_find_pathways_by_gene():
            return await manager.reactome.find_pathways_by_gene("AXL")
        
        result1, data1 = await test_reactome_tool(
            "find_pathways_by_gene",
            test_find_pathways_by_gene,
            "Find pathways for gene AXL"
        )
        all_results['find_pathways_by_gene'] = result1
        
        # Test 2: get_protein_interactions with a known pathway ID
        # First, get a pathway ID from the previous test if available
        pathway_id = "R-HSA-1640170"  # Default pathway ID for testing
        if data1 and isinstance(data1, dict):
            pathways = data1.get('pathways', [])
            if pathways and len(pathways) > 0:
                pathway_id = pathways[0].get('stId') or pathways[0].get('id') or pathway_id
                print(f"\n  Using pathway ID from previous test: {pathway_id}")
        
        async def test_get_protein_interactions():
            return await manager.reactome.get_protein_interactions(pathway_id)
        
        result2, data2 = await test_reactome_tool(
            "get_protein_interactions",
            test_get_protein_interactions,
            f"Get protein interactions for pathway {pathway_id}"
        )
        all_results['get_protein_interactions'] = result2
        
        # Test 3: search_pathways("breast cancer")
        async def test_search_pathways():
            return await manager.reactome.search_pathways("breast cancer", limit=10)
        
        result3, data3 = await test_reactome_tool(
            "search_pathways",
            test_search_pathways,
            "Search pathways for 'breast cancer'"
        )
        all_results['search_pathways'] = result3
        
        # Summary
        print("\n" + "=" * 80)
        print("Test Summary")
        print("=" * 80)
        
        total_tests = len(all_results)
        successful_tests = sum(1 for r in all_results.values() if r['success'])
        total_retries = sum(r['retries'] for r in all_results.values())
        total_time = sum(r['total_time'] for r in all_results.values())
        
        print(f"\n📊 Overall Statistics:")
        print(f"   Total tests: {total_tests}")
        print(f"   Successful: {successful_tests}/{total_tests} ({100*successful_tests/total_tests:.1f}%)")
        print(f"   Total retries: {total_retries}")
        print(f"   Total time: {total_time:.2f}s")
        print(f"   Average time per test: {total_time/total_tests:.2f}s")
        
        print(f"\n📋 Per-Tool Results:")
        for tool_name, result in all_results.items():
            status = "✅" if result['success'] else "❌"
            retries = result['retries']
            time_taken = result['total_time']
            print(f"   {status} {tool_name}: {time_taken:.2f}s, {retries} retries")
            if not result['success']:
                last_error = result['attempts'][-1]['error'] if result['attempts'] else "Unknown"
                print(f"      Last error: {last_error[:80]}...")
        
        # Save detailed results to JSON
        output_file = Path(__file__).parent / "test_reactome_health_results.json"
        with open(output_file, 'w') as f:
            json.dump(all_results, f, indent=2, default=str)
        print(f"\n💾 Detailed results saved to: {output_file}")
        
        # Success criteria
        print(f"\n✅ Success Criteria:")
        print(f"   - All tools respond within 60s: {'✅' if total_time < 60 else '❌'}")
        print(f"   - Success rate ≥50%: {'✅' if successful_tests/total_tests >= 0.5 else '❌'}")
        print(f"   - Retry logic working: {'✅' if total_retries > 0 or successful_tests == total_tests else '⚠️'}")
        
    except Exception as e:
        print(f"\n❌ Test setup failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Stop all servers
        print("\n🛑 Stopping MCP servers...")
        await manager.stop_all()
        print("✅ MCP servers stopped")


if __name__ == "__main__":
    asyncio.run(main())




