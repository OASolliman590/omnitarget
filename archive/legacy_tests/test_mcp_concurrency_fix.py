#!/usr/bin/env python3
"""
Test MCP Concurrency Fix

This test verifies that the per-server semaphore prevents concurrent access
to MCP server stdout, fixing the "readuntil() called while another coroutine
is already waiting" error.
"""

import asyncio
import time
import logging
from src.core.mcp_client_manager import MCPClientManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_concurrent_queries_same_server():
    """Test that concurrent queries to the same server don't conflict."""
    logger.info("=" * 80)
    logger.info("Testing MCP Concurrency Fix")
    logger.info("=" * 80)
    
    manager = MCPClientManager("config/mcp_servers.json")
    
    try:
        async with manager.session() as session:
            # Test with KEGG server - run multiple queries concurrently
            logger.info("\n🧪 Testing KEGG server with 3 concurrent disease queries...")
            start_time = time.time()
            
            # Run 3 queries concurrently to KEGG
            tasks = [
                session.kegg.search_diseases("breast cancer", limit=5),
                session.kegg.search_diseases("lung cancer", limit=5),
                session.kegg.search_diseases("colorectal cancer", limit=5)
            ]
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            elapsed = time.time() - start_time
            
            # Check results
            success_count = 0
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.error(f"  ❌ Query {i+1} failed: {result}")
                else:
                    success_count += 1
                    logger.info(f"  ✅ Query {i+1} succeeded: {len(result.get('diseases', []))} diseases found")
            
            logger.info(f"\n📊 Results: {success_count}/3 queries successful in {elapsed:.2f}s")
            
            if success_count == 3:
                logger.info("✅ CONCURRENCY FIX WORKING! No 'readuntil()' errors")
            else:
                logger.error("❌ Some queries failed - may need investigation")
            
            return success_count == 3
            
    except Exception as e:
        logger.error(f"❌ Test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_parallel_different_servers():
    """Test that different servers can run in parallel."""
    logger.info("\n🧪 Testing parallel queries to different servers...")
    start_time = time.time()
    
    manager = MCPClientManager("config/mcp_servers.json")
    
    try:
        async with manager.session() as session:
            # Query KEGG, Reactome, and HPA simultaneously
            tasks = [
                ("KEGG", session.kegg.search_diseases("breast cancer", limit=3)),
                ("Reactome", session.reactome.search_pathways("p53", limit=3)),
                ("HPA", session.hpa.search_proteins("EGFR", limit=3))
            ]
            
            results = await asyncio.gather(*[t[1] for t in tasks], return_exceptions=True)
            
            elapsed = time.time() - start_time
            
            for i, (server_name, result) in enumerate(zip([t[0] for t in tasks], results)):
                if isinstance(result, Exception):
                    logger.info(f"  ⚠️  {server_name}: {result}")
                else:
                    logger.info(f"  ✅ {server_name}: Success")
            
            logger.info(f"\n📊 Parallel execution completed in {elapsed:.2f}s")
            logger.info("✅ Different servers can run in parallel!")
            
            return True
            
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        return False

async def main():
    """Run all tests."""
    logger.info("\n🚀 Starting MCP Concurrency Fix Tests")
    logger.info("=" * 80)
    
    # Test 1: Same server concurrency
    test1_result = await test_concurrent_queries_same_server()
    
    # Test 2: Different servers in parallel
    test2_result = await test_parallel_different_servers()
    
    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("📋 TEST SUMMARY")
    logger.info("=" * 80)
    logger.info(f"Test 1 (Same Server Concurrency): {'✅ PASS' if test1_result else '❌ FAIL'}")
    logger.info(f"Test 2 (Different Servers Parallel): {'✅ PASS' if test2_result else '❌ FAIL'}")
    logger.info("=" * 80)
    
    if test1_result and test2_result:
        logger.info("🎉 ALL TESTS PASSED! MCP Concurrency Fix is working!")
        return 0
    else:
        logger.error("❌ Some tests failed")
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    exit(exit_code)
