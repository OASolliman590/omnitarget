#!/usr/bin/env python3
"""
Test ChEMBL Caching System
Validates LRU cache, TTL expiration, batch processing, and performance.
"""
import asyncio
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.core.chembl_cache import (
    ChEMBLCache,
    CacheEntry,
    CacheStats,
    get_cache,
    cached,
    BatchProcessor,
    CacheWarmer,
    benchmark_cache_performance
)


async def test_cache_basic_operations():
    """Test 1: Basic cache get/set operations."""
    print("\n" + "=" * 80)
    print("Test 1: Basic Cache Operations")
    print("=" * 80)

    cache = ChEMBLCache(max_size=5, ttl=10)

    # Test set and get
    await cache.set("key1", "value1")
    value = await cache.get("key1")
    assert value == "value1", f"Expected 'value1', got {value}"
    print("✅ Basic set/get working")

    # Test cache miss
    value = await cache.get("nonexistent")
    assert value is None, f"Expected None for missing key, got {value}"
    print("✅ Cache miss returns None")

    # Test update existing key
    await cache.set("key1", "updated_value1")
    value = await cache.get("key1")
    assert value == "updated_value1", f"Expected 'updated_value1', got {value}"
    print("✅ Cache update working")

    # Test delete
    deleted = await cache.delete("key1")
    assert deleted is True, "Delete should return True"
    value = await cache.get("key1")
    assert value is None, "Deleted key should return None"
    print("✅ Cache delete working")

    return True


async def test_lru_eviction():
    """Test 2: LRU eviction when cache is full."""
    print("\n" + "=" * 80)
    print("Test 2: LRU Eviction")
    print("=" * 80)

    cache = ChEMBLCache(max_size=3, ttl=100)

    # Fill cache
    await cache.set("key1", "value1")
    await cache.set("key2", "value2")
    await cache.set("key3", "value3")

    stats = await cache.get_stats()
    assert stats.size == 3, f"Cache should have 3 items, has {stats.size}"
    print(f"✅ Cache filled: {stats.size} items")

    # Add 4th item - should evict oldest (key1)
    await cache.set("key4", "value4")

    # Check key1 was evicted
    value = await cache.get("key1")
    assert value is None, "key1 should have been evicted"
    print("✅ Oldest key evicted")

    # Check other keys still exist
    value2 = await cache.get("key2")
    value3 = await cache.get("key3")
    value4 = await cache.get("key4")
    assert value2 == "value2" and value3 == "value3" and value4 == "value4"
    print("✅ Recent keys retained")

    stats = await cache.get_stats()
    assert stats.evictions == 1, f"Expected 1 eviction, got {stats.evictions}"
    print(f"✅ Eviction count correct: {stats.evictions}")

    # Test LRU ordering - access key2, then add key5
    # key2 becomes most recent, so key3 should be evicted
    await cache.get("key2")  # Move key2 to end
    await cache.set("key5", "value5")

    value3 = await cache.get("key3")
    assert value3 is None, "key3 should have been evicted"
    value2 = await cache.get("key2")
    assert value2 == "value2", "key2 should still exist (was recently accessed)"
    print("✅ LRU ordering working correctly")

    return True


async def test_ttl_expiration():
    """Test 3: TTL-based expiration."""
    print("\n" + "=" * 80)
    print("Test 3: TTL Expiration")
    print("=" * 80)

    cache = ChEMBLCache(max_size=10, ttl=1)  # 1 second TTL

    # Add item
    await cache.set("temp_key", "temp_value")
    value = await cache.get("temp_key")
    assert value == "temp_value", "Fresh value should be available"
    print("✅ Fresh value retrieved")

    # Wait for expiration
    print("   Waiting 1.5 seconds for expiration...")
    await asyncio.sleep(1.5)

    # Try to get expired item
    value = await cache.get("temp_key")
    assert value is None, "Expired value should return None"
    print("✅ Expired value correctly removed")

    stats = await cache.get_stats()
    assert stats.misses >= 1, "Should have recorded miss for expired item"
    assert stats.evictions >= 1, "Should have recorded eviction for expired item"
    print(f"✅ Stats updated: {stats.misses} misses, {stats.evictions} evictions")

    # Test prune_expired
    await cache.set("key1", "value1")
    await cache.set("key2", "value2")
    await asyncio.sleep(1.5)

    pruned = await cache.prune_expired()
    assert pruned == 2, f"Should have pruned 2 items, pruned {pruned}"
    print(f"✅ Prune removed {pruned} expired entries")

    return True


async def test_cache_stats():
    """Test 4: Cache statistics tracking."""
    print("\n" + "=" * 80)
    print("Test 4: Cache Statistics")
    print("=" * 80)

    cache = ChEMBLCache(max_size=10, ttl=100)

    # Generate hits and misses
    await cache.set("key1", "value1")
    await cache.set("key2", "value2")

    # 2 hits
    await cache.get("key1")
    await cache.get("key2")

    # 2 misses
    await cache.get("nonexistent1")
    await cache.get("nonexistent2")

    stats = await cache.get_stats()

    assert stats.hits >= 2, f"Expected at least 2 hits, got {stats.hits}"
    assert stats.misses >= 2, f"Expected at least 2 misses, got {stats.misses}"
    assert stats.size == 2, f"Expected size 2, got {stats.size}"
    assert stats.max_size == 10, f"Expected max_size 10, got {stats.max_size}"

    hit_rate = stats.hit_rate
    assert 0.0 <= hit_rate <= 1.0, f"Hit rate should be 0-1, got {hit_rate}"

    print(f"✅ Cache stats: {stats}")
    print(f"   Hit rate: {stats.hit_rate:.1%}")
    print(f"   Miss rate: {stats.miss_rate:.1%}")

    # Test clear
    await cache.clear()
    stats = await cache.get_stats()
    assert stats.size == 0, f"Cache should be empty after clear, size={stats.size}"
    print("✅ Cache clear working")

    return True


async def test_cached_decorator():
    """Test 5: @cached decorator."""
    print("\n" + "=" * 80)
    print("Test 5: @cached Decorator")
    print("=" * 80)

    # Reset global cache
    import src.core.chembl_cache as cache_module
    cache_module._global_cache = None

    call_count = 0

    @cached(ttl=10)
    async def expensive_function(x: int) -> int:
        nonlocal call_count
        call_count += 1
        await asyncio.sleep(0.1)  # Simulate expensive operation
        return x * 2

    # First call - should execute function
    result1 = await expensive_function(5)
    assert result1 == 10, f"Expected 10, got {result1}"
    assert call_count == 1, f"Should have called function once, called {call_count} times"
    print(f"✅ First call executed function (result={result1})")

    # Second call with same args - should use cache
    result2 = await expensive_function(5)
    assert result2 == 10, f"Expected 10, got {result2}"
    assert call_count == 1, f"Should still be 1 call (cached), got {call_count} calls"
    print(f"✅ Second call used cache (no function execution)")

    # Different args - should execute function again
    result3 = await expensive_function(10)
    assert result3 == 20, f"Expected 20, got {result3}"
    assert call_count == 2, f"Should have called function twice, called {call_count} times"
    print(f"✅ Different args executed function (result={result3})")

    # Check cache stats
    cache = get_cache()
    stats = await cache.get_stats()
    assert stats.hits >= 1, f"Expected at least 1 cache hit, got {stats.hits}"
    print(f"✅ Cache stats: {stats.hits} hits, {stats.misses} misses")

    return True


async def test_batch_processor():
    """Test 6: Parallel batch processing."""
    print("\n" + "=" * 80)
    print("Test 6: Batch Processor")
    print("=" * 80)

    processor = BatchProcessor(max_concurrent=3, delay=0.05)

    # Test function that tracks execution order
    execution_times = []

    async def process_item(item: int) -> int:
        start = time.time()
        await asyncio.sleep(0.1)  # Simulate work
        execution_times.append((item, time.time() - start))
        return item * 2

    # Process 6 items (2 batches of 3)
    items = [1, 2, 3, 4, 5, 6]

    start_time = time.time()
    results = await processor.process_batch(items, process_item)
    elapsed = time.time() - start_time

    # Check results are correct
    expected = [2, 4, 6, 8, 10, 12]
    assert results == expected, f"Expected {expected}, got {results}"
    print(f"✅ All items processed correctly: {results}")

    # Check parallel execution
    # With max_concurrent=3, 6 items should take ~2*0.1 + delays = ~0.3s
    # Sequential would take 6*0.1 = 0.6s
    assert elapsed < 0.5, f"Parallel processing too slow: {elapsed:.2f}s"
    print(f"✅ Parallel processing working (completed in {elapsed:.2f}s)")

    # Test error handling
    async def failing_function(item: int) -> int:
        if item == 3:
            raise ValueError(f"Error processing {item}")
        return item * 2

    results_with_errors = await processor.process_batch(
        [1, 2, 3, 4],
        failing_function,
        return_exceptions=True
    )

    # Should have None for failed item
    assert results_with_errors[2] is None, "Failed item should return None"
    assert results_with_errors[0] == 2, "Other items should succeed"
    print("✅ Error handling working (returns None for failed items)")

    return True


async def test_cache_warmer():
    """Test 7: Cache warmer (mock)."""
    print("\n" + "=" * 80)
    print("Test 7: Cache Warmer (Mock)")
    print("=" * 80)

    cache = ChEMBLCache(max_size=100, ttl=3600)

    # Mock ChEMBL client
    class MockChEMBLClient:
        async def search_targets(self, gene: str, limit: int):
            return {'targets': [{'target_chembl_id': f'CHEMBL_{gene}'}]}

        async def search_compounds(self, compound: str, limit: int):
            return {'compounds': [{'molecule_chembl_id': f'CHEMBL_{compound}'}]}

    mock_client = MockChEMBLClient()
    warmer = CacheWarmer(cache, mock_client)

    # Warm targets
    genes = ['EGFR', 'BRCA1', 'TP53']
    count = await warmer.warm_common_targets(genes)
    assert count == 3, f"Expected 3 cached targets, got {count}"
    print(f"✅ Warmed {count} targets")

    # Warm compounds
    compounds = ['ASPIRIN', 'GEFITINIB', 'ERLOTINIB']
    count = await warmer.warm_common_compounds(compounds)
    assert count == 3, f"Expected 3 cached compounds, got {count}"
    print(f"✅ Warmed {count} compounds")

    # Check cache populated
    stats = await cache.get_stats()
    assert stats.size == 6, f"Expected 6 cached items, got {stats.size}"
    print(f"✅ Cache populated: {stats.size} items")

    return True


async def test_performance_benchmark():
    """Test 8: Performance benchmarking."""
    print("\n" + "=" * 80)
    print("Test 8: Performance Benchmark")
    print("=" * 80)

    cache = ChEMBLCache(max_size=1000, ttl=3600)

    # Benchmark with small test data
    test_key = "benchmark_test"
    test_value = {"data": "x" * 1000}  # 1KB data

    results = await benchmark_cache_performance(
        cache,
        test_key,
        test_value,
        iterations=50  # Reduced for quick test
    )

    print(f"\n   Performance Results:")
    print(f"   Write: {results['writes_per_sec']:.0f} ops/sec")
    print(f"   Read:  {results['reads_per_sec']:.0f} ops/sec")
    print(f"   Write latency: {results['write_time_per_op']*1000:.2f}ms")
    print(f"   Read latency:  {results['read_time_per_op']*1000:.2f}ms")

    # Basic performance assertions
    assert results['writes_per_sec'] > 100, "Write performance too slow"
    assert results['reads_per_sec'] > 100, "Read performance too slow"
    assert results['write_time_per_op'] < 0.1, "Write latency too high"
    assert results['read_time_per_op'] < 0.1, "Read latency too high"

    print(f"\n✅ Performance acceptable")

    return True


async def test_cache_key_generation():
    """Test 9: Cache key generation."""
    print("\n" + "=" * 80)
    print("Test 9: Cache Key Generation")
    print("=" * 80)

    cache = ChEMBLCache()

    # Test simple key
    key1 = cache._make_key("func", "arg1", "arg2")
    assert key1 == "func:arg1:arg2", f"Expected 'func:arg1:arg2', got {key1}"
    print(f"✅ Simple key: {key1}")

    # Test with kwargs
    key2 = cache._make_key("func", "arg1", limit=10, offset=0)
    assert "limit=10" in key2 and "offset=0" in key2
    print(f"✅ Key with kwargs: {key2}")

    # Test long key (should hash)
    long_args = ["x" * 50] * 10
    key3 = cache._make_key("func", *long_args)
    assert len(key3) < 100, f"Long key should be hashed, length={len(key3)}"
    assert "func:" in key3, "Hashed key should contain function name"
    print(f"✅ Long key hashed: {key3}")

    # Test complex objects (should use type name)
    key4 = cache._make_key("func", {"data": "value"}, [1, 2, 3])
    assert "dict" in key4 or "list" in key4
    print(f"✅ Complex object key: {key4}")

    return True


async def run_all_tests():
    """Run all cache tests."""
    print("=" * 80)
    print("ChEMBL Cache System Test Suite")
    print("=" * 80)
    print("\nTesting cache implementation, LRU eviction, TTL, and performance...")

    tests = [
        ("Basic Operations", test_cache_basic_operations),
        ("LRU Eviction", test_lru_eviction),
        ("TTL Expiration", test_ttl_expiration),
        ("Cache Statistics", test_cache_stats),
        ("@cached Decorator", test_cached_decorator),
        ("Batch Processor", test_batch_processor),
        ("Cache Warmer", test_cache_warmer),
        ("Performance Benchmark", test_performance_benchmark),
        ("Cache Key Generation", test_cache_key_generation),
    ]

    passed = 0
    failed = 0

    for test_name, test_func in tests:
        try:
            result = await test_func()
            if result:
                passed += 1
            else:
                failed += 1
                print(f"❌ {test_name} FAILED")
        except Exception as e:
            failed += 1
            print(f"❌ {test_name} FAILED with error: {e}")
            import traceback
            traceback.print_exc()

    # Summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    print(f"Total tests: {len(tests)}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")

    if failed == 0:
        print("\n✅ ALL CACHE TESTS PASSED!")
        print("=" * 80)
        print("\nCache system features validated:")
        print("  ✅ LRU eviction policy")
        print("  ✅ TTL-based expiration")
        print("  ✅ Cache statistics (hits, misses, evictions)")
        print("  ✅ @cached decorator for async functions")
        print("  ✅ Parallel batch processing with rate limiting")
        print("  ✅ Cache warming for common queries")
        print("  ✅ Performance benchmarking")
        print("  ✅ Key generation and hashing")
        print("\nPhase 6 caching implementation complete!")
    else:
        print(f"\n⚠️  {failed} test(s) failed")

    return failed == 0


if __name__ == "__main__":
    success = asyncio.run(run_all_tests())
    sys.exit(0 if success else 1)
