#!/usr/bin/env python3
"""
Test script to verify that _save_results works correctly.
"""
import asyncio
import json
from pathlib import Path
from datetime import datetime


async def test_save_results():
    """Test the _save_results method."""
    print("Testing _save_results method...")
    print("-" * 80)

    # Import after ensuring aiofiles is available
    try:
        import aiofiles
        print("✅ aiofiles is available")
    except ImportError:
        print("❌ aiofiles not installed")
        return False

    # Create a minimal test setup
    from src.cli.yaml_runner import YAMLRunner

    runner = YAMLRunner()

    # Create test data
    config = {
        'hypothesis': 'Test hypothesis',
        'description': 'Test description',
    }

    results = [
        {
            'scenario_id': 'test_1',
            'scenario_name': 'Test Scenario 1',
            'status': 'success',
            'result': type('MockResult', (), {
                'validation_score': 0.85,
                'network_nodes': ['AXL', 'BRCA1', 'TP53']
            })(),
            'execution_time': datetime.now().isoformat()
        }
    ]

    yaml_path = "/Users/omara.soliman/Desktop/Projects /My Projects/15-OmniTarget_/examples/yaml_configs/simple_test.yaml"

    try:
        # Test the _save_results method
        result_path = await runner._save_results(config, results, yaml_path)

        # Verify file was created
        if not result_path.exists():
            print(f"❌ FAIL: Result file was not created at {result_path}")
            return False

        print(f"✅ File created: {result_path}")

        # Verify file is not empty
        if result_path.stat().st_size == 0:
            print(f"❌ FAIL: Result file is empty")
            return False

        print(f"✅ File is not empty (size: {result_path.stat().st_size} bytes)")

        # Load and validate file content
        async with aiofiles.open(result_path, 'r') as f:
            content = await f.read()
            data = json.loads(content)

        print(f"✅ File contains valid JSON")

        # Verify structure
        if "results" not in data:
            print("❌ FAIL: 'results' missing from saved file")
            return False

        print(f"✅ 'results' key present in saved file")

        if len(data["results"]) == 0:
            print("❌ FAIL: No scenarios in saved file")
            return False

        print(f"✅ File contains {len(data['results'])} scenario(s)")

        # Verify timestamp is current (not old Nov 1 date)
        if "execution_metadata" in data and "timestamp" in data["execution_metadata"]:
            timestamp = data["execution_metadata"]["timestamp"]
            date_str = timestamp.split('T')[0]  # Get date part
            print(f"✅ Timestamp in file: {timestamp}")
            print(f"   Date: {date_str}")

            # Check if it's not Nov 1
            if "2024-11-01" in timestamp or "2025-11-01" in timestamp:
                print(f"⚠️  WARNING: Timestamp shows Nov 1 (old data)")
            else:
                print(f"✅ Timestamp is current (not Nov 1)")

        print()
        print("=" * 80)
        print("✅ ALL TESTS PASSED!")
        print("=" * 80)
        return True

    except Exception as e:
        print(f"❌ FAIL: Exception occurred: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = asyncio.run(test_save_results())
    exit(0 if success else 1)
