#!/usr/bin/env python3
"""
Quick test script to verify ChEMBL MCP server functionality.
"""
import asyncio
import subprocess
import json
import sys

async def test_chembl_server():
    """Test ChEMBL server startup and basic operations."""

    server_path = "/Users/omara.soliman/Documents/mcp/chembl/build/index.js"

    print("=" * 80)
    print("ChEMBL MCP Server Test")
    print("=" * 80)

    # Start server
    print("\n1. Starting ChEMBL server...")
    process = subprocess.Popen(
        ['node', server_path],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

    try:
        # Test 1: Initialize
        print("\n2. Testing MCP initialization...")
        init_request = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {
                    "name": "omnitarget-test",
                    "version": "1.0.0"
                }
            }
        }

        process.stdin.write(json.dumps(init_request) + '\n')
        process.stdin.flush()

        response_line = process.stdout.readline()
        response = json.loads(response_line)

        if 'result' in response:
            print("   ✅ Server initialized successfully")
            print(f"   Server: {response['result'].get('serverInfo', {}).get('name', 'unknown')}")
            print(f"   Protocol: {response['result'].get('protocolVersion', 'unknown')}")
        else:
            print(f"   ❌ Initialization failed: {response.get('error', 'unknown error')}")
            return False

        # Test 2: List tools
        print("\n3. Testing tools/list...")
        list_tools_request = {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/list",
            "params": {}
        }

        process.stdin.write(json.dumps(list_tools_request) + '\n')
        process.stdin.flush()

        response_line = process.stdout.readline()
        response = json.loads(response_line)

        if 'result' in response and 'tools' in response['result']:
            tools = response['result']['tools']
            print(f"   ✅ Found {len(tools)} tools")
            print("\n   Available tools:")
            for i, tool in enumerate(tools[:10], 1):
                print(f"      {i}. {tool['name']}")
            if len(tools) > 10:
                print(f"      ... and {len(tools) - 10} more")
        else:
            print(f"   ❌ Failed to list tools: {response.get('error', 'unknown error')}")
            return False

        # Test 3: Search compounds (aspirin)
        print("\n4. Testing search_compounds with 'aspirin'...")
        search_request = {
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {
                "name": "search_compounds",
                "arguments": {
                    "query": "aspirin",
                    "limit": 3
                }
            }
        }

        process.stdin.write(json.dumps(search_request) + '\n')
        process.stdin.flush()

        response_line = process.stdout.readline()
        response = json.loads(response_line)

        if 'result' in response:
            # Parse the content
            content = response['result'].get('content', [])
            if content and len(content) > 0:
                text_content = content[0].get('text', '{}')
                data = json.loads(text_content)

                if 'molecules' in data:
                    molecules = data['molecules']
                    print(f"   ✅ Found {len(molecules)} compound(s)")
                    for mol in molecules[:3]:
                        chembl_id = mol.get('molecule_chembl_id', 'N/A')
                        name = mol.get('pref_name', 'N/A')
                        print(f"      - {chembl_id}: {name}")
                else:
                    print(f"   ⚠️  No molecules in response: {data}")
            else:
                print(f"   ⚠️  Empty content in response")
        else:
            error_msg = response.get('error', {}).get('message', 'unknown error')
            print(f"   ❌ Search failed: {error_msg}")
            return False

        print("\n" + "=" * 80)
        print("✅ ALL TESTS PASSED - ChEMBL server is working correctly!")
        print("=" * 80)
        return True

    except json.JSONDecodeError as e:
        print(f"\n❌ JSON decode error: {e}")
        print(f"Response: {response_line if 'response_line' in locals() else 'N/A'}")
        return False
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        # Cleanup
        process.terminate()
        try:
            process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            process.kill()

if __name__ == "__main__":
    success = asyncio.run(test_chembl_server())
    sys.exit(0 if success else 1)
