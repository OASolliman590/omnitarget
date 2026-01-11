#!/usr/bin/env python3
"""
Debug script to capture UniProt MCP server stderr output.
"""
import asyncio
import sys
import traceback
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

from src.mcp_clients.uniprot_client import UniProtClient


async def test_uniprot_stderr():
    """Test UniProt MCP and capture stderr."""
    print("=" * 80)
    print("🧪 UNIPROT MCP STDERR CAPTURE")
    print("=" * 80)
    
    # Read config to get UniProt server path
    import json
    with open("config/mcp_servers.json") as f:
        config = json.load(f)
    
    uniprot_config = config.get("uniprot", {})
    server_path = uniprot_config.get("path")
    server_args = uniprot_config.get("args", [])
    
    print(f"\nServer path: {server_path}")
    print(f"Server args: {server_args}")
    
    client = UniProtClient(server_path, server_args)
    
    try:
        print("\n🔄 Starting UniProt MCP server...")
        await client.start()
        
        print(f"✅ Server started (PID: {client.process.pid})")
        
        # Give it a moment
        await asyncio.sleep(1)
        
        # Check if still running
        if client.process.returncode is not None:
            print(f"\n❌ Server exited with code: {client.process.returncode}")
            
            # Read stderr
            stderr_data = await client.process.stderr.read()
            if stderr_data:
                print(f"\n📋 STDERR OUTPUT:")
                print(stderr_data.decode())
            else:
                print("\n📋 No stderr output")
        else:
            print(f"\n✅ Server still running")
            
            # Try to call a tool
            print(f"\n🔄 Calling get_protein_info...")
            try:
                result = await client.call_tool("get_protein_info", {"accession": "P27694"})
                print(f"✅ Result: {result}")
            except Exception as e:
                print(f"❌ Call failed: {e}")
                
                # Check stderr again
                try:
                    stderr_data = await asyncio.wait_for(
                        client.process.stderr.read(1024),
                        timeout=0.5
                    )
                    if stderr_data:
                        print(f"\n📋 STDERR after call:")
                        print(stderr_data.decode())
                except:
                    pass
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        traceback.print_exc()
    finally:
        await client.stop()


if __name__ == "__main__":
    asyncio.run(test_uniprot_stderr())

