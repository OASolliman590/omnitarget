#!/usr/bin/env python3
"""
Debug script to get detailed UniProt MCP error with full traceback.
"""
import asyncio
import sys
import traceback
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

from src.core.mcp_client_manager import MCPClientManager


async def test_uniprot():
    """Test UniProt MCP with detailed error reporting."""
    config_path = "config/mcp_servers.json"
    
    print("=" * 80)
    print("🧪 DETAILED UNIPROT MCP DEBUG")
    print("=" * 80)
    
    manager = MCPClientManager(config_path)
    async with manager.session() as mcp_manager:
        # Test with a known UniProt ID
        test_accession = "P27694"  # One that failed in logs
        
        print(f"\n📍 Testing UniProt ID: {test_accession}")
        print(f"   Client available: {mcp_manager.uniprot is not None}")
        
        if not mcp_manager.uniprot:
            print("❌ UniProt client not available!")
            return
        
        try:
            print(f"\n🔄 Calling get_protein_info({test_accession})...")
            result = await mcp_manager.uniprot.get_protein_info(test_accession)
            print(f"✅ Success!")
            print(f"   Result type: {type(result)}")
            print(f"   Result keys: {result.keys() if isinstance(result, dict) else 'not a dict'}")
            print(f"   Result: {result}")
            
        except Exception as e:
            print(f"\n❌ ERROR CAUGHT:")
            print(f"   Type: {type(e).__name__}")
            print(f"   Message: {str(e)}")
            print(f"\n📋 FULL TRACEBACK:")
            traceback.print_exc()
            
            # Try to get more info from the exception
            if hasattr(e, '__cause__'):
                print(f"\n🔗 Caused by:")
                print(f"   {e.__cause__}")
            
            if hasattr(e, '__context__'):
                print(f"\n🔗 Context:")
                print(f"   {e.__context__}")


if __name__ == "__main__":
    asyncio.run(test_uniprot())

