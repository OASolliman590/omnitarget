#!/usr/bin/env python3
"""
MCP Server Startup Script
Purpose: Start all MCP servers and verify they're working
Created: 2025-10-27
"""

import asyncio
import json
import subprocess
import time
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from mcp_clients.reactome_client import ReactomeClient
from mcp_clients.hpa_client import HPAClient
from mcp_clients.kegg_client import KEGGClient
from mcp_clients.string_client import StringClient

class MCPServerManager:
    def __init__(self):
        self.config_path = "config/mcp_servers.json"
        self.servers = {}
        
    async def start_all_servers(self):
        """Start all MCP servers and test them"""
        print("🚀 Starting MCP Servers...")
        print("=" * 50)
        
        # Load server configs
        with open(self.config_path, 'r') as f:
            config = json.load(f)
            
        # Start each server
        for server_name, server_config in config.items():
            print(f"\nStarting {server_name.upper()} server...")
            try:
                # Start server process
                process = subprocess.Popen(
                    ['node', server_config['path']],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True
                )
                
                self.servers[server_name] = process
                print(f"✅ {server_name} started (PID: {process.pid})")
                
                # Give server time to start
                await asyncio.sleep(2)
                
            except Exception as e:
                print(f"❌ Failed to start {server_name}: {e}")
                
        # Test all servers
        print(f"\n🧪 Testing MCP Servers...")
        print("-" * 30)
        
        await self.test_kegg()
        await self.test_reactome()
        await self.test_hpa()
        await self.test_string()
        
        # Summary
        print(f"\n📊 MCP Server Status Summary")
        print("=" * 50)
        for server_name, process in self.servers.items():
            if process.poll() is None:
                print(f"✅ {server_name}: Running (PID: {process.pid})")
            else:
                print(f"❌ {server_name}: Not running")
                
    async def test_kegg(self):
        """Test KEGG server"""
        print(f"\nTesting KEGG...")
        try:
            client = KEGGClient(self.config_path)
            result = await client.search_pathways('breast cancer', limit=1)
            if result.get('pathways'):
                print(f"✅ KEGG: Working - {len(result['pathways'])} pathways found")
            else:
                print(f"⚠️ KEGG: Responding but no pathways found")
        except Exception as e:
            print(f"❌ KEGG: {e}")
            
    async def test_reactome(self):
        """Test Reactome server"""
        print(f"\nTesting Reactome...")
        try:
            client = ReactomeClient(self.config_path)
            result = await client.find_pathways_by_disease('breast cancer')
            if result.get('pathways'):
                print(f"✅ Reactome: Working - {len(result['pathways'])} pathways found")
            else:
                print(f"⚠️ Reactome: Responding but no pathways found")
        except Exception as e:
            print(f"❌ Reactome: {e}")
            
    async def test_hpa(self):
        """Test HPA server"""
        print(f"\nTesting HPA...")
        try:
            client = HPAClient(self.config_path)
            result = await client.get_protein_info('BRCA1')
            if result.get('uniprot'):
                print(f"✅ HPA: Working - UniProt ID: {result['uniprot']}")
            else:
                print(f"⚠️ HPA: Responding but no UniProt ID found")
        except Exception as e:
            print(f"❌ HPA: {e}")
            
    async def test_string(self):
        """Test STRING server"""
        print(f"\nTesting STRING...")
        try:
            client = StringClient(self.config_path)
            result = await client.get_interaction_network(['BRCA1', 'TP53'])
            if result.get('nodes'):
                print(f"✅ STRING: Working - {len(result['nodes'])} nodes found")
            else:
                print(f"⚠️ STRING: Responding but no nodes found")
        except Exception as e:
            print(f"❌ STRING: {e}")
            
    def stop_all_servers(self):
        """Stop all MCP servers"""
        print(f"\n🛑 Stopping MCP Servers...")
        for server_name, process in self.servers.items():
            if process.poll() is None:
                process.terminate()
                print(f"✅ {server_name} stopped")
            else:
                print(f"⚠️ {server_name} already stopped")

async def main():
    """Main function"""
    manager = MCPServerManager()
    
    try:
        await manager.start_all_servers()
        
        print(f"\n🎯 MCP Servers are ready!")
        print(f"Press Ctrl+C to stop all servers...")
        
        # Keep running until interrupted
        while True:
            await asyncio.sleep(1)
            
    except KeyboardInterrupt:
        print(f"\n\nStopping...")
        manager.stop_all_servers()
        print(f"✅ All servers stopped")

if __name__ == "__main__":
    asyncio.run(main())
