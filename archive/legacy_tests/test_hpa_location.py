#!/usr/bin/env python3
"""Test HPA subcellular location data format."""

import asyncio
import sys
import logging
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.core.mcp_client_manager import MCPClientManager

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


async def test_hpa_location():
    """Test HPA subcellular location response format."""
    
    config_path = "config/mcp_servers.json"
    manager = MCPClientManager(config_path)
    
    async with manager.session():
        logger.info("Testing HPA subcellular location for BRCA1...")
        
        try:
            location_data = await manager.hpa.get_subcellular_location("BRCA1")
            
            logger.info(f"Response type: {type(location_data)}")
            logger.info(f"Response: {json.dumps(location_data, indent=2) if location_data else 'None'}")
            
        except Exception as e:
            logger.error(f"Error: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(test_hpa_location())

