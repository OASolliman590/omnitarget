"""
MCP Client Manager

Unified manager for all MCP clients with lifecycle management and parallel execution.
"""

import asyncio
import json
import logging
from typing import Dict, Any, Optional
from contextlib import asynccontextmanager
from pathlib import Path

from ..mcp_clients import KEGGClient, ReactomeClient, STRINGClient, HPAClient, UniProtClient, ChEMBLClient
from .exceptions import DatabaseConnectionError, DatabaseUnavailableError, format_error_for_logging

logger = logging.getLogger(__name__)


class MCPClientManager:
    """Unified manager for all MCP clients."""
    
    def __init__(self, config_path: str):
        """
        Initialize MCP client manager.
        
        Args:
            config_path: Path to MCP servers configuration file
        """
        self.config_path = config_path
        self.config = self._load_config()
        self.clients = self._initialize_clients()
        self._started = False
    
    def _load_config(self) -> Dict[str, Any]:
        """Load MCP server configuration."""
        try:
            with open(self.config_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            raise DatabaseConnectionError(
                server_name="MCP_CONFIG",
                message=f"MCP configuration file not found: {self.config_path}",
                details={"config_path": self.config_path}
            )
        except json.JSONDecodeError as e:
            raise DatabaseConnectionError(
                server_name="MCP_CONFIG",
                message=f"Invalid JSON in MCP configuration file: {e}",
                details={"config_path": self.config_path, "error": str(e)}
            )
        except (OSError, IOError) as e:
            raise DatabaseConnectionError(
                server_name="MCP_CONFIG",
                message=f"Failed to read MCP configuration: {e}",
                details={"config_path": self.config_path, "error_type": type(e).__name__}
            )
    
    def _initialize_clients(self) -> Dict[str, Any]:
        """Initialize all MCP clients."""
        clients = {}
        
        # Initialize KEGG client
        if 'kegg' in self.config:
            clients['kegg'] = KEGGClient(self.config['kegg']['path'])
        else:
            raise ValueError("KEGG server configuration not found")
        
        # Initialize Reactome client
        if 'reactome' in self.config:
            clients['reactome'] = ReactomeClient(self.config['reactome']['path'])
        else:
            raise ValueError("Reactome server configuration not found")
        
        # Initialize STRING client
        if 'string' in self.config:
            clients['string'] = STRINGClient(self.config['string']['path'])
        else:
            raise ValueError("STRING server configuration not found")
        
        # Initialize HPA client
        if 'hpa' in self.config:
            clients['hpa'] = HPAClient(self.config['hpa']['path'])
        else:
            raise ValueError("HPA server configuration not found")
        
        # Initialize UniProt client (optional)
        if 'uniprot' in self.config:
            uniprot_config = self.config['uniprot']
            clients['uniprot'] = UniProtClient(
                uniprot_config['path'],
                uniprot_config.get('args')
            )
            logger.info("UniProt client initialized")
        else:
            logger.info("UniProt server not configured, skipping (will use HPA fallback)")

        # Initialize ChEMBL client (optional but recommended for drug discovery)
        if 'chembl' in self.config:
            clients['chembl'] = ChEMBLClient(self.config['chembl']['path'])
            logger.info("ChEMBL client initialized")
        else:
            logger.info("ChEMBL server not configured, skipping (drug discovery features limited)")

        return clients
    
    async def start_all(self) -> None:
        """Start all MCP servers in parallel."""
        if self._started:
            logger.warning("MCP clients already started")
            return
        
        logger.info("Starting all MCP servers...")
        
        # Start all clients in parallel
        start_tasks = [
            client.start() for client in self.clients.values()
        ]
        
        try:
            await asyncio.gather(*start_tasks)
            self._started = True
            logger.info("All MCP servers started successfully")
        except (DatabaseConnectionError, DatabaseUnavailableError) as e:
            # Database connection errors - log and re-raise
            logger.error(
                "Failed to start MCP servers - database connection error",
                extra=format_error_for_logging(e)
            )
            # Clean up any partially started servers
            await self.stop_all()
            raise
        except Exception as e:
            # Unexpected errors during startup
            logger.error(
                f"Failed to start MCP servers - unexpected error: {type(e).__name__}: {e}",
                extra=format_error_for_logging(e)
            )
            # Clean up any partially started servers
            await self.stop_all()
            raise
    
    async def stop_all(self) -> None:
        """Stop all MCP servers."""
        if not self._started:
            return
        
        logger.info("Stopping all MCP servers...")
        
        # Stop all clients in parallel
        stop_tasks = [
            client.stop() for client in self.clients.values()
        ]
        
        try:
            await asyncio.gather(*stop_tasks, return_exceptions=True)
            self._started = False
            logger.info("All MCP servers stopped")
        except (DatabaseConnectionError, DatabaseUnavailableError) as e:
            # Log database errors during shutdown but don't raise (shutdown should be best-effort)
            logger.warning(
                "Database error during MCP server shutdown",
                extra=format_error_for_logging(e)
            )
        except Exception as e:
            logger.error(
                f"Error stopping MCP servers: {type(e).__name__}: {e}",
                extra=format_error_for_logging(e)
            )
    
    @asynccontextmanager
    async def session(self):
        """Context manager for MCP server lifecycle."""
        try:
            await self.start_all()
            yield self
        finally:
            await self.stop_all()
    
    # Convenience methods for accessing specific clients
    @property
    def kegg(self) -> KEGGClient:
        """Get KEGG client."""
        if not self._started:
            raise RuntimeError("MCP clients not started. Use session() context manager.")
        return self.clients['kegg']
    
    @property
    def reactome(self) -> ReactomeClient:
        """Get Reactome client."""
        if not self._started:
            raise RuntimeError("MCP clients not started. Use session() context manager.")
        return self.clients['reactome']
    
    @property
    def string(self) -> STRINGClient:
        """Get STRING client."""
        if not self._started:
            raise RuntimeError("MCP clients not started. Use session() context manager.")
        return self.clients['string']
    
    @property
    def hpa(self) -> HPAClient:
        """Get HPA client."""
        if not self._started:
            raise RuntimeError("MCP clients not started. Use session() context manager.")
        return self.clients['hpa']
    
    @property
    def uniprot(self) -> Optional[UniProtClient]:
        """Get UniProt client (optional, may be None)."""
        if not self._started:
            raise RuntimeError("MCP clients not started. Use session() context manager.")
        return self.clients.get('uniprot')

    @property
    def chembl(self) -> Optional[ChEMBLClient]:
        """Get ChEMBL client (optional, may be None)."""
        if not self._started:
            raise RuntimeError("MCP clients not started. Use session() context manager.")
        return self.clients.get('chembl')

    async def health_check(self) -> Dict[str, bool]:
        """Check health of all MCP servers."""
        health_status = {}
        
        for name, client in self.clients.items():
            try:
                # Try a simple operation to test connectivity
                if name == 'kegg':
                    await client.get_database_info()
                elif name == 'reactome':
                    await client.search_pathways("test", limit=1)
                elif name == 'string':
                    await client.search_proteins("test", limit=1)
                elif name == 'hpa':
                    # HPA responses can be very large, skip actual call
                    # Just check if client is initialized
                    if client.process and client.process.returncode is None:
                        pass  # Server is running
                    else:
                        raise Exception("HPA server not running")
                elif name == 'chembl':
                    # Test ChEMBL with simple aspirin lookup
                    await client.search_compounds("aspirin", limit=1)
                elif name == 'uniprot':
                    # Skip UniProt health check (optional server)
                    if client.process and client.process.returncode is None:
                        pass  # Server is running
                    else:
                        raise Exception("UniProt server not running")

                health_status[name] = True
            except (DatabaseConnectionError, DatabaseTimeoutError, MCPServerError) as e:
                # Database/MCP errors
                logger.warning(
                    f"Health check failed for {name}",
                    extra=format_error_for_logging(e)
                )
                health_status[name] = False
            except Exception as e:
                # Unexpected errors
                logger.warning(
                    f"Health check failed for {name} with unexpected error: {type(e).__name__}: {e}",
                    extra=format_error_for_logging(e)
                )
                health_status[name] = False
        
        return health_status
    
    def get_client_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status information for all clients."""
        status = {}
        
        for name, client in self.clients.items():
            status[name] = {
                'name': client.server_name,
                'path': client.server_path,
                'started': self._started,
                'process_pid': client.process.pid if client.process else None
            }
        
        return status
    
    def __repr__(self) -> str:
        return f"MCPClientManager(config='{self.config_path}', started={self._started})"
