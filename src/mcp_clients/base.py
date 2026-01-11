"""
Base MCP Client for subprocess communication with JSON-RPC.

Provides the foundation for all MCP server communication via Node.js subprocess execution.

Updated with P0-2: Custom exceptions and retry logic.
Updated with P0-3: Connection pooling for parallel database queries (10-20x improvement).
"""

import asyncio
import json
import logging
from typing import Dict, Any, Optional
from contextlib import asynccontextmanager

from ..core.exceptions import (
    DatabaseConnectionError,
    DatabaseTimeoutError,
    MCPServerError,
    DatabaseUnavailableError,
)
from ..core.retry import async_retry_with_backoff, RetryConfig, DEFAULT_RETRY_CONFIG
# P0-3: Connection pooling imported - but we'll use it differently
# The actual integration happens at the scenario level with batch/parallel queries
# from ..core.connection_pooling import get_global_pool_manager

logger = logging.getLogger(__name__)


# Legacy compatibility - MCPError is now MCPServerError
# Keep for backward compatibility but log deprecation
class MCPError(MCPServerError):
    """
    Legacy MCP error class (DEPRECATED).

    Use MCPServerError instead.
    This class is kept for backward compatibility only.
    """
    def __init__(self, message: str):
        logger.warning(
            "MCPError is deprecated. Use MCPServerError or specific exception types instead."
        )
        # Parse legacy message format to extract server name if possible
        parts = message.split(" server")
        server_name = parts[0] if parts else "Unknown"
        super().__init__(
            server_name=server_name,
            error_code=None,
            error_message=message,
            tool_name=None
        )


class MCPSubprocessClient:
    """Base class for all MCP server communication via subprocess."""

    def __init__(self, server_path: str, server_name: str, timeout: int = 30, server_args: Optional[list] = None):
        """
        Initialize MCP client.

        Args:
            server_path: Path to the MCP server executable (Node.js index.js or Python executable)
            server_name: Human-readable name for logging
            timeout: Request timeout in seconds
            server_args: Optional list of additional arguments (for Python servers: ['-m', 'module_name'])
        """
        self.server_path = server_path
        self.server_name = server_name
        self.timeout = timeout
        self.server_args = server_args or []
        self.process: Optional[asyncio.subprocess.Process] = None
        self.request_id = 0
        # P0-3 UPDATE P0-6: Add per-server semaphore to prevent concurrent stdio access
        # This serializes requests to each server (one at a time) but allows different servers to run in parallel
        # Fixes: "readuntil() called while another coroutine is already waiting" error
        self._server_semaphore = asyncio.Semaphore(1)

        
    async def start(self) -> None:
        """Start MCP server subprocess (supports both Node.js and Python servers)."""
        try:
            # Determine if this is a Node.js or Python server
            # Node.js servers: path ends with .js
            # Python servers: path is a Python executable (python/python3)
            if self.server_path.endswith('.js'):
                # Node.js server
                cmd = ['node', self.server_path]
            else:
                # Python server or other executable
                cmd = [self.server_path] + self.server_args

            # Create subprocess with larger buffer limit for HPA server large responses
            # HPA can return very large protein info responses (10+ MB)
            self.process = await asyncio.create_subprocess_exec(
                *cmd,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                limit=10 * 1024 * 1024  # 10MB buffer limit (was 1MB, increased for HPA)
            )
            logger.info(f"Started {self.server_name} MCP server (PID: {self.process.pid}, cmd: {' '.join(cmd)})")

            # Perform MCP initialization handshake
            # Required for Python-based MCP servers (e.g., UniProt)
            await self._initialize_mcp()

        except FileNotFoundError as e:
            raise DatabaseConnectionError(
                server_name=self.server_name,
                message=f"Server executable not found: {self.server_path}",
                details={"error": str(e), "path": self.server_path}
            )
        except PermissionError as e:
            raise DatabaseConnectionError(
                server_name=self.server_name,
                message=f"Permission denied when starting server",
                details={"error": str(e), "path": self.server_path}
            )
        except (OSError, RuntimeError) as e:
            # Catch specific OS and runtime errors that can occur during server startup
            raise DatabaseConnectionError(
                server_name=self.server_name,
                message=f"Failed to start server: {e}",
                details={"error": str(e), "error_type": type(e).__name__}
            )
    
    async def _initialize_mcp(self) -> None:
        """Perform MCP protocol initialization handshake."""
        try:
            self.request_id += 1
            init_request = {
                "jsonrpc": "2.0",
                "id": self.request_id,
                "method": "initialize",
                "params": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {},
                    "clientInfo": {
                        "name": "omnitarget-pipeline",
                        "version": "1.0.0"
                    }
                }
            }
            
            # Send initialization request
            request_json = json.dumps(init_request) + '\n'
            self.process.stdin.write(request_json.encode())
            await self.process.stdin.drain()
            
            # Read initialization response
            response_line = await asyncio.wait_for(
                self.process.stdout.readline(),
                timeout=5.0
            )
            
            if not response_line:
                logger.warning(f"{self.server_name} initialization: no response (may not support MCP protocol)")
                return
            
            response = json.loads(response_line.decode().strip())
            
            if 'error' in response:
                logger.warning(f"{self.server_name} initialization error (may not support MCP protocol): {response['error']}")
                return
            
            logger.debug(f"{self.server_name} MCP initialized successfully")
            
        except asyncio.TimeoutError:
            # Some servers may not require initialization, ignore timeout
            logger.debug(f"{self.server_name} initialization timeout (may not support MCP protocol)")
        except Exception as e:
            # Don't fail startup if initialization fails (backward compatibility with Node.js servers)
            logger.debug(f"{self.server_name} initialization skipped: {e}")
    
    async def stop(self) -> None:
        """Gracefully terminate MCP server."""
        if self.process:
            try:
                self.process.terminate()
                await asyncio.wait_for(self.process.wait(), timeout=5.0)
                logger.info(f"Stopped {self.server_name} MCP server")
            except asyncio.TimeoutError:
                logger.warning(f"Force killing {self.server_name} server")
                self.process.kill()
                await self.process.wait()
            except ProcessLookupError:
                # Process already terminated
                logger.debug(f"{self.server_name} server already stopped")
            except (OSError, asyncio.CancelledError) as e:
                # Catch specific errors that can occur during cleanup
                logger.warning(
                    f"Error during {self.server_name} server cleanup: {type(e).__name__}: {e}"
                )
            finally:
                self.process = None
    
    async def call_tool(self, tool_name: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Call MCP tool via JSON-RPC.

        Args:
            tool_name: Name of the MCP tool to call
            params: Tool parameters

        Returns:
            Tool response data

        Raises:
            DatabaseConnectionError: If server not started or connection closed
            DatabaseTimeoutError: If request times out
            MCPServerError: If server returns error response
        """
        if not self.process:
            raise DatabaseConnectionError(
                server_name=self.server_name,
                message="Server not started - call start() first",
                details={"tool": tool_name}
            )

        self.request_id += 1

        request = {
            "jsonrpc": "2.0",
            "id": self.request_id,
            "method": "tools/call",
            "params": {
                "name": tool_name,
                "arguments": params
            }
        }

        try:
            # P0-3 UPDATE P0-6: Use per-server semaphore to serialize access to this server
            # This prevents concurrent read operations on stdout which cause "readuntil() called while
            # another coroutine is already waiting" errors in Node.js MCP servers
            async with self._server_semaphore:
                # Send request
                request_json = json.dumps(request) + '\n'
                self.process.stdin.write(request_json.encode())
                await self.process.stdin.drain()

                # Read response with timeout and increased buffer limit
                # HPA server can return very large responses (>64KB)
                response_line = await asyncio.wait_for(
                    self.process.stdout.readline(),
                    timeout=self.timeout
                )

                if not response_line:
                    raise DatabaseConnectionError(
                        server_name=self.server_name,
                        message="Server closed connection unexpectedly",
                        details={"tool": tool_name, "request_id": self.request_id}
                    )

                response = json.loads(response_line.decode().strip())

                if 'error' in response:
                    error_data = response['error']
                    error_code = error_data.get('code')
                    error_msg = error_data.get('message', 'Unknown error')

                    raise MCPServerError(
                        server_name=self.server_name,
                        error_code=error_code,
                        error_message=error_msg,
                        tool_name=tool_name
                    )

                result = response.get('result', {})

                # Some MCP servers return success responses with an isError flag
                if isinstance(result, dict) and result.get('isError'):
                    error_text = ""
                    content = result.get('content')
                    if isinstance(content, list) and content:
                        text_value = content[0].get('text')
                        if isinstance(text_value, str):
                            error_text = text_value
                    raise MCPServerError(
                        server_name=self.server_name,
                        error_code=-32603,
                        error_message=error_text or f"{self.server_name} reported isError for {tool_name}",
                        tool_name=tool_name
                    )

                # Parse MCP content format if present
                # MCP servers return data in format: {"content": [{"type": "text", "text": "..."}]}
                if 'content' in result and isinstance(result['content'], list) and len(result['content']) > 0:
                    content_item = result['content'][0]
                    if isinstance(content_item, dict) and 'text' in content_item:
                        try:
                            # Parse the JSON string inside the text field
                            parsed_data = json.loads(content_item['text'])
                            return parsed_data
                        except json.JSONDecodeError:
                            # If not JSON, return as-is
                            return result

                return result

        except asyncio.TimeoutError:
            raise DatabaseTimeoutError(
                server_name=self.server_name,
                timeout=self.timeout,
                query=f"{tool_name}({params})"
            )
        except json.JSONDecodeError as e:
            raise MCPServerError(
                server_name=self.server_name,
                error_code=-32700,  # Parse error
                error_message=f"Invalid JSON response: {e}",
                tool_name=tool_name
            )
        except (DatabaseConnectionError, DatabaseTimeoutError, MCPServerError):
            # Re-raise our custom exceptions
            raise
        except BrokenPipeError as e:
            raise DatabaseConnectionError(
                server_name=self.server_name,
                message=f"Connection broken (pipe error)",
                details={"tool": tool_name, "error": str(e)}
            )
        except ConnectionError as e:
            raise DatabaseConnectionError(
                server_name=self.server_name,
                message=f"Connection error: {e}",
                details={"tool": tool_name, "error_type": type(e).__name__}
            )
        except (AttributeError, ValueError, TypeError) as e:
            # Catch programming errors that should be fixed, not retried
            raise MCPServerError(
                server_name=self.server_name,
                error_code=-32603,  # Internal error
                error_message=f"Programming error in request/response handling: {type(e).__name__}: {e}",
                tool_name=tool_name
            )
        except Exception as e:
            # Unexpected error - wrap in MCPServerError with internal error code
            raise MCPServerError(
                server_name=self.server_name,
                error_code=-32603,  # Internal error
                error_message=f"Unexpected error: {type(e).__name__}: {e}",
                tool_name=tool_name
            )
    
    async def call_tool_with_retry(
        self,
        tool_name: str,
        params: Dict[str, Any],
        max_retries: int = 3,
        retry_config: Optional[RetryConfig] = None
    ) -> Dict[str, Any]:
        """
        Call MCP tool with exponential backoff retry logic.

        Uses the new retry module with custom exceptions for better error handling.

        Args:
            tool_name: Name of the MCP tool to call
            params: Tool parameters
            max_retries: Maximum number of retry attempts (overrides config if provided)
            retry_config: Optional custom retry configuration

        Returns:
            Tool response data

        Raises:
            DatabaseConnectionError: If connection fails after all retries
            DatabaseTimeoutError: If request times out after all retries
            MCPServerError: If server returns non-retryable error
        """
        # Use provided config or create one with max_retries
        cfg = retry_config or RetryConfig(max_attempts=max_retries)

        # Use the retry_async_operation function for clean retry logic
        from ..core.retry import retry_async_operation

        return await retry_async_operation(
            self.call_tool,
            tool_name,
            params,
            config=cfg,
            operation_name=f"{self.server_name}.{tool_name}"
        )
    
    @asynccontextmanager
    async def session(self):
        """Context manager for MCP server lifecycle."""
        try:
            await self.start()
            yield self
        finally:
            await self.stop()
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(server='{self.server_name}', path='{self.server_path}')"
