"""
MCP Client Layer

Provides unified interface for communicating with MCP servers via subprocess execution.
"""

from .base import MCPSubprocessClient, MCPError
from .kegg_client import KEGGClient
from .reactome_client import ReactomeClient
from .string_client import STRINGClient
from .hpa_client import HPAClient
from .uniprot_client import UniProtClient
from .chembl_client import ChEMBLClient

__all__ = [
    'MCPSubprocessClient',
    'MCPError',
    'KEGGClient',
    'ReactomeClient',
    'STRINGClient',
    'HPAClient',
    'UniProtClient',
    'ChEMBLClient'
]
