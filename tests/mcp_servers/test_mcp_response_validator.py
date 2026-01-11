#!/usr/bin/env python3
"""
MCP Response Validator

Validates MCP server responses against expected formats from documentation.
Logs actual vs expected response structures and identifies mismatches.
"""

import json
import logging
from typing import Dict, Any, List, Optional, Set
from pathlib import Path

logger = logging.getLogger(__name__)


class MCPResponseValidator:
    """Validates MCP server responses against expected structures."""
    
    def __init__(self):
        self.mismatches = []
        self.warnings = []
    
    def validate_response_structure(
        self,
        response: Any,
        expected_type: type,
        expected_keys: Optional[List[str]] = None,
        server_name: str = "Unknown",
        tool_name: str = "Unknown"
    ) -> Dict[str, Any]:
        """
        Validate response structure against expectations.
        
        Args:
            response: The actual response from MCP server
            expected_type: Expected response type (dict, list, etc.)
            expected_keys: List of expected keys (for dict responses)
            server_name: Name of MCP server
            tool_name: Name of MCP tool
            
        Returns:
            Validation result dictionary
        """
        result = {
            'valid': True,
            'type_match': False,
            'keys_match': False,
            'missing_keys': [],
            'unexpected_keys': [],
            'warnings': []
        }
        
        # Check type
        if isinstance(response, expected_type):
            result['type_match'] = True
        else:
            result['valid'] = False
            result['warnings'].append(
                f"Type mismatch: expected {expected_type.__name__}, got {type(response).__name__}"
            )
            self.warnings.append(
                f"[{server_name}.{tool_name}] Type mismatch: "
                f"expected {expected_type.__name__}, got {type(response).__name__}"
            )
        
        # Check keys (for dict responses)
        if isinstance(response, dict) and expected_keys:
            actual_keys = set(response.keys())
            expected_keys_set = set(expected_keys)
            
            missing = expected_keys_set - actual_keys
            unexpected = actual_keys - expected_keys_set
            
            if missing:
                result['missing_keys'] = list(missing)
                result['valid'] = False
                result['warnings'].append(f"Missing keys: {list(missing)}")
                self.warnings.append(
                    f"[{server_name}.{tool_name}] Missing expected keys: {list(missing)}"
                )
            
            if unexpected:
                result['unexpected_keys'] = list(unexpected)
                result['warnings'].append(f"Unexpected keys: {list(unexpected)}")
                self.warnings.append(
                    f"[{server_name}.{tool_name}] Unexpected keys: {list(unexpected)}"
                )
            
            if not missing:
                result['keys_match'] = True
        
        if not result['valid']:
            self.mismatches.append({
                'server': server_name,
                'tool': tool_name,
                'result': result
            })
        
        return result
    
    def log_response_structure(
        self,
        response: Any,
        server_name: str,
        tool_name: str,
        log_level: int = logging.DEBUG
    ) -> None:
        """
        Log detailed response structure for debugging.
        
        Args:
            response: The response to log
            server_name: Name of MCP server
            tool_name: Name of MCP tool
            log_level: Logging level to use
        """
        logger.log(log_level, f"[{server_name}.{tool_name}] Response type: {type(response).__name__}")
        
        if isinstance(response, dict):
            logger.log(log_level, f"[{server_name}.{tool_name}] Response keys: {list(response.keys())}")
            for key in list(response.keys())[:5]:  # Log first 5 keys
                value = response[key]
                logger.log(log_level, 
                    f"[{server_name}.{tool_name}] Response['{key}']: "
                    f"type={type(value).__name__}, "
                    f"is_list={isinstance(value, list)}, "
                    f"len={len(value) if isinstance(value, (list, dict, str)) else 'N/A'}"
                )
                if isinstance(value, list) and len(value) > 0:
                    first_item = value[0]
                    if isinstance(first_item, dict):
                        logger.log(log_level,
                            f"[{server_name}.{tool_name}] Response['{key}'][0] keys: "
                            f"{list(first_item.keys())[:5]}"
                        )
        elif isinstance(response, list):
            logger.log(log_level, f"[{server_name}.{tool_name}] Response list length: {len(response)}")
            if len(response) > 0:
                logger.log(log_level,
                    f"[{server_name}.{tool_name}] Response[0] type: {type(response[0]).__name__}"
                )
                if isinstance(response[0], dict):
                    logger.log(log_level,
                        f"[{server_name}.{tool_name}] Response[0] keys: "
                        f"{list(response[0].keys())[:5]}"
                    )
    
    def get_summary(self) -> Dict[str, Any]:
        """Get validation summary."""
        return {
            'total_mismatches': len(self.mismatches),
            'total_warnings': len(self.warnings),
            'mismatches': self.mismatches,
            'warnings': self.warnings
        }
    
    def reset(self):
        """Reset validator state."""
        self.mismatches = []
        self.warnings = []




