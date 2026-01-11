"""
Configuration Management for OmniTarget Pipeline

This module provides environment-based configuration management with validation
and support for multiple deployment environments.
"""

import os
import json
from pathlib import Path
from typing import Dict, Any, Optional
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class Environment(Enum):
    """Deployment environments."""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    TESTING = "testing"


class Config:
    """
    Configuration manager with environment-based settings.
    
    Supports configuration via:
    1. Environment variables
    2. Configuration files
    3. Default values
    """
    
    def __init__(self, env: Optional[str] = None):
        """
        Initialize configuration.
        
        Args:
            env: Environment name (development, staging, production, testing)
        """
        self.env = Environment(env or os.getenv('OMNITARGET_ENV', 'development'))
        self._config = self._load_config()
        self._validate()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from environment and files."""
        config = {
            # Environment
            'environment': self.env.value,
            
            # MCP Server Configuration
            'mcp_base_path': os.getenv(
                'MCP_BASE_PATH',
                '/Users/omara.soliman/Documents/mcp/'
            ),
            'mcp_config_path': os.getenv(
                'MCP_CONFIG_PATH',
                'config/mcp_servers.json'
            ),
            'mcp_timeout': int(os.getenv('MCP_TIMEOUT', '30')),
            'mcp_max_retries': int(os.getenv('MCP_MAX_RETRIES', '3')),
            'mcp_retry_delay': int(os.getenv('MCP_RETRY_DELAY', '1')),
            
            # Performance Settings
            'max_workers': int(os.getenv('MAX_WORKERS', '4')),
            'chunk_size': int(os.getenv('CHUNK_SIZE', '1000')),
            'memory_limit_gb': float(os.getenv('MEMORY_LIMIT_GB', '2.0')),
            
            # Caching Configuration
            'cache_enabled': os.getenv('CACHE_ENABLED', 'true').lower() == 'true',
            'cache_ttl': int(os.getenv('CACHE_TTL', '3600')),
            'redis_enabled': os.getenv('REDIS_ENABLED', 'false').lower() == 'true',
            'redis_host': os.getenv('REDIS_HOST', 'localhost'),
            'redis_port': int(os.getenv('REDIS_PORT', '6379')),
            
            # Logging Configuration
            'log_level': os.getenv('LOG_LEVEL', 'INFO'),
            'log_file': os.getenv('LOG_FILE', 'logs/omnitarget.log'),
            'log_format': os.getenv(
                'LOG_FORMAT',
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            ),
            
            # Output Configuration
            'output_dir': os.getenv('OUTPUT_DIR', 'results'),
            'temp_dir': os.getenv('TEMP_DIR', 'tmp'),
            
            # Feature Flags
            'enable_parallel_processing': os.getenv(
                'ENABLE_PARALLEL_PROCESSING', 'true'
            ).lower() == 'true',
            'enable_memory_optimization': os.getenv(
                'ENABLE_MEMORY_OPTIMIZATION', 'true'
            ).lower() == 'true',
            'enable_benchmarking': os.getenv(
                'ENABLE_BENCHMARKING', 'false'
            ).lower() == 'true',
            
            # Scientific Parameters
            'min_confidence_score': float(os.getenv('MIN_CONFIDENCE_SCORE', '0.5')),
            'min_interaction_score': float(os.getenv('MIN_INTERACTION_SCORE', '0.7')),
            'max_network_size': int(os.getenv('MAX_NETWORK_SIZE', '10000')),
        }
        
        # Environment-specific overrides
        if self.env == Environment.PRODUCTION:
            config.update(self._get_production_overrides())
        elif self.env == Environment.TESTING:
            config.update(self._get_testing_overrides())
        
        return config
    
    def _get_production_overrides(self) -> Dict[str, Any]:
        """Get production-specific configuration overrides."""
        return {
            'log_level': 'WARNING',
            'enable_benchmarking': False,
            'mcp_max_retries': 5,
            'cache_enabled': True,
            'redis_enabled': True,
        }
    
    def _get_testing_overrides(self) -> Dict[str, Any]:
        """Get testing-specific configuration overrides."""
        return {
            'log_level': 'DEBUG',
            'mcp_timeout': 10,
            'cache_enabled': False,
            'output_dir': 'test_results',
        }
    
    def _validate(self):
        """Validate configuration parameters."""
        errors = []
        
        # Validate paths
        mcp_base = Path(self._config['mcp_base_path'])
        if not mcp_base.exists():
            errors.append(f"MCP base path does not exist: {mcp_base}")
        
        # Validate numeric ranges
        if not 0 < self._config['min_confidence_score'] <= 1:
            errors.append("min_confidence_score must be between 0 and 1")
        
        if not 0 < self._config['min_interaction_score'] <= 1:
            errors.append("min_interaction_score must be between 0 and 1")
        
        if self._config['mcp_timeout'] < 1:
            errors.append("mcp_timeout must be at least 1 second")
        
        if self._config['max_workers'] < 1:
            errors.append("max_workers must be at least 1")
        
        # Validate Redis configuration if enabled
        if self._config['redis_enabled']:
            if not self._config['redis_host']:
                errors.append("redis_host is required when redis_enabled is true")
        
        # Create required directories
        for dir_key in ['output_dir', 'temp_dir']:
            dir_path = Path(self._config[dir_key])
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Log directory
        log_file = Path(self._config['log_file'])
        log_file.parent.mkdir(parents=True, exist_ok=True)
        
        if errors:
            error_msg = "Configuration validation failed:\n" + "\n".join(f"  - {e}" for e in errors)
            raise ValueError(error_msg)
        
        logger.info(f"✅ Configuration validated successfully for {self.env.value} environment")
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value."""
        return self._config.get(key, default)
    
    def __getitem__(self, key: str) -> Any:
        """Get configuration value using dict-like access."""
        return self._config[key]
    
    def __getattr__(self, key: str) -> Any:
        """Get configuration value using attribute access."""
        if key.startswith('_'):
            return object.__getattribute__(self, key)
        return self._config.get(key)
    
    def to_dict(self) -> Dict[str, Any]:
        """Export configuration as dictionary."""
        return self._config.copy()
    
    def save_to_file(self, filepath: str):
        """Save configuration to JSON file."""
        with open(filepath, 'w') as f:
            json.dump(self._config, f, indent=2)
        logger.info(f"Configuration saved to {filepath}")
    
    @classmethod
    def from_file(cls, filepath: str) -> 'Config':
        """Load configuration from JSON file."""
        with open(filepath, 'r') as f:
            config_data = json.load(f)
        
        instance = cls()
        instance._config.update(config_data)
        instance._validate()
        return instance


# Global configuration instance
_config: Optional[Config] = None


def get_config() -> Config:
    """Get global configuration instance."""
    global _config
    if _config is None:
        _config = Config()
    return _config


def reset_config():
    """Reset global configuration (useful for testing)."""
    global _config
    _config = None


def configure_logging(config: Optional[Config] = None):
    """Configure logging based on configuration."""
    if config is None:
        config = get_config()
    
    logging.basicConfig(
        level=getattr(logging, config.log_level),
        format=config.log_format,
        handlers=[
            logging.FileHandler(config.log_file),
            logging.StreamHandler()
        ]
    )
    
    logger.info(f"✅ Logging configured: level={config.log_level}, file={config.log_file}")

