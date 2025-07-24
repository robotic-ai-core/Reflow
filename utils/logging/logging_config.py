"""
Logging configuration for VibeDiffusion project.
"""

import logging
import sys
import os
from typing import Optional, Dict, Any


def setup_logging(config: Optional[Dict[str, Any]] = None) -> None:
    """
    Setup logging configuration for the entire project.
    
    Args:
        config: Logging configuration dictionary from YAML
    """
    if config is None:
        config = {
            "level": "INFO",
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        }
    
    # Get log level
    level_str = config.get("level", "INFO").upper()
    level = getattr(logging, level_str, logging.INFO)
    
    # Get format
    log_format = config.get("format", "%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    
    # Configure root logger
    logging.basicConfig(
        level=level,
        format=log_format,
        handlers=[
            logging.StreamHandler(sys.stdout)
        ],
        force=True  # Override any existing configuration
    )
    
    # Set specific loggers to appropriate levels
    dataset_config = config.get("dataset_logging", {})
    if dataset_config.get("enabled", True):
        dataset_level_str = dataset_config.get("level", level_str).upper()
        dataset_level = getattr(logging, dataset_level_str, level)
        
        # Configure dataset-specific loggers
        logging.getLogger("modules.streaming_datasets").setLevel(dataset_level)
        logging.getLogger("modules.data_modules").setLevel(dataset_level)
    else:
        # Disable dataset logging
        logging.getLogger("modules.streaming_datasets").setLevel(logging.CRITICAL)
        logging.getLogger("modules.data_modules").setLevel(logging.CRITICAL)


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger with the specified name and configured level.
    
    Args:
        name: Logger name (typically __name__)
        
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    
    # Set the log level based on environment variable
    log_level = os.getenv('VIBE_LOG_LEVEL', 'INFO').upper()
    logger.setLevel(getattr(logging, log_level, logging.INFO))
    
    return logger


class DatasetLogger:
    """
    Simplified logger for dataset operations - just uses standard log levels.
    """
    
    def __init__(self, name: str, config: Optional[Dict[str, Any]] = None):
        self.logger = logging.getLogger(name)
    
    def debug(self, msg: str, *args, **kwargs):
        self.logger.debug(msg, *args, **kwargs)
    
    def info(self, msg: str, *args, **kwargs):
        self.logger.info(msg, *args, **kwargs)
    
    def warning(self, msg: str, *args, **kwargs):
        self.logger.warning(msg, *args, **kwargs)
    
    def error(self, msg: str, *args, **kwargs):
        self.logger.error(msg, *args, **kwargs)
    
    def critical(self, msg: str, *args, **kwargs):
        self.logger.critical(msg, *args, **kwargs)
    
    def log_worker_info(self, msg: str, *args, **kwargs):
        """Log worker-specific information at INFO level."""
        self.info(msg, *args, **kwargs)
    
    def log_shard_info(self, msg: str, *args, **kwargs):
        """Log shard assignment information at INFO level."""
        self.info(msg, *args, **kwargs)
    
    def log_item_count(self, msg: str, *args, **kwargs):
        """Log item count information at INFO level."""
        self.info(msg, *args, **kwargs)


def configure_logging():
    """Configure logging for the entire application."""
    # Get the main log level
    log_level = os.getenv('VIBE_LOG_LEVEL', 'INFO').upper()
    
    # Configure root logger
    logging.basicConfig(
        level=getattr(logging, log_level, logging.INFO),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Suppress noisy third-party loggers
    noisy_loggers = [
        'fsspec',           # HuggingFace file system operations
        'fsspec.http',      # HTTP operations
        'fsspec.core',      # Core fsspec operations
        'urllib3',          # HTTP library
        'requests',         # HTTP requests
        'datasets',         # HuggingFace datasets (some operations)
        'filelock',         # File locking operations
    ]
    
    for logger_name in noisy_loggers:
        logging.getLogger(logger_name).setLevel(logging.WARNING)
    
    # Special handling for dataset logging if explicitly disabled
    dataset_log_enabled = os.getenv('VIBE_DATASET_LOG_ENABLED', 'true').lower() == 'true'
    if not dataset_log_enabled:
        logging.getLogger('datasets').setLevel(logging.ERROR)
        logging.getLogger('modules.streaming_datasets').setLevel(logging.WARNING)
    
    # Configure specific loggers based on environment variables
    if os.getenv('VIBE_FSSPEC_DEBUG', 'false').lower() == 'true':
        # Allow fsspec debug if explicitly requested
        logging.getLogger('fsspec').setLevel(logging.DEBUG)
    
    if os.getenv('VIBE_TORCH_DEBUG', 'false').lower() == 'true':
        # Allow torch debug if explicitly requested
        logging.getLogger('torch').setLevel(logging.DEBUG)
    else:
        # Suppress torch debug messages by default
        logging.getLogger('torch').setLevel(logging.WARNING) 