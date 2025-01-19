"""
The `utils` package provides utility functions and classes for common operations,
such as logging, exception handling, and helper functions, ensuring modularity,
scalability, and reusability throughout the trading system.
"""

from .logger import Logger
from .helpers import Helpers
from .exception_handler import ExceptionHandler
import os

__all__ = ["Logger", "Helpers", "ExceptionHandler"]

# Optional: Initialize global settings or dependencies
def initialize_utils():
    """
    Initializes utility modules with global configurations or pre-checks.
    Ensures that the log directory exists, logger is tested, and utilities are ready for use.
    """
    try:
        # Initialize logger
        logger = Logger.get_logger("UtilsInitializer")
        logger.info("Initializing utility modules...")

        # Verify and create logs directory
        if not os.path.exists(Logger.LOG_DIR):
            os.makedirs(Logger.LOG_DIR)
            logger.info("Created logs directory at %s", Logger.LOG_DIR)
        else:
            logger.info("Log directory verified: %s", Logger.LOG_DIR)

        # Test logger functionality
        Logger.test_logger()

        # Finalize initialization
        logger.info("Utility modules initialized successfully.")
    except Exception as e:
        # Log initialization failure and raise for visibility
        logger = Logger.get_logger("UtilsInitializerFallback")
        logger.error("Failed to initialize utility modules: %s", e, exc_info=True)
        raise RuntimeError(f"Utils package initialization failed: {e}")

# Perform initialization when the package is imported
initialize_utils()
