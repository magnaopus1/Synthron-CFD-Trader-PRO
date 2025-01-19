import logging
from importlib.util import find_spec

# Configure logging for the data package
def configure_logger(name: str):
    """Configure the logger for the data package."""
    logger = logging.getLogger(name)
    if not logger.handlers:  # Avoid adding multiple handlers
        handler = logging.StreamHandler()
        formatter = logging.Formatter("%(asctime)s [%(levelname)s]: %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger

logger = configure_logger(__name__)
logger.info("Initializing the data package.")

# Check for module dependencies
def check_dependencies(modules):
    """Ensure all required modules are available."""
    missing_modules = [mod for mod in modules if not find_spec(mod)]
    if missing_modules:
        logger.error(f"Missing required modules: {', '.join(missing_modules)}")
        raise ImportError(f"Missing required modules: {', '.join(missing_modules)}")

required_modules = ["pandas", "numpy", "sklearn", "scipy", "statsmodels"]
check_dependencies(required_modules)

# Import core modules
try:
    from .data_loader import DataLoader
    from .data_processing import DataProcessing
    from .indicators import Indicators
    logger.info("Successfully imported core modules: DataLoader, DataProcessing, Indicators.")
except ImportError as e:
    logger.error(f"Error importing core modules: {e}")
    raise

# Define public interface
__all__ = ["DataLoader", "DataProcessing", "Indicators"]
