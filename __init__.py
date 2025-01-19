"""
SYNTHRON CFD TRADER by Magna Opus Technologies

This package provides the core components for the SYNTHRON CFD Trading System, including:
- Configuration management
- Data loading and preprocessing
- Strategy execution
- Performance monitoring
- Utility functions

Modules:
- `config`: Manages system configurations and MetaTrader 5 API integration.
- `data`: Handles data fetching, cleaning, and indicator calculations.
- `strategies`: Implements trading strategies, position management, and risk control.
- `performance`: Provides backtesting, performance metrics, and reporting tools.
- `utils`: Contains helper functions, logging, and exception handling.

Author: Magna Opus Technologies
Version: 1.1.0
License: MIT
"""

import logging
import os
import sys
from importlib import import_module

# Configure package-wide logging
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger("SYNTHRON_CFD_TRADER")
logger.info("Initializing SYNTHRON CFD TRADER package...")

# Validate required dependencies
REQUIRED_MODULES = [
    "pandas",
    "numpy",
    "matplotlib",
    "scipy",
    "sklearn",
    "statsmodels",
    "dotenv",
    "MetaTrader5",  # Ensure MT5 API library is installed
]

def validate_dependencies():
    """
    Checks for the availability of required dependencies and reports missing modules.
    """
    missing_modules = []
    for mod in REQUIRED_MODULES:
        try:
            import_module(mod)
        except ImportError:
            missing_modules.append(mod)

    if missing_modules:
        logger.error(f"Missing required dependencies: {', '.join(missing_modules)}")
        raise ImportError(f"Install missing modules: {', '.join(missing_modules)}")

    logger.info("All dependencies validated successfully.")

# Perform dependency validation
try:
    validate_dependencies()
except ImportError as e:
    logger.critical(f"Dependency validation failed: {e}")
    sys.exit(1)

# Exporting key modules for streamlined imports
from config import settings
from data import DataLoader, DataProcessing, Indicators
from strategies import (
    EntryExitStrategy,
    PositionManagement,
    RiskManagement,
    StrategySelector,
)
from performance import BacktestingEngine, PerformanceMetrics, Reporting
from utils import Logger, Helpers, ExceptionHandler
from models import (
    ARIMAModel,
    GRUModel,
    LSTMModel,
    TransformerModel,
    Autoencoder,
    IsolationForestModel,
    OneClassSVMModel,
    BayesianOptimization,
    GeneticAlgorithm,
    ParticleSwarmOptimization,
    SVRModel,
    RandomForestRegressorModel,
    DNNRegressor,
    LogisticRegressionModel,
    GradientBoostingModel,
    DBSCANModel,
    GMMModel,
    KMeansModel,
    ActorCritic,
    DQN,
    PPO,
    VaderAnalyzer,
    BERTSentimentAnalyzer,
)

# Public API for the package
__all__ = [
    "settings",
    "DataLoader",
    "DataProcessing",
    "Indicators",
    "EntryExitStrategy",
    "PositionManagement",
    "RiskManagement",
    "StrategySelector",
    "BacktestingEngine",
    "PerformanceMetrics",
    "Reporting",
    "Logger",
    "Helpers",
    "ExceptionHandler",
    "ARIMAModel",
    "GRUModel",
    "LSTMModel",
    "TransformerModel",
    "Autoencoder",
    "IsolationForestModel",
    "OneClassSVMModel",
    "BayesianOptimization",
    "GeneticAlgorithm",
    "ParticleSwarmOptimization",
    "SVRModel",
    "RandomForestRegressorModel",
    "DNNRegressor",
    "LogisticRegressionModel",
    "GradientBoostingModel",
    "DBSCANModel",
    "GMMModel",
    "KMeansModel",
    "ActorCritic",
    "DQN",
    "PPO",
    "VaderAnalyzer",
    "BERTSentimentAnalyzer",
]

# Initialize key components
def initialize():
    """
    Performs global initialization for the SYNTHRON CFD TRADER package.

    - Validates configurations.
    - Checks for MetaTrader 5 API connection.
    - Ensures log directories are available.
    """
    try:
        logger.info("Performing global initialization...")

        # Validate configurations
        settings.validate_configurations()

        # Verify MetaTrader 5 API connection
        from data import DataLoader
        data_loader = DataLoader()
        if not data_loader.connect_to_mt5():
            raise ConnectionError("MetaTrader 5 API connection failed.")

        # Ensure log directory exists
        log_dir = os.path.join(os.getcwd(), "logs")
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
            logger.info(f"Log directory created at: {log_dir}")

        logger.info("Global initialization completed successfully.")
    except Exception as e:
        logger.critical(f"Global initialization failed: {e}")
        sys.exit(1)

# Perform global initialization at package import
try:
    initialize()
except Exception as e:
    logger.critical(f"Failed to initialize the package: {e}")
    sys.exit(1)

# Log package metadata
logger.info("SYNTHRON CFD TRADER by Magna Opus Technologies")
logger.info("Version: 1.1.0 | License: MIT")
