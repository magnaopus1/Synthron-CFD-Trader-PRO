"""
Config Package Initialization.

This package provides configurations and utilities for the CFD Trading System,
including settings, API credentials, and environment variable management.

Features:
- Secure environment variable loading with `.env` file support.
- Centralized access to configurations.
- Logging for configuration loading and validation.
"""

import os
import logging
from dotenv import load_dotenv

# Initialize logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("Config")

# Load environment variables from the .env file
dotenv_path = os.path.join(os.path.dirname(__file__), ".env")
if os.path.exists(dotenv_path):
    load_dotenv(dotenv_path)
    logger.info(".env file loaded successfully.")
else:
    logger.warning(".env file not found. Using system environment variables.")

# Import specific configurations
try:
    from config.settings import (
        ACCOUNT_ID,
        ACCOUNT_PASSWORD,
        BROKER_NAME,
        LEVERAGE,
        MAX_DRAWDOWN,
        RISK_PER_TRADE,
        MT5_API_USERNAME,
        MT5_API_PASSWORD,
        ICM_API_KEY,
        TRADING_PAIRS,
        TRADE_THRESHOLD,
        STOP_LOSS_BUFFER,
        TAKE_PROFIT_BUFFER,
        MIN_SPREAD_THRESHOLD,
        MAX_SPREAD_THRESHOLD,
        TRADING_START_HOUR,
        TRADING_END_HOUR,
        LOGGING_LEVEL,
        LOG_FILE,
        ENABLE_NOTIFICATIONS,
        EMAIL_ALERTS,
        SLACK_WEBHOOK_URL,
        DEFAULT_CURRENCY,
        DEFAULT_LOT_SIZE,
        ALLOW_HEDGING,
        MAX_OPEN_TRADES,
    )
except ImportError as e:
    logger.error(f"Error importing settings: {e}")
    raise

# Exported configurations for public use
__all__ = [
    "ACCOUNT_ID",
    "ACCOUNT_PASSWORD",
    "BROKER_NAME",
    "LEVERAGE",
    "MAX_DRAWDOWN",
    "RISK_PER_TRADE",
    "MT5_API_USERNAME",
    "MT5_API_PASSWORD",
    "ICM_API_KEY",
    "TRADING_PAIRS",
    "TRADE_THRESHOLD",
    "STOP_LOSS_BUFFER",
    "TAKE_PROFIT_BUFFER",
    "MIN_SPREAD_THRESHOLD",
    "MAX_SPREAD_THRESHOLD",
    "TRADING_START_HOUR",
    "TRADING_END_HOUR",
    "LOGGING_LEVEL",
    "LOG_FILE",
    "ENABLE_NOTIFICATIONS",
    "EMAIL_ALERTS",
    "SLACK_WEBHOOK_URL",
    "DEFAULT_CURRENCY",
    "DEFAULT_LOT_SIZE",
    "ALLOW_HEDGING",
    "MAX_OPEN_TRADES",
]

# Validate critical configurations
def validate_configurations():
    """
    Validates the configurations to ensure correctness and prevent runtime errors.
    """
    if not ACCOUNT_ID or not ACCOUNT_PASSWORD:
        logger.error("Account ID and password are mandatory. Check your .env file.")
        raise ValueError("ACCOUNT_ID and ACCOUNT_PASSWORD must be provided.")

    if LEVERAGE <= 0:
        logger.error("LEVERAGE must be a positive integer.")
        raise ValueError("LEVERAGE must be a positive integer.")

    if not (0 < MAX_DRAWDOWN <= 1):
        logger.error("MAX_DRAWDOWN must be a percentage between 0 and 1.")
        raise ValueError("MAX_DRAWDOWN must be between 0 and 1.")

    if not TRADING_PAIRS:
        logger.error("TRADING_PAIRS cannot be empty. Specify at least one trading pair.")
        raise ValueError("TRADING_PAIRS cannot be empty.")

    if TRADING_START_HOUR >= TRADING_END_HOUR:
        logger.error("TRADING_START_HOUR must be earlier than TRADING_END_HOUR.")
        raise ValueError("TRADING_START_HOUR must be earlier than TRADING_END_HOUR.")

    logger.info("All configurations validated successfully.")

# Validate configurations at import time
try:
    validate_configurations()
except Exception as e:
    logger.critical(f"Configuration validation failed: {e}")
    raise
