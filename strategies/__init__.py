"""
Strategies package for managing entry, exit, risk management, position scaling, and dynamic strategy selection.

This package integrates with system-wide configurations and provides:
1. Entry and Exit Strategy: Defines rules for entering and exiting trades.
2. Position Management: Handles scaling, trailing stops, and profit-locking mechanisms.
3. Risk Management: Manages stop-loss, take-profit, and position sizing based on account balance and drawdown limits.
4. Strategy Selector: Dynamically assigns and runs strategies based on market conditions for single and pairwise assets.
"""

import logging

from .entry_exit import EntryExitStrategy
from .position_management import PositionManagement
from .risk_management import RiskManagement
from .strategy_selector import StrategySelector

__all__ = [
    "EntryExitStrategy",
    "PositionManagement",
    "RiskManagement",
    "StrategySelector",
]

def validate_environment():
    """
    Validates that required configurations are loaded from settings.
    """
    try:
        import config.settings as settings
        
        # Required critical settings
        required_settings = [
            "ACCOUNT_ID",
            "ACCOUNT_PASSWORD",
            "LEVERAGE",
            "MAX_DRAWDOWN",
            "RISK_PER_TRADE",
            "TRADING_PAIRS",
            "TRADING_START_HOUR",
            "TRADING_END_HOUR",
        ]

        for setting in required_settings:
            if not getattr(settings, setting, None):
                raise ValueError(f"Missing critical setting: {setting}")

        # Log success
        logger = logging.getLogger(__name__)
        logger.info("Environment settings validated successfully.")

    except ImportError:
        raise ImportError("The 'config.settings' module is required but not found.")
    except ValueError as ve:
        raise ve


# Validate the environment at package initialization
validate_environment()
