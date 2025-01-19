"""
Performance Package

The `performance` package provides comprehensive tools to evaluate and report the performance
of the CFD trading system. This includes backtesting, performance metrics calculation, and detailed reporting.

Modules:
- Backtesting: Simulate trading strategies on historical data, accounting for transaction costs.
- Metrics: Calculate key performance indicators (KPIs) like Sharpe Ratio, Maximum Drawdown, Risk of Ruin, and more.
- Reporting: Generate detailed reports, log significant events, and maintain historical performance records.

Features:
- Modular and extensible structure for integration into trading systems.
- Designed for scalability and production-grade reliability.
- Includes robust logging and error handling for real-world applications.
"""

from .backtesting import BacktestingEngine
from .metrics import PerformanceMetrics
from .reporting import Reporting

# Define public API
__all__ = ["BacktestingEngine", "PerformanceMetrics", "Reporting"]

# Logging configuration for the performance package
import logging

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger("performance")
logger.info("Performance package initialized.")
