import signal
import sys
import MetaTrader5 as mt5
import pandas as pd
from datetime import datetime
from config import settings
from data import DataLoader, DataProcessing
from strategies import (
    EntryExitStrategy,
    PositionManagement,
    RiskManagement,
    StrategySelector,
)
from performance import BacktestingEngine, PerformanceMetrics, Reporting
from utils import Logger, ExceptionHandler

# Initialize logging
logger = Logger.get_logger("SYNTHRON_CFD_TRADER")

# Global flag for stopping the bot gracefully
RUNNING = True

def signal_handler(sig, frame):
    """
    Handles termination signals for graceful shutdown.
    """
    global RUNNING
    logger.info("Termination signal received. Stopping the bot...")
    RUNNING = False

def display_logo():
    """
    Displays the ASCII art logo for the system.
    """
    logo = """
    ███████╗██╗   ██╗███╗   ██╗████████╗██╗  ██╗██████╗ ██████╗ ███╗   ██╗
    ██╔════╝██║   ██║████╗  ██║╚══██╔══╝██║  ██║██╔══██╗██╔══██╗████╗  ██║
    █████╗  ██║   ██║██╔██╗ ██║   ██║   ███████║██████╔╝██████╔╝██╔██╗ ██║
    ██╔══╝  ██║   ██║██║╚██╗██║   ██║   ██╔══██║██╔═══╝ ██╔═══╝ ██║╚██╗██║
    ██║     ╚██████╔╝██║ ╚████║   ██║   ██║  ██║██║     ██║     ██║ ╚████║
    ╚═╝      ╚═════╝ ╚═╝  ╚═══╝   ╚═╝   ╚═╝  ╚═╝╚═╝     ╚═╝     ╚═╝  ╚═══╝

        SYNTHRON CFD TRADER BY MAGNA OPUS TECHNOLOGIES
    """
    print("\n" + logo + "\n")
    logger.info("Displayed system logo.")

def initialize_system():
    """
    Initializes the CFD Trading System.
    """
    try:
        # Load and validate configurations
        logger.info("Initializing system configurations...")
        settings.validate_configurations()

        # Authenticate with MetaTrader 5 API
        logger.info("Authenticating with MetaTrader 5 API...")
        data_loader = DataLoader()
        if not data_loader.connect_to_mt5():
            raise ConnectionError("Failed to connect to MetaTrader 5 API.")
        logger.info("Authenticated with MetaTrader 5 API.")

        # Return initialized components
        return data_loader
    except Exception as e:
        ExceptionHandler.log_and_handle_exception(e, "System Initialization")
        sys.exit(1)

def load_data(data_loader):
    """
    Loads and preprocesses data.
    """
    try:
        logger.info("Loading market data...")
        symbols = settings.TRADING_PAIRS.split(",")
        historical_data = {
            symbol: data_loader.fetch_historical_data(
                symbol, mt5.TIMEFRAME_H1, datetime(2024, 1, 1), datetime.now()
            )
            for symbol in symbols
        }

        logger.info("Preprocessing market data...")
        data_processor = DataProcessing()
        processed_data = {
            symbol: data_processor.clean_data(data) for symbol, data in historical_data.items()
        }
        return processed_data
    except Exception as e:
        ExceptionHandler.log_and_handle_exception(e, "Data Loading")
        sys.exit(1)

def run_strategies(processed_data, pairwise_data=None):
    """
    Executes trading strategies for single assets and pairs, and monitors performance.
    """
    try:
        logger.info("Initializing strategies...")

        # Initialize strategy components
        entry_exit = EntryExitStrategy()
        position_manager = PositionManagement()
        risk_manager = RiskManagement(
            account_balance=settings.ACCOUNT_BALANCE,
            leverage=settings.LEVERAGE,
            max_drawdown=settings.MAX_DRAWDOWN,
            risk_per_trade=settings.RISK_PER_TRADE,
            default_lot_size=settings.DEFAULT_LOT_SIZE,
        )
        strategy_selector = StrategySelector(
            strategies={
                "trend_following": entry_exit.trend_following,
                "mean_reversion": entry_exit.mean_reversion,
                "breakout_strategy": entry_exit.breakout_strategy,
                "momentum_strategy": entry_exit.momentum_strategy,
                "scalping_strategy": entry_exit.scalping_strategy,
                "cointegration_strategy": entry_exit.cointegration_strategy,
            }
        )

        logger.info("Executing single-asset strategies...")
        market_conditions = {symbol: "trend" for symbol in processed_data.keys()}  # Default market condition
        time_frames = {symbol: "1h" for symbol in processed_data.keys()}  # Default time frame
        single_results = strategy_selector.run_multiple_assets(
            assets_data=processed_data,
            market_conditions=market_conditions,
            time_frames=time_frames,
        )

        if pairwise_data:
            logger.info("Executing pairwise strategies...")
            pairwise_results = strategy_selector.run_multiple_assets(
                assets_data={},
                market_conditions={},
                pairwise_data=pairwise_data,
                time_frames={"pairwise": "1h"},  # Default time frame for pairwise
            )
            logger.info(f"Pairwise results: {pairwise_results}")

        logger.info(f"Strategy execution results: {single_results}")

    except Exception as e:
        ExceptionHandler.log_and_handle_exception(e, "Strategy Execution")

def run_backtesting(data_loader):
    """
    Performs backtesting on historical data for single assets and pairwise strategies.
    """
    try:
        logger.info("Starting backtesting...")
        symbols = settings.TRADING_PAIRS.split(",")
        pairwise_pairs = settings.PAIRWISE_PAIRS.split(",")
        backtesting_engine = BacktestingEngine(
            historical_data=None, strategy=None, initial_balance=settings.INITIAL_BALANCE
        )
        metrics = PerformanceMetrics()
        reporting = Reporting()

        overall_results = {"single_assets": {}, "pairwise": {}}

        for symbol in symbols:
            data = data_loader.fetch_historical_data(symbol, mt5.TIMEFRAME_H1, datetime(2024, 1, 1), datetime(2024, 12, 31))
            for strategy_name, strategy_func in EntryExitStrategy().get_strategies().items():
                backtesting_engine.data = data
                backtesting_engine.strategy = strategy_func
                trades, balance_history = backtesting_engine.run_backtest()
                metrics.trades = trades
                summary = metrics.calculate_metrics_summary()
                overall_results["single_assets"][f"{symbol}_{strategy_name}"] = summary

        for pair in pairwise_pairs:
            asset1, asset2 = pair.split("_")
            data1 = data_loader.fetch_historical_data(asset1, mt5.TIMEFRAME_H1, datetime(2024, 1, 1), datetime(2024, 12, 31))
            data2 = data_loader.fetch_historical_data(asset2, mt5.TIMEFRAME_H1, datetime(2024, 1, 1), datetime(2024, 12, 31))
            for strategy_name, strategy_func in {"cointegration_strategy": EntryExitStrategy().cointegration_strategy}.items():
                backtesting_engine.data = pd.concat([data1["Close"], data2["Close"]], axis=1)
                backtesting_engine.strategy = lambda row: strategy_func(row[0], row[1])
                trades, balance_history = backtesting_engine.run_backtest()
                metrics.trades = trades
                summary = metrics.calculate_metrics_summary()
                overall_results["pairwise"][f"{pair}_{strategy_name}"] = summary

        logger.info(f"Backtesting completed: {overall_results}")

    except Exception as e:
        ExceptionHandler.log_and_handle_exception(e, "Backtesting")

def display_menu():
    """
    Displays the main menu options.
    """
    print("\n--- SYNTHRON CFD TRADER MENU ---")
    print("1. Start Live Trading")
    print("2. Run Backtesting")
    print("3. View Performance Metrics")
    print("4. Exit")
    print("-------------------------------")

def main():
    """
    Main function to initialize and run the trading bot.
    """
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    display_logo()
    data_loader = initialize_system()
    processed_data = load_data(data_loader)

    while RUNNING:
        display_menu()
        choice = input("Enter your choice: ").strip()
        if choice == "1":
            run_strategies(processed_data)
        elif choice == "2":
            run_backtesting(data_loader)
        elif choice == "4":
            global RUNNING
            RUNNING = False
        else:
            print("Invalid choice.")

if __name__ == "__main__":
    main()
