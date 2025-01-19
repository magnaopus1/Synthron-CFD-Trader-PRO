"""
Unit Tests for BacktestingEngine.
"""

import unittest
import pandas as pd
from datetime import datetime
from data.backtesting import BacktestingEngine


class TestBacktestingEngine(unittest.TestCase):

    def setUp(self):
        """Set up sample historical data and a dummy strategy for testing."""
        self.historical_data = pd.DataFrame({
            "Open": [1.1, 1.2, 1.3, 1.4],
            "High": [1.15, 1.25, 1.35, 1.45],
            "Low": [1.05, 1.15, 1.25, 1.35],
            "Close": [1.1, 1.2, 1.3, 1.4],
            "Volume": [1000, 1200, 1100, 1300],
        }, index=pd.date_range(start="2023-01-01", periods=4, freq="D"))

        self.initial_balance = 1000

        # Define a simple dummy strategy
        def dummy_strategy(row):
            """Buy 1 unit when price is greater than 1.15, sell 1 unit otherwise."""
            return 1 if row["Close"] > 1.15 else -1

        self.dummy_strategy = dummy_strategy

        self.engine = BacktestingEngine(
            historical_data=self.historical_data,
            strategy=self.dummy_strategy,
            initial_balance=self.initial_balance,
        )

    def test_initialization(self):
        """Test initialization of the backtesting engine."""
        self.assertEqual(self.engine.current_balance, self.initial_balance)
        self.assertEqual(self.engine.current_position, 0)
        self.assertIsInstance(self.engine.data, pd.DataFrame)

    def test_validate_data(self):
        """Test validation of historical data."""
        with self.assertRaises(ValueError):
            BacktestingEngine(historical_data=pd.DataFrame({"Invalid": [1, 2, 3]}),
                              strategy=self.dummy_strategy,
                              initial_balance=self.initial_balance)

    def test_execute_trade_buy(self):
        """Test execution of a buy trade."""
        self.engine._execute_trade(signal=1, price=1.2, timestamp="2023-01-02")
        self.assertEqual(self.engine.current_position, 1)
        self.assertLess(self.engine.current_balance, self.initial_balance)

    def test_execute_trade_sell(self):
        """Test execution of a sell trade."""
        # First, open a buy position
        self.engine._execute_trade(signal=1, price=1.2, timestamp="2023-01-02")
        # Then, sell it
        self.engine._execute_trade(signal=-1, price=1.3, timestamp="2023-01-03")
        self.assertEqual(self.engine.current_position, 0)
        self.assertGreater(self.engine.current_balance, self.initial_balance)

    def test_execute_trade_insufficient_balance(self):
        """Test insufficient balance for buy trade."""
        self.engine.current_balance = 0  # Set balance to 0
        self.engine._execute_trade(signal=1, price=1.2, timestamp="2023-01-02")
        self.assertEqual(self.engine.current_position, 0)
        self.assertEqual(self.engine.current_balance, 0)

    def test_execute_trade_insufficient_position(self):
        """Test insufficient position for sell trade."""
        self.engine._execute_trade(signal=-1, price=1.2, timestamp="2023-01-02")
        self.assertEqual(self.engine.current_position, 0)  # Position should remain unchanged

    def test_run_backtest(self):
        """Test the full backtesting process."""
        trades, balance_history = self.engine.run_backtest()
        self.assertIsInstance(trades, pd.DataFrame)
        self.assertIsInstance(balance_history, pd.DataFrame)
        self.assertFalse(trades.empty)
        self.assertFalse(balance_history.empty)

    def test_load_historical_data_from_csv(self):
        """Test loading historical data from a CSV file."""
        with self.assertRaises(RuntimeError):
            BacktestingEngine.load_historical_data_from_csv("invalid_file_path.csv")

    def test_load_historical_data_from_mt5(self):
        """Test loading historical data from MetaTrader 5."""
        with self.assertRaises(RuntimeError):
            BacktestingEngine.load_historical_data_from_mt5("EURUSD", mt5.TIMEFRAME_H1, datetime(2023, 1, 1), datetime(2023, 1, 2))


if __name__ == "__main__":
    unittest.main()
