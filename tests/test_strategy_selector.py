"""
Unit Tests for StrategySelector module.
"""

import unittest
from strategies.strategy_selector import StrategySelector
from unittest.mock import MagicMock


class TestStrategySelector(unittest.TestCase):

    def setUp(self):
        """Set up a StrategySelector instance with mock strategies."""
        # Mock strategies
        self.mock_trend_following = MagicMock(return_value="Trend Following Signal")
        self.mock_mean_reversion = MagicMock(return_value="Mean Reversion Signal")
        self.mock_breakout_strategy = MagicMock(return_value="Breakout Signal")
        self.mock_momentum_strategy = MagicMock(return_value="Momentum Signal")
        self.mock_cointegration_strategy = MagicMock(return_value="Cointegration Signal")

        # Dictionary of strategies
        strategies = {
            "trend_following": self.mock_trend_following,
            "mean_reversion": self.mock_mean_reversion,
            "breakout_strategy": self.mock_breakout_strategy,
            "momentum_strategy": self.mock_momentum_strategy,
            "cointegration_strategy": self.mock_cointegration_strategy,
        }

        # Create StrategySelector instance
        self.selector = StrategySelector(strategies)

    def test_select_strategy_trend(self):
        """Test selecting strategies based on market condition 'trend'."""
        selected_strategies = self.selector.select_strategy(market_condition="trend")
        self.assertIn("trend_following", selected_strategies)
        self.assertIn("scalping", selected_strategies)

    def test_select_strategy_range(self):
        """Test selecting strategies based on market condition 'range'."""
        selected_strategies = self.selector.select_strategy(market_condition="range")
        self.assertIn("mean_reversion", selected_strategies)
        self.assertIn("scalping", selected_strategies)

    def test_select_strategy_volatility(self):
        """Test selecting strategies based on market condition 'volatility'."""
        selected_strategies = self.selector.select_strategy(market_condition="volatility")
        self.assertIn("breakout_strategy", selected_strategies)
        self.assertIn("momentum_strategy", selected_strategies)

    def test_select_strategy_pairwise(self):
        """Test selecting pairwise strategies based on market condition 'volatility'."""
        selected_strategies = self.selector.select_strategy(market_condition="volatility", pairwise=True)
        self.assertIn("cointegration_strategy", selected_strategies)

    def test_execute_strategy(self):
        """Test executing a specific strategy."""
        result = self.selector.execute_strategy("trend_following", "EURUSD", [100, 102, 105])
        self.mock_trend_following.assert_called_once_with([100, 102, 105])
        self.assertEqual(result, "Trend Following Signal")

    def test_execute_strategy_invalid(self):
        """Test executing an invalid strategy."""
        result = self.selector.execute_strategy("invalid_strategy", "EURUSD", [100, 102, 105])
        self.assertIsNone(result)

    def test_run_concurrent_strategies(self):
        """Test running multiple strategies concurrently on the same asset."""
        selected_strategies = ["trend_following", "mean_reversion"]
        results = self.selector.run_concurrent_strategies("EURUSD", [100, 102, 105], selected_strategies)
        self.assertEqual(results["trend_following"], "Trend Following Signal")
        self.assertEqual(results["mean_reversion"], "Mean Reversion Signal")

    def test_run_multiple_assets(self):
        """Test running strategies across multiple assets concurrently."""
        assets_data = {
            "EURUSD": [100, 102, 105],
            "GBPUSD": [101, 103, 106],
        }
        market_conditions = {
            "EURUSD": "trend",
            "GBPUSD": "range",
        }
        results = self.selector.run_multiple_assets(assets_data, market_conditions)
        self.assertIn("EURUSD", results)
        self.assertIn("GBPUSD", results)

    def test_run_pairwise_strategy(self):
        """Test running pairwise strategies concurrently."""
        pairwise_data = {
            "EURGBP": ([100, 102, 105], [101, 103, 106]),
        }
        market_conditions = {"EURGBP": "volatility"}
        results = self.selector.run_multiple_assets({}, market_conditions, pairwise_data)
        self.assertIn("EURGBP", results)
        self.assertEqual(results["EURGBP"]["Cointegration"], "Cointegration Signal")


if __name__ == "__main__":
    unittest.main()
