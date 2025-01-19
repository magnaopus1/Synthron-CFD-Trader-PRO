"""
Unit Tests for EntryExitStrategy module.
"""

import unittest
import pandas as pd
from strategies.entry_exit import EntryExitStrategy
from data.indicators import Indicators
from unittest.mock import patch


class TestEntryExitStrategy(unittest.TestCase):

    def setUp(self):
        """Set up sample data and strategy instance for testing."""
        self.data = pd.Series([100, 102, 104, 103, 101, 105, 107, 109, 108, 110])
        self.data_pair_1 = pd.Series([100, 101, 102, 103, 104, 105, 106, 107, 108, 109])
        self.data_pair_2 = pd.Series([100, 100.5, 101, 101.5, 102, 102.5, 103, 103.5, 104, 104.5])
        self.strategy = EntryExitStrategy(max_exposure_per_asset=0.02, sharpe_ratio_target=2)

    @patch("data.indicators.Indicators.moving_average")
    def test_trend_following_buy(self, mock_moving_average):
        mock_moving_average.side_effect = [pd.Series([102] * len(self.data)), pd.Series([100] * len(self.data))]
        signal = self.strategy.trend_following(self.data)
        self.assertEqual(signal, "BUY")

    @patch("data.indicators.Indicators.moving_average")
    def test_trend_following_sell(self, mock_moving_average):
        mock_moving_average.side_effect = [pd.Series([98] * len(self.data)), pd.Series([100] * len(self.data))]
        signal = self.strategy.trend_following(self.data)
        self.assertEqual(signal, "SELL")

    def test_mean_reversion_buy(self):
        with patch("data.indicators.Indicators.z_score", return_value=pd.Series([-2.5] * len(self.data))), \
             patch("data.indicators.Indicators.relative_strength_index", return_value=pd.Series([25] * len(self.data))):
            signal = self.strategy.mean_reversion(self.data)
            self.assertEqual(signal, "BUY")

    def test_mean_reversion_sell(self):
        with patch("data.indicators.Indicators.z_score", return_value=pd.Series([2.5] * len(self.data))), \
             patch("data.indicators.Indicators.relative_strength_index", return_value=pd.Series([75] * len(self.data))):
            signal = self.strategy.mean_reversion(self.data)
            self.assertEqual(signal, "SELL")

    @patch("data.indicators.Indicators.bollinger_bands")
    @patch("data.indicators.Indicators.exponential_moving_average")
    def test_breakout_buy(self, mock_ema, mock_bands):
        mock_ema.return_value = pd.Series([108] * len(self.data))
        mock_bands.return_value = pd.DataFrame({
            "Upper Band": [107] * len(self.data),
            "Lower Band": [95] * len(self.data),
        })
        signal = self.strategy.breakout_strategy(self.data)
        self.assertEqual(signal, "BUY")

    @patch("data.indicators.Indicators.bollinger_bands")
    @patch("data.indicators.Indicators.exponential_moving_average")
    def test_breakout_sell(self, mock_ema, mock_bands):
        mock_ema.return_value = pd.Series([98] * len(self.data))
        mock_bands.return_value = pd.DataFrame({
            "Upper Band": [120] * len(self.data),
            "Lower Band": [99] * len(self.data),
        })
        signal = self.strategy.breakout_strategy(self.data)
        self.assertEqual(signal, "SELL")

    def test_momentum_buy(self):
        with patch("data.indicators.Indicators.relative_strength_index", return_value=pd.Series([25] * len(self.data))), \
             patch("data.indicators.Indicators.z_score", return_value=pd.Series([-2.5] * len(self.data))):
            signal = self.strategy.momentum_strategy(self.data)
            self.assertEqual(signal, "BUY")

    def test_momentum_sell(self):
        with patch("data.indicators.Indicators.relative_strength_index", return_value=pd.Series([75] * len(self.data))), \
             patch("data.indicators.Indicators.z_score", return_value=pd.Series([2.5] * len(self.data))):
            signal = self.strategy.momentum_strategy(self.data)
            self.assertEqual(signal, "SELL")

    def test_cointegration_buy(self):
        with patch("data.indicators.Indicators.cointegration", return_value=0.01), \
             patch("data.indicators.Indicators.z_score", return_value=pd.Series([-2.5] * len(self.data))):
            signal = self.strategy.cointegration_strategy(self.data_pair_1, self.data_pair_2)
            self.assertEqual(signal, "BUY")

    def test_cointegration_sell(self):
        with patch("data.indicators.Indicators.cointegration", return_value=0.01), \
             patch("data.indicators.Indicators.z_score", return_value=pd.Series([2.5] * len(self.data))):
            signal = self.strategy.cointegration_strategy(self.data_pair_1, self.data_pair_2)
            self.assertEqual(signal, "SELL")

    def test_execute_trade_buy(self):
        signal = self.strategy.execute_trade(symbol="EURUSD", signal="BUY", current_exposure=0.01)
        self.assertEqual(signal, "BUY")

    def test_execute_trade_sell(self):
        signal = self.strategy.execute_trade(symbol="EURUSD", signal="SELL", current_exposure=0.02)
        self.assertEqual(signal, "SELL")

    def test_execute_trade_hold(self):
        signal = self.strategy.execute_trade(symbol="EURUSD", signal="BUY", current_exposure=0.02)
        self.assertEqual(signal, "HOLD")


if __name__ == "__main__":
    unittest.main()
