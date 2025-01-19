"""
Unit Tests for Indicators module.
"""

import unittest
import pandas as pd
import numpy as np
from data.indicators import Indicators


class TestIndicators(unittest.TestCase):

    def setUp(self):
        """Set up sample data for testing."""
        self.data = pd.Series([100, 102, 104, 103, 101, 105, 107, 109, 108, 110])
        self.high = pd.Series([101, 103, 105, 104, 102, 106, 108, 110, 109, 111])
        self.low = pd.Series([99, 101, 103, 102, 100, 104, 106, 108, 107, 109])
        self.volume = pd.Series([1000, 1200, 1100, 1300, 1500, 1400, 1600, 1700, 1800, 1900])

    def test_moving_average(self):
        sma = Indicators.moving_average(self.data, period=3)
        self.assertEqual(len(sma), len(self.data))
        self.assertAlmostEqual(sma.iloc[2], 102)

    def test_exponential_moving_average(self):
        ema = Indicators.exponential_moving_average(self.data, period=3)
        self.assertEqual(len(ema), len(self.data))
        self.assertTrue(ema.iloc[-1] > self.data.iloc[-1])

    def test_bollinger_bands(self):
        bands = Indicators.bollinger_bands(self.data, period=3)
        self.assertIn("MA", bands.columns)
        self.assertIn("Upper Band", bands.columns)
        self.assertIn("Lower Band", bands.columns)

    def test_relative_strength_index(self):
        rsi = Indicators.relative_strength_index(self.data, period=3)
        self.assertEqual(len(rsi), len(self.data))
        self.assertTrue((rsi >= 0).all() and (rsi <= 100).all())

    def test_macd(self):
        macd = Indicators.macd(self.data)
        self.assertIn("MACD", macd.columns)
        self.assertIn("Signal Line", macd.columns)
        self.assertIn("Histogram", macd.columns)

    def test_stochastic(self):
        stochastic = Indicators.stochastic(self.data, self.high, self.low, period=3)
        self.assertIn("%K", stochastic.columns)
        self.assertIn("%D", stochastic.columns)

    def test_z_score(self):
        zscore = Indicators.z_score(self.data, window=3)
        self.assertEqual(len(zscore), len(self.data))
        self.assertAlmostEqual(zscore.mean(), 0, delta=1e-6)

    def test_correlation_matrix(self):
        df = pd.DataFrame({
            "Series1": self.data,
            "Series2": self.data * 1.5,
            "Series3": self.data[::-1]
        })
        corr_matrix = Indicators.correlation_matrix(df)
        self.assertAlmostEqual(corr_matrix.loc["Series1", "Series2"], 1, delta=1e-6)
        self.assertAlmostEqual(corr_matrix.loc["Series1", "Series3"], -1, delta=1e-6)

    def test_cointegration(self):
        series1 = self.data
        series2 = self.data * 1.01
        p_value = Indicators.cointegration(series1, series2)
        self.assertTrue(0 <= p_value <= 1)

    def test_on_balance_volume(self):
        obv = Indicators.on_balance_volume(self.data, self.volume)
        self.assertEqual(len(obv), len(self.data))
        self.assertTrue((obv.diff().iloc[1:] != 0).any())

    def test_invalid_series(self):
        with self.assertRaises(ValueError):
            Indicators.moving_average([100, 102, 104], period=3)

        with self.assertRaises(ValueError):
            Indicators.exponential_moving_average("invalid_series", period=3)

        with self.assertRaises(ValueError):
            Indicators.bollinger_bands({"invalid": "data"}, period=3)

    def test_invalid_periods(self):
        with self.assertRaises(ValueError):
            Indicators.moving_average(self.data, period=-1)

        with self.assertRaises(ValueError):
            Indicators.exponential_moving_average(self.data, period=0)

        with self.assertRaises(ValueError):
            Indicators.bollinger_bands(self.data, period=3, std_dev=0)


if __name__ == "__main__":
    unittest.main()
