"""
Unit Tests for DataProcessing module.
"""

import unittest
import pandas as pd
import numpy as np
from data.data_processing import DataProcessing


class TestDataProcessing(unittest.TestCase):

    def setUp(self):
        """Set up sample data for testing."""
        self.data = pd.DataFrame({
            "Open": [100, 101, 102, np.nan, 104],
            "High": [101, 102, 103, 104, 105],
            "Low": [99, 100, 101, 102, 103],
            "Close": [100, 101, 102, 103, 104],
            "Volume": [1000, 1100, 1050, 1200, 1300]
        })

    def test_clean_data_ffill(self):
        cleaned_data = DataProcessing.clean_data(self.data, fill_method="ffill", handle_outliers=False)
        self.assertFalse(cleaned_data.isnull().any().any())

    def test_clean_data_bfill(self):
        cleaned_data = DataProcessing.clean_data(self.data, fill_method="bfill", handle_outliers=False)
        self.assertFalse(cleaned_data.isnull().any().any())

    def test_clean_data_mean(self):
        cleaned_data = DataProcessing.clean_data(self.data, fill_method="mean", handle_outliers=False)
        self.assertFalse(cleaned_data.isnull().any().any())
        self.assertAlmostEqual(cleaned_data.loc[3, "Open"], 101.75)

    def test_clean_data_invalid_fill_method(self):
        with self.assertRaises(ValueError):
            DataProcessing.clean_data(self.data, fill_method="invalid", handle_outliers=False)

    def test_normalize_data_minmax(self):
        normalized_data = DataProcessing.normalize_data(self.data.dropna(), method="minmax")
        self.assertTrue((normalized_data.min().min() == 0) and (normalized_data.max().max() == 1))

    def test_normalize_data_zscore(self):
        normalized_data = DataProcessing.normalize_data(self.data.dropna(), method="zscore")
        self.assertAlmostEqual(normalized_data.mean().mean(), 0, places=6)
        self.assertAlmostEqual(normalized_data.std().mean(), 1, places=6)

    def test_normalize_data_invalid_method(self):
        with self.assertRaises(ValueError):
            DataProcessing.normalize_data(self.data.dropna(), method="invalid")

    def test_add_technical_indicators(self):
        indicators = {
            "SMA_2": lambda x: x.rolling(window=2).mean(),
            "EMA_2": lambda x: x.ewm(span=2, adjust=False).mean(),
        }
        data_with_indicators = DataProcessing.add_technical_indicators(self.data, price_column="Close", indicators=indicators)
        self.assertIn("SMA_2", data_with_indicators.columns)
        self.assertIn("EMA_2", data_with_indicators.columns)

    def test_add_technical_indicators_invalid_column(self):
        with self.assertRaises(ValueError):
            DataProcessing.add_technical_indicators(self.data, price_column="Invalid", indicators={})

    def test_add_custom_features(self):
        data_with_features = DataProcessing.add_custom_features(self.data)
        self.assertIn("Log Return", data_with_features.columns)
        self.assertIn("Skewness", data_with_features.columns)
        self.assertIn("Kurtosis", data_with_features.columns)
        self.assertIn("Volume Change", data_with_features.columns)

    def test_preprocess_data(self):
        X, y = DataProcessing.preprocess_data(self.data, target_column="Close")
        self.assertIsInstance(X, pd.DataFrame)
        self.assertIsInstance(y, pd.Series)
        self.assertNotIn("Close", X.columns)

    def test_preprocess_data_invalid_target_column(self):
        with self.assertRaises(ValueError):
            DataProcessing.preprocess_data(self.data, target_column="Invalid")

    def test_generate_lag_features(self):
        lagged_data = DataProcessing.generate_lag_features(self.data, columns=["Close"], lags=2)
        self.assertIn("Close_lag1", lagged_data.columns)
        self.assertIn("Close_lag2", lagged_data.columns)

    def test_generate_lag_features_invalid_column(self):
        with self.assertRaises(ValueError):
            DataProcessing.generate_lag_features(self.data, columns=["Invalid"], lags=2)


if __name__ == "__main__":
    unittest.main()
