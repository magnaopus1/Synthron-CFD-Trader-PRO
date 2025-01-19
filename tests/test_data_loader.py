"""
Unit Tests for DataLoader module.
"""

import unittest
from unittest.mock import patch, MagicMock
from datetime import datetime
import MetaTrader5 as mt5
from data.data_loader import DataLoader


class TestDataLoader(unittest.TestCase):
    def setUp(self):
        # Initialize DataLoader with mocked MT5
        self.data_loader = DataLoader(max_workers=3)

    @patch("data.data_loader.mt5.initialize")
    def test_initialize_success(self, mock_initialize):
        mock_initialize.return_value = True
        data_loader = DataLoader(max_workers=3)
        self.assertIsInstance(data_loader, DataLoader)
        mock_initialize.assert_called_once()

    @patch("data.data_loader.mt5.initialize")
    def test_initialize_failure(self, mock_initialize):
        mock_initialize.return_value = False
        with self.assertRaises(RuntimeError):
            DataLoader(max_workers=3)
        mock_initialize.assert_called_once()

    @patch("data.data_loader.mt5.symbol_info")
    def test_fetch_symbol_info_success(self, mock_symbol_info):
        mock_symbol_info.return_value = MagicMock(_asdict=lambda: {"name": "EURUSD", "ask": 1.2345})
        result = self.data_loader.fetch_symbol_info("EURUSD")
        self.assertEqual(result["name"], "EURUSD")
        self.assertEqual(result["ask"], 1.2345)
        mock_symbol_info.assert_called_once_with("EURUSD")

    @patch("data.data_loader.mt5.symbol_info")
    def test_fetch_symbol_info_failure(self, mock_symbol_info):
        mock_symbol_info.return_value = None
        result = self.data_loader.fetch_symbol_info("INVALID")
        self.assertIsNone(result)
        mock_symbol_info.assert_called_once_with("INVALID")

    @patch("data.data_loader.mt5.symbol_info_tick")
    def test_fetch_symbol_tick_success(self, mock_symbol_tick):
        mock_symbol_tick.return_value = MagicMock(_asdict=lambda: {"bid": 1.1234, "ask": 1.1236})
        result = self.data_loader.fetch_symbol_tick("EURUSD")
        self.assertEqual(result["bid"], 1.1234)
        self.assertEqual(result["ask"], 1.1236)
        mock_symbol_tick.assert_called_once_with("EURUSD")

    @patch("data.data_loader.mt5.symbol_info_tick")
    def test_fetch_symbol_tick_failure(self, mock_symbol_tick):
        mock_symbol_tick.return_value = None
        result = self.data_loader.fetch_symbol_tick("INVALID")
        self.assertIsNone(result)
        mock_symbol_tick.assert_called_once_with("INVALID")

    @patch("data.data_loader.mt5.copy_rates_range")
    def test_fetch_historical_data_success(self, mock_copy_rates_range):
        mock_copy_rates_range.return_value = [
            {"time": 1609459200, "open": 1.1, "high": 1.2, "low": 1.0, "close": 1.15, "tick_volume": 1000}
        ]
        result = self.data_loader.fetch_historical_data(
            "EURUSD", mt5.TIMEFRAME_M1, datetime(2021, 1, 1), datetime(2021, 1, 2)
        )
        self.assertEqual(len(result), 1)
        self.assertIn("time", result.columns)
        self.assertEqual(result.iloc[0]["close"], 1.15)
        mock_copy_rates_range.assert_called_once()

    @patch("data.data_loader.mt5.copy_rates_range")
    def test_fetch_historical_data_failure(self, mock_copy_rates_range):
        mock_copy_rates_range.return_value = None
        result = self.data_loader.fetch_historical_data(
            "EURUSD", mt5.TIMEFRAME_M1, datetime(2021, 1, 2), datetime(2021, 1, 1)
        )
        self.assertIsNone(result)
        mock_copy_rates_range.assert_called_once()

    @patch("data.data_loader.mt5.positions_get")
    def test_fetch_open_positions_success(self, mock_positions_get):
        mock_positions_get.return_value = [
            MagicMock(_asdict=lambda: {"symbol": "EURUSD", "volume": 1.0, "profit": 50})
        ]
        result = self.data_loader.fetch_open_positions()
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["symbol"], "EURUSD")
        mock_positions_get.assert_called_once()

    @patch("data.data_loader.mt5.positions_get")
    def test_fetch_open_positions_failure(self, mock_positions_get):
        mock_positions_get.return_value = None
        result = self.data_loader.fetch_open_positions()
        self.assertEqual(result, [])
        mock_positions_get.assert_called_once()

    @patch("data.data_loader.mt5.order_calc_margin")
    def test_fetch_margin_requirements_success(self, mock_order_calc_margin):
        mock_order_calc_margin.return_value = 100.0
        result = self.data_loader.fetch_margin_requirements("EURUSD", 1.0, mt5.ORDER_BUY)
        self.assertEqual(result, 100.0)
        mock_order_calc_margin.assert_called_once()

    @patch("data.data_loader.mt5.order_calc_margin")
    def test_fetch_margin_requirements_failure(self, mock_order_calc_margin):
        mock_order_calc_margin.return_value = None
        result = self.data_loader.fetch_margin_requirements("EURUSD", 1.0, mt5.ORDER_BUY)
        self.assertIsNone(result)
        mock_order_calc_margin.assert_called_once()

    @patch("data.data_loader.DataLoader.fetch_symbol_tick")
    @patch("data.data_loader.ThreadPoolExecutor")
    def test_fetch_live_data(self, mock_executor, mock_fetch_symbol_tick):
        mock_fetch_symbol_tick.side_effect = [{"symbol": "EURUSD", "ask": 1.2345}, {"symbol": "GBPUSD", "ask": 1.3456}]
        mock_executor.return_value.__enter__.return_value.submit.side_effect = [MagicMock(result=lambda: res) for res in mock_fetch_symbol_tick.side_effect]
        result = self.data_loader.fetch_live_data(["EURUSD", "GBPUSD"])
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]["symbol"], "EURUSD")
        self.assertEqual(result[1]["symbol"], "GBPUSD")

    @patch("data.data_loader.mt5.shutdown")
    def test_shutdown(self, mock_shutdown):
        self.data_loader.shutdown()
        mock_shutdown.assert_called_once()


if __name__ == "__main__":
    unittest.main()
