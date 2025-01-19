"""
Unit Tests for MetaTrader 5 API Configuration.
"""

import unittest
from unittest.mock import patch, MagicMock
from config.api_config import (
    initialize_mt5,
    shutdown_mt5,
    is_connected,
    reconnect_mt5,
    get_account_info,
    log_account_info,
)
import MetaTrader5 as mt5


class TestMetaTrader5API(unittest.TestCase):
    @patch("config.api_config.mt5.initialize")
    def test_initialize_mt5_success(self, mock_initialize):
        mock_initialize.return_value = True
        with patch("config.api_config.mt5.login") as mock_login:
            mock_login.return_value = True
            self.assertTrue(initialize_mt5())
            mock_initialize.assert_called_once()
            mock_login.assert_called_once()

    @patch("config.api_config.mt5.initialize")
    def test_initialize_mt5_failure(self, mock_initialize):
        mock_initialize.return_value = False
        self.assertFalse(initialize_mt5())
        mock_initialize.assert_called_once()

    @patch("config.api_config.mt5.shutdown")
    def test_shutdown_mt5(self, mock_shutdown):
        shutdown_mt5()
        mock_shutdown.assert_called_once()

    @patch("config.api_config.mt5.connection_state")
    def test_is_connected_true(self, mock_connection_state):
        mock_connection_state.return_value = mt5.CONNECTION_STATUS_CONNECTED
        self.assertTrue(is_connected())
        mock_connection_state.assert_called_once()

    @patch("config.api_config.mt5.connection_state")
    def test_is_connected_false(self, mock_connection_state):
        mock_connection_state.return_value = mt5.CONNECTION_STATUS_DISCONNECTED
        self.assertFalse(is_connected())
        mock_connection_state.assert_called_once()

    @patch("config.api_config.initialize_mt5")
    @patch("config.api_config.shutdown_mt5")
    def test_reconnect_mt5_success(self, mock_shutdown, mock_initialize):
        mock_initialize.side_effect = [False, True]
        self.assertTrue(reconnect_mt5())
        self.assertEqual(mock_initialize.call_count, 2)
        self.assertEqual(mock_shutdown.call_count, 1)

    @patch("config.api_config.initialize_mt5")
    @patch("config.api_config.shutdown_mt5")
    def test_reconnect_mt5_failure(self, mock_shutdown, mock_initialize):
        mock_initialize.return_value = False
        self.assertFalse(reconnect_mt5())
        self.assertEqual(mock_initialize.call_count, 3)
        self.assertEqual(mock_shutdown.call_count, 3)

    @patch("config.api_config.mt5.account_info")
    def test_get_account_info_success(self, mock_account_info):
        mock_account_info.return_value = MagicMock(_asdict=lambda: {"balance": 1000})
        result = get_account_info()
        self.assertIsNotNone(result)
        self.assertIn("balance", result)
        self.assertEqual(result["balance"], 1000)
        mock_account_info.assert_called_once()

    @patch("config.api_config.mt5.account_info")
    def test_get_account_info_failure(self, mock_account_info):
        mock_account_info.return_value = None
        result = get_account_info()
        self.assertIsNone(result)
        mock_account_info.assert_called_once()

    @patch("config.api_config.get_account_info")
    def test_log_account_info(self, mock_get_account_info):
        mock_get_account_info.return_value = {"balance": 1000, "equity": 950}
        with self.assertLogs("MT5_API", level="INFO") as log:
            log_account_info()
            self.assertIn("INFO:MT5_API:Account Details:", log.output[0])
            self.assertIn("INFO:MT5_API:balance: 1000", log.output[1])
            self.assertIn("INFO:MT5_API:equity: 950", log.output[2])


if __name__ == "__main__":
    unittest.main()
