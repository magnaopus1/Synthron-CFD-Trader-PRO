"""
Unit Tests for Reporting module.
"""

import unittest
import pandas as pd
from performance.reporting import Reporting
from unittest.mock import patch


class TestReporting(unittest.TestCase):

    def setUp(self):
        """Set up sample trades and balance history for testing."""
        self.trades = pd.DataFrame({
            "timestamp": pd.date_range(start="2023-01-01", periods=10, freq="D"),
            "signal": [1, -1, 1, -1, 1, -1, 1, -1, 1, -1],
            "price": [1.1, 1.2, 1.3, 1.4, 1.3, 1.5, 1.4, 1.6, 1.5, 1.7],
            "balance": [1000, 1010, 1020, 1015, 1030, 1020, 1040, 1035, 1050, 1060],
            "position": [1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
        })

        self.balance_history = pd.DataFrame({
            "timestamp": pd.date_range(start="2023-01-01", periods=10, freq="D"),
            "balance": [1000, 1010, 1020, 1015, 1030, 1020, 1040, 1035, 1050, 1060],
            "position": [1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
        })

        self.reporting = Reporting(
            trades=self.trades,
            balance_history=self.balance_history,
            log_file="test_trading_report.log"
        )

    def test_log_event(self):
        """Test logging significant events."""
        with patch("logging.Logger.log") as mock_log:
            self.reporting.log_event("Test event", level="info")
            mock_log.assert_called_once()

    def test_generate_trade_summary(self):
        """Test trade summary generation."""
        summary = self.reporting.generate_trade_summary()
        self.assertIsInstance(summary, pd.DataFrame)
        self.assertIn("Metric", summary.columns)
        self.assertIn("Value", summary.columns)

    def test_generate_performance_table(self):
        """Test performance table generation."""
        performance_tables = self.reporting.generate_performance_table()
        self.assertIn("daily", performance_tables)
        self.assertIn("weekly", performance_tables)
        self.assertIn("monthly", performance_tables)

        for key in ["daily", "weekly", "monthly"]:
            self.assertIsInstance(performance_tables[key], pd.DataFrame)
            self.assertIn("Balance", performance_tables[key].columns)
            self.assertIn(f"{key.capitalize()} Change (%)", performance_tables[key].columns)

    @patch("pandas.ExcelWriter")
    def test_export_report(self, mock_excel_writer):
        """Test exporting report to Excel."""
        try:
            self.reporting.export_report(file_path="test_report.xlsx")
            mock_excel_writer.assert_called_once_with("test_report.xlsx", engine="xlsxwriter")
        except Exception as e:
            self.fail(f"export_report raised an exception: {e}")

    def test_display_trade_summary(self):
        """Test displaying trade summary."""
        try:
            self.reporting.display_trade_summary()
        except Exception as e:
            self.fail(f"display_trade_summary raised an exception: {e}")

    def test_display_historical_performance(self):
        """Test displaying historical performance."""
        try:
            self.reporting.display_historical_performance()
        except Exception as e:
            self.fail(f"display_historical_performance raised an exception: {e}")


if __name__ == "__main__":
    unittest.main()
