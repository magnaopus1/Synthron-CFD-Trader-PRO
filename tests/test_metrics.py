"""
Unit Tests for PerformanceMetrics module.
"""

import unittest
import pandas as pd
from performance.metrics import PerformanceMetrics


class TestPerformanceMetrics(unittest.TestCase):

    def setUp(self):
        """Set up sample trades and balance history for testing."""
        self.trades = pd.DataFrame({
            "timestamp": pd.date_range(start="2023-01-01", periods=10, freq="D"),
            "signal": [1, -1, 1, 1, -1, 1, -1, 1, -1, 1],
            "price": [1.1, 1.2, 1.3, 1.4, 1.3, 1.5, 1.4, 1.6, 1.5, 1.7],
            "balance": [1000, 1010, 1020, 1030, 1025, 1040, 1030, 1050, 1045, 1060],
            "position": [1, 0, 1, 2, 1, 2, 1, 2, 1, 2],
        })

        self.balance_history = pd.DataFrame({
            "timestamp": pd.date_range(start="2023-01-01", periods=10, freq="D"),
            "balance": [1000, 1010, 1020, 1030, 1025, 1040, 1030, 1050, 1045, 1060],
            "position": [1, 0, 1, 2, 1, 2, 1, 2, 1, 2],
        })

        self.metrics = PerformanceMetrics(
            trades=self.trades,
            balance_history=self.balance_history,
            risk_free_rate=0.02
        )

    def test_calculate_sharpe_ratio(self):
        """Test Sharpe Ratio calculation."""
        sharpe_ratio = self.metrics.calculate_sharpe_ratio()
        self.assertIsInstance(sharpe_ratio, float)

    def test_calculate_win_ratio(self):
        """Test Win Ratio calculation."""
        win_ratio = self.metrics.calculate_win_ratio()
        self.assertGreaterEqual(win_ratio, 0)
        self.assertLessEqual(win_ratio, 100)

    def test_calculate_risk_of_ruin(self):
        """Test Risk of Ruin calculation."""
        risk_of_ruin = self.metrics.calculate_risk_of_ruin()
        self.assertGreaterEqual(risk_of_ruin, 0)
        self.assertLessEqual(risk_of_ruin, 100)

    def test_calculate_max_drawdown(self):
        """Test Maximum Drawdown calculation."""
        max_drawdown = self.metrics.calculate_max_drawdown()
        self.assertLessEqual(max_drawdown, 0)

    def test_calculate_avg_daily_gain(self):
        """Test Average Daily Gain calculation."""
        avg_daily_gain = self.metrics.calculate_avg_daily_gain()
        self.assertIsInstance(avg_daily_gain, float)

    def test_calculate_avg_weekly_gain(self):
        """Test Average Weekly Gain calculation."""
        avg_weekly_gain = self.metrics.calculate_avg_weekly_gain()
        self.assertIsInstance(avg_weekly_gain, float)

    def test_calculate_metrics_summary(self):
        """Test metrics summary calculation."""
        summary = self.metrics.calculate_metrics_summary()
        self.assertIsInstance(summary, dict)
        self.assertIn("Sharpe Ratio", summary)
        self.assertIn("Win Ratio (%)", summary)
        self.assertIn("Maximum Drawdown (%)", summary)

    def test_display_metrics_table(self):
        """Test metrics table display."""
        metrics_table = self.metrics.display_metrics_table(display_prompt=False)
        self.assertIsInstance(metrics_table, pd.DataFrame)
        self.assertIn("Metric", metrics_table.columns)
        self.assertIn("Value", metrics_table.columns)

    def test_plot_performance(self):
        """Test performance plotting (silently execute)."""
        try:
            self.metrics.plot_performance(display_prompt=False)
        except Exception as e:
            self.fail(f"plot_performance raised an exception: {e}")


if __name__ == "__main__":
    unittest.main()
