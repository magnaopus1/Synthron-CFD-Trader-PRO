"""
Unit Tests for RiskManagement module.
"""

import unittest
from strategies.risk_management import RiskManagement


class TestRiskManagement(unittest.TestCase):

    def setUp(self):
        """Set up a RiskManagement instance for testing."""
        self.rm = RiskManagement(
            account_balance=10000,
            leverage=100,
            max_drawdown=0.2,
            risk_per_trade=0.01,
            default_lot_size=0.1
        )

    def test_calculate_position_size(self):
        """Test position size calculation."""
        position_size = self.rm.calculate_position_size(stop_loss_pips=50, pip_value=10)
        self.assertEqual(position_size, 2.0)  # Expected size based on risk per trade

    def test_calculate_stop_loss_long(self):
        """Test stop-loss calculation for a long position."""
        stop_loss = self.rm.calculate_stop_loss(entry_price=1.2000, direction="long", stop_loss_buffer=0.01)
        self.assertAlmostEqual(stop_loss, 1.1880, places=4)

    def test_calculate_stop_loss_short(self):
        """Test stop-loss calculation for a short position."""
        stop_loss = self.rm.calculate_stop_loss(entry_price=1.2000, direction="short", stop_loss_buffer=0.01)
        self.assertAlmostEqual(stop_loss, 1.2120, places=4)

    def test_calculate_stop_loss_invalid_direction(self):
        """Test invalid direction in stop-loss calculation."""
        with self.assertRaises(ValueError):
            self.rm.calculate_stop_loss(entry_price=1.2000, direction="invalid", stop_loss_buffer=0.01)

    def test_calculate_take_profit_long(self):
        """Test take-profit calculation for a long position."""
        take_profit = self.rm.calculate_take_profit(entry_price=1.2000, direction="long", take_profit_buffer=0.02)
        self.assertAlmostEqual(take_profit, 1.2240, places=4)

    def test_calculate_take_profit_short(self):
        """Test take-profit calculation for a short position."""
        take_profit = self.rm.calculate_take_profit(entry_price=1.2000, direction="short", take_profit_buffer=0.02)
        self.assertAlmostEqual(take_profit, 1.1760, places=4)

    def test_calculate_take_profit_invalid_direction(self):
        """Test invalid direction in take-profit calculation."""
        with self.assertRaises(ValueError):
            self.rm.calculate_take_profit(entry_price=1.2000, direction="invalid", take_profit_buffer=0.02)

    def test_check_drawdown_limit_within_limit(self):
        """Test drawdown limit check within acceptable limits."""
        result = self.rm.check_drawdown_limit(current_drawdown=0.1)
        self.assertTrue(result)

    def test_check_drawdown_limit_exceeded(self):
        """Test drawdown limit check when exceeded."""
        result = self.rm.check_drawdown_limit(current_drawdown=0.25)
        self.assertFalse(result)

    def test_validate_trade_conditions_valid(self):
        """Test trade condition validation with valid conditions."""
        result = self.rm.validate_trade_conditions(
            spread=1.5,
            min_spread_threshold=1.0,
            max_spread_threshold=2.0,
            current_open_trades=3,
            max_open_trades=5
        )
        self.assertTrue(result)

    def test_validate_trade_conditions_invalid_spread(self):
        """Test trade condition validation with invalid spread."""
        result = self.rm.validate_trade_conditions(
            spread=2.5,
            min_spread_threshold=1.0,
            max_spread_threshold=2.0,
            current_open_trades=3,
            max_open_trades=5
        )
        self.assertFalse(result)

    def test_validate_trade_conditions_exceeded_open_trades(self):
        """Test trade condition validation with exceeded open trades."""
        result = self.rm.validate_trade_conditions(
            spread=1.5,
            min_spread_threshold=1.0,
            max_spread_threshold=2.0,
            current_open_trades=6,
            max_open_trades=5
        )
        self.assertFalse(result)

    def test_apply_stop_loss_take_profit(self):
        """Test applying stop-loss and take-profit levels."""
        levels = self.rm.apply_stop_loss_take_profit(
            entry_price=1.2000,
            direction="long",
            stop_loss_buffer=0.01,
            take_profit_buffer=0.02
        )
        self.assertAlmostEqual(levels["stop_loss"], 1.1880, places=4)
        self.assertAlmostEqual(levels["take_profit"], 1.2240, places=4)


if __name__ == "__main__":
    unittest.main()
