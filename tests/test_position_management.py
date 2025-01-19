"""
Unit Tests for PositionManagement module.
"""

import unittest
from strategies.position_management import PositionManagement


class TestPositionManagement(unittest.TestCase):

    def setUp(self):
        """Set up a PositionManagement instance for testing."""
        self.pm = PositionManagement(trailing_stop_buffer=0.01, scale_in_threshold=0.005, scale_out_threshold=0.01)

    def test_scale_in_conditions_met(self):
        """Test scaling in when conditions are met."""
        new_position = self.pm.scale_in(
            current_price=105, entry_price=100, current_position=1, max_position=5, scale_step=0.1
        )
        self.assertEqual(new_position, 1.5)  # 10% of max_position added

    def test_scale_in_conditions_not_met(self):
        """Test no scaling in when conditions are not met."""
        new_position = self.pm.scale_in(
            current_price=100, entry_price=100, current_position=1, max_position=5, scale_step=0.1
        )
        self.assertEqual(new_position, 1)

    def test_scale_out_conditions_met(self):
        """Test scaling out when conditions are met."""
        new_position = self.pm.scale_out(
            current_price=95, entry_price=100, current_position=5, min_position=2, scale_step=0.1
        )
        self.assertEqual(new_position, 4.5)  # 10% of current_position removed

    def test_scale_out_conditions_not_met(self):
        """Test no scaling out when conditions are not met."""
        new_position = self.pm.scale_out(
            current_price=100, entry_price=100, current_position=5, min_position=2, scale_step=0.1
        )
        self.assertEqual(new_position, 5)

    def test_apply_trailing_stop_long_adjustment(self):
        """Test trailing stop adjustment for long positions."""
        new_stop = self.pm.apply_trailing_stop(
            current_price=110, trailing_stop_price=100, direction="long"
        )
        self.assertEqual(new_stop, 108.9)  # Current price adjusted by buffer

    def test_apply_trailing_stop_short_adjustment(self):
        """Test trailing stop adjustment for short positions."""
        new_stop = self.pm.apply_trailing_stop(
            current_price=90, trailing_stop_price=100, direction="short"
        )
        self.assertEqual(new_stop, 90.9)  # Current price adjusted by buffer

    def test_apply_trailing_stop_no_adjustment(self):
        """Test no adjustment to trailing stop when conditions are not met."""
        new_stop = self.pm.apply_trailing_stop(
            current_price=100, trailing_stop_price=100, direction="long"
        )
        self.assertEqual(new_stop, 100)

    def test_lock_profit_conditions_met(self):
        """Test profit locking when conditions are met."""
        new_position, locked_profit = self.pm.lock_profit(
            current_price=110, entry_price=100, position_size=10, lock_threshold=0.05
        )
        self.assertEqual(new_position, 7.5)  # 25% of position locked
        self.assertEqual(locked_profit, 2.5)

    def test_lock_profit_conditions_not_met(self):
        """Test no profit locking when conditions are not met."""
        new_position, locked_profit = self.pm.lock_profit(
            current_price=102, entry_price=100, position_size=10, lock_threshold=0.05
        )
        self.assertEqual(new_position, 10)
        self.assertEqual(locked_profit, 0)

    def test_partial_closing(self):
        """Test partial closing at multiple profit levels."""
        new_position = self.pm.partial_closing(
            current_price=110, entry_price=100, position_size=10, levels=[0.05, 0.1]
        )
        self.assertEqual(new_position, 8)  # Two partial closes (10% each)

    def test_partial_closing_no_conditions_met(self):
        """Test no partial closing when conditions are not met."""
        new_position = self.pm.partial_closing(
            current_price=102, entry_price=100, position_size=10, levels=[0.05, 0.1]
        )
        self.assertEqual(new_position, 10)


if __name__ == "__main__":
    unittest.main()
