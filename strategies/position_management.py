import logging
from models.forecasting import ARIMAModel, GRUModel, LSTMModel, TransformerModel

logger = logging.getLogger(__name__)

class PositionManagement:
    """Handles position scaling, trailing stops, and conditional partial position closures with forecasting integration."""

    def __init__(self, trailing_stop_buffer=0.01, scale_in_threshold=0.005, scale_out_threshold=0.01):
        """
        Initialize position management logic with forecasting models.
        
        :param trailing_stop_buffer: Percentage buffer for trailing stop adjustments.
        :param scale_in_threshold: Threshold for scaling into a position.
        :param scale_out_threshold: Threshold for scaling out of a position.
        """
        self.trailing_stop_buffer = trailing_stop_buffer
        self.scale_in_threshold = scale_in_threshold
        self.scale_out_threshold = scale_out_threshold

        # Initialize forecasting models
        self.forecasting_models = {
            "ARIMA": ARIMAModel(),
            "GRU": GRUModel(),
            "LSTM": LSTMModel(),
            "Transformer": TransformerModel(),
        }

    def forecast_prices(self, data, model_name, forecast_steps=5):
        """
        Generate price forecasts using the specified forecasting model.
        
        :param data: Input price series.
        :param model_name: The forecasting model to use ('ARIMA', 'GRU', 'LSTM', 'Transformer').
        :param forecast_steps: Number of steps to forecast into the future.
        :return: Forecasted price value.
        """
        if model_name not in self.forecasting_models:
            logger.error(f"Invalid model name: {model_name}. Choose from {list(self.forecasting_models.keys())}.")
            raise ValueError(f"Invalid model name: {model_name}")

        model = self.forecasting_models[model_name]
        try:
            logger.info(f"Generating forecasts using {model_name} for {forecast_steps} steps.")
            forecast = model.forecast(data, forecast_steps)
            return forecast[-1]  # Return the last forecasted value
        except Exception as e:
            logger.error(f"Error generating forecast with {model_name}: {e}")
            return None

    def scale_in(self, current_price, entry_price, current_position, max_position, scale_step=0.1, time_frame="1H", model_name="ARIMA"):
        """
        Scale into a position using forecasting models for confirmation.
        
        :param current_price: Current market price.
        :param entry_price: Entry price of the position.
        :param current_position: Current size of the position.
        :param max_position: Maximum allowable position size.
        :param scale_step: Incremental scale-in percentage of max_position.
        :param time_frame: Time frame for strategy (e.g., "1m", "5m", "1h").
        :param model_name: Forecasting model to use for scale-in confirmation.
        :return: Adjusted position size.
        """
        valid_timeframes = {
            "1m": 0.5, "5m": 1, "15m": 1.5, "30m": 2, "1H": 3, "4H": 5, "1D": 10
        }

        if time_frame not in valid_timeframes:
            logger.warning(f"Time frame {time_frame} not supported for scale-in. Defaulting to 1H.")
            time_frame = "1H"

        scale_threshold = valid_timeframes[time_frame]
        forecast_price = self.forecast_prices(current_price, model_name)

        if forecast_price is not None and forecast_price > current_price * (1 + scale_threshold / 100):
            logger.info(f"Forecast {model_name}: Price expected to rise, scaling in.")
            if current_position < max_position:
                scale_amount = min(scale_step * max_position, max_position - current_position)
                new_position = current_position + scale_amount
                logger.info(f"Scaling in: Added {scale_amount} to position. New position size: {new_position}")
                return new_position
        logger.info("Scale-in conditions not met based on forecast.")
        return current_position

    def scale_out(self, current_price, entry_price, current_position, min_position, scale_step=0.1, time_frame="1H", model_name="ARIMA"):
        """
        Scale out of a position using forecasting models for confirmation.
        
        :param current_price: Current market price.
        :param entry_price: Entry price of the position.
        :param current_position: Current size of the position.
        :param min_position: Minimum allowable position size.
        :param scale_step: Incremental scale-out percentage of current position.
        :param time_frame: Time frame for strategy (e.g., "1m", "5m", "1h").
        :param model_name: Forecasting model to use for scale-out confirmation.
        :return: Adjusted position size.
        """
        valid_timeframes = {
            "1m": 0.5, "5m": 1, "15m": 1.5, "30m": 2, "1H": 3, "4H": 5, "1D": 10
        }

        if time_frame not in valid_timeframes:
            logger.warning(f"Time frame {time_frame} not supported for scale-out. Defaulting to 1H.")
            time_frame = "1H"

        scale_threshold = valid_timeframes[time_frame]
        forecast_price = self.forecast_prices(current_price, model_name)

        if forecast_price is not None and forecast_price < current_price * (1 - scale_threshold / 100):
            logger.info(f"Forecast {model_name}: Price expected to drop, scaling out.")
            if current_position > min_position:
                scale_amount = min(scale_step * current_position, current_position - min_position)
                new_position = current_position - scale_amount
                logger.info(f"Scaling out: Reduced position by {scale_amount}. New position size: {new_position}")
                return new_position
        logger.info("Scale-out conditions not met based on forecast.")
        return current_position

    def apply_trailing_stop(self, current_price, trailing_stop_price, direction="long", time_frame="1H", model_name="ARIMA"):
        """
        Adjust trailing stop using forecasting models for confirmation.
        
        :param current_price: Current market price.
        :param trailing_stop_price: Current trailing stop price.
        :param direction: Direction of the position ('long' or 'short').
        :param time_frame: Time frame for strategy (e.g., "1m", "5m", "1h").
        :param model_name: Forecasting model to use for confirmation.
        :return: Updated trailing stop price.
        """
        valid_timeframes = {
            "1m": 0.001, "5m": 0.002, "1H": 0.005, "1D": 0.01
        }

        if time_frame not in valid_timeframes:
            logger.warning(f"Time frame {time_frame} not supported for trailing stop. Defaulting to 1H.")
            time_frame = "1H"

        buffer = valid_timeframes[time_frame]
        forecast_price = self.forecast_prices(current_price, model_name)

        if forecast_price is not None:
            if direction == "long" and forecast_price > current_price:
                new_stop = current_price * (1 - self.trailing_stop_buffer)
                logger.info(f"Trailing stop adjusted for long position to {new_stop}.")
                return new_stop
            elif direction == "short" and forecast_price < current_price:
                new_stop = current_price * (1 + self.trailing_stop_buffer)
                logger.info(f"Trailing stop adjusted for short position to {new_stop}.")
                return new_stop
        logger.info("No trailing stop adjustment needed based on forecast.")
        return trailing_stop_price

    def lock_profit(self, current_price, entry_price, position_size, lock_threshold=0.02, time_frame="1H", model_name="ARIMA"):
        """
        Lock in profit using forecasting models to validate conditions.
        
        :param current_price: Current market price.
        :param entry_price: Entry price of the position.
        :param position_size: Current size of the position.
        :param lock_threshold: Profit percentage threshold for locking in profits.
        :param time_frame: Time frame for strategy (e.g., "1m", "5m", "1h").
        :param model_name: Forecasting model to use for profit-locking confirmation.
        :return: Adjusted position size and profit locked.
        """
        valid_timeframes = {
            "1m": 0.5, "5m": 1, "1H": 2, "1D": 5
        }

        if time_frame not in valid_timeframes:
            logger.warning(f"Time frame {time_frame} not supported for profit locking. Defaulting to 1H.")
            time_frame = "1H"

        lock_level = valid_timeframes[time_frame]
        forecast_price = self.forecast_prices(current_price, model_name)
        profit_percent = (current_price - entry_price) / entry_price if current_price > entry_price else 0

        if forecast_price is not None and profit_percent >= lock_level / 100:
            lock_size = position_size * 0.25  # Lock 25% of the position
            new_position_size = position_size - lock_size
            logger.info(f"Profit locked: Closed {lock_size} of position. Remaining size: {new_position_size}")
            return new_position_size, lock_size
        logger.info("Profit-locking conditions not met based on forecast.")
        return position_size, 0

    def partial_closing(self, current_price, entry_price, position_size, levels=None, time_frame="1H", model_name="ARIMA"):
        """
        Conditionally close partial positions at multiple profit levels using forecasting models.
        
        :param current_price: Current market price.
        :param entry_price: Entry price of the position.
        :param position_size: Current size of the position.
        :param levels: List of profit percentage levels for partial closures.
        :param time_frame: Time frame for strategy (e.g., "1m", "5m", "1h").
        :param model_name: Forecasting model to use for confirmation.
        :return: Adjusted position size.
        """
        valid_timeframes = {
            "1m": [0.02], "5m": [0.02, 0.05], "1H": [0.05, 0.1], "1D": [0.1, 0.2]
        }

        if time_frame not in valid_timeframes:
            logger.warning(f"Time frame {time_frame} not supported for partial closing. Defaulting to 1H.")
            time_frame = "1H"

        levels = valid_timeframes[time_frame] if levels is None else levels
        forecast_price = self.forecast_prices(current_price, model_name)

        for level in levels:
            if current_price >= entry_price * (1 + level) and (forecast_price is None or forecast_price >= current_price):
                partial_close = position_size * 0.1  # Close 10% of the position at each level
                position_size -= partial_close
                logger.info(f"Partial close: Closed {partial_close} of position at profit level {level * 100}%.")

        return position_size
