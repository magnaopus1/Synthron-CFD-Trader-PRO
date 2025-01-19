import pandas as pd
from data.indicators import Indicators
from utils.logger import logger
from models.forecasting import ARIMAModel, GRUModel, LSTMModel, TransformerModel
from models.reinforcement_learning import ActorCritic, DQN, PPO


class EntryExitStrategy:
    """
    Defines entry and exit strategies based on indicators, market conditions,
    forecasting models, and reinforcement learning models.
    """

    def __init__(self, max_exposure_per_asset=0.01, sharpe_ratio_target=3):
        """
        Initialize the entry and exit strategy.
        
        :param max_exposure_per_asset: Maximum allowable exposure per asset.
        :param sharpe_ratio_target: Target Sharpe ratio for strategy selection.
        """
        self.max_exposure_per_asset = max_exposure_per_asset
        self.sharpe_ratio_target = sharpe_ratio_target

        # Initialize forecasting models
        self.forecasting_models = {
            "ARIMA": ARIMAModel(),
            "GRU": GRUModel(),
            "LSTM": LSTMModel(),
            "Transformer": TransformerModel(),
        }

        # Initialize reinforcement learning models
        self.rl_models = {
            "ActorCritic": ActorCritic(),
            "DQN": DQN(),
            "PPO": PPO(),
        }

    def forecast_prices(self, data: pd.Series, model_name: str, forecast_steps: int = 5):
        """Generate price forecasts using the specified forecasting model."""
        if model_name not in self.forecasting_models:
            logger.error(f"Invalid model name: {model_name}. Choose from {list(self.forecasting_models.keys())}.")
            raise ValueError(f"Invalid model name: {model_name}")

        model = self.forecasting_models[model_name]
        try:
            logger.info(f"Generating forecasts using {model_name} for {forecast_steps} steps.")
            forecast = model.forecast(data, forecast_steps)
            return pd.Series(forecast, name=f"{model_name}_Forecast")
        except Exception as e:
            logger.error(f"Error generating forecasts with {model_name}: {e}")
            return pd.Series()

    def apply_rl_model(self, state, model_name: str):
        """Apply a reinforcement learning model to predict actions."""
        if model_name not in self.rl_models:
            logger.error(f"Invalid RL model name: {model_name}. Choose from {list(self.rl_models.keys())}.")
            raise ValueError(f"Invalid RL model name: {model_name}")

        model = self.rl_models[model_name]
        try:
            logger.info(f"Applying RL model {model_name}.")
            action = model.predict(state)
            return action
        except Exception as e:
            logger.error(f"Error applying RL model {model_name}: {e}")
            return "HOLD"

    def trend_following(self, data: pd.Series, model_name: str, rl_model: str, period: int = 50, confirmation_period: int = 200):
        """Trend-following strategy using moving averages, forecasting, and RL."""
        sma_short = Indicators.moving_average(data, period)
        sma_long = Indicators.moving_average(data, confirmation_period)
        current_price = data.iloc[-1]
        forecast = self.forecast_prices(data, model_name)

        state = {
            "current_price": current_price,
            "sma_short": sma_short.iloc[-1],
            "sma_long": sma_long.iloc[-1],
        }
        rl_action = self.apply_rl_model(state, rl_model)

        if rl_action == "BUY" and current_price > sma_short.iloc[-1] and sma_short.iloc[-1] > sma_long.iloc[-1]:
            return "BUY"
        elif rl_action == "SELL" and current_price < sma_short.iloc[-1] and sma_short.iloc[-1] < sma_long.iloc[-1]:
            return "SELL"
        return "HOLD"

    def mean_reversion(self, data: pd.Series, model_name: str, rl_model: str, z_window: int = 20, confirmation_rsi: int = 14):
        """Mean reversion strategy using Z-score, RSI, forecasting, and RL."""
        z_score = Indicators.z_score(data, z_window)
        rsi = Indicators.relative_strength_index(data, confirmation_rsi)
        forecast = self.forecast_prices(data, model_name)

        state = {
            "z_score": z_score.iloc[-1],
            "rsi": rsi.iloc[-1],
        }
        rl_action = self.apply_rl_model(state, rl_model)

        if rl_action == "BUY" and z_score.iloc[-1] < -2 and rsi.iloc[-1] < 30:
            return "BUY"
        elif rl_action == "SELL" and z_score.iloc[-1] > 2 and rsi.iloc[-1] > 70:
            return "SELL"
        return "HOLD"

    def scalping_strategy(self, data: pd.Series, model_name: str, rl_model: str, period_fast: int = 5, period_slow: int = 20, confirmation_rsi: int = 14):
        """Scalping strategy using fast/slow EMAs, RSI, forecasting, and RL."""
        ema_fast = Indicators.exponential_moving_average(data, period_fast)
        ema_slow = Indicators.exponential_moving_average(data, period_slow)
        rsi = Indicators.relative_strength_index(data, confirmation_rsi)
        forecast = self.forecast_prices(data, model_name)

        state = {
            "ema_fast": ema_fast.iloc[-1],
            "ema_slow": ema_slow.iloc[-1],
            "rsi": rsi.iloc[-1],
        }
        rl_action = self.apply_rl_model(state, rl_model)

        if rl_action == "BUY" and ema_fast.iloc[-1] > ema_slow.iloc[-1] and 30 < rsi.iloc[-1] < 70:
            return "BUY"
        elif rl_action == "SELL" and ema_fast.iloc[-1] < ema_slow.iloc[-1] and 30 < rsi.iloc[-1] < 70:
            return "SELL"
        return "HOLD"

    def breakout_strategy(self, data: pd.Series, model_name: str, rl_model: str, period: int = 14, confirmation_ema: int = 20):
        """Breakout strategy using Bollinger Bands, EMA, forecasting, and RL."""
        bands = Indicators.bollinger_bands(data, period)
        ema = Indicators.exponential_moving_average(data, confirmation_ema)
        current_price = data.iloc[-1]
        forecast = self.forecast_prices(data, model_name)

        state = {
            "current_price": current_price,
            "upper_band": bands["Upper Band"].iloc[-1],
            "lower_band": bands["Lower Band"].iloc[-1],
            "ema": ema.iloc[-1],
        }
        rl_action = self.apply_rl_model(state, rl_model)

        if rl_action == "BUY" and current_price > bands["Upper Band"].iloc[-1] and current_price > ema.iloc[-1]:
            return "BUY"
        elif rl_action == "SELL" and current_price < bands["Lower Band"].iloc[-1] and current_price < ema.iloc[-1]:
            return "SELL"
        return "HOLD"

    def momentum_strategy(self, data: pd.Series, model_name: str, rl_model: str, period: int = 14, confirmation_z: int = 20):
        """Momentum strategy using RSI, Z-score, forecasting, and RL."""
        rsi = Indicators.relative_strength_index(data, period)
        z_score = Indicators.z_score(data, confirmation_z)
        forecast = self.forecast_prices(data, model_name)

        state = {
            "rsi": rsi.iloc[-1],
            "z_score": z_score.iloc[-1],
        }
        rl_action = self.apply_rl_model(state, rl_model)

        if rl_action == "BUY" and rsi.iloc[-1] < 30 and z_score.iloc[-1] < -2:
            return "BUY"
        elif rl_action == "SELL" and rsi.iloc[-1] > 70 and z_score.iloc[-1] > 2:
            return "SELL"
        return "HOLD"

    def cointegration_strategy(self, series1: pd.Series, series2: pd.Series, model_name: str, rl_model: str, confirmation_z: int = 20):
        """Cointegration strategy using Z-score, forecasting, and RL."""
        p_value = Indicators.cointegration(series1, series2)
        if p_value < 0.05:
            spread = series1 - series2
            z_score = Indicators.z_score(spread, confirmation_z)
            forecast = self.forecast_prices(spread, model_name)

            state = {
                "spread": spread.iloc[-1],
                "z_score": z_score.iloc[-1],
            }
            rl_action = self.apply_rl_model(state, rl_model)

            if rl_action == "BUY" and z_score.iloc[-1] < -2:
                return "BUY"
            elif rl_action == "SELL" and z_score.iloc[-1] > 2:
                return "SELL"
        return "HOLD"

    def combine_strategies(self, data: pd.DataFrame, pairwise_data: dict = None, time_frame="1H", model_name="ARIMA", rl_model="ActorCritic"):
        """
        Combine all strategies (single and pairwise) into one system, incorporating forecasting and RL.
        
        :param data: DataFrame containing price data for single-asset strategies.
        :param pairwise_data: Dictionary containing paired data for cointegration strategies.
        :param time_frame: Time frame for strategy execution.
        :param model_name: Forecasting model to use in the forecasting strategy.
        :param rl_model: RL model to use for action predictions.
        :return: Dictionary of trade signals for each strategy.
        """
        signals = {}

        # Single-asset strategies
        for symbol, series in data.iteritems():
            trend_signal = self.trend_following(series, model_name=model_name, rl_model=rl_model)
            mean_reversion_signal = self.mean_reversion(series, model_name=model_name, rl_model=rl_model)
            breakout_signal = self.breakout_strategy(series, model_name=model_name, rl_model=rl_model)
            momentum_signal = self.momentum_strategy(series, model_name=model_name, rl_model=rl_model)
            scalping_signal = self.scalping_strategy(series, model_name=model_name, rl_model=rl_model)

            signals[symbol] = {
                "TrendFollowing": trend_signal,
                "MeanReversion": mean_reversion_signal,
                "Breakout": breakout_signal,
                "Momentum": momentum_signal,
                "Scalping": scalping_signal,
            }

        # Pairwise strategies
        if pairwise_data:
            for pair_name, (series1, series2) in pairwise_data.items():
                cointegration_signal = self.cointegration_strategy(series1, series2, model_name=model_name, rl_model=rl_model)
                signals[pair_name] = {"Cointegration": cointegration_signal}

        return signals
