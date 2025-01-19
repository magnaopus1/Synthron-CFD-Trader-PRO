import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import coint
import logging

# Configure logging
logging.basicConfig(
    format="%(asctime)s [%(levelname)s]: %(message)s",
    level=logging.INFO,
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

class Indicators:
    """Comprehensive technical indicators for trading systems."""

    @staticmethod
    def validate_series(data: pd.Series):
        if not isinstance(data, pd.Series):
            logger.error("Input data must be a pandas Series.")
            raise ValueError("Input data must be a pandas Series.")

    @staticmethod
    def moving_average(data: pd.Series, period: int) -> pd.Series:
        """Calculate Simple Moving Average (SMA)."""
        Indicators.validate_series(data)
        if period <= 0:
            logger.error("Period must be a positive integer.")
            raise ValueError("Period must be positive.")
        return data.rolling(window=period).mean()

    @staticmethod
    def exponential_moving_average(data: pd.Series, period: int) -> pd.Series:
        """Calculate Exponential Moving Average (EMA)."""
        Indicators.validate_series(data)
        if period <= 0:
            logger.error("Period must be a positive integer.")
            raise ValueError("Period must be positive.")
        return data.ewm(span=period, adjust=False).mean()

    @staticmethod
    def bollinger_bands(data: pd.Series, period: int, std_dev: int = 2) -> pd.DataFrame:
        """Calculate Bollinger Bands."""
        Indicators.validate_series(data)
        if period <= 0 or std_dev <= 0:
            logger.error("Period and standard deviation must be positive integers.")
            raise ValueError("Period and standard deviation must be positive.")
        ma = data.rolling(window=period).mean()
        std = data.rolling(window=period).std()
        upper_band = ma + (std_dev * std)
        lower_band = ma - (std_dev * std)
        return pd.DataFrame({'MA': ma, 'Upper Band': upper_band, 'Lower Band': lower_band})

    @staticmethod
    def relative_strength_index(data: pd.Series, period: int = 14) -> pd.Series:
        """Calculate Relative Strength Index (RSI)."""
        Indicators.validate_series(data)
        if period <= 0:
            logger.error("Period must be a positive integer.")
            raise ValueError("Period must be positive.")
        delta = data.diff(1)
        gain = np.maximum(delta, 0)
        loss = -np.minimum(delta, 0)
        avg_gain = pd.Series(gain).rolling(window=period).mean()
        avg_loss = pd.Series(loss).rolling(window=period).mean()
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        rsi = rsi.fillna(50)  # Default RSI value when no data is available
        return pd.Series(rsi, index=data.index)

    @staticmethod
    def macd(data: pd.Series, fast_period: int = 12, slow_period: int = 26, signal_period: int = 9) -> pd.DataFrame:
        """Calculate Moving Average Convergence Divergence (MACD)."""
        Indicators.validate_series(data)
        if fast_period <= 0 or slow_period <= 0 or signal_period <= 0:
            logger.error("Periods must be positive integers.")
            raise ValueError("Periods must be positive.")
        fast_ema = Indicators.exponential_moving_average(data, fast_period)
        slow_ema = Indicators.exponential_moving_average(data, slow_period)
        macd_line = fast_ema - slow_ema
        signal_line = Indicators.exponential_moving_average(macd_line, signal_period)
        histogram = macd_line - signal_line
        return pd.DataFrame({'MACD': macd_line, 'Signal Line': signal_line, 'Histogram': histogram})

    @staticmethod
    def stochastic(data: pd.Series, high: pd.Series, low: pd.Series, period: int = 14) -> pd.DataFrame:
        """Calculate Stochastic Oscillator."""
        Indicators.validate_series(data)
        Indicators.validate_series(high)
        Indicators.validate_series(low)
        if period <= 0:
            logger.error("Period must be a positive integer.")
            raise ValueError("Period must be positive.")
        low_min = low.rolling(window=period).min()
        high_max = high.rolling(window=period).max()
        k = ((data - low_min) / (high_max - low_min)) * 100
        d = k.rolling(window=3).mean()
        return pd.DataFrame({'%K': k, '%D': d})

    @staticmethod
    def z_score(data: pd.Series, window: int = 20) -> pd.Series:
        """Calculate Z-Score."""
        Indicators.validate_series(data)
        if window <= 0:
            logger.error("Window must be a positive integer.")
            raise ValueError("Window must be positive.")
        rolling_mean = data.rolling(window=window).mean()
        rolling_std = data.rolling(window=window).std()
        zscore = (data - rolling_mean) / rolling_std
        return zscore.fillna(0)

    @staticmethod
    def correlation_matrix(data: pd.DataFrame) -> pd.DataFrame:
        """Calculate correlation matrix."""
        if not isinstance(data, pd.DataFrame):
            logger.error("Input data must be a pandas DataFrame.")
            raise ValueError("Input data must be a pandas DataFrame.")
        return data.corr()

    @staticmethod
    def cointegration(series1: pd.Series, series2: pd.Series) -> float:
        """Calculate cointegration between two series."""
        Indicators.validate_series(series1)
        Indicators.validate_series(series2)
        score, p_value, _ = coint(series1, series2)
        return p_value

    @staticmethod
    def on_balance_volume(close: pd.Series, volume: pd.Series) -> pd.Series:
        """Calculate On-Balance Volume (OBV)."""
        Indicators.validate_series(close)
        Indicators.validate_series(volume)
        obv = (np.sign(close.diff()) * volume).fillna(0).cumsum()
        return obv
