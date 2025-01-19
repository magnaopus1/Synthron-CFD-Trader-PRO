"""
Forecasting Models Package

This package contains advanced forecasting models designed for time series analysis
and trading signal generation in the Synthron CFD Trader Pro system.

Modules:
    - arima_model: Classical ARIMA model for trend and volatility forecasting.
    - gru_model: GRU-based forecasting model for efficient memory usage.
    - lstm_model: LSTM-based forecasting model for capturing long-term dependencies.
    - transformer_model: Transformer-based model for advanced sequence forecasting.

Usage:
    from models.forecasting import ARIMAModel, GRUModel, LSTMModel, TransformerModel

Each model class includes methods for training, predicting, evaluation, and saving/loading models.
"""

from models.forecasting.arima_model import ARIMAModel
from models.forecasting.gru_model import GRUModel
from models.forecasting.lstm_model import LSTMModel
from models.forecasting.transformer_model import TransformerModel

__all__ = [
    "ARIMAModel",
    "GRUModel",
    "LSTMModel",
    "TransformerModel"
]
