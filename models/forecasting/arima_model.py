import logging
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from utils.exception_handler import log_exception
from models.config.ml_settings import ARIMA_SETTINGS

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ARIMAModel:
    def __init__(self):
        """
        Initialize the ARIMA model using configuration settings.
        """
        try:
            # Load configuration parameters
            self.order = ARIMA_SETTINGS["order"]
            self.seasonal_order = ARIMA_SETTINGS.get("seasonal_order", (0, 0, 0, 0))
            self.trend = ARIMA_SETTINGS.get("trend", None)

            logger.info(f"ARIMA Model initialized with order={self.order}, "
                        f"seasonal_order={self.seasonal_order}, trend={self.trend}.")
        except KeyError as e:
            logger.error(f"Configuration key missing: {e}")
            raise

    def fit(self, time_series):
        """
        Fit the ARIMA model to the provided time series data.
        :param time_series: Time series data (Pandas Series).
        :return: Fitted ARIMA model.
        """
        try:
            self.model = SARIMAX(
                time_series,
                order=self.order,
                seasonal_order=self.seasonal_order,
                trend=self.trend,
                enforce_stationarity=False,
                enforce_invertibility=False
            )
            self.results = self.model.fit(disp=False)
            logger.info("ARIMA model fitting completed.")
            return self.results
        except Exception as e:
            logger.error("ARIMA model fitting failed.")
            log_exception(e)
            raise

    def forecast(self, steps=1):
        """
        Forecast future values using the fitted ARIMA model.
        :param steps: Number of steps to forecast.
        :return: Forecasted values (Pandas Series).
        """
        try:
            forecast = self.results.get_forecast(steps=steps)
            forecast_index = pd.date_range(
                start=self.results.data.dates[-1],
                periods=steps + 1, freq=self.results.data.freq
            )[1:]
            forecast_values = forecast.predicted_mean
            logger.info(f"Forecast for {steps} steps completed.")
            return pd.Series(forecast_values, index=forecast_index)
        except Exception as e:
            logger.error("Forecasting failed.")
            log_exception(e)
            raise

    def evaluate(self, test_series):
        """
        Evaluate the ARIMA model using a test dataset.
        :param test_series: Test dataset (Pandas Series).
        :return: Mean Squared Error (MSE) of the forecast.
        """
        try:
            forecast = self.forecast(steps=len(test_series))
            mse = ((test_series - forecast) ** 2).mean()
            logger.info(f"Model evaluation completed with MSE={mse:.4f}.")
            return mse
        except Exception as e:
            logger.error("Model evaluation failed.")
            log_exception(e)
            raise

    def save_model(self, path):
        """
        Save the ARIMA model to disk.
        :param path: Path to save the model.
        """
        try:
            self.results.save(path)
            logger.info(f"ARIMA model saved to {path}.")
        except Exception as e:
            logger.error("Failed to save ARIMA model.")
            log_exception(e)
            raise

    def load_model(self, path):
        """
        Load a previously saved ARIMA model from disk.
        :param path: Path to the saved model.
        """
        try:
            from statsmodels.tsa.statespace.sarimax import SARIMAXResults
            self.results = SARIMAXResults.load(path)
            logger.info(f"ARIMA model loaded from {path}.")
        except Exception as e:
            logger.error("Failed to load ARIMA model.")
            log_exception(e)
            raise
