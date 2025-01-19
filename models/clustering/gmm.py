from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
import numpy as np
import logging
from utils.exception_handler import log_exception
from models.config.ml_settings import GMM_SETTINGS
import os
import joblib

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GMMModel:
    def __init__(self):
        """
        Initialize the Gaussian Mixture Model (GMM) for clustering and volatility pattern detection
        using configuration settings.
        """
        try:
            # Load configuration parameters
            self.n_components = GMM_SETTINGS["n_components"]
            self.covariance_type = GMM_SETTINGS.get("covariance_type", "full")
            self.random_state = GMM_SETTINGS.get("random_state", 42)

            # Initialize the GMM model
            self.model = GaussianMixture(
                n_components=self.n_components,
                covariance_type=self.covariance_type,
                random_state=self.random_state
            )
            logger.info(f"GMM model initialized with n_components={self.n_components}, "
                        f"covariance_type={self.covariance_type}, random_state={self.random_state}.")
        except KeyError as e:
            logger.error(f"Configuration key missing: {e}")
            raise

    def fit(self, X):
        """
        Fit the Gaussian Mixture Model.
        :param X: Input feature set.
        """
        try:
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            self.model.fit(X_scaled)
            logger.info("GMM model fitting completed.")
        except Exception as e:
            logger.error("Model fitting failed.")
            log_exception(e)
            raise

    def predict(self, X):
        """
        Predict cluster labels for the input data.
        :param X: Input feature set.
        :return: Predicted cluster labels.
        """
        try:
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            labels = self.model.predict(X_scaled)
            logger.info("GMM prediction completed.")
            return labels
        except Exception as e:
            logger.error("Prediction failed.")
            log_exception(e)
            raise

    def predict_proba(self, X):
        """
        Predict probabilities for each cluster.
        :param X: Input feature set.
        :return: Probabilities for each cluster.
        """
        try:
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            probabilities = self.model.predict_proba(X_scaled)
            logger.info("GMM probability prediction completed.")
            return probabilities
        except Exception as e:
            logger.error("Probability prediction failed.")
            log_exception(e)
            raise

    def save_model(self, path):
        """
        Save the trained model to disk.
        :param path: Path to save the model (file path).
        """
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            joblib.dump(self.model, path)
            logger.info(f"Model saved to {path}.")
        except Exception as e:
            logger.error("Failed to save model.")
            log_exception(e)
            raise

    def load_model(self, path):
        """
        Load a trained model from disk.
        :param path: Path to the saved model (file path).
        """
        try:
            self.model = joblib.load(path)
            logger.info(f"Model loaded from {path}.")
        except Exception as e:
            logger.error("Failed to load model.")
            log_exception(e)
            raise
