from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import numpy as np
import logging
from utils.exception_handler import log_exception
from models.config.ml_settings import KMEANS_SETTINGS
import os
import joblib

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class KMeansModel:
    def __init__(self):
        """
        Initialize the KMeans model for clustering similar trading patterns using configuration settings.
        """
        try:
            # Load configuration parameters
            self.n_clusters = KMEANS_SETTINGS["n_clusters"]
            self.init = KMEANS_SETTINGS.get("init", "k-means++")
            self.max_iter = KMEANS_SETTINGS.get("max_iter", 300)
            self.random_state = KMEANS_SETTINGS.get("random_state", 42)

            # Initialize the KMeans model
            self.model = KMeans(
                n_clusters=self.n_clusters,
                init=self.init,
                max_iter=self.max_iter,
                random_state=self.random_state
            )
            logger.info(f"KMeans model initialized with n_clusters={self.n_clusters}, init={self.init}, "
                        f"max_iter={self.max_iter}, random_state={self.random_state}.")
        except KeyError as e:
            logger.error(f"Configuration key missing: {e}")
            raise

    def fit(self, X):
        """
        Fit the KMeans model to the data.
        :param X: Input feature set.
        """
        try:
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            self.model.fit(X_scaled)
            logger.info("KMeans model fitting completed.")
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
            logger.info("KMeans prediction completed.")
            return labels
        except Exception as e:
            logger.error("Prediction failed.")
            log_exception(e)
            raise

    def inertia(self):
        """
        Return the inertia of the current model.
        :return: Inertia value.
        """
        try:
            inertia_value = self.model.inertia_
            logger.info(f"KMeans inertia value: {inertia_value}")
            return inertia_value
        except Exception as e:
            logger.error("Failed to retrieve inertia value.")
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
