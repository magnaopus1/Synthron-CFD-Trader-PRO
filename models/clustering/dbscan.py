from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import numpy as np
import logging
from utils.exception_handler import log_exception
from models.config.ml_settings import DBSCAN_SETTINGS
import os
import joblib

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DBSCANModel:
    def __init__(self):
        """
        Initialize the DBSCAN model for market state anomaly detection using configuration settings.
        """
        try:
            self.epsilon = DBSCAN_SETTINGS["epsilon"]
            self.min_samples = DBSCAN_SETTINGS["min_samples"]

            # Initialize the DBSCAN model
            self.model = DBSCAN(eps=self.epsilon, min_samples=self.min_samples)
            logger.info(f"DBSCAN model initialized with eps={self.epsilon} and min_samples={self.min_samples}.")
        except KeyError as e:
            logger.error(f"Configuration key missing: {e}")
            raise

    def fit_predict(self, X):
        """
        Fit the DBSCAN model and predict cluster labels.
        :param X: Input feature set.
        :return: Cluster labels (outliers marked as -1).
        """
        try:
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            labels = self.model.fit_predict(X_scaled)
            logger.info("DBSCAN fit and prediction completed.")
            return labels
        except Exception as e:
            logger.error("Fit and prediction failed.")
            log_exception(e)
            raise

    def detect_anomalies(self, labels):
        """
        Detect anomalies from cluster labels.
        :param labels: Cluster labels (outliers marked as -1).
        :return: Indices of anomalies in the dataset.
        """
        try:
            anomalies = np.where(labels == -1)[0]
            logger.info(f"Anomalies detected: {len(anomalies)} instances.")
            return anomalies
        except Exception as e:
            logger.error("Anomaly detection failed.")
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
