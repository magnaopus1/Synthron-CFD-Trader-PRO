from sklearn.ensemble import IsolationForest
import numpy as np
import logging
from utils.exception_handler import log_exception
from models.config.ml_settings import ISOLATION_FOREST_SETTINGS

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class IsolationForestModel:
    def __init__(self):
        """
        Initialize the Isolation Forest model for anomaly detection using configuration settings.
        """
        try:
            self.contamination = ISOLATION_FOREST_SETTINGS["contamination"]
            self.n_estimators = ISOLATION_FOREST_SETTINGS["n_estimators"]
            self.max_samples = ISOLATION_FOREST_SETTINGS["max_samples"]
            self.random_state = ISOLATION_FOREST_SETTINGS["random_state"]
            self.max_features = ISOLATION_FOREST_SETTINGS["max_features"]

            self.model = IsolationForest(
                contamination=self.contamination,
                n_estimators=self.n_estimators,
                max_samples=self.max_samples,
                random_state=self.random_state,
                max_features=self.max_features
            )
            logger.info("Isolation Forest model initialized with configuration settings.")
        except KeyError as e:
            logger.error(f"Configuration key missing: {e}")
            raise

    def train(self, X):
        """
        Train the Isolation Forest model.
        :param X: Training data (numpy array or DataFrame).
        """
        try:
            self.model.fit(X)
            logger.info("Isolation Forest training completed.")
        except Exception as e:
            logger.error("Training failed.")
            log_exception(e)
            raise

    def predict(self, X):
        """
        Predict anomalies using the trained model.
        :param X: Input data (numpy array or DataFrame).
        :return: Predictions (-1 for anomalies, 1 for normal instances).
        """
        try:
            predictions = self.model.predict(X)
            logger.info("Prediction completed.")
            return predictions
        except Exception as e:
            logger.error("Prediction failed.")
            log_exception(e)
            raise

    def evaluate(self, X):
        """
        Evaluate the anomaly scores of input data.
        :param X: Input data (numpy array or DataFrame).
        :return: Anomaly scores (lower scores indicate anomalies).
        """
        try:
            scores = self.model.decision_function(X)
            logger.info("Anomaly scoring completed.")
            return scores
        except Exception as e:
            logger.error("Evaluation failed.")
            log_exception(e)
            raise

    def save_model(self, path):
        """
        Save the trained model to disk.
        :param path: Path to save the model (file path).
        """
        try:
            import joblib
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
            import joblib
            self.model = joblib.load(path)
            logger.info(f"Model loaded from {path}.")
        except Exception as e:
            logger.error("Failed to load model.")
            log_exception(e)
            raise
