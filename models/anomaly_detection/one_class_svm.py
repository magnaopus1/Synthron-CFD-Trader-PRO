from sklearn.svm import OneClassSVM
import numpy as np
import logging
from utils.exception_handler import log_exception
from models.config.ml_settings import ONE_CLASS_SVM_SETTINGS
import joblib
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OneClassSVMModel:
    def __init__(self):
        """
        Initialize the One-Class SVM model for anomaly detection using configuration settings.
        """
        try:
            self.nu = ONE_CLASS_SVM_SETTINGS["nu"]
            self.kernel = ONE_CLASS_SVM_SETTINGS["kernel"]
            self.gamma = ONE_CLASS_SVM_SETTINGS.get("gamma", "scale")  # Optional gamma parameter

            self.model = OneClassSVM(nu=self.nu, kernel=self.kernel, gamma=self.gamma)
            logger.info(f"One-Class SVM initialized with nu={self.nu}, kernel={self.kernel}, gamma={self.gamma}")
        except KeyError as e:
            logger.error(f"Configuration key missing: {e}")
            raise

    def train(self, X):
        """
        Train the One-Class SVM model.
        :param X: Training data (numpy array or DataFrame).
        """
        try:
            self.model.fit(X)
            logger.info("One-Class SVM training completed.")
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

    def get_model_params(self):
        """
        Get current model parameters for inspection or debugging.
        :return: Model parameters as a dictionary.
        """
        try:
            params = self.model.get_params()
            logger.info(f"Model parameters: {params}")
            return params
        except Exception as e:
            logger.error("Failed to retrieve model parameters.")
            log_exception(e)
            raise
