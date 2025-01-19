import logging
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from utils.exception_handler import log_exception
from models.config.ml_settings import RANDOM_FOREST_REGRESSOR_SETTINGS

# Configure logging for the regression model
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RandomForestRegressorModel:
    """
    Random Forest Regressor for predicting asset returns or volatility using ensemble learning.
    This model is designed for high performance and scalability in real-world trading systems.
    """

    def __init__(self):
        """
        Initialize the Random Forest Regressor model using configuration settings.
        """
        try:
            # Load configuration parameters from settings
            self.n_estimators = RANDOM_FOREST_REGRESSOR_SETTINGS["n_estimators"]
            self.max_depth = RANDOM_FOREST_REGRESSOR_SETTINGS.get("max_depth", None)
            self.random_state = RANDOM_FOREST_REGRESSOR_SETTINGS.get("random_state", 42)

            # Build the model
            self.model = self._build_model()
            logger.info(f"Random Forest Regressor initialized with {self.n_estimators} estimators, "
                        f"max_depth={self.max_depth}, random_state={self.random_state}.")
        except KeyError as e:
            logger.error(f"Configuration key missing: {e}")
            raise

    def _build_model(self):
        """
        Build the Random Forest Regressor model architecture.
        :return: Compiled Random Forest model.
        """
        try:
            model = RandomForestRegressor(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                random_state=self.random_state
            )
            logger.info("Random Forest Regressor model architecture created.")
            return model
        except Exception as e:
            logger.error("Failed to build Random Forest model.")
            log_exception(e)
            raise

    def train(self, X_train, y_train):
        """
        Train the Random Forest model.
        :param X_train: Training feature set (NumPy array).
        :param y_train: Training labels (NumPy array).
        :return: Training history.
        """
        try:
            self.model.fit(X_train, y_train)
            logger.info("Random Forest model training completed.")
        except Exception as e:
            logger.error("Training failed.")
            log_exception(e)
            raise

    def predict(self, X):
        """
        Predict continuous values using the trained Random Forest model.
        :param X: Input feature set (NumPy array).
        :return: Predicted values.
        """
        try:
            predictions = self.model.predict(X)
            logger.info("Prediction completed.")
            return predictions
        except Exception as e:
            logger.error("Prediction failed.")
            log_exception(e)
            raise

    def evaluate(self, X, y):
        """
        Evaluate the Random Forest model using a test dataset.
        :param X: Test feature set (NumPy array).
        :param y: Test labels (NumPy array).
        :return: R^2 score of the model.
        """
        try:
            r2_score = self.model.score(X, y)
            logger.info(f"Model evaluation completed with R^2 score={r2_score:.4f}.")
            return r2_score
        except Exception as e:
            logger.error("Evaluation failed.")
            log_exception(e)
            raise

    def save_model(self, path):
        """
        Save the trained Random Forest model to disk.
        :param path: Path to save the model (file path).
        """
        try:
            import joblib
            joblib.dump(self.model, path)
            logger.info(f"Random Forest model saved to {path}.")
        except Exception as e:
            logger.error("Failed to save Random Forest model.")
            log_exception(e)
            raise

    def load_model(self, path):
        """
        Load a previously saved Random Forest model from disk.
        :param path: Path to the saved model (file path).
        """
        try:
            import joblib
            self.model = joblib.load(path)
            logger.info(f"Random Forest model loaded from {path}.")
        except Exception as e:
            logger.error("Failed to load Random Forest model.")
            log_exception(e)
            raise
