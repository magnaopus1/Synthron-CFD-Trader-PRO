import logging
import numpy as np
from sklearn.svm import SVR
from utils.exception_handler import log_exception
from models.config.ml_settings import SVR_MODEL_SETTINGS

# Configure logging for the regression model
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SVRModel:
    """
    Support Vector Regression (SVR) model for predicting continuous variables like price changes.
    The model is designed for real-time trading systems and can be integrated for strategy optimization.
    """

    def __init__(self):
        """
        Initialize the SVR model using configuration settings.
        """
        try:
            # Load configuration parameters from settings
            self.kernel = SVR_MODEL_SETTINGS["kernel"]
            self.C = SVR_MODEL_SETTINGS["C"]
            self.epsilon = SVR_MODEL_SETTINGS.get("epsilon", 0.1)
            self.gamma = SVR_MODEL_SETTINGS.get("gamma", 'scale')
            
            # Build the model
            self.model = self._build_model()
            logger.info(f"SVR model initialized with kernel={self.kernel}, C={self.C}, "
                        f"epsilon={self.epsilon}, gamma={self.gamma}.")
        except KeyError as e:
            logger.error(f"Configuration key missing: {e}")
            raise

    def _build_model(self):
        """
        Build the SVR model architecture.
        :return: Compiled SVR model.
        """
        try:
            model = SVR(kernel=self.kernel, C=self.C, epsilon=self.epsilon, gamma=self.gamma)
            logger.info("SVR model architecture created.")
            return model
        except Exception as e:
            logger.error("Failed to build SVR model.")
            log_exception(e)
            raise

    def train(self, X_train, y_train):
        """
        Train the SVR model.
        :param X_train: Training feature set (NumPy array).
        :param y_train: Training labels (NumPy array).
        :return: Training history.
        """
        try:
            self.model.fit(X_train, y_train)
            logger.info("SVR model training completed.")
        except Exception as e:
            logger.error("Training failed.")
            log_exception(e)
            raise

    def predict(self, X):
        """
        Predict continuous values using the trained SVR model.
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
        Evaluate the SVR model using a test dataset.
        :param X: Test feature set (NumPy array).
        :param y: Test labels (NumPy array).
        :return: Evaluation metrics, such as R^2 score.
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
        Save the trained SVR model to disk.
        :param path: Path to save the model (file path).
        """
        try:
            import joblib
            joblib.dump(self.model, path)
            logger.info(f"SVR model saved to {path}.")
        except Exception as e:
            logger.error("Failed to save SVR model.")
            log_exception(e)
            raise

    def load_model(self, path):
        """
        Load a previously saved SVR model from disk.
        :param path: Path to the saved model (file path).
        """
        try:
            import joblib
            self.model = joblib.load(path)
            logger.info(f"SVR model loaded from {path}.")
        except Exception as e:
            logger.error("Failed to load SVR model.")
            log_exception(e)
            raise
