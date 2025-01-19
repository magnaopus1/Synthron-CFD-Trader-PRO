from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import logging
import joblib
import os
from utils.exception_handler import log_exception
from models.config.ml_settings import LOGISTIC_REGRESSION_SETTINGS

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LogisticRegressionModel:
    def __init__(self):
        """
        Initialize the Logistic Regression model for binary classification using configuration settings.
        """
        try:
            self.penalty = LOGISTIC_REGRESSION_SETTINGS["penalty"]
            self.C = LOGISTIC_REGRESSION_SETTINGS["C"]
            self.solver = LOGISTIC_REGRESSION_SETTINGS["solver"]
            self.random_state = LOGISTIC_REGRESSION_SETTINGS["random_state"]

            self.buy_threshold = LOGISTIC_REGRESSION_SETTINGS["buy_threshold"]
            self.sell_threshold = LOGISTIC_REGRESSION_SETTINGS["sell_threshold"]

            self.model = LogisticRegression(
                penalty=self.penalty,
                C=self.C,
                solver=self.solver,
                random_state=self.random_state,
            )
            logger.info("Logistic Regression model initialized with configuration settings.")
        except KeyError as e:
            logger.error(f"Configuration key missing: {e}")
            raise

    def train(self, X_train, y_train):
        """
        Train the Logistic Regression model.
        :param X_train: Training feature set.
        :param y_train: Training labels.
        """
        try:
            self.model.fit(X_train, y_train)
            logger.info("Logistic Regression model training completed.")
        except Exception as e:
            logger.error("Training failed.")
            log_exception(e)
            raise

    def predict(self, X):
        """
        Predict class labels using the trained model.
        :param X: Input feature set.
        :return: Predicted labels.
        """
        try:
            predictions = self.model.predict(X)
            logger.info("Prediction completed.")
            return predictions
        except Exception as e:
            logger.error("Prediction failed.")
            log_exception(e)
            raise

    def predict_proba(self, X):
        """
        Predict probabilities for each class.
        :param X: Input feature set.
        :return: Predicted probabilities.
        """
        try:
            probabilities = self.model.predict_proba(X)
            logger.info("Probability prediction completed.")
            return probabilities
        except Exception as e:
            logger.error("Probability prediction failed.")
            log_exception(e)
            raise

    def classify_signals(self, probabilities):
        """
        Classify signals based on thresholds.
        :param probabilities: Predicted probabilities for the positive class.
        :return: Buy, sell, or hold signals.
        """
        try:
            signals = []
            for prob in probabilities[:, 1]:  # Positive class probabilities
                if prob >= self.buy_threshold:
                    signals.append("BUY")
                elif prob <= self.sell_threshold:
                    signals.append("SELL")
                else:
                    signals.append("HOLD")
            logger.info("Signal classification completed.")
            return signals
        except Exception as e:
            logger.error("Signal classification failed.")
            log_exception(e)
            raise

    def evaluate(self, X, y_true):
        """
        Evaluate the model's performance.
        :param X: Feature set.
        :param y_true: True labels.
        :return: Dictionary with evaluation metrics.
        """
        try:
            y_pred = self.predict(X)
            metrics = {
                "accuracy": accuracy_score(y_true, y_pred),
                "f1_score": f1_score(y_true, y_pred),
                "precision": precision_score(y_true, y_pred),
                "recall": recall_score(y_true, y_pred),
            }
            logger.info(f"Model evaluation completed: {metrics}")
            return metrics
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
