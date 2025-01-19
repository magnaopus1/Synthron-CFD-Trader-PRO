from xgboost import XGBClassifier
import logging
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from utils.exception_handler import log_exception
from models.config.ml_settings import GRADIENT_BOOSTING_SETTINGS
import joblib
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GradientBoostingModel:
    def __init__(self):
        """
        Initialize the Gradient Boosting model for advanced classification using configuration settings.
        """
        try:
            self.learning_rate = GRADIENT_BOOSTING_SETTINGS["learning_rate"]
            self.n_estimators = GRADIENT_BOOSTING_SETTINGS["n_estimators"]
            self.max_depth = GRADIENT_BOOSTING_SETTINGS["max_depth"]
            self.boosting_type = GRADIENT_BOOSTING_SETTINGS["boosting_type"]
            self.random_state = GRADIENT_BOOSTING_SETTINGS["random_state"]

            self.model = XGBClassifier(
                learning_rate=self.learning_rate,
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                objective="binary:logistic",
                booster=self.boosting_type,
                random_state=self.random_state,
                use_label_encoder=False,
                eval_metric="logloss",
            )
            logger.info("Gradient Boosting model initialized with configuration settings.")
        except KeyError as e:
            logger.error(f"Configuration key missing: {e}")
            raise

    def train(self, X_train, y_train, X_val=None, y_val=None, early_stopping_rounds=10):
        """
        Train the Gradient Boosting model with early stopping support.
        :param X_train: Training feature set.
        :param y_train: Training labels.
        :param X_val: Validation feature set (optional).
        :param y_val: Validation labels (optional).
        :param early_stopping_rounds: Number of rounds for early stopping.
        """
        try:
            if X_val is not None and y_val is not None:
                self.model.fit(
                    X_train,
                    y_train,
                    eval_set=[(X_val, y_val)],
                    early_stopping_rounds=early_stopping_rounds,
                    verbose=True,
                )
            else:
                self.model.fit(X_train, y_train)
            logger.info("Gradient Boosting model training completed.")
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
        Predict class probabilities using the trained model.
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
