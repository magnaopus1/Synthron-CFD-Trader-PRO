import logging
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout
from utils.exception_handler import log_exception
from models.config.ml_settings import RNN_SETTINGS

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GRUModel:
    def __init__(self):
        """
        Initialize the GRU-based forecasting model using shared RNN configuration settings.
        """
        try:
            # Load configuration parameters
            self.units = RNN_SETTINGS["units"]
            self.input_shape = RNN_SETTINGS["input_shape"]
            self.learning_rate = RNN_SETTINGS["learning_rate"]
            self.epochs = RNN_SETTINGS["epochs"]
            self.batch_size = RNN_SETTINGS["batch_size"]
            self.validation_split = RNN_SETTINGS.get("validation_split", 0.2)
            self.dropout_rate = RNN_SETTINGS.get("dropout_rate", 0.2)

            # Build the GRU model
            self.model = self._build_model()
            logger.info(f"GRU model initialized with {self.units} units, input_shape={self.input_shape}.")
        except KeyError as e:
            logger.error(f"Configuration key missing: {e}")
            raise

    def _build_model(self):
        """
        Build the GRU model architecture.
        :return: Compiled GRU model.
        """
        try:
            model = Sequential([
                GRU(units=self.units, activation='tanh', input_shape=self.input_shape, return_sequences=True),
                Dropout(self.dropout_rate),
                GRU(units=self.units // 2, activation='tanh'),
                Dense(1)  # Output layer
            ])
            model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate), loss='mse')
            logger.info("GRU model architecture created and compiled.")
            return model
        except Exception as e:
            logger.error("Failed to build GRU model.")
            log_exception(e)
            raise

    def train(self, X_train, y_train, X_val=None, y_val=None):
        """
        Train the GRU model.
        :param X_train: Training feature set (NumPy array).
        :param y_train: Training labels (NumPy array).
        :param X_val: Validation feature set (NumPy array, optional).
        :param y_val: Validation labels (NumPy array, optional).
        :return: Training history.
        """
        try:
            validation_data = (X_val, y_val) if X_val is not None and y_val is not None else None
            history = self.model.fit(
                X_train, y_train,
                validation_data=validation_data,
                epochs=self.epochs,
                batch_size=self.batch_size,
                validation_split=self.validation_split if validation_data is None else 0,
                verbose=1
            )
            logger.info("GRU model training completed.")
            return history
        except Exception as e:
            logger.error("Training failed.")
            log_exception(e)
            raise

    def predict(self, X):
        """
        Predict future values using the trained GRU model.
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
        Evaluate the GRU model using a test dataset.
        :param X: Test feature set (NumPy array).
        :param y: Test labels (NumPy array).
        :return: Evaluation loss.
        """
        try:
            loss = self.model.evaluate(X, y, verbose=0)
            logger.info(f"Model evaluation completed with loss={loss:.4f}.")
            return loss
        except Exception as e:
            logger.error("Evaluation failed.")
            log_exception(e)
            raise

    def save_model(self, path):
        """
        Save the trained GRU model to disk.
        :param path: Path to save the model.
        """
        try:
            self.model.save(path)
            logger.info(f"GRU model saved to {path}.")
        except Exception as e:
            logger.error("Failed to save GRU model.")
            log_exception(e)
            raise

    def load_model(self, path):
        """
        Load a previously saved GRU model from disk.
        :param path: Path to the saved model.
        """
        try:
            self.model = tf.keras.models.load_model(path)
            logger.info(f"GRU model loaded from {path}.")
        except Exception as e:
            logger.error("Failed to load GRU model.")
            log_exception(e)
            raise
