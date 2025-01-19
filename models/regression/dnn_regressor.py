import logging
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from utils.exception_handler import log_exception
from models.config.ml_settings import DNN_REGRESSOR_SETTINGS

# Configure logging for detailed tracking of processes
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DNNRegressor:
    """
    Deep Neural Network (DNN) Regressor for regression tasks using deep learning.
    This model is designed for high performance and scalability in real-world trading systems.
    """

    def __init__(self):
        """
        Initialize the DNN Regressor model using configuration settings.
        """
        try:
            # Load configuration parameters from settings
            self.input_dim = DNN_REGRESSOR_SETTINGS["input_dim"]
            self.layers = DNN_REGRESSOR_SETTINGS["layers"]
            self.units = DNN_REGRESSOR_SETTINGS["units"]
            self.dropout_rate = DNN_REGRESSOR_SETTINGS.get("dropout_rate", 0.2)
            self.learning_rate = DNN_REGRESSOR_SETTINGS.get("learning_rate", 0.001)
            self.epochs = DNN_REGRESSOR_SETTINGS["epochs"]
            self.batch_size = DNN_REGRESSOR_SETTINGS["batch_size"]

            # Build the model
            self.model = self._build_model()
            logger.info(f"DNN Regressor initialized with {self.layers} layers, "
                        f"{self.units} units per layer, learning rate={self.learning_rate}.")
        except KeyError as e:
            logger.error(f"Configuration key missing: {e}")
            raise

    def _build_model(self):
        """
        Build the DNN Regressor model architecture.
        :return: Compiled DNN model.
        """
        try:
            model = Sequential()
            # Input layer
            model.add(Dense(self.units, input_dim=self.input_dim, activation='relu'))

            # Hidden layers
            for _ in range(self.layers - 1):
                model.add(Dense(self.units, activation='relu'))
                if self.dropout_rate > 0:
                    model.add(Dropout(self.dropout_rate))

            # Output layer
            model.add(Dense(1))  # Single neuron for regression output

            # Compile the model
            model.compile(optimizer=Adam(learning_rate=self.learning_rate), loss='mean_squared_error')
            logger.info("DNN Regressor model architecture created and compiled.")
            return model
        except Exception as e:
            logger.error("Failed to build DNN model.")
            log_exception(e)
            raise

    def train(self, X_train, y_train, X_val=None, y_val=None):
        """
        Train the DNN model.
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
                verbose=1
            )
            logger.info("DNN model training completed.")
            return history
        except Exception as e:
            logger.error("Training failed.")
            log_exception(e)
            raise

    def predict(self, X):
        """
        Predict continuous values using the trained DNN model.
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
        Evaluate the DNN model using a test dataset.
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
        Save the trained DNN model to disk.
        :param path: Path to save the model (file path).
        """
        try:
            self.model.save(path)
            logger.info(f"DNN model saved to {path}.")
        except Exception as e:
            logger.error("Failed to save DNN model.")
            log_exception(e)
            raise

    def load_model(self, path):
        """
        Load a previously saved DNN model from disk.
        :param path: Path to the saved model (file path).
        """
        try:
            from tensorflow.keras.models import load_model
            self.model = load_model(path)
            logger.info(f"DNN model loaded from {path}.")
        except Exception as e:
            logger.error("Failed to load DNN model.")
            log_exception(e)
            raise
