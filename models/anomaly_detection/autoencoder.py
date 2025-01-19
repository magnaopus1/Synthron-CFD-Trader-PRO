import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import numpy as np
import logging
import os
from models.config.ml_settings import AUTOENCODER_SETTINGS

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Autoencoder:
    def __init__(self):
        """
        Initialize the Autoencoder for anomaly detection using settings from the configuration file.
        """
        self.input_dim = AUTOENCODER_SETTINGS["input_dim"]
        self.encoding_dim = AUTOENCODER_SETTINGS["encoding_dim"]
        self.learning_rate = AUTOENCODER_SETTINGS["learning_rate"]
        self.seed = AUTOENCODER_SETTINGS.get("seed", 42)  # Optional seed parameter
        self.epochs = AUTOENCODER_SETTINGS["epochs"]
        self.batch_size = AUTOENCODER_SETTINGS["batch_size"]
        self.validation_split = AUTOENCODER_SETTINGS["validation_split"]

        # Set random seeds for reproducibility
        tf.random.set_seed(self.seed)
        np.random.seed(self.seed)

        # Build and compile the model
        self.model = self._build_autoencoder()
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss='mse',
            metrics=['mae']
        )
        logger.info("Autoencoder model initialized with configuration settings.")

    def _build_autoencoder(self):
        """
        Builds the autoencoder architecture.
        :return: Compiled Keras model.
        """
        input_layer = Input(shape=(self.input_dim,))
        encoded = Dense(self.encoding_dim, activation='relu')(input_layer)
        decoded = Dense(self.input_dim, activation='sigmoid')(encoded)

        model = Model(inputs=input_layer, outputs=decoded)
        logger.info("Autoencoder architecture created.")
        return model

    def train(self, X_train, X_val, checkpoint_path="model_checkpoints/autoencoder.h5"):
        """
        Train the autoencoder with callbacks for early stopping and model checkpointing.
        :param X_train: Training data.
        :param X_val: Validation data.
        :param checkpoint_path: Path to save model checkpoints.
        """
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)

        callbacks = [
            EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1),
            ModelCheckpoint(filepath=checkpoint_path, monitor='val_loss', save_best_only=True, verbose=1),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1)
        ]

        history = self.model.fit(
            X_train, X_train,
            validation_data=(X_val, X_val),
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_split=self.validation_split,
            verbose=1,
            callbacks=callbacks
        )
        logger.info("Training completed.")
        return history

    def predict(self, X):
        """
        Predict reconstruction for input data.
        :param X: Input data.
        :return: Reconstructed data.
        """
        try:
            predictions = self.model.predict(X)
            return predictions
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            raise

    def evaluate(self, X):
        """
        Evaluate reconstruction error.
        :param X: Input data.
        :return: Reconstruction error.
        """
        try:
            predictions = self.predict(X)
            reconstruction_error = np.mean(np.power(X - predictions, 2), axis=1)
            return reconstruction_error
        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            raise

    def save_model(self, path):
        """
        Save the model to disk.
        :param path: Path to save the model.
        """
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            self.model.save(path)
            logger.info(f"Model saved to {path}.")
        except Exception as e:
            logger.error(f"Failed to save model: {e}")
            raise

    def load_model(self, path):
        """
        Load a model from disk.
        :param path: Path to the saved model.
        """
        try:
            self.model = tf.keras.models.load_model(path)
            logger.info(f"Model loaded from {path}.")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
