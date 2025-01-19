import logging
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, LayerNormalization, Dropout
from tensorflow.keras.layers import MultiHeadAttention, Add
from utils.exception_handler import log_exception
from models.config.ml_settings import TRANSFORMER_SETTINGS

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TransformerBlock(tf.keras.layers.Layer):
    """
    Implements a single Transformer block with multi-head attention and feedforward layers.
    """
    def __init__(self, embed_dim, num_heads, ff_dim, dropout_rate=0.1):
        super(TransformerBlock, self).__init__()
        self.attention = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.feed_forward = tf.keras.Sequential([
            Dense(ff_dim, activation="relu"),
            Dense(embed_dim),
        ])
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(dropout_rate)
        self.dropout2 = Dropout(dropout_rate)

    def call(self, inputs, training):
        attn_output = self.attention(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ff_output = self.feed_forward(out1)
        ff_output = self.dropout2(ff_output, training=training)
        return self.layernorm2(out1 + ff_output)

class TransformerModel:
    def __init__(self):
        """
        Initialize the Transformer-based forecasting model using shared configuration settings.
        """
        try:
            # Load configuration parameters
            self.embed_dim = TRANSFORMER_SETTINGS["embed_dim"]
            self.num_heads = TRANSFORMER_SETTINGS["num_heads"]
            self.depth = TRANSFORMER_SETTINGS["depth"]
            self.ff_dim = TRANSFORMER_SETTINGS["ff_dim"]
            self.input_shape = TRANSFORMER_SETTINGS["input_shape"]
            self.learning_rate = TRANSFORMER_SETTINGS["learning_rate"]
            self.epochs = TRANSFORMER_SETTINGS["epochs"]
            self.batch_size = TRANSFORMER_SETTINGS["batch_size"]
            self.dropout_rate = TRANSFORMER_SETTINGS.get("dropout_rate", 0.1)

            # Build the Transformer model
            self.model = self._build_model()
            logger.info(f"Transformer model initialized with embed_dim={self.embed_dim}, "
                        f"num_heads={self.num_heads}, depth={self.depth}.")
        except KeyError as e:
            logger.error(f"Configuration key missing: {e}")
            raise

    def _build_model(self):
        """
        Build the Transformer model architecture.
        :return: Compiled Transformer model.
        """
        try:
            inputs = Input(shape=self.input_shape)
            x = inputs
            for _ in range(self.depth):
                x = TransformerBlock(self.embed_dim, self.num_heads, self.ff_dim, self.dropout_rate)(x)
            x = Dense(1, activation="linear")(x)
            outputs = tf.squeeze(x, axis=-1)  # Squeeze for 1D output
            model = Model(inputs, outputs)
            model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate), loss='mse')
            logger.info("Transformer model architecture created and compiled.")
            return model
        except Exception as e:
            logger.error("Failed to build Transformer model.")
            log_exception(e)
            raise

    def train(self, X_train, y_train, X_val=None, y_val=None):
        """
        Train the Transformer model.
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
            logger.info("Transformer model training completed.")
            return history
        except Exception as e:
            logger.error("Training failed.")
            log_exception(e)
            raise

    def predict(self, X):
        """
        Predict future values using the trained Transformer model.
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
        Evaluate the Transformer model using a test dataset.
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
        Save the Transformer model to disk.
        :param path: Path to save the model.
        """
        try:
            self.model.save(path)
            logger.info(f"Transformer model saved to {path}.")
        except Exception as e:
            logger.error("Failed to save Transformer model.")
            log_exception(e)
            raise

    def load_model(self, path):
        """
        Load a previously saved Transformer model from disk.
        :param path: Path to the saved model.
        """
        try:
            self.model = tf.keras.models.load_model(path)
            logger.info(f"Transformer model loaded from {path}.")
        except Exception as e:
            logger.error("Failed to load Transformer model.")
            log_exception(e)
            raise
