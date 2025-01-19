import logging
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
from utils.exception_handler import log_exception
from models.config.ml_settings import DQN_SETTINGS

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DQN:
    def __init__(self):
        """
        Initialize the Deep Q-Network (DQN) model using configuration settings.
        """
        try:
            # Load configuration parameters from settings
            self.gamma = DQN_SETTINGS["gamma"]
            self.learning_rate = DQN_SETTINGS["learning_rate"]
            self.epsilon = DQN_SETTINGS["epsilon"]
            self.epsilon_decay = DQN_SETTINGS["epsilon_decay"]
            self.epsilon_min = DQN_SETTINGS["epsilon_min"]
            self.state_space_dim = DQN_SETTINGS["state_space_dim"]
            self.action_space_dim = DQN_SETTINGS["action_space_dim"]
            self.batch_size = DQN_SETTINGS["batch_size"]

            # Build the Q-Network model
            self.model = self._build_model()

            # Optimizer for training
            self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)

            logger.info("DQN model initialized with state_space_dim=%d, action_space_dim=%d, gamma=%.2f.",
                        self.state_space_dim, self.action_space_dim, self.gamma)
        except KeyError as e:
            logger.error(f"Configuration key missing: {e}")
            raise

    def _build_model(self):
        """
        Build the Q-Network architecture using a simple deep neural network.
        :return: Compiled model.
        """
        try:
            state_input = layers.Input(shape=(self.state_space_dim,))
            x = layers.Dense(128, activation='relu')(state_input)
            x = layers.Dense(64, activation='relu')(x)
            output = layers.Dense(self.action_space_dim, activation='linear')(x)

            model = Model(inputs=state_input, outputs=output)
            model.compile(optimizer=self.optimizer, loss='mse')

            logger.info("DQN model built successfully.")
            return model
        except Exception as e:
            logger.error("Failed to build DQN model.")
            log_exception(e)
            raise

    def choose_action(self, state):
        """
        Choose an action based on epsilon-greedy strategy.
        :param state: Current state (NumPy array).
        :return: Action (int).
        """
        try:
            if np.random.rand() <= self.epsilon:
                # Random action (exploration)
                action = np.random.choice(self.action_space_dim)
            else:
                # Best action based on the Q-network (exploitation)
                q_values = self.model(state)
                action = np.argmax(q_values.numpy())

            logger.info(f"Chosen action: {action}")
            return action
        except Exception as e:
            logger.error("Failed to choose action.")
            log_exception(e)
            raise

    def train(self, replay_memory):
        """
        Perform a single training step by updating the Q-network.
        :param replay_memory: List of past experiences for training.
        """
        try:
            if len(replay_memory) < self.batch_size:
                return

            batch = np.random.choice(replay_memory, self.batch_size)

            states = np.array([sample[0] for sample in batch])
            actions = np.array([sample[1] for sample in batch])
            rewards = np.array([sample[2] for sample in batch])
            next_states = np.array([sample[3] for sample in batch])
            dones = np.array([sample[4] for sample in batch])

            target = rewards + self.gamma * (1 - dones) * np.max(self.model(next_states), axis=1)

            with tf.GradientTape() as tape:
                q_values = self.model(states)
                one_hot_actions = tf.one_hot(actions, self.action_space_dim)
                q_value = tf.reduce_sum(q_values * one_hot_actions, axis=1)
                loss = tf.reduce_mean(tf.square(target - q_value))

            grads = tape.gradient(loss, self.model.trainable_variables)
            self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

            # Decay epsilon after each training step to reduce exploration over time
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay

            logger.info(f"Training step completed. Epsilon: {self.epsilon:.4f}")
        except Exception as e:
            logger.error("Training failed.")
            log_exception(e)
            raise

    def save_model(self, path):
        """
        Save the trained model to disk.
        :param path: Path to save the model (file path).
        """
        try:
            self.model.save(path)
            logger.info(f"DQN model saved to {path}.")
        except Exception as e:
            logger.error("Failed to save DQN model.")
            log_exception(e)
            raise

    def load_model(self, path):
        """
        Load a previously saved model from disk.
        :param path: Path to load the model from (file path).
        """
        try:
            self.model = tf.keras.models.load_model(path)
            logger.info(f"DQN model loaded from {path}.")
        except Exception as e:
            logger.error("Failed to load DQN model.")
            log_exception(e)
            raise
