import logging
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
from utils.exception_handler import log_exception
from models.config.ml_settings import PPO_SETTINGS

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PPO:
    def __init__(self):
        """
        Initialize the PPO model using configuration settings.
        """
        try:
            # Load configuration parameters from settings
            self.gamma = PPO_SETTINGS["gamma"]
            self.learning_rate = PPO_SETTINGS["learning_rate"]
            self.epsilon = PPO_SETTINGS["epsilon"]
            self.state_space_dim = PPO_SETTINGS["state_space_dim"]
            self.action_space_dim = PPO_SETTINGS["action_space_dim"]
            self.batch_size = PPO_SETTINGS["batch_size"]
            self.ppo_clip_value = PPO_SETTINGS["ppo_clip_value"]

            # Build the actor and critic networks
            self.actor, self.critic = self._build_models()

            # Optimizers for actor and critic
            self.actor_optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
            self.critic_optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)

            logger.info(f"PPO model initialized with state_space_dim={self.state_space_dim}, "
                        f"action_space_dim={self.action_space_dim}, gamma={self.gamma}, "
                        f"epsilon={self.epsilon}, batch_size={self.batch_size}.")
        except KeyError as e:
            logger.error(f"Configuration key missing: {e}")
            raise

    def _build_models(self):
        """
        Build the actor and critic networks using a simple dense network.
        :return: actor and critic models.
        """
        try:
            # Actor Network
            state_input = layers.Input(shape=(self.state_space_dim,))
            x = layers.Dense(128, activation='relu')(state_input)
            x = layers.Dense(64, activation='relu')(x)
            output = layers.Dense(self.action_space_dim, activation='softmax')(x)
            actor = Model(inputs=state_input, outputs=output)

            # Critic Network
            state_input = layers.Input(shape=(self.state_space_dim,))
            x = layers.Dense(128, activation='relu')(state_input)
            x = layers.Dense(64, activation='relu')(x)
            output = layers.Dense(1)(x)  # Single value output for the state value
            critic = Model(inputs=state_input, outputs=output)

            logger.info("Actor and Critic models built successfully.")
            return actor, critic
        except Exception as e:
            logger.error("Failed to build Actor-Critic models.")
            log_exception(e)
            raise

    def compute_loss(self, states, actions, rewards, next_states, done, old_probs):
        """
        Compute the actor and critic losses for PPO.
        :param states: Current states (NumPy array).
        :param actions: Actions taken (NumPy array).
        :param rewards: Rewards received (NumPy array).
        :param next_states: Next states (NumPy array).
        :param done: Whether the episode has ended (NumPy array).
        :param old_probs: Probabilities of actions taken (NumPy array).
        :return: Total loss (actor loss + critic loss).
        """
        try:
            # Compute the value of the current and next states
            values = self.critic(states)
            next_values = self.critic(next_states)
            
            # Compute advantage
            target = rewards + self.gamma * (1 - done) * next_values
            advantage = target - values

            # Actor loss (Negative log probability * advantage, clipped)
            action_probs = self.actor(states)
            action_indices = tf.argmax(actions, axis=1)
            action_log_probs = tf.math.log(tf.gather(action_probs, action_indices, axis=1))
            ratio = tf.exp(action_log_probs - old_probs)
            clip_ratio = tf.clip_by_value(ratio, 1 - self.ppo_clip_value, 1 + self.ppo_clip_value)
            actor_loss = -tf.reduce_mean(tf.minimum(ratio * advantage, clip_ratio * advantage))

            # Critic loss (Mean Squared Error between target and predicted value)
            critic_loss = tf.reduce_mean(tf.square(advantage))

            total_loss = actor_loss + critic_loss
            logger.info(f"Computed loss: Actor Loss={actor_loss.numpy()}, Critic Loss={critic_loss.numpy()}, Total Loss={total_loss.numpy()}")
            return total_loss
        except Exception as e:
            logger.error("Failed to compute loss.")
            log_exception(e)
            raise

    def train(self, states, actions, rewards, next_states, done, old_probs):
        """
        Perform a single training step by updating actor and critic networks.
        :param states: Current states.
        :param actions: Actions taken.
        :param rewards: Rewards received.
        :param next_states: Next states.
        :param done: Whether the episode is finished.
        :param old_probs: Old action probabilities.
        """
        try:
            with tf.GradientTape(persistent=True) as tape:
                loss = self.compute_loss(states, actions, rewards, next_states, done, old_probs)

            # Compute gradients and update actor and critic models
            actor_gradients = tape.gradient(loss, self.actor.trainable_variables)
            critic_gradients = tape.gradient(loss, self.critic.trainable_variables)

            self.actor_optimizer.apply_gradients(zip(actor_gradients, self.actor.trainable_variables))
            self.critic_optimizer.apply_gradients(zip(critic_gradients, self.critic.trainable_variables))

            logger.info(f"Training step completed with loss={loss.numpy()}.")
        except Exception as e:
            logger.error("Training step failed.")
            log_exception(e)
            raise

    def choose_action(self, state):
        """
        Choose an action using the actor network based on the current state.
        :param state: Current state.
        :return: Action (Buy, Sell, Hold).
        """
        try:
            action_probs = self.actor(state)
            action = np.random.choice(self.action_space_dim, p=action_probs.numpy()[0])
            logger.info(f"Chosen action: {action}")
            return action
        except Exception as e:
            logger.error("Failed to choose action.")
            log_exception(e)
            raise

    def save_model(self, path):
        """
        Save the actor and critic models to disk.
        :param path: Path to save the models.
        """
        try:
            self.actor.save(path + '_actor.h5')
            self.critic.save(path + '_critic.h5')
            logger.info(f"Actor-Critic models saved to {path}.")
        except Exception as e:
            logger.error("Failed to save Actor-Critic models.")
            log_exception(e)
            raise

    def load_model(self, path):
        """
        Load the actor and critic models from disk.
        :param path: Path to load the models.
        """
        try:
            self.actor = tf.keras.models.load_model(path + '_actor.h5')
            self.critic = tf.keras.models.load_model(path + '_critic.h5')
            logger.info(f"Actor-Critic models loaded from {path}.")
        except Exception as e:
            logger.error("Failed to load Actor-Critic models.")
            log_exception(e)
            raise
