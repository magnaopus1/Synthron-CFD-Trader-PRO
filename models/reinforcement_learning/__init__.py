# Importing the required models for reinforcement learning
from .actor_critic import ActorCritic
from .dqn import DQN
from .ppo import PPO

# Make the reinforcement learning models accessible when importing the module
__all__ = [
    "ActorCritic",  # Actor-Critic model for policy optimization
    "DQN",          # Deep Q-Network model for reinforcement learning
    "PPO"           # Proximal Policy Optimization model for policy optimization
]

# Ensure proper logging and error handling during module imports
import logging
logger = logging.getLogger(__name__)

try:
    logger.info("Reinforcement learning models (ActorCritic, DQN, PPO) successfully loaded.")
except Exception as e:
    logger.error(f"Error loading reinforcement learning models: {e}")
    raise
