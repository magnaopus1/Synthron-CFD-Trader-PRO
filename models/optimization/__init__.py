# __init__.py for models/optimization

import logging

# Import optimization models
from .bayesian_optimization import BayesianOptimization
from .genetic_algorithm import GeneticAlgorithm
from .pso import ParticleSwarmOptimization

# Configure logging for the optimization module
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    # Ensure that the classes are available for external use
    __all__ = [
        "BayesianOptimization",  # Optimizer for hyperparameter tuning using Gaussian Process
        "GeneticAlgorithm",      # Genetic Algorithm for hyperparameter optimization
        "ParticleSwarmOptimization"  # Particle Swarm Optimization for hyperparameter optimization
    ]
    logger.info("Optimization module initialized successfully.")
except Exception as e:
    logger.error("Failed to initialize the optimization module.")
    logger.exception(e)
    raise
