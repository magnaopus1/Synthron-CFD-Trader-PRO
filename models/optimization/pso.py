import logging
import numpy as np
from random import random
from models.config.ml_settings import PSO_SETTINGS
from utils.exception_handler import log_exception

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ParticleSwarmOptimization:
    """
    Particle Swarm Optimization (PSO) for optimizing hyperparameters for strategies.
    """

    def __init__(self, objective_function):
        """
        Initialize the Particle Swarm Optimization (PSO) object using configuration settings.
        :param objective_function: Function to evaluate fitness based on current hyperparameters.
        """
        try:
            # Load configuration parameters
            self.swarm_size = PSO_SETTINGS["swarm_size"]
            self.inertia_weight = PSO_SETTINGS["inertia_weight"]
            self.cognitive_weight = PSO_SETTINGS["cognitive_weight"]
            self.social_weight = PSO_SETTINGS["social_weight"]
            self.generations = PSO_SETTINGS["generations"]
            self.bounds = PSO_SETTINGS["bounds"]

            self.objective_function = objective_function
            self.swarm = self.initialize_swarm()
            logger.info(f"PSO initialized with swarm_size={self.swarm_size}, "
                        f"inertia_weight={self.inertia_weight}, cognitive_weight={self.cognitive_weight}, "
                        f"social_weight={self.social_weight}, generations={self.generations}.")
        except KeyError as e:
            logger.error(f"Configuration key missing: {e}")
            raise

    def initialize_swarm(self):
        """
        Initialize the swarm with random positions and velocities within bounds.
        :return: Initial swarm (list of particles).
        """
        swarm = []
        for _ in range(self.swarm_size):
            position = [np.random.uniform(low, high) for low, high in self.bounds]
            velocity = [np.random.uniform(-abs(high - low), abs(high - low)) for low, high in self.bounds]
            best_position = position.copy()
            swarm.append({"position": position, "velocity": velocity, "best_position": best_position, "best_fitness": float('inf')})
        logger.info(f"Swarm initialized with {self.swarm_size} particles.")
        return swarm

    def fitness(self, particle):
        """
        Evaluate the fitness of a particle (candidate solution).
        :param particle: Particle (candidate solution with position).
        :return: Fitness value (lower is better).
        """
        try:
            return self.objective_function(particle['position'])
        except Exception as e:
            logger.error("Fitness evaluation failed.")
            log_exception(e)
            raise

    def update_velocity(self, particle, global_best_position):
        """
        Update the velocity of a particle.
        :param particle: Particle whose velocity is to be updated.
        :param global_best_position: The global best position found so far.
        :return: Updated velocity.
        """
        try:
            new_velocity = []
            for i in range(len(particle['position'])):
                inertia = self.inertia_weight * particle['velocity'][i]
                cognitive = self.cognitive_weight * random() * (particle['best_position'][i] - particle['position'][i])
                social = self.social_weight * random() * (global_best_position[i] - particle['position'][i])
                new_velocity.append(inertia + cognitive + social)
            return new_velocity
        except Exception as e:
            logger.error("Velocity update failed.")
            log_exception(e)
            raise

    def update_position(self, particle):
        """
        Update the position of a particle.
        :param particle: Particle whose position is to be updated.
        :return: Updated position.
        """
        try:
            new_position = []
            for i in range(len(particle['position'])):
                new_position.append(particle['position'][i] + particle['velocity'][i])
            return new_position
        except Exception as e:
            logger.error("Position update failed.")
            log_exception(e)
            raise

    def run(self):
        """
        Run the Particle Swarm Optimization to find the optimal parameters.
        :return: Optimal hyperparameters.
        """
        try:
            global_best_position = None
            global_best_fitness = float('inf')

            for generation in range(self.generations):
                logger.info(f"Generation {generation + 1}/{self.generations} started.")

                # Evaluate fitness of each particle and update best positions
                for particle in self.swarm:
                    fitness_value = self.fitness(particle)
                    if fitness_value < particle['best_fitness']:
                        particle['best_fitness'] = fitness_value
                        particle['best_position'] = particle['position'].copy()

                    if fitness_value < global_best_fitness:
                        global_best_fitness = fitness_value
                        global_best_position = particle['position'].copy()

                # Update velocity and position of each particle
                for particle in self.swarm:
                    particle['velocity'] = self.update_velocity(particle, global_best_position)
                    particle['position'] = self.update_position(particle)

                logger.info(f"Generation {generation + 1} completed.")

            # Return the best solution
            logger.info(f"Optimization completed. Optimal hyperparameters: {global_best_position}")
            return global_best_position
        except Exception as e:
            logger.error("Optimization process failed.")
            log_exception(e)
            raise
