import logging
import numpy as np
from random import randint, random, sample
from models.config.ml_settings import GENETIC_ALGORITHM_SETTINGS
from utils.exception_handler import log_exception

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GeneticAlgorithm:
    """
    Genetic Algorithm for optimizing hyperparameters for strategies.
    """

    def __init__(self, objective_function):
        """
        Initialize the Genetic Algorithm object using configuration settings.
        :param objective_function: Function to evaluate fitness based on current hyperparameters.
        """
        try:
            # Load configuration parameters
            self.population_size = GENETIC_ALGORITHM_SETTINGS["population_size"]
            self.mutation_rate = GENETIC_ALGORITHM_SETTINGS["mutation_rate"]
            self.crossover_rate = GENETIC_ALGORITHM_SETTINGS["crossover_rate"]
            self.generations = GENETIC_ALGORITHM_SETTINGS["generations"]
            self.bounds = GENETIC_ALGORITHM_SETTINGS["bounds"]

            self.objective_function = objective_function
            self.population = self.initialize_population()
            logger.info(f"Genetic Algorithm initialized with population_size={self.population_size}, "
                        f"mutation_rate={self.mutation_rate}, crossover_rate={self.crossover_rate}, "
                        f"generations={self.generations}.")
        except KeyError as e:
            logger.error(f"Configuration key missing: {e}")
            raise

    def initialize_population(self):
        """
        Initialize the population randomly within bounds.
        :return: Initial population (list of hyperparameters).
        """
        population = []
        for _ in range(self.population_size):
            individual = [np.random.uniform(low, high) for low, high in self.bounds]
            population.append(individual)
        logger.info(f"Population initialized with {self.population_size} individuals.")
        return population

    def fitness(self, individual):
        """
        Evaluate the fitness of an individual.
        :param individual: Candidate solution (hyperparameters).
        :return: Fitness value (lower is better).
        """
        try:
            return self.objective_function(individual)
        except Exception as e:
            logger.error("Fitness evaluation failed.")
            log_exception(e)
            raise

    def selection(self):
        """
        Select the best individuals using a roulette wheel selection mechanism.
        :return: Selected parents for crossover.
        """
        try:
            fitness_values = [self.fitness(ind) for ind in self.population]
            fitness_sum = sum(fitness_values)
            probabilities = [f / fitness_sum for f in fitness_values]
            parents = sample(self.population, len(self.population) // 2, weights=probabilities)
            logger.info("Selection completed.")
            return parents
        except Exception as e:
            logger.error("Selection failed.")
            log_exception(e)
            raise

    def crossover(self, parent1, parent2):
        """
        Perform crossover between two parents to create a child.
        :param parent1: First parent.
        :param parent2: Second parent.
        :return: Offspring (child).
        """
        try:
            child = []
            for i in range(len(parent1)):
                if random() < self.crossover_rate:
                    child.append(parent1[i])
                else:
                    child.append(parent2[i])
            logger.info("Crossover completed.")
            return child
        except Exception as e:
            logger.error("Crossover failed.")
            log_exception(e)
            raise

    def mutate(self, individual):
        """
        Mutate an individual with a certain mutation rate.
        :param individual: Candidate solution (hyperparameters).
        :return: Mutated individual.
        """
        try:
            for i in range(len(individual)):
                if random() < self.mutation_rate:
                    individual[i] = np.random.uniform(self.bounds[i][0], self.bounds[i][1])
            logger.info("Mutation completed.")
            return individual
        except Exception as e:
            logger.error("Mutation failed.")
            log_exception(e)
            raise

    def run(self):
        """
        Run the genetic algorithm to optimize hyperparameters.
        :return: Optimal hyperparameters.
        """
        try:
            for generation in range(self.generations):
                logger.info(f"Generation {generation + 1}/{self.generations} started.")

                # Selection
                parents = self.selection()

                # Crossover
                offspring = []
                for i in range(0, len(parents), 2):
                    parent1, parent2 = parents[i], parents[i + 1]
                    child = self.crossover(parent1, parent2)
                    offspring.append(child)

                # Mutation
                for i in range(len(offspring)):
                    offspring[i] = self.mutate(offspring[i])

                # Evaluate the new population
                self.population = parents + offspring
                self.population = sorted(self.population, key=lambda ind: self.fitness(ind))[:self.population_size]

                logger.info(f"Generation {generation + 1} completed.")

            # Return the best solution
            best_solution = self.population[0]
            logger.info(f"Optimization completed. Optimal hyperparameters: {best_solution}")
            return best_solution
        except Exception as e:
            logger.error("Optimization process failed.")
            log_exception(e)
            raise
