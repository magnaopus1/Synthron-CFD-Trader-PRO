import logging
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from scipy.optimize import minimize
from scipy.stats import norm
from utils.exception_handler import log_exception
from models.config.ml_settings import BAYESIAN_OPTIMIZATION_SETTINGS

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BayesianOptimization:
    """
    Bayesian Optimization for parameter tuning using Gaussian Process Regression.
    """

    def __init__(self):
        """
        Initialize the Bayesian Optimization object using configuration settings.
        """
        try:
            # Load configuration parameters
            self.bounds = BAYESIAN_OPTIMIZATION_SETTINGS["bounds"]
            self.n_iterations = BAYESIAN_OPTIMIZATION_SETTINGS["n_iterations"]
            self.acquisition_function = BAYESIAN_OPTIMIZATION_SETTINGS.get("acquisition_function", "EI")
            self.kernel = Matern(length_scale=1.0, nu=2.5)

            self.gp = GaussianProcessRegressor(kernel=self.kernel, normalize_y=True)
            self.X_sample = np.array([]).reshape(0, len(self.bounds))
            self.Y_sample = np.array([]).reshape(0, 1)

            logger.info(f"Bayesian Optimization initialized with bounds={self.bounds} "
                        f"and {self.n_iterations} iterations.")
        except KeyError as e:
            logger.error(f"Configuration key missing: {e}")
            raise

    def acquisition(self, X, xi=0.01):
        """
        Compute the acquisition function value for given inputs.
        :param X: Points to evaluate (2D array).
        :param xi: Exploration-exploitation tradeoff parameter.
        :return: Acquisition function values.
        """
        try:
            mu, sigma = self.gp.predict(X, return_std=True)
            mu_sample_opt = np.max(self.Y_sample)

            if self.acquisition_function == "EI":  # Expected Improvement
                with np.errstate(divide='warn'):
                    imp = mu - mu_sample_opt - xi
                    Z = imp / sigma
                    ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
                    ei[sigma == 0.0] = 0.0
                return ei
            else:
                raise ValueError(f"Unsupported acquisition function: {self.acquisition_function}")
        except Exception as e:
            logger.error("Failed to compute acquisition function.")
            log_exception(e)
            raise

    def propose_location(self):
        """
        Propose the next sampling point by optimizing the acquisition function.
        :return: Next sampling point (1D array).
        """
        try:
            def min_obj(X):
                return -self.acquisition(X.reshape(1, -1))

            x0 = np.random.uniform(self.bounds[:, 0], self.bounds[:, 1], size=self.bounds.shape[0])
            res = minimize(min_obj, x0=x0, bounds=self.bounds, method="L-BFGS-B")
            logger.info("Proposed new location for sampling.")
            return res.x
        except Exception as e:
            logger.error("Failed to propose a new sampling location.")
            log_exception(e)
            raise

    def optimize(self, objective_function):
        """
        Perform Bayesian Optimization to find the optimal parameters.
        :param objective_function: The objective function to minimize.
        :return: Optimal parameters and corresponding objective value.
        """
        try:
            for i in range(self.n_iterations):
                logger.info(f"Starting iteration {i+1}/{self.n_iterations}")

                if self.X_sample.shape[0] == 0:
                    X_next = np.random.uniform(self.bounds[:, 0], self.bounds[:, 1], size=self.bounds.shape[0])
                else:
                    X_next = self.propose_location()

                Y_next = objective_function(X_next)

                self.X_sample = np.vstack((self.X_sample, X_next.reshape(1, -1)))
                self.Y_sample = np.vstack((self.Y_sample, Y_next))

                self.gp.fit(self.X_sample, self.Y_sample)
                logger.info(f"Iteration {i+1} completed. Sampled point: {X_next}, Value: {Y_next}")

            optimal_idx = np.argmax(self.Y_sample)
            optimal_params = self.X_sample[optimal_idx]
            optimal_value = self.Y_sample[optimal_idx]

            logger.info(f"Optimization completed. Optimal parameters: {optimal_params}, Value: {optimal_value}")
            return optimal_params, optimal_value
        except Exception as e:
            logger.error("Optimization failed.")
            log_exception(e)
            raise
