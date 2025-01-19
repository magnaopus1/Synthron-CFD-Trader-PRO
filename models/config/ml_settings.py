# Machine Learning Model Settings
import numpy as np

AUTOENCODER_SETTINGS = {
    "input_dim": 30,  # Number of input features
    "encoding_dim": 16,  # Size of the bottleneck layer
    "learning_rate": 0.001,  # Learning rate for optimizer
    "epochs": 100,  # Number of training epochs
    "batch_size": 64,  # Batch size
    "validation_split": 0.2,  # Fraction of data used for validation
    "seed": 42  # Seed for reproducibility
}

ISOLATION_FOREST_SETTINGS = {
    "n_estimators": 100,        # Number of estimators
    "max_samples": "auto",      # Maximum samples for base estimators
    "contamination": 0.1,       # Proportion of anomalies in the data
    "max_features": 1.0,        # Maximum number of features to consider
    "random_state": 42          # Random seed for reproducibility
}

ONE_CLASS_SVM_SETTINGS = {
    "nu": 0.05,        # Upper bound on proportion of anomalies in the data
    "kernel": "rbf",   # Kernel type for SVM
    "gamma": "scale"   # Kernel coefficient (scale or auto)
}

GRADIENT_BOOSTING_SETTINGS = {
    "learning_rate": 0.1,       # Step size shrinkage
    "n_estimators": 100,        # Number of boosting rounds
    "max_depth": 6,             # Maximum depth of a tree
    "boosting_type": "gbtree",  # Boosting type (gbtree, dart, or gblinear)
    "random_state": 42          # Seed for reproducibility
}

RANDOM_FOREST_SETTINGS = {
    "n_estimators": 100,       # Number of trees in the forest
    "max_depth": 10,           # Maximum depth of a tree
    "random_state": 42,        # Seed for reproducibility
    "buy_threshold": 0.7,      # Probability threshold to generate a BUY signal
    "sell_threshold": 0.3,     # Probability threshold to generate a SELL signal
}

LOGISTIC_REGRESSION_SETTINGS = {
    "penalty": "l2",           # Regularization type
    "C": 1.0,                  # Inverse of regularization strength
    "solver": "lbfgs",         # Algorithm to use in optimization
    "random_state": 42,        # Random seed for reproducibility
    "buy_threshold": 0.7,      # Probability threshold to generate a BUY signal
    "sell_threshold": 0.3,     # Probability threshold to generate a SELL signal
}

DBSCAN_SETTINGS = {
    "epsilon": 0.5,        # Maximum distance between two samples for them to be considered as in the same neighborhood
    "min_samples": 5,      # Number of samples in a neighborhood for a point to be considered as a core point
}

GMM_SETTINGS = {
    "n_components": 3,         # Number of Gaussian components in the mixture model
    "covariance_type": "full", # Type of covariance matrix (full, tied, diag, spherical)
    "random_state": 42         # Seed for reproducibility
}

KMEANS_SETTINGS = {
    "n_clusters": 5,           # Number of clusters
    "init": "k-means++",       # Method for initialization
    "max_iter": 300,           # Maximum number of iterations for a single run
    "random_state": 42         # Seed for reproducibility
}

MUTUAL_INFO_SETTINGS = {
    "num_features": 10,        # Number of top features to select
    "random_state": 42         # Seed for reproducibility
}

PCA_SETTINGS = {
    "variance_ratio": 0.95  # Retain 95% of the variance in the dataset
}

RFE_SETTINGS = {
    "num_features": 10,        # Number of features to select
    "random_state": 42         # Seed for reproducibility
}

ARIMA_SETTINGS = {
    "order": (1, 1, 1),            # ARIMA (p, d, q) order
    "seasonal_order": (0, 0, 0, 0),# SARIMA (P, D, Q, s) order
    "trend": "n"                   # Trend component: "n" (no trend), "c" (constant), "t" (linear trend), "ct" (constant + trend)
}

RNN_SETTINGS = {
    "units": 128,               # Number of LSTM units
    "input_shape": (30, 1),     # Input shape for time series data (timesteps, features)
    "learning_rate": 0.001,     # Learning rate for optimizer
    "epochs": 50,               # Number of training epochs
    "batch_size": 64,           # Batch size for training
    "validation_split": 0.2,    # Fraction of training data for validation
    "dropout_rate": 0.2         # Dropout rate for regularization
}

TRANSFORMER_SETTINGS = {
    "embed_dim": 64,             # Embedding dimension for each input feature
    "num_heads": 4,              # Number of attention heads
    "depth": 3,                  # Number of Transformer blocks
    "ff_dim": 128,               # Dimension of the feedforward network
    "input_shape": (30, 1),      # Input shape (time steps, features)
    "learning_rate": 0.001,      # Learning rate for optimizer
    "epochs": 50,                # Number of training epochs
    "batch_size": 32,            # Batch size for training
    "dropout_rate": 0.1          # Dropout rate for regularization
}

BAYESIAN_OPTIMIZATION_SETTINGS = {
    "bounds": np.array([[0, 1], [0, 10], [1, 100]]),  # Example bounds for 3 parameters
    "n_iterations": 20,                              # Number of optimization iterations
    "acquisition_function": "EI"                     # Acquisition function (e.g., "EI" for Expected Improvement)
}

GENETIC_ALGORITHM_SETTINGS = {
    "population_size": 50,         # Size of the population
    "mutation_rate": 0.1,          # Rate of mutation in offspring
    "crossover_rate": 0.7,         # Rate of crossover between parents
    "generations": 100,            # Number of generations for optimization
    "bounds": [
        [0, 1],                    # Bounds for first hyperparameter
        [0, 10],                   # Bounds for second hyperparameter
        [1, 100]                   # Bounds for third hyperparameter
    ]                             # Hyperparameter bounds for optimization
}

PSO_SETTINGS = {
    "swarm_size": 30,               # Number of particles in the swarm
    "inertia_weight": 0.5,          # Inertia weight
    "cognitive_weight": 1.5,        # Cognitive (personal best) weight
    "social_weight": 1.5,           # Social (global best) weight
    "generations": 100,             # Number of generations for optimization
    "bounds": [
        [0, 1],                     # Bounds for first hyperparameter
        [0, 10],                    # Bounds for second hyperparameter
        [1, 100]                    # Bounds for third hyperparameter
    ]                              # Hyperparameter bounds for optimization
}

DNN_REGRESSOR_SETTINGS = {
    "input_dim": 10,              # Number of input features
    "layers": 3,                  # Number of hidden layers
    "units": 64,                  # Number of units in each hidden layer
    "dropout_rate": 0.2,          # Dropout rate for regularization
    "learning_rate": 0.001,       # Learning rate for Adam optimizer
    "epochs": 100,                # Number of training epochs
    "batch_size": 32              # Batch size for training
}

RANDOM_FOREST_REGRESSOR_SETTINGS = {
    "n_estimators": 100,        # Number of trees in the forest
    "max_depth": None,           # Maximum depth of the tree (None means fully expanded)
    "random_state": 42           # Random seed for reproducibility
}

SVR_MODEL_SETTINGS = {
    "kernel": "rbf",            # Kernel type (options: 'linear', 'poly', 'rbf', 'sigmoid')
    "C": 1.0,                   # Regularization parameter (higher values mean less regularization)
    "epsilon": 0.1,             # Epsilon parameter (margin of tolerance for error)
    "gamma": "scale"            # Kernel coefficient (options: 'scale', 'auto')
}

ACTOR_CRITIC_SETTINGS = {
    "gamma": 0.99,                          # Discount factor for rewards (Gamma)
    "learning_rate_actor": 0.001,            # Learning rate for the actor network
    "learning_rate_critic": 0.001,           # Learning rate for the critic network
    "state_space_dim": 10,                   # State space dimension (number of features in state representation)
    "action_space_dim": 3,                   # Action space dimension (e.g., Buy, Sell, Hold)
}

DQN_SETTINGS = {
    "gamma": 0.99,                          # Discount factor for future rewards
    "learning_rate": 0.001,                  # Learning rate for the optimizer
    "epsilon": 1.0,                          # Initial exploration rate (epsilon-greedy strategy)
    "epsilon_decay": 0.995,                  # Epsilon decay rate
    "epsilon_min": 0.01,                     # Minimum value for epsilon (avoid zero exploration)
    "state_space_dim": 10,                   # Dimension of the state space (e.g., market indicators)
    "action_space_dim": 3,                   # Number of possible actions (e.g., Buy, Sell, Hold)
    "batch_size": 64                         # Batch size for training
}

PPO_SETTINGS = {
    "gamma": 0.99,                          # Discount factor for future rewards
    "learning_rate": 0.0003,                 # Learning rate for the optimizer
    "epsilon": 0.2,                          # Clipping parameter for PPO loss function
    "state_space_dim": 10,                   # Dimension of the state space (e.g., market indicators)
    "action_space_dim": 3,                   # Number of possible actions (e.g., Buy, Sell, Hold)
    "batch_size": 64,                        # Batch size for training
    "ppo_clip_value": 0.2                    # Clipping value for the PPO loss function
}

SENTIMENT_ANALYSIS_SETTINGS = {
    "sentiment_sources": ["newsapi", "bloomberg"],
    "news_api_key": "your_news_api_key_here",
    "bloomberg_api_key": "your_bloomberg_api_key_here"
}

# Shared Model Logging Settings
LOGGING_LEVEL = "INFO"

