# __init__.py for models/regression

import logging

# Configure logging for the regression module
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import models
from .svr_model import SVRModel
from .random_forest_regressor import RandomForestRegressorModel
from .dnn_regressor import DNNRegressor

# Log when the module is initialized
logger.info("Initializing regression models module")

# Make the regression models accessible when importing the module
__all__ = [
    "SVRModel",               # Support Vector Regression Model
    "RandomForestRegressorModel",  # Random Forest Regressor Model
    "DNNRegressor"            # Deep Neural Network Regressor Model
]

# Log available models after successful imports
logger.info("Successfully imported the following regression models: SVRModel, RandomForestRegressorModel, DNNRegressor")
