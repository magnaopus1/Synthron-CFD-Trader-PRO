"""
Classification Module

This module provides classification models for predicting market states and making trading decisions.
Supported models:
- Logistic Regression: Binary classification for buy/sell signals.
- Random Forest: Ensemble learning method for robust classification.
- Gradient Boosting: Advanced classification using boosting algorithms.

Module Usage:
1. Import the required model:
    from models.classification import (
        LogisticRegressionModel,
        RandomForestModel,
        GradientBoostingModel,
    )

2. Initialize the model with appropriate configurations:
    logistic_model = LogisticRegressionModel()
    random_forest_model = RandomForestModel()
    gradient_boosting_model = GradientBoostingModel()

3. Train the model on labeled data:
    logistic_model.train(X_train, y_train)
    random_forest_model.train(X_train, y_train)

4. Make predictions or classify signals:
    probabilities = logistic_model.predict_proba(X_test)
    signals = random_forest_model.classify_signals(probabilities)

5. Save and load models for persistent usage:
    logistic_model.save_model("path/to/save")
    random_forest_model.load_model("path/to/model")
"""

from .logistic_regression import LogisticRegressionModel
from .random_forest import RandomForestModel
from .gradient_boosting import GradientBoostingModel

__all__ = [
    "LogisticRegressionModel",
    "RandomForestModel",
    "GradientBoostingModel",
]

# Logging
import logging
from models.config.ml_settings import LOGGING_LEVEL

logging.basicConfig(level=LOGGING_LEVEL)
logger = logging.getLogger(__name__)
logger.info("Classification module initialized.")
