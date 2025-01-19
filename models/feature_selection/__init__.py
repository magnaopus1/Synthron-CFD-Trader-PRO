"""
Feature Selection Module

This module provides various feature selection techniques for optimizing trading models by reducing dimensionality,
selecting relevant features, and improving interpretability.

Supported feature selection methods:
- Mutual Information: Select features based on mutual information scores.
- PCA: Dimensionality reduction while retaining a specified variance ratio.
- RFE: Recursive Feature Elimination for selecting features based on model importance.

Module Usage:
1. Import the required feature selector:
    from models.feature_selection import (
        MutualInfoFeatureSelector,
        PCAFeatureReducer,
        RFEFeatureSelector,
    )

2. Initialize the selector with appropriate configurations:
    mutual_info_selector = MutualInfoFeatureSelector()
    pca_reducer = PCAFeatureReducer()
    rfe_selector = RFEFeatureSelector()

3. Perform feature selection or dimensionality reduction:
    selected_features, scores = mutual_info_selector.select_features(X_train, y_train)
    reduced_data, components, variance = pca_reducer.reduce_dimensions(X_train)
    ranked_features = rfe_selector.rank_features(X_train, y_train)

4. Integrate into preprocessing pipelines for trading models.
"""

from .mutual_info import MutualInfoFeatureSelector
from .pca import PCAFeatureReducer
from .rfe import RFEFeatureSelector

__all__ = [
    "MutualInfoFeatureSelector",
    "PCAFeatureReducer",
    "RFEFeatureSelector",
]

# Logging
import logging
from models.config.ml_settings import LOGGING_LEVEL

logging.basicConfig(level=LOGGING_LEVEL)
logger = logging.getLogger(__name__)
logger.info("Feature Selection module initialized.")
