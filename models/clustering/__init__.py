"""
Clustering Module

This module provides clustering models for identifying patterns in trading data, detecting anomalies, and grouping
similar market behaviors.

Supported models:
- DBSCAN: Density-based clustering for anomaly detection.
- GMM: Gaussian Mixture Model for probabilistic clustering and volatility pattern detection.
- KMeans: K-means clustering for identifying similar trading patterns.

Module Usage:
1. Import the required model:
    from models.clustering import (
        DBSCANModel,
        GMMModel,
        KMeansModel,
    )

2. Initialize the model with appropriate configurations:
    dbscan_model = DBSCANModel()
    gmm_model = GMMModel()
    kmeans_model = KMeansModel()

3. Fit the model and predict clusters or anomalies:
    dbscan_model.fit_predict(X_train)
    gmm_model.fit(X_train)
    kmeans_model.fit(X_train)

4. Save and load models for reuse:
    dbscan_model.save_model("path/to/dbscan_model.pkl")
    gmm_model.load_model("path/to/gmm_model.pkl")
"""

from .dbscan import DBSCANModel
from .gmm import GMMModel
from .kmeans import KMeansModel

__all__ = [
    "DBSCANModel",
    "GMMModel",
    "KMeansModel",
]

# Logging
import logging
from models.config.ml_settings import LOGGING_LEVEL

logging.basicConfig(level=LOGGING_LEVEL)
logger = logging.getLogger(__name__)
logger.info("Clustering module initialized.")
