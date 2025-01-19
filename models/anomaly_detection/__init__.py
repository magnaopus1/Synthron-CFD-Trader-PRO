"""
Anomaly Detection Module

This module provides machine learning models for detecting anomalies in trading data.
Supported models:
- Autoencoder: Deep learning-based anomaly detection using reconstruction errors.
- Isolation Forest: Tree ensemble-based anomaly detection for high-dimensional data.
- One-Class SVM: Anomaly detection using support vector machines.

Each model is designed for real-world trading system integration and adheres to production-level standards.

Module Usage:
1. Import the required model:
    from models.anomaly_detection import AutoencoderModel, IsolationForestModel, OneClassSVMModel

2. Initialize the model with appropriate configurations:
    autoencoder = AutoencoderModel()
    isolation_forest = IsolationForestModel()
    one_class_svm = OneClassSVMModel()

3. Train the model on trading data:
    autoencoder.train(X_train, X_val)
    isolation_forest.train(X_train)
    one_class_svm.train(X_train)

4. Make predictions or evaluate anomalies:
    predictions = autoencoder.predict(X_test)
    scores = one_class_svm.evaluate(X_test)
    anomalies = isolation_forest.predict(X_test)

5. Save and load models for persistent usage:
    autoencoder.save_model("path/to/save")
    isolation_forest.load_model("path/to/model")
"""

from .autoencoder import Autoencoder
from .isolation_forest import IsolationForestModel
from .one_class_svm import OneClassSVMModel

__all__ = [
    "Autoencoder",
    "IsolationForestModel",
    "OneClassSVMModel",
]

# Logging
import logging
from models.config.ml_settings import LOGGING_LEVEL

logging.basicConfig(level=LOGGING_LEVEL)
logger = logging.getLogger(__name__)
logger.info("Anomaly Detection module initialized.")
