from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import logging
from utils.exception_handler import log_exception
from models.config.ml_settings import RFE_SETTINGS

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RFEFeatureSelector:
    def __init__(self):
        """
        Initialize the Recursive Feature Elimination (RFE) selector using configuration settings.
        """
        try:
            # Load configuration parameters
            self.num_features = RFE_SETTINGS["num_features"]
            self.random_state = RFE_SETTINGS.get("random_state", 42)

            # Initialize estimator for RFE
            self.estimator = RandomForestClassifier(random_state=self.random_state)
            logger.info(f"RFE Feature Selector initialized with num_features={self.num_features}, "
                        f"random_state={self.random_state}.")
        except KeyError as e:
            logger.error(f"Configuration key missing: {e}")
            raise

    def select_features(self, X, y):
        """
        Perform feature selection using Recursive Feature Elimination (RFE).
        :param X: Feature matrix (Pandas DataFrame or NumPy array).
        :param y: Target vector (Pandas Series or NumPy array).
        :return: Selected feature names and their rankings.
        """
        try:
            # Extract feature names
            feature_names = X.columns if isinstance(X, pd.DataFrame) else [f"Feature_{i}" for i in range(X.shape[1])]

            # Standardize features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            # Initialize and fit RFE
            rfe = RFE(estimator=self.estimator, n_features_to_select=self.num_features)
            rfe.fit(X_scaled, y)

            # Get selected features and rankings
            selected_features = [feature_names[i] for i in range(len(feature_names)) if rfe.support_[i]]
            feature_rankings = {feature_names[i]: rfe.ranking_[i] for i in range(len(feature_names))}

            logger.info(f"RFE selected top {self.num_features} features.")
            return selected_features, feature_rankings
        except Exception as e:
            logger.error("Feature selection using RFE failed.")
            log_exception(e)
            raise

    def rank_features(self, X, y):
        """
        Rank all features based on their importance in RFE.
        :param X: Feature matrix (Pandas DataFrame or NumPy array).
        :param y: Target vector (Pandas Series or NumPy array).
        :return: Ranked feature names and their rankings.
        """
        try:
            # Extract feature names
            feature_names = X.columns if isinstance(X, pd.DataFrame) else [f"Feature_{i}" for i in range(X.shape[1])]

            # Standardize features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            # Initialize and fit RFE
            rfe = RFE(estimator=self.estimator, n_features_to_select=self.num_features)
            rfe.fit(X_scaled, y)

            # Rank all features
            feature_rankings = {feature_names[i]: rfe.ranking_[i] for i in range(len(feature_names))}
            logger.info("RFE feature rankings computed.")
            return dict(sorted(feature_rankings.items(), key=lambda item: item[1]))
        except Exception as e:
            logger.error("Ranking features using RFE failed.")
            log_exception(e)
            raise
