from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import logging
from utils.exception_handler import log_exception
from models.config.ml_settings import MUTUAL_INFO_SETTINGS

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MutualInfoFeatureSelector:
    def __init__(self):
        """
        Initialize the Mutual Information-based feature selector using configuration settings.
        """
        try:
            self.num_features = MUTUAL_INFO_SETTINGS["num_features"]
            self.random_state = MUTUAL_INFO_SETTINGS.get("random_state", 42)
            logger.info(f"Mutual Information Feature Selector initialized with num_features={self.num_features}.")
        except KeyError as e:
            logger.error(f"Configuration key missing: {e}")
            raise

    def select_features(self, X, y):
        """
        Select features based on mutual information scores.
        :param X: Feature matrix (Pandas DataFrame or NumPy array).
        :param y: Target vector (Pandas Series or NumPy array).
        :return: Selected feature names and scores.
        """
        try:
            feature_names = X.columns if isinstance(X, pd.DataFrame) else [f"Feature_{i}" for i in range(X.shape[1])]

            # Standardize features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            # Compute mutual information scores
            mi_scores = mutual_info_classif(X_scaled, y, random_state=self.random_state)
            mi_scores_series = pd.Series(mi_scores, index=feature_names).sort_values(ascending=False)

            # Select top features
            selected_features = mi_scores_series.head(self.num_features)
            logger.info(f"Selected top {self.num_features} features based on mutual information.")

            return selected_features.index.tolist(), selected_features.to_dict()
        except Exception as e:
            logger.error("Feature selection failed.")
            log_exception(e)
            raise

    def rank_features(self, X, y):
        """
        Rank all features based on their mutual information scores.
        :param X: Feature matrix (Pandas DataFrame or NumPy array).
        :param y: Target vector (Pandas Series or NumPy array).
        :return: Ranked feature names and scores.
        """
        try:
            feature_names = X.columns if isinstance(X, pd.DataFrame) else [f"Feature_{i}" for i in range(X.shape[1])]

            # Standardize features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            # Compute mutual information scores
            mi_scores = mutual_info_classif(X_scaled, y, random_state=self.random_state)
            mi_scores_series = pd.Series(mi_scores, index=feature_names).sort_values(ascending=False)

            logger.info("Ranked all features based on mutual information scores.")
            return mi_scores_series.to_dict()
        except Exception as e:
            logger.error("Feature ranking failed.")
            log_exception(e)
            raise
