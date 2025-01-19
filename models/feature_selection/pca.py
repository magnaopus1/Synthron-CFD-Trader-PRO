from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import logging
from utils.exception_handler import log_exception
from models.config.ml_settings import PCA_SETTINGS

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PCAFeatureReducer:
    def __init__(self):
        """
        Initialize the PCA-based dimensionality reducer using configuration settings.
        """
        try:
            # Load variance retention ratio from settings
            self.variance_ratio = PCA_SETTINGS["variance_ratio"]
            logger.info(f"PCA Feature Reducer initialized with variance_ratio={self.variance_ratio}.")
        except KeyError as e:
            logger.error(f"Configuration key missing: {e}")
            raise

    def reduce_dimensions(self, X):
        """
        Reduce dimensions of the dataset using PCA.
        :param X: Feature matrix (Pandas DataFrame or NumPy array).
        :return: Transformed dataset, PCA components, and explained variance ratio.
        """
        try:
            # Determine feature names if input is a DataFrame
            feature_names = X.columns if isinstance(X, pd.DataFrame) else None

            # Standardize features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            # Initialize PCA with the specified variance ratio
            pca = PCA(n_components=self.variance_ratio)
            X_reduced = pca.fit_transform(X_scaled)

            logger.info(f"PCA dimensionality reduction completed. "
                        f"Explained variance: {np.sum(pca.explained_variance_ratio_):.2f}")

            # Return reduced data, components, and explained variance
            return {
                "reduced_data": pd.DataFrame(X_reduced, columns=[f"PC{i+1}" for i in range(X_reduced.shape[1])]),
                "components": pd.DataFrame(pca.components_, columns=feature_names),
                "explained_variance": pca.explained_variance_ratio_.tolist()
            }
        except Exception as e:
            logger.error("PCA dimensionality reduction failed.")
            log_exception(e)
            raise

    def get_explained_variance(self, X):
        """
        Compute the explained variance ratio for each principal component.
        :param X: Feature matrix (Pandas DataFrame or NumPy array).
        :return: Explained variance ratio as a list.
        """
        try:
            # Standardize features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            # Initialize PCA and fit to data
            pca = PCA()
            pca.fit(X_scaled)

            explained_variance = pca.explained_variance_ratio_
            logger.info("PCA explained variance ratio computed for all components.")
            return explained_variance.tolist()
        except Exception as e:
            logger.error("Failed to compute explained variance ratio.")
            log_exception(e)
            raise
