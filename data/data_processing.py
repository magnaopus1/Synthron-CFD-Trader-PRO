import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import logging
from models.feature_selection import (
    MutualInfoFeatureSelector,
    PCAFeatureReducer,
    RFEFeatureSelector,
)

# Configure logging
logging.basicConfig(
    format="%(asctime)s [%(levelname)s]: %(message)s",
    level=logging.INFO,
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

class DataProcessing:
    """Class for data cleaning, normalization, preprocessing, and feature engineering."""

    def __init__(self):
        """
        Initialize feature selection and reduction models.
        """
        self.feature_selection_models = {
            "MutualInfo": MutualInfoFeatureSelector(),
            "PCA": PCAFeatureReducer(),
            "RFE": RFEFeatureSelector(),
        }

    @staticmethod
    def validate_dataframe(data: pd.DataFrame):
        """
        Validate if the input is a valid pandas DataFrame.
        """
        if not isinstance(data, pd.DataFrame):
            logger.error("Input data must be a pandas DataFrame.")
            raise ValueError("Input data must be a pandas DataFrame.")

    def clean_data(self, data: pd.DataFrame, fill_method: str = 'ffill', handle_outliers: bool = False) -> pd.DataFrame:
        """
        Clean the input data by handling missing values, duplicates, and outliers.
        :param data: Input DataFrame.
        :param fill_method: Method for filling missing values ('ffill', 'bfill', or 'mean').
        :param handle_outliers: Whether to handle outliers (replace with NaN).
        :return: Cleaned DataFrame.
        """
        self.validate_dataframe(data)
        logger.info("Cleaning data: Handling missing values, duplicates, and outliers.")
        
        # Remove duplicates
        data = data.drop_duplicates()

        # Handle outliers
        if handle_outliers:
            z_scores = np.abs((data - data.mean()) / data.std())
            data[z_scores > 3] = np.nan
            logger.info("Outliers replaced with NaN based on Z-score.")

        # Fill missing values
        if fill_method == 'mean':
            data = data.fillna(data.mean())
        elif fill_method in ['ffill', 'bfill']:
            data = data.fillna(method=fill_method)
        else:
            logger.error("Invalid fill_method. Choose 'ffill', 'bfill', or 'mean'.")
            raise ValueError("Invalid fill_method. Choose 'ffill', 'bfill', or 'mean'.")
        
        return data

    def normalize_data(self, data: pd.DataFrame, method: str = 'minmax', feature_range=(0, 1)) -> pd.DataFrame:
        """
        Normalize the input data using the specified method.
        :param data: Input DataFrame.
        :param method: Normalization method ('minmax' or 'zscore').
        :param feature_range: Feature range for MinMaxScaler (used only with 'minmax').
        :return: Normalized DataFrame.
        """
        self.validate_dataframe(data)
        logger.info(f"Normalizing data using {method} method.")
        
        if method == 'minmax':
            scaler = MinMaxScaler(feature_range=feature_range)
            normalized_data = pd.DataFrame(scaler.fit_transform(data), columns=data.columns, index=data.index)
        elif method == 'zscore':
            scaler = StandardScaler()
            normalized_data = pd.DataFrame(scaler.fit_transform(data), columns=data.columns, index=data.index)
        else:
            logger.error("Invalid normalization method. Choose 'minmax' or 'zscore'.")
            raise ValueError("Invalid normalization method. Choose 'minmax' or 'zscore'.")
        
        return normalized_data

    def select_features(self, X: pd.DataFrame, y: pd.Series, method: str = 'MutualInfo', n_features: int = 10) -> pd.DataFrame:
        """
        Perform feature selection using the specified method.
        :param X: Input features DataFrame.
        :param y: Target variable Series.
        :param method: Feature selection method ('MutualInfo', 'PCA', 'RFE').
        :param n_features: Number of features to select (for applicable methods).
        :return: DataFrame with selected features.
        """
        self.validate_dataframe(X)
        logger.info(f"Selecting features using {method} method.")
        
        if method not in self.feature_selection_models:
            logger.error(f"Invalid feature selection method: {method}. Choose 'MutualInfo', 'PCA', or 'RFE'.")
            raise ValueError(f"Invalid feature selection method: {method}")
        
        model = self.feature_selection_models[method]
        selected_features = model.select_features(X, y, n_features=n_features)
        logger.info(f"Selected features using {method}: {selected_features.columns.tolist()}")
        return selected_features

    def preprocess_data(self, data: pd.DataFrame, target_column: str, normalize: bool = True) -> tuple[pd.DataFrame, pd.Series]:
        """
        Preprocess the data for machine learning models.
        :param data: Input DataFrame.
        :param target_column: The target column for prediction.
        :param normalize: Whether to normalize the features.
        :return: Tuple of features (X) and target (y).
        """
        self.validate_dataframe(data)
        
        if target_column not in data.columns:
            logger.error(f"Target column {target_column} not found in data.")
            raise ValueError(f"Target column {target_column} not found in data.")
        
        logger.info("Preprocessing data for model training.")
        y = data[target_column]
        X = data.drop(columns=[target_column])

        if normalize:
            logger.info("Normalizing feature data.")
            X = self.normalize_data(X)
        
        return X, y

    def add_custom_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Add custom features to the dataset.
        :param data: Input DataFrame.
        :return: DataFrame with added custom features.
        """
        self.validate_dataframe(data)
        logger.info("Adding custom features to the dataset.")
        
        if 'Close' in data.columns:
            data['Log Return'] = np.log(data['Close'] / data['Close'].shift(1))
            data['Skewness'] = data['Close'].rolling(window=14).skew()
            data['Kurtosis'] = data['Close'].rolling(window=14).kurt()
            logger.info("Added 'Log Return', 'Skewness', and 'Kurtosis'.")

        if 'Volume' in data.columns:
            data['Volume Change'] = data['Volume'].pct_change()
            logger.info("Added 'Volume Change'.")
        
        return data

    def generate_lag_features(self, data: pd.DataFrame, columns: list, lags: int) -> pd.DataFrame:
        """
        Generate lag features for time series data.
        :param data: Input DataFrame.
        :param columns: List of columns to generate lag features for.
        :param lags: Number of lags to generate.
        :return: DataFrame with lag features.
        """
        self.validate_dataframe(data)
        logger.info(f"Generating lag features for columns: {columns}")
        
        for col in columns:
            for lag in range(1, lags + 1):
                data[f"{col}_lag{lag}"] = data[col].shift(lag)
                logger.info(f"Added lag feature: {col}_lag{lag}")
        
        return data
