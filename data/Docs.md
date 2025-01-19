# Data Package Documentation

The `data` package provides a robust framework for handling data in trading systems. It integrates advanced tools for fetching, cleaning, preprocessing, and transforming raw market data. The package also includes modules for generating technical indicators and leveraging Machine Learning Models (MLMs) for feature selection, sentiment analysis, and advanced analytics.

---

## Table of Contents

1. [Data Loader (`data_loader.py`)](#data-loader)
2. [Data Processing (`data_processing.py`)](#data-processing)
3. [Indicators (`indicators.py`)](#indicators)
4. [Machine Learning Models Integration](#machine-learning-models-integration)

---

## Data Loader

### File: `data_loader.py`

The `data_loader` module provides efficient tools for fetching market data using the MetaTrader 5 (MT5) API. It supports real-time data retrieval, historical data fetching, and sentiment analysis integration.

#### Key Features

1. **Concurrent Data Fetching**:
   - Multi-threaded support for fetching live data for multiple symbols simultaneously.
2. **Real-Time Market Sentiment**:
   - Integrated with Vader and BERT models for analyzing market sentiment from custom text or live price movements.
3. **Account and Symbol Management**:
   - Retrieves account information, symbol details, and margin requirements.

#### Classes and Methods

- `DataLoader`: Manages API connections and fetches data.
  - `fetch_symbol_info(symbol)`: Retrieves metadata for a trading symbol.
  - `fetch_historical_data(symbol, timeframe, start, end)`: Retrieves historical OHLC data.
  - `fetch_live_data(symbols)`: Fetches live tick data for multiple symbols concurrently.
  - `perform_sentiment_analysis(text_data)`: Analyzes text sentiment using integrated Vader and BERT models.
  - `analyze_live_sentiment(symbols)`: Performs sentiment analysis on live data.

---

## Data Processing

### File: `data_processing.py`

The `data_processing` module is designed for preparing data for machine learning models and trading strategies. It includes advanced preprocessing techniques such as lag feature generation, outlier detection, and custom feature engineering.

#### Key Features

1. **Advanced Feature Engineering**:
   - Custom features like log returns, skewness, and kurtosis are calculated for better market insights.
2. **Dynamic Normalization**:
   - Supports Min-Max scaling and Z-score normalization.
3. **Machine Learning Feature Selection**:
   - Integrated with models like Mutual Information, PCA, and RFE for dynamic feature selection.

#### Classes and Methods

- `DataProcessing`: Handles data cleaning, normalization, and feature engineering.
  - `clean_data(data, fill_method, handle_outliers)`: Removes duplicates, fills missing values, and handles outliers.
  - `normalize_data(data, method, feature_range)`: Normalizes data using specified methods.
  - `add_custom_features(data)`: Adds advanced features like log returns and skewness.
  - `generate_lag_features(data, columns, lags)`: Creates lagged features for time-series analysis.
  - `select_features(X, y, method, n_features)`: Uses ML models to select the most relevant features.

---

## Indicators

### File: `indicators.py`

The `indicators` module provides a comprehensive suite of technical indicators, essential for developing, analyzing, and testing trading strategies. It includes statistical, momentum-based, and trend-following indicators to support various trading models and strategies.

---

#### Key Features

1. **Technical Indicators**:
   - Simple Moving Average (SMA), Exponential Moving Average (EMA), Bollinger Bands, MACD, RSI, and more.
2. **Statistical Analytics**:
   - Includes tools for Z-score calculation, correlation matrix analysis, and cointegration testing.
3. **Momentum and Volume Indicators**:
   - Stochastic Oscillator and On-Balance Volume (OBV) for analyzing market momentum and volume trends.

---

#### Classes and Methods

- **`Indicators`**: Computes a wide range of technical indicators for financial data.

  - **Moving Average**:
    - `moving_average(data, period)`: Computes the Simple Moving Average (SMA) over the specified period.

  - **Exponential Moving Average**:
    - `exponential_moving_average(data, period)`: Computes the Exponential Moving Average (EMA) for a given period.

  - **Bollinger Bands**:
    - `bollinger_bands(data, period, std_dev)`: Calculates Bollinger Bands, including the Moving Average (MA), Upper Band, and Lower Band, using the specified standard deviation.

  - **Relative Strength Index (RSI)**:
    - `relative_strength_index(data, period)`: Calculates the RSI, a momentum oscillator that measures the speed and change of price movements.

  - **Moving Average Convergence Divergence (MACD)**:
    - `macd(data, fast_period, slow_period, signal_period)`: Calculates MACD values, including the MACD line, Signal line, and Histogram.

  - **Stochastic Oscillator**:
    - `stochastic(data, high, low, period)`: Computes the Stochastic Oscillator (%K and %D lines) to analyze price momentum relative to high-low ranges.

  - **Z-Score**:
    - `z_score(data, window)`: Calculates the Z-Score for a rolling window, identifying how far values deviate from the mean.

  - **Correlation Matrix**:
    - `correlation_matrix(data)`: Computes the correlation matrix for a given DataFrame, analyzing relationships between multiple time series.

  - **Cointegration**:
    - `cointegration(series1, series2)`: Tests for cointegration between two time-series, useful for pair trading strategies.

  - **On-Balance Volume (OBV)**:
    - `on_balance_volume(close, volume)`: Calculates the OBV, which uses price and volume to measure buying and selling pressure.


---

## Machine Learning Models Integration

The `data` package incorporates Machine Learning Models (MLMs) for advanced feature selection, sentiment analysis, and analytics.

#### Key Updates

1. **Sentiment Analysis**:
   - Integrated `VaderAnalyzer` (rule-based) and `BERTSentimentAnalyzer` (deep learning) for analyzing market sentiment.

2. **Feature Selection**:
   - Utilizes Mutual Information, PCA, and Recursive Feature Elimination (RFE) for selecting the most predictive features in a dataset.

3. **Feature Engineering**:
   - Generates features tailored for machine learning models, including lagged features, rolling statistics, and time-series transformations.

---

## Summary

The `data` package provides a comprehensive and production-ready framework for managing trading data, integrating Machine Learning Models for feature selection, and leveraging advanced technical indicators for strategy development. The updates incorporate robust sentiment analysis, multi-threaded live data fetching, and cutting-edge feature engineering for real-world trading systems.
