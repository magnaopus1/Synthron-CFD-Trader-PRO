# Performance Package Documentation

The `performance` package is designed to analyze, evaluate, and report the results of trading strategies within the CFD Trading System. It integrates advanced Machine Learning Models (MLMs) for forecasting, optimization, clustering, and regression analysis, enabling a robust framework for backtesting, calculating performance metrics, and generating insightful reports.

---

## Table of Contents

1. [Backtesting (`backtesting.py`)](#backtesting)
2. [Performance Metrics (`metrics.py`)](#performance-metrics)
3. [Reporting (`reporting.py`)](#reporting)

---

## Backtesting

### File: `backtesting.py`

The `backtesting` module simulates trading strategies on historical data, incorporating advanced features such as machine learning-based forecasting, reinforcement learning for decision-making, and optimization for parameter tuning.

#### Key Features

1. **Forecasting Models**:
   - Supports `ARIMA`, `GRU`, `LSTM`, and `Transformer` models for price forecasting.

2. **Reinforcement Learning Models**:
   - Integrates `ActorCritic`, `DQN`, and `PPO` models for dynamic action prediction.

3. **Optimization Models**:
   - Includes `Bayesian Optimization`, `Genetic Algorithm`, and `Particle Swarm Optimization` for strategy parameter tuning.

#### Classes and Methods

- **`BacktestingEngine`**:
  - Simulates trading strategies using historical data.

  **Attributes**:
  - `data`: Historical OHLCV data (DataFrame).
  - `strategy`: User-defined function generating buy/sell signals.
  - `forecasting_models`: Dictionary of forecasting models (`ARIMA`, `GRU`, `LSTM`, `Transformer`).
  - `rl_models`: Dictionary of reinforcement learning models (`ActorCritic`, `DQN`, `PPO`).
  - `optimizers`: Dictionary of optimization models (`BayesianOptimization`, `GeneticAlgorithm`, `ParticleSwarmOptimization`).
  - `trades`: List of executed trades.
  - `balance_history`: Tracks balance and position changes.

  **Methods**:
  - `_validate_data(data)`: Ensures input data has required columns (`Open`, `High`, `Low`, `Close`, `Volume`).
  - `forecast_prices(data, model_name, forecast_steps)`: Generates forecasts using specified forecasting models.
  - `apply_rl_model(state, model_name)`: Predicts actions using reinforcement learning models.
  - `optimize_strategy(objective_function, optimizer_name, bounds, iterations)`: Optimizes strategy parameters using specified optimization models.
  - `_execute_trade(signal, price, timestamp)`: Executes trades and updates balances.
  - `run_backtest(model_name, rl_model, forecast_steps)`: Runs the backtest using forecasting and RL models.
  - `load_historical_data_from_csv(file_path)`: Loads historical data from a CSV file.
  - `load_historical_data_from_mt5(symbol, timeframe, start, end)`: Fetches historical data from MetaTrader 5.

---

## Performance Metrics

### File: `metrics.py`

The `metrics` module calculates quantitative performance metrics to assess trading strategies. It integrates regression models to predict performance metrics based on historical data.

#### Key Features

1. **Regression Models**:
   - Supports `SVR`, `Random Forest Regressor`, and `DNN` models for performance prediction.

2. **Comprehensive Metrics**:
   - Calculates profitability, risk, and efficiency metrics.

3. **Visualization Tools**:
   - Provides visualizations for balance over time, drawdown, and return distributions.

#### Classes and Methods

- **`PerformanceMetrics`**:
  - Computes key performance metrics and applies regression models for predictions.

  **Attributes**:
  - `trades`: Trade details (timestamp, signal, price, balance, position).
  - `balance_history`: Balance and position data over time.
  - `risk_free_rate`: Risk-free rate for calculating Sharpe Ratio.
  - `regression_models`: Dictionary of regression models (`SVR`, `Random Forest Regressor`, `DNN`).

  **Methods**:
  - `calculate_sharpe_ratio()`: Computes the Sharpe Ratio.
  - `calculate_win_ratio()`: Calculates the percentage of profitable trades.
  - `calculate_risk_of_ruin()`: Estimates the likelihood of losing all capital.
  - `calculate_max_drawdown()`: Calculates the maximum drawdown as a percentage.
  - `calculate_avg_daily_gain()`: Computes average daily returns.
  - `calculate_avg_weekly_gain()`: Computes average weekly returns.
  - `apply_regression_models(feature_data, target_data, model_name)`: Predicts metrics using regression models.
  - `calculate_metrics_summary()`: Summarizes all metrics in a dictionary.
  - `display_metrics_table(display_prompt)`: Displays metrics as a table.
  - `plot_performance(display_prompt)`: Plots balance, drawdown, and return distributions.

---

## Reporting

### File: `reporting.py`

The `reporting` module generates summaries and visualizations of trading performance. It integrates clustering models to analyze patterns in performance and trade data.

#### Key Features

1. **Clustering Models**:
   - Includes `DBSCAN`, `GMM`, and `KMeans` models for pattern detection in trade and balance data.

2. **Historical Performance**:
   - Provides daily, weekly, and monthly performance summaries.

3. **Export Functionality**:
   - Exports reports to Excel for offline analysis.

#### Classes and Methods

- **`Reporting`**:
  - Generates and exports performance reports with clustering insights.

  **Attributes**:
  - `trades`: DataFrame containing executed trade details.
  - `balance_history`: DataFrame with balance and performance history.
  - `clustering_models`: Dictionary of clustering models (`DBSCAN`, `GMM`, `KMeans`).
  - `log_file`: Path to the log file for tracking reporting actions.

  **Methods**:
  - `log_event(message, level)`: Logs significant events or errors.
  - `apply_clustering_models(features)`: Applies clustering models to analyze trade and balance data.
  - `generate_trade_summary()`: Summarizes trading activity with clustering insights.
  - `generate_performance_table()`: Creates daily, weekly, and monthly performance tables.
  - `export_report(file_path)`: Exports the report to an Excel file.

---

## Summary

The `performance` package provides a comprehensive framework for evaluating trading strategies. The integration of Machine Learning Models enhances forecasting, optimization, regression analysis, and clustering. These advanced tools, combined with detailed reporting capabilities, ensure robust evaluation and actionable insights into trading performance.
