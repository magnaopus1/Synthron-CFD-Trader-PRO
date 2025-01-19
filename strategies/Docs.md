# CFD Trading System - Strategies Documentation

## Overview

The strategies package in the CFD Trading System manages various aspects of trading strategies, including **Entry/Exit**, **Position Management**, **Risk Management**, and **Strategy Selection**. This package integrates advanced Machine Learning Models (MLMs), forecasting methods, anomaly detection, and reinforcement learning to dynamically identify opportunities, manage risk, and optimize trading strategies.

### Key Components

- **EntryExitStrategy**: Defines rules for entering and exiting positions based on market indicators, forecasting models, and reinforcement learning.
- **PositionManagement**: Handles position scaling, trailing stops, profit-locking mechanisms, and anomaly detection.
- **RiskManagement**: Manages risk, stop-loss, take-profit, position sizing, and advanced anomaly detection integrated with forecasting and optimization models.
- **StrategySelector**: Dynamically selects and executes multiple strategies based on market conditions, sentiment analysis, and clustering results.

---

## 1. EntryExitStrategy

### Purpose:
The **EntryExitStrategy** class manages the logic for entering and exiting trades. It uses a combination of technical indicators, forecasting models, and reinforcement learning to determine when to buy or sell an asset.

### Functions:

#### `__init__(self, max_exposure_per_asset=0.01, sharpe_ratio_target=3)`
- **max_exposure_per_asset**: Maximum percentage of total portfolio risk allowed for each trade.
- **sharpe_ratio_target**: Target Sharpe ratio to evaluate the profitability of strategies.

#### `forecast_prices(self, data: pd.Series, model_name: str, forecast_steps: int = 5)`
- **Purpose**: Generate price forecasts using specified forecasting models (ARIMA, GRU, LSTM, Transformer).
- **Logic**: Predict future price movements for strategy validation.

#### `apply_rl_model(self, state, model_name: str)`
- **Purpose**: Use reinforcement learning models (ActorCritic, PPO, DQN) to predict actions.
- **Logic**: Leverages RL models to optimize trade decisions dynamically.

#### `trend_following(self, data: pd.Series, model_name: str, rl_model: str, period: int = 50, confirmation_period: int = 200)`
- **Purpose**: Identifies trends using moving averages and RL confirmation.
- **Logic**: Combines forecast and RL model predictions with SMA crossovers for signals.

#### `mean_reversion(self, data: pd.Series, model_name: str, rl_model: str, z_window: int = 20, confirmation_rsi: int = 14)`
- **Purpose**: Detects mean reversion opportunities using Z-score, RSI, and forecasting.
- **Logic**: Validates signals with forecasting and RL predictions.

#### `breakout_strategy(self, data: pd.Series, model_name: str, rl_model: str, period: int = 14, confirmation_ema: int = 20)`
- **Purpose**: Identifies breakout opportunities using Bollinger Bands and EMA with RL validation.

#### `momentum_strategy(self, data: pd.Series, model_name: str, rl_model: str, period: int = 14, confirmation_z: int = 20)`
- **Purpose**: Executes momentum trades based on RSI, Z-score, and forecasting.

#### `scalping_strategy(self, data: pd.Series, model_name: str, rl_model: str, period_fast: int = 5, period_slow: int = 20, confirmation_rsi: int = 14)`
- **Purpose**: Implements scalping with fast/slow EMAs and RL.

#### `cointegration_strategy(self, series1: pd.Series, series2: pd.Series, model_name: str, rl_model: str, confirmation_z: int = 20)`
- **Purpose**: Executes pair trading using cointegration and forecasting confirmation.

#### `combine_strategies(self, data: pd.DataFrame, pairwise_data: dict = None)`
- **Purpose**: Combines single-asset and pairwise strategies for enhanced decision-making.

#### `execute_trade(self, symbol: str, signal: str, current_exposure: float)`
- **Purpose**: Executes trade actions based on validated signals.

---

## 2. PositionManagement

### Purpose:
The **PositionManagement** class handles position scaling, trailing stops, and profit-locking mechanisms integrated with forecasting models for enhanced risk control.

### Functions:

#### `__init__(self, trailing_stop_buffer=0.01, scale_in_threshold=0.005, scale_out_threshold=0.01)`
- **trailing_stop_buffer**: Adjusts trailing stops dynamically.
- **scale_in_threshold**: Determines thresholds for scaling into positions.
- **scale_out_threshold**: Sets thresholds for scaling out.

#### `scale_in(self, current_price, entry_price, current_position, max_position, scale_step=0.1)`
- **Purpose**: Scales into a position when conditions improve, validated by forecasting.

#### `scale_out(self, current_price, entry_price, current_position, min_position, scale_step=0.1)`
- **Purpose**: Reduces exposure when conditions deteriorate.

#### `apply_trailing_stop(self, current_price, trailing_stop_price, direction="long")`
- **Purpose**: Adjusts trailing stops dynamically based on forecasting and RL.

#### `lock_profit(self, current_price, entry_price, position_size, lock_threshold=0.02)`
- **Purpose**: Locks profits by closing partial positions at predefined thresholds.

#### `partial_closing(self, current_price, entry_price, position_size, levels=None)`
- **Purpose**: Implements partial closures at multiple profit levels.

---

## 3. RiskManagement

### Purpose:
The **RiskManagement** class employs advanced risk control methods, including stop-loss, take-profit, position sizing, anomaly detection, and dynamic optimization.

### Functions:

#### `__init__(self, account_balance, leverage, max_drawdown, risk_per_trade, default_lot_size)`
- **Parameters**:
  - **account_balance**: Current balance.
  - **leverage**: Leverage applied.
  - **max_drawdown**: Maximum allowable drawdown.
  - **risk_per_trade**: Risk allocated per trade.
  - **default_lot_size**: Default trade size.

#### `forecast_prices(self, data, model_name, forecast_steps=5)`
- **Purpose**: Generates forecasts using models for risk management.

#### `validate_trade_conditions(self, spread, min_spread_threshold, max_spread_threshold, current_open_trades, max_open_trades)`
- **Purpose**: Validates trade conditions such as spreads and trade limits.

#### `calculate_position_size(self, stop_loss_pips, pip_value)`
- **Purpose**: Determines position size based on risk parameters.

#### `calculate_stop_loss(self, entry_price, direction, stop_loss_buffer)`
- **Purpose**: Calculates stop-loss levels dynamically.

#### `calculate_take_profit(self, entry_price, direction, take_profit_buffer)`
- **Purpose**: Determines take-profit levels based on market conditions.

#### `detect_anomalies(self, feature_vector)`
- **Purpose**: Identifies trade anomalies using Autoencoder, Isolation Forest, and One-Class SVM models.

#### `dynamic_risk_adjustment(self, market_conditions, bounds, optimizer_name="BayesianOptimization")`
- **Purpose**: Dynamically adjusts risk parameters using optimization.

---

## 4. StrategySelector

### Purpose:
The **StrategySelector** class dynamically selects and executes strategies based on market conditions, sentiment analysis, and clustering results.

### Functions:

#### `__init__(self, strategies, max_concurrent_strategies=5)`
- **strategies**: Dictionary of strategy objects.
- **max_concurrent_strategies**: Maximum concurrent strategies.

#### `select_strategy(self, market_condition, time_frame, pairwise=False, series1=None, series2=None, sentiment_data=None)`
- **Purpose**: Selects strategies based on market conditions and additional features like sentiment and clustering.

#### `execute_strategy(self, strategy_name, asset_identifier, asset_data, time_frame)`
- **Purpose**: Executes a specific strategy for an asset.

#### `run_concurrent_strategies(self, asset_identifier, asset_data, selected_strategies, time_frame)`
- **Purpose**: Runs multiple strategies concurrently.

#### `run_multiple_assets(self, assets_data, market_conditions, time_frames, pairwise_data=None)`
- **Purpose**: Executes strategies for multiple assets and pairs simultaneously.

---

## Conclusion

This comprehensive documentation outlines the full set of functionalities available in the strategies package of the CFD Trading System. The integration of machine learning models, forecasting, and reinforcement learning ensures a robust, adaptive, and scalable trading environment. The package is designed to adapt dynamically to varying market conditions while maintaining rigorous risk controls and optimizing execution efficiency.
