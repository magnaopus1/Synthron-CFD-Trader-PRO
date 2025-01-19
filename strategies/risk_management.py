import logging
from models.forecasting import ARIMAModel, GRUModel, LSTMModel, TransformerModel
from models.anomaly_detection import Autoencoder, IsolationForestModel, OneClassSVMModel
from models.optimization import BayesianOptimization, GeneticAlgorithm, ParticleSwarmOptimization
from models.regression import SVRModel, RandomForestRegressorModel, DNNRegressor

logger = logging.getLogger(__name__)

class RiskManagement:
    """
    Manages risk via stop-loss, take-profit, position sizing, anomaly detection, forecasting models,
    regression models, and hyperparameter optimization for enhanced decision-making.
    """

    def __init__(self, account_balance, leverage, max_drawdown, risk_per_trade, default_lot_size):
        """
        Initialize the risk management module.

        :param account_balance: Current account balance.
        :param leverage: Account leverage.
        :param max_drawdown: Maximum allowable drawdown as a percentage.
        :param risk_per_trade: Risk per trade as a percentage of account balance.
        :param default_lot_size: Default lot size for trading.
        """
        self.account_balance = account_balance
        self.leverage = leverage
        self.max_drawdown = max_drawdown
        self.risk_per_trade = risk_per_trade
        self.default_lot_size = default_lot_size

        # Initialize forecasting models
        self.forecasting_models = {
            "ARIMA": ARIMAModel(),
            "GRU": GRUModel(),
            "LSTM": LSTMModel(),
            "Transformer": TransformerModel(),
        }

        # Initialize anomaly detection models
        self.autoencoder = Autoencoder()
        self.isolation_forest = IsolationForestModel()
        self.one_class_svm = OneClassSVMModel()

        # Initialize optimization models
        self.optimizers = {
            "BayesianOptimization": BayesianOptimization(),
            "GeneticAlgorithm": GeneticAlgorithm(),
            "ParticleSwarmOptimization": ParticleSwarmOptimization(),
        }

        # Initialize regression models
        self.regression_models = {
            "SVR": SVRModel(),
            "RandomForest": RandomForestRegressorModel(),
            "DNN": DNNRegressor(),
        }

        logger.info(f"RiskManagement initialized with account balance: {self.account_balance}, leverage: {self.leverage}")

    def forecast_prices(self, data, model_name, forecast_steps=5):
        """
        Generate price forecasts using the specified forecasting or regression model.

        :param data: Input price series or feature set.
        :param model_name: The forecasting or regression model to use.
        :param forecast_steps: Number of steps to forecast into the future (for forecasting models).
        :return: Forecasted price value.
        """
        if model_name in self.forecasting_models:
            model = self.forecasting_models[model_name]
        elif model_name in self.regression_models:
            model = self.regression_models[model_name]
        else:
            logger.error(f"Invalid model name: {model_name}. Choose from forecasting or regression models.")
            raise ValueError(f"Invalid model name: {model_name}")

        try:
            logger.info(f"Generating forecasts using {model_name}.")
            if model_name in self.forecasting_models:
                forecast = model.forecast(data, forecast_steps)
            else:
                forecast = model.predict(data)  # Regression models use a `predict` method
            return forecast[-1] if isinstance(forecast, list) else forecast
        except Exception as e:
            logger.error(f"Error generating forecast with {model_name}: {e}")
            return None

    def predict_market_conditions(self, features, model_name="RandomForest"):
        """
        Use regression models to predict market conditions.

        :param features: Feature vector representing market conditions.
        :param model_name: Regression model to use ('SVR', 'RandomForest', 'DNN').
        :return: Predicted market condition value.
        """
        if model_name not in self.regression_models:
            logger.error(f"Invalid regression model name: {model_name}. Choose from {list(self.regression_models.keys())}.")
            raise ValueError(f"Invalid model name: {model_name}")

        model = self.regression_models[model_name]
        try:
            prediction = model.predict(features)
            logger.info(f"Market condition predicted using {model_name}: {prediction}")
            return prediction
        except Exception as e:
            logger.error(f"Error predicting market condition with {model_name}: {e}")
            return None

    def optimize_hyperparameters(self, objective_function, optimizer_name, bounds, iterations=50):
        """
        Optimize hyperparameters for risk management using the specified optimizer.

        :param objective_function: Callable function to minimize or maximize.
        :param optimizer_name: The optimization model to use ('BayesianOptimization', 'GeneticAlgorithm', 'ParticleSwarmOptimization').
        :param bounds: Bounds for hyperparameters.
        :param iterations: Number of iterations for optimization.
        :return: Optimized hyperparameters and score.
        """
        if optimizer_name not in self.optimizers:
            logger.error(f"Invalid optimizer name: {optimizer_name}. Choose from {list(self.optimizers.keys())}.")
            raise ValueError(f"Invalid optimizer name: {optimizer_name}")

        optimizer = self.optimizers[optimizer_name]
        try:
            logger.info(f"Optimizing hyperparameters using {optimizer_name}.")
            best_params, best_score = optimizer.optimize(objective_function, bounds, iterations)
            logger.info(f"Optimization complete. Best parameters: {best_params}, Best score: {best_score}")
            return best_params, best_score
        except Exception as e:
            logger.error(f"Error during optimization with {optimizer_name}: {e}")
            return None, None

    def validate_trade_conditions(self, spread, min_spread_threshold, max_spread_threshold, current_open_trades, max_open_trades):
        """
        Validate if a trade can be executed based on conditions.

        :param spread: Current spread of the trading pair.
        :param min_spread_threshold: Minimum acceptable spread.
        :param max_spread_threshold: Maximum acceptable spread.
        :param current_open_trades: Number of currently open trades.
        :param max_open_trades: Maximum allowable open trades.
        :return: Boolean indicating whether trade conditions are met.
        """
        if not (min_spread_threshold <= spread <= max_spread_threshold):
            logger.warning(f"Spread {spread} out of acceptable range [{min_spread_threshold}, {max_spread_threshold}].")
            return False
        if current_open_trades >= max_open_trades:
            logger.warning(f"Max open trades limit reached: {current_open_trades}/{max_open_trades}.")
            return False
        logger.info("Trade conditions validated successfully.")
        return True


    def assess_risk_with_anomalies(self, feature_vector, threshold=0.5):
        """
        Assess trade risk based on anomaly detection and return a final decision.

        :param feature_vector: A numerical vector representing trade-related features.
        :param threshold: Threshold for anomaly confidence to reject trade.
        :return: Boolean indicating whether the trade is acceptable.
        """
        anomalies = self.detect_anomalies(feature_vector)
        average_score = sum(anomalies.values()) / len(anomalies)

        if average_score > threshold:
            logger.warning(f"Anomaly score {average_score} exceeds threshold {threshold}. Trade rejected.")
            return False

        logger.info(f"Anomaly score {average_score} within threshold {threshold}. Trade approved.")
        return True

    def calculate_position_size(self, stop_loss_pips, pip_value):
        """
        Calculate the appropriate position size based on risk per trade.

        :param stop_loss_pips: Stop-loss in pips.
        :param pip_value: Value of one pip for the trading pair.
        :return: Position size in lots.
        """
        try:
            risk_amount = self.account_balance * self.risk_per_trade
            position_size = risk_amount / (stop_loss_pips * pip_value)
            logger.info(f"Calculated position size: {position_size} lots for risk amount: {risk_amount}.")
            return round(position_size, 2)
        except ZeroDivisionError:
            logger.error("Division by zero in position size calculation. Check stop_loss_pips or pip_value.")
            return 0

    def calculate_stop_loss(self, entry_price, direction, stop_loss_buffer, time_frame="1H", model_name="ARIMA"):
        """
        Calculate stop-loss price based on entry price, buffer, and forecasting.

        :param entry_price: The entry price of the trade.
        :param direction: 'long' or 'short'.
        :param stop_loss_buffer: Buffer for stop-loss in percentage.
        :param time_frame: Time frame for strategy (e.g., "1m", "5m", "1h").
        :param model_name: Forecasting model to use for confirmation.
        :return: Stop-loss price.
        """
        valid_timeframes = {
            "1m": 0.001,
            "5m": 0.002,
            "1h": 0.005,
            "4h": 0.01,
            "1D": 0.02,
        }
        buffer = valid_timeframes.get(time_frame, 0.005)
        forecast_price = self.forecast_prices(entry_price, model_name)

        if forecast_price is not None:
            if direction == "long" and forecast_price < entry_price:
                buffer += 0.001  # Add additional buffer for risk management
            elif direction == "short" and forecast_price > entry_price:
                buffer += 0.001

        stop_loss = entry_price * (1 - buffer) if direction == "long" else entry_price * (1 + buffer)
        logger.debug(f"Stop-loss calculated: {stop_loss} for direction: {direction}")
        return stop_loss

    def calculate_take_profit(self, entry_price, direction, take_profit_buffer, time_frame="1H", model_name="ARIMA"):
        """
        Calculate take-profit price based on entry price, buffer, and forecasting.

        :param entry_price: The entry price of the trade.
        :param direction: 'long' or 'short'.
        :param take_profit_buffer: Buffer for take-profit in percentage.
        :param time_frame: Time frame for strategy (e.g., "1m", "5m", "1h").
        :param model_name: Forecasting model to use for confirmation.
        :return: Take-profit price.
        """
        valid_timeframes = {
            "1m": 0.002,
            "5m": 0.005,
            "1h": 0.01,
            "4h": 0.02,
            "1D": 0.05,
        }
        buffer = valid_timeframes.get(time_frame, 0.01)
        forecast_price = self.forecast_prices(entry_price, model_name)

        if forecast_price is not None:
            if direction == "long" and forecast_price > entry_price:
                buffer += 0.002
            elif direction == "short" and forecast_price < entry_price:
                buffer += 0.002

        take_profit = entry_price * (1 + buffer) if direction == "long" else entry_price * (1 - buffer)
        logger.debug(f"Take-profit calculated: {take_profit} for direction: {direction}")
        return take_profit

    def detect_anomalies(self, feature_vector):
        """
        Detect anomalies in the provided feature vector using anomaly detection models.

        :param feature_vector: A numerical vector representing trade-related features.
        :return: Dictionary with anomaly detection results.
        """
        try:
            autoencoder_score = self.autoencoder.detect_anomaly(feature_vector)
            isolation_forest_score = self.isolation_forest.detect_anomaly(feature_vector)
            one_class_svm_score = self.one_class_svm.detect_anomaly(feature_vector)
            
            return {
                'autoencoder': autoencoder_score,
                'isolation_forest': isolation_forest_score,
                'one_class_svm': one_class_svm_score
            }
        except Exception as e:
            logger.error(f"Error during anomaly detection: {e}")
            return None

    def dynamic_risk_adjustment(self, market_conditions, bounds, optimizer_name="BayesianOptimization"):
        """
        Dynamically adjust risk parameters based on current market conditions using optimization.

        :param market_conditions: Dictionary containing market condition metrics (e.g., volatility, liquidity).
        :param bounds: Bounds for optimization variables (e.g., risk_per_trade, leverage).
        :param optimizer_name: Optimization model to use ('BayesianOptimization', 'GeneticAlgorithm', 'ParticleSwarmOptimization').
        :return: Optimized risk parameters.
        """
        def objective_function(params):
            risk_per_trade, leverage = params
            score = -1 * (market_conditions["volatility"] * risk_per_trade + market_conditions["liquidity"] * leverage)
            return score

        best_params, best_score = self.optimize_hyperparameters(objective_function, optimizer_name, bounds)
        if best_params:
            self.risk_per_trade, self.leverage = best_params
            logger.info(f"Dynamically adjusted risk parameters: risk_per_trade={self.risk_per_trade}, leverage={self.leverage}")
        else:
            logger.warning("Failed to optimize risk parameters.")

    def adaptive_position_sizing(self, market_conditions, model_name="LSTM"):
        """
        Adjust position sizing dynamically based on market conditions and forecasting.

        :param market_conditions: Dictionary containing metrics like volatility, trend strength.
        :param model_name: Forecasting model to validate conditions.
        :return: Adjusted position size.
        """
        volatility = market_conditions.get("volatility", 1.0)
        trend_strength = market_conditions.get("trend_strength", 1.0)

        forecast = self.forecast_prices(market_conditions["price_series"], model_name)
        if forecast:
            adjusted_position = self.calculate_position_size(stop_loss_pips=volatility * 10, pip_value=trend_strength * 0.1)
            logger.info(f"Adaptive position sizing calculated: {adjusted_position}")
            return adjusted_position
        else:
            logger.warning("Forecast unavailable; using default position size.")
            return self.default_lot_size

    def advanced_scaling(self, entry_price, market_conditions, time_frame="1H", model_name="GRU"):
        """
        Dynamically adjust stop-loss and take-profit based on market conditions and scaling logic.

        :param entry_price: Trade entry price.
        :param market_conditions: Dictionary containing metrics like trend, volatility.
        :param time_frame: Time frame for strategy.
        :param model_name: Forecasting model to validate scaling.
        :return: Tuple of (stop_loss, take_profit).
        """
        trend = market_conditions.get("trend", 0.02)
        volatility = market_conditions.get("volatility", 0.01)

        stop_loss = self.calculate_stop_loss(entry_price, "long", stop_loss_buffer=volatility, time_frame=time_frame, model_name=model_name)
        take_profit = self.calculate_take_profit(entry_price, "long", take_profit_buffer=trend, time_frame=time_frame, model_name=model_name)

        logger.info(f"Advanced scaling: stop_loss={stop_loss}, take_profit={take_profit}")
        return stop_loss, take_profit

    def portfolio_risk_evaluation(self, portfolio, max_correlation_threshold=0.8):
        """
        Evaluate and manage portfolio-level risk based on asset correlations and diversification.

        :param portfolio: Dictionary containing asset holdings and weights.
        :param max_correlation_threshold: Maximum acceptable correlation between assets.
        :return: Boolean indicating if the portfolio risk is acceptable.
        """
        correlations = portfolio.get("correlations")
        if correlations:
            max_correlation = correlations.max().max()
            if max_correlation > max_correlation_threshold:
                logger.warning(f"High correlation detected in portfolio: {max_correlation}")
                return False
        logger.info("Portfolio risk evaluation passed.")
        return True