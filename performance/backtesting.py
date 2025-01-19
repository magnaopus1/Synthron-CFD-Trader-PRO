import pandas as pd
import logging
import MetaTrader5 as mt5  # Ensure MetaTrader5 Python API is installed
from models.forecasting import ARIMAModel, GRUModel, LSTMModel, TransformerModel
from models.optimization import BayesianOptimization, GeneticAlgorithm, ParticleSwarmOptimization
from models.reinforcement_learning import ActorCritic, DQN, PPO

class BacktestingEngine:
    def __init__(self, historical_data, strategy, initial_balance, commission=0.0005, spread=0.0002, slippage=0.0001):
        """
        Initialize the BacktestingEngine.

        :param historical_data: DataFrame containing historical price data (OHLCV).
        :param strategy: Callable that generates signals (buy/sell) based on the data.
        :param initial_balance: Starting balance for the backtest.
        :param commission: Commission per trade as a fraction of the trade value.
        :param spread: Bid-ask spread as a fraction of the price.
        :param slippage: Slippage as a fraction of the price.
        """
        self.data = self._validate_data(historical_data)
        self.strategy = strategy
        self.initial_balance = initial_balance
        self.commission = commission
        self.spread = spread
        self.slippage = slippage
        self.trades = []
        self.balance_history = []
        self.current_balance = initial_balance
        self.current_position = 0
        self.logger = logging.getLogger("BacktestingEngine")
        logging.basicConfig(level=logging.INFO)

        # Initialize forecasting models
        self.forecasting_models = {
            "ARIMA": ARIMAModel(),
            "GRU": GRUModel(),
            "LSTM": LSTMModel(),
            "Transformer": TransformerModel(),
        }

        # Initialize optimization models
        self.optimizers = {
            "BayesianOptimization": BayesianOptimization(),
            "GeneticAlgorithm": GeneticAlgorithm(),
            "ParticleSwarmOptimization": ParticleSwarmOptimization(),
        }

        # Initialize reinforcement learning models
        self.rl_models = {
            "ActorCritic": ActorCritic(),
            "DQN": DQN(),
            "PPO": PPO(),
        }

    @staticmethod
    def _validate_data(data):
        """
        Validate and preprocess the historical data.
        :param data: Input DataFrame.
        :return: Cleaned DataFrame.
        """
        required_columns = {"Open", "High", "Low", "Close", "Volume"}
        if not required_columns.issubset(data.columns):
            raise ValueError(f"Historical data must contain columns: {required_columns}")
        return data.dropna()

    def forecast_prices(self, data, model_name="ARIMA", forecast_steps=5):
        """
        Generate price forecasts using the specified forecasting model.

        :param data: Input price series (e.g., "Close").
        :param model_name: Forecasting model to use ('ARIMA', 'GRU', 'LSTM', 'Transformer').
        :param forecast_steps: Number of steps to forecast into the future.
        :return: Forecasted price value.
        """
        if model_name not in self.forecasting_models:
            self.logger.error(f"Invalid model name: {model_name}. Choose from {list(self.forecasting_models.keys())}.")
            raise ValueError(f"Invalid model name: {model_name}")

        model = self.forecasting_models[model_name]
        try:
            self.logger.info(f"Generating forecasts using {model_name} for {forecast_steps} steps.")
            forecast = model.forecast(data, forecast_steps)
            return forecast[-1]  # Return the last forecasted value
        except Exception as e:
            self.logger.error(f"Error generating forecast with {model_name}: {e}")
            return None

    def apply_rl_model(self, state, model_name="ActorCritic"):
        """
        Apply reinforcement learning model to determine actions.

        :param state: Current market state as a dictionary.
        :param model_name: Reinforcement learning model name ('ActorCritic', 'DQN', 'PPO').
        :return: Predicted action (e.g., "BUY", "SELL", "HOLD").
        """
        if model_name not in self.rl_models:
            self.logger.error(f"Invalid RL model name: {model_name}. Choose from {list(self.rl_models.keys())}.")
            raise ValueError(f"Invalid RL model name: {model_name}")

        model = self.rl_models[model_name]
        try:
            self.logger.info(f"Applying RL model {model_name}.")
            action = model.predict(state)
            return action
        except Exception as e:
            self.logger.error(f"Error applying RL model {model_name}: {e}")
            return "HOLD"

    def optimize_strategy(self, objective_function, optimizer_name="BayesianOptimization", bounds=None, iterations=50):
        """
        Optimize the strategy parameters using the specified optimization model.

        :param objective_function: Function to evaluate the strategy performance.
        :param optimizer_name: Optimization model to use ('BayesianOptimization', 'GeneticAlgorithm', 'ParticleSwarmOptimization').
        :param bounds: Parameter bounds for the optimizer.
        :param iterations: Number of optimization iterations.
        :return: Optimized parameters and the best score.
        """
        if optimizer_name not in self.optimizers:
            self.logger.error(f"Invalid optimizer name: {optimizer_name}. Choose from {list(self.optimizers.keys())}.")
            raise ValueError(f"Invalid optimizer name: {optimizer_name}")

        optimizer = self.optimizers[optimizer_name]
        try:
            self.logger.info(f"Optimizing strategy using {optimizer_name}.")
            best_params, best_score = optimizer.optimize(objective_function, bounds, iterations)
            self.logger.info(f"Optimization completed. Best parameters: {best_params}, Best score: {best_score}")
            return best_params, best_score
        except Exception as e:
            self.logger.error(f"Error during optimization with {optimizer_name}: {e}")
            return None, None

    def _execute_trade(self, signal, price, timestamp):
        """
        Execute a trade based on the signal.

        :param signal: Trade signal (positive for buy, negative for sell).
        :param price: Current market price.
        :param timestamp: Time of the trade execution.
        """
        trade_price = price * (1 + self.spread + self.slippage)
        trade_value = trade_price * abs(signal)
        commission_cost = trade_value * self.commission
        total_cost = trade_value + commission_cost

        if signal > 0:  # Buy
            if self.current_balance >= total_cost:
                self.current_balance -= total_cost
                self.current_position += signal
                self.logger.info(f"[{timestamp}] Buy: {signal} units at {trade_price:.4f}, Balance: {self.current_balance:.2f}")
            else:
                self.logger.warning(f"[{timestamp}] Insufficient balance for buy order.")
        elif signal < 0:  # Sell
            if abs(signal) <= self.current_position:
                self.current_balance += trade_value - commission_cost
                self.current_position += signal
                self.logger.info(f"[{timestamp}] Sell: {abs(signal)} units at {trade_price:.4f}, Balance: {self.current_balance:.2f}")
            else:
                self.logger.warning(f"[{timestamp}] Insufficient holdings for sell order.")

        self.trades.append({
            "timestamp": timestamp,
            "signal": signal,
            "price": trade_price,
            "cost": total_cost,
            "balance": self.current_balance,
            "position": self.current_position,
        })

    def run_backtest(self, model_name="ARIMA", rl_model="ActorCritic", forecast_steps=5):
        """
        Run the backtest using the strategy on the historical data with integrated forecasting and RL.

        :param model_name: Forecasting model to use ('ARIMA', 'GRU', 'LSTM', 'Transformer').
        :param rl_model: Reinforcement learning model to use ('ActorCritic', 'DQN', 'PPO').
        :param forecast_steps: Number of steps to forecast into the future.
        """
        self.logger.info("Starting backtest...")
        for index, row in self.data.iterrows():
            forecast_price = self.forecast_prices(row['Close'], model_name, forecast_steps)
            state = {
                "current_price": row['Close'],
                "volume": row['Volume'],
                "forecast_price": forecast_price,
            }
            rl_action = self.apply_rl_model(state, rl_model)
            signal = self.strategy(row, rl_action)
            if signal:
                self._execute_trade(signal, row['Close'], index)
            self.balance_history.append({
                "timestamp": index,
                "balance": self.current_balance,
                "position": self.current_position,
            })

        self.logger.info("Backtest completed.")
        return pd.DataFrame(self.trades), pd.DataFrame(self.balance_history)

    @staticmethod
    def load_historical_data_from_csv(file_path):
        """
        Load historical data from a CSV file.
        :param file_path: Path to the CSV file.
        :return: DataFrame with historical data.
        """
        try:
            data = pd.read_csv(file_path, parse_dates=True, index_col="timestamp")
            logging.info(f"Loaded historical data from {file_path}.")
            return data
        except Exception as e:
            raise RuntimeError(f"Error loading historical data: {e}")

    @staticmethod
    def load_historical_data_from_mt5(symbol, timeframe, start, end):
        """
        Fetch historical data from the MetaTrader 5 API.
        :param symbol: Trading symbol (e.g., "EURUSD").
        :param timeframe: MT5 timeframe (e.g., mt5.TIMEFRAME_H1).
        :param start: Start date (datetime).
        :param end: End date (datetime).
        :return: DataFrame with historical data.
        """
        if not mt5.initialize():
            raise RuntimeError("MetaTrader 5 initialization failed.")

        rates = mt5.copy_rates_range(symbol, timeframe, start, end)
        if rates is None:
            raise RuntimeError(f"Failed to fetch data for {symbol} from MT5.")
        data = pd.DataFrame(rates)
        data['timestamp'] = pd.to_datetime(data['time'], unit='s')
        data.set_index('timestamp', inplace=True)
        logging.info(f"Loaded historical data for {symbol} from MT5.")
        return data
