import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging
from models.regression import SVRModel, RandomForestRegressorModel, DNNRegressor

class PerformanceMetrics:
    def __init__(self, trades, balance_history, risk_free_rate=0.02):
        """
        Initialize PerformanceMetrics.

        :param trades: DataFrame containing trade details (timestamp, signal, price, balance, position).
        :param balance_history: DataFrame with balance and position history over time.
        :param risk_free_rate: Risk-free rate for calculating Sharpe Ratio.
        """
        self.trades = trades
        self.balance_history = balance_history
        self.risk_free_rate = risk_free_rate
        self.logger = logging.getLogger("PerformanceMetrics")
        logging.basicConfig(level=logging.INFO)

        # Initialize regression models
        self.regression_models = {
            "SVR": SVRModel(),
            "RandomForest": RandomForestRegressorModel(),
            "DNN": DNNRegressor(),
        }

    def calculate_sharpe_ratio(self):
        """Calculate the Sharpe Ratio."""
        returns = self.balance_history['balance'].pct_change().dropna()
        excess_returns = returns - (self.risk_free_rate / 252)
        sharpe_ratio = excess_returns.mean() / excess_returns.std() * np.sqrt(252)
        return sharpe_ratio

    def calculate_win_ratio(self):
        """Calculate the Win Ratio (percentage of profitable trades)."""
        wins = self.trades[self.trades['balance'].diff() > 0]
        win_ratio = len(wins) / len(self.trades) * 100 if len(self.trades) > 0 else 0
        return win_ratio

    def calculate_risk_of_ruin(self):
        """Estimate the Risk of Ruin."""
        total_trades = len(self.trades)
        losing_trades = self.trades[self.trades['balance'].diff() < 0]
        p_loss = len(losing_trades) / total_trades if total_trades > 0 else 0
        risk_of_ruin = (p_loss / (1 - p_loss)) ** total_trades if p_loss < 1 else 100
        return risk_of_ruin * 100

    def calculate_max_drawdown(self):
        """Calculate the Maximum Drawdown."""
        balance = self.balance_history['balance']
        cumulative_max = balance.cummax()
        drawdown = (balance - cumulative_max) / cumulative_max
        max_drawdown = drawdown.min() * 100
        return max_drawdown

    def calculate_avg_daily_gain(self):
        """Calculate the average daily gain."""
        daily_returns = self.balance_history['balance'].pct_change().dropna()
        avg_daily_gain = daily_returns.mean() * 100
        return avg_daily_gain

    def calculate_avg_weekly_gain(self):
        """Calculate the average weekly gain."""
        balance = self.balance_history.set_index("timestamp").resample("W")['balance'].last()
        weekly_returns = balance.pct_change().dropna()
        avg_weekly_gain = weekly_returns.mean() * 100
        return avg_weekly_gain

    def calculate_metrics_summary(self):
        """
        Calculate and summarize all metrics in a table.
        :return: Dictionary of key metrics.
        """
        summary = {
            "Sharpe Ratio": self.calculate_sharpe_ratio(),
            "Win Ratio (%)": self.calculate_win_ratio(),
            "Risk of Ruin (%)": self.calculate_risk_of_ruin(),
            "Maximum Drawdown (%)": self.calculate_max_drawdown(),
            "Average Daily Gain (%)": self.calculate_avg_daily_gain(),
            "Average Weekly Gain (%)": self.calculate_avg_weekly_gain(),
            "Final Balance": self.balance_history['balance'].iloc[-1],
            "Total Gain (%)": ((self.balance_history['balance'].iloc[-1] / self.balance_history['balance'].iloc[0]) - 1) * 100,
        }
        return summary

    def apply_regression_models(self, feature_data, target_data, model_name="SVR"):
        """
        Apply regression models to predict performance metrics.

        :param feature_data: Features for regression model.
        :param target_data: Target data for regression model.
        :param model_name: Name of the regression model ('SVR', 'RandomForest', 'DNN').
        :return: Predicted values.
        """
        if model_name not in self.regression_models:
            self.logger.error(f"Invalid model name: {model_name}. Choose from {list(self.regression_models.keys())}.")
            raise ValueError(f"Invalid model name: {model_name}")

        model = self.regression_models[model_name]
        try:
            self.logger.info(f"Applying regression model {model_name}.")
            predictions = model.predict(feature_data)
            return predictions
        except Exception as e:
            self.logger.error(f"Error applying regression model {model_name}: {e}")
            return None

    def display_metrics_table(self, display_prompt=True):
        """
        Display key metrics as a table.
        :param display_prompt: If True, show the table in the console.
        :return: DataFrame containing metrics.
        """
        metrics = self.calculate_metrics_summary()
        metrics_df = pd.DataFrame(list(metrics.items()), columns=["Metric", "Value"])
        if display_prompt:
            print(metrics_df)
        return metrics_df

    def plot_performance(self, display_prompt=True):
        """
        Plot performance metrics over time.
        :param display_prompt: If True, display plots.
        :return: None.
        """
        if not display_prompt:
            return

        # Plot Balance Over Time
        plt.figure(figsize=(10, 6))
        plt.plot(self.balance_history['timestamp'], self.balance_history['balance'], label="Balance")
        plt.title("Balance Over Time")
        plt.xlabel("Time")
        plt.ylabel("Balance")
        plt.legend()
        plt.grid(True)
        plt.show()

        # Plot Drawdown
        balance = self.balance_history['balance']
        cumulative_max = balance.cummax()
        drawdown = (balance - cumulative_max) / cumulative_max

        plt.figure(figsize=(10, 6))
        plt.plot(self.balance_history['timestamp'], drawdown, label="Drawdown", color="red")
        plt.title("Drawdown Over Time")
        plt.xlabel("Time")
        plt.ylabel("Drawdown")
        plt.legend()
        plt.grid(True)
        plt.show()

        # Histogram of Trade Returns
        returns = self.balance_history['balance'].pct_change().dropna()

        plt.figure(figsize=(10, 6))
        plt.hist(returns, bins=50, alpha=0.7, label="Returns")
        plt.title("Distribution of Returns")
        plt.xlabel("Returns")
        plt.ylabel("Frequency")
        plt.legend()
        plt.grid(True)
        plt.show()
