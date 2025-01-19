import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from models.classification import (
    LogisticRegressionModel,
    RandomForestModel,
    GradientBoostingModel,
)
from models.clustering import (
    DBSCANModel,
    GMMModel,
    KMeansModel,
)
from models.reinforcement_learning import ActorCritic, DQN, PPO
from models.sentiment_analysis import VaderAnalyzer, BERTSentimentAnalyzer
from statsmodels.tsa.stattools import coint
import numpy as np

logger = logging.getLogger(__name__)

class StrategySelector:
    """Manages strategy selection and execution for different assets."""

    def __init__(self, strategies, max_concurrent_strategies=5):
        """
        Initialize the strategy selector.

        :param strategies: Dictionary of strategy objects.
        :param max_concurrent_strategies: Maximum number of concurrent strategies.
        """
        self.strategies = strategies  # {'strategy_name': strategy_object}
        self.max_concurrent_strategies = max_concurrent_strategies
        self.classification_models = {
            "LogisticRegression": LogisticRegressionModel(),
            "RandomForest": RandomForestModel(),
            "GradientBoosting": GradientBoostingModel(),
        }
        self.clustering_models = {
            "DBSCAN": DBSCANModel(),
            "GMM": GMMModel(),
            "KMeans": KMeansModel(),
        }
        self.rl_models = {
            "ActorCritic": ActorCritic(),
            "DQN": DQN(),
            "PPO": PPO(),
        }
        self.sentiment_models = {
            "Vader": VaderAnalyzer(),
            "BERT": BERTSentimentAnalyzer(),
        }

    def check_cointegration(self, series1, series2):
        """
        Check if two time series are cointegrated.

        :param series1: First time series.
        :param series2: Second time series.
        :return: Cointegration p-value.
        """
        try:
            score, p_value, _ = coint(series1, series2)
            return p_value
        except Exception as e:
            logger.error(f"Error checking cointegration: {e}")
            return None

    def generate_market_features(self, asset_data):
        """
        Generate market features for classification and clustering models.

        :param asset_data: Market data to extract features from.
        :return: Feature array or DataFrame.
        """
        return np.random.rand(100, 5)

    def apply_clustering_models(self, market_features):
        """
        Apply clustering models to segment market features.

        :param market_features: Features extracted from market data.
        :return: Dictionary with clustering results.
        """
        clustering_results = {}
        for name, model in self.clustering_models.items():
            try:
                labels = model.cluster(market_features)
                clustering_results[name] = labels
                logger.info(f"Clustering result for {name}: {np.unique(labels)}")
            except Exception as e:
                logger.error(f"Error applying {name}: {e}")
                clustering_results[name] = None
        return clustering_results

    def apply_sentiment_analysis(self, text_data, model_name="Vader"):
        """
        Apply sentiment analysis to textual data.

        :param text_data: List or string of textual data for sentiment analysis.
        :param model_name: Sentiment model to use ('Vader', 'BERT').
        :return: Sentiment scores or analysis result.
        """
        if model_name not in self.sentiment_models:
            logger.error(f"Invalid sentiment model name: {model_name}. Choose from {list(self.sentiment_models.keys())}.")
            raise ValueError(f"Invalid sentiment model name: {model_name}")

        model = self.sentiment_models[model_name]
        try:
            logger.info(f"Applying sentiment analysis using {model_name}.")
            sentiment_result = model.analyze(text_data)
            return sentiment_result
        except Exception as e:
            logger.error(f"Error applying sentiment analysis with {model_name}: {e}")
            return None

    def apply_rl_model(self, state, model_name="ActorCritic"):
        """
        Apply reinforcement learning model to determine actions.

        :param state: Current market state as a dictionary.
        :param model_name: RL model name ('ActorCritic', 'DQN', 'PPO').
        :return: Predicted action (e.g., "BUY", "SELL", "HOLD").
        """
        if model_name not in self.rl_models:
            logger.error(f"Invalid RL model name: {model_name}. Choose from {list(self.rl_models.keys())}.")
            raise ValueError(f"Invalid RL model name: {model_name}")

        model = self.rl_models[model_name]
        try:
            logger.info(f"Applying RL model {model_name}.")
            action = model.predict(state)
            return action
        except Exception as e:
            logger.error(f"Error applying RL model {model_name}: {e}")
            return "HOLD"

    def select_strategy(self, market_condition, time_frame, pairwise=False, series1=None, series2=None, sentiment_data=None):
        """
        Select an appropriate strategy based on market conditions, clustering results, sentiment, and models.

        :param market_condition: Current market condition ('trend', 'range', 'volatility', etc.).
        :param time_frame: Time frame for the strategy (e.g., "1m", "1h", "1D").
        :param pairwise: Boolean indicating if the strategy applies to pairwise data.
        :param series1: Optional first time series for pairwise strategies.
        :param series2: Optional second time series for pairwise strategies.
        :param sentiment_data: Text data for sentiment analysis.
        :return: List of selected strategy names.
        """
        strategy_map = {
            "trend": {"allowed_timeframes": ["1m", "5m", "1h", "4h", "1D"], "strategies": ["trend_following", "scalping"]},
            "range": {"allowed_timeframes": ["5m", "15m", "30m"], "strategies": ["mean_reversion", "scalping"]},
            "volatility": {"allowed_timeframes": ["15m", "1h", "4h", "1D"], "strategies": ["breakout_strategy", "momentum_strategy"]},
        }

        if sentiment_data:
            sentiment_result = self.apply_sentiment_analysis(sentiment_data, model_name="BERT")
            if sentiment_result.get("sentiment_score", 0) > 0.7:
                logger.info("Positive sentiment detected, favoring long strategies.")
                return ["trend_following", "momentum_strategy"]
            elif sentiment_result.get("sentiment_score", 0) < -0.7:
                logger.info("Negative sentiment detected, favoring short strategies.")
                return ["mean_reversion", "scalping"]

        if pairwise and series1 is not None and series2 is not None:
            cointegration_p_value = self.check_cointegration(series1, series2)
            if cointegration_p_value and cointegration_p_value < 0.05:
                logger.info(f"Pair is cointegrated (p-value={cointegration_p_value}). Using mean-reversion strategy.")
                return ["mean_reversion"]

        if market_condition not in strategy_map:
            logger.warning(f"Unrecognized market condition: {market_condition}. Defaulting to trend-following.")
            market_condition = "trend"

        if time_frame not in strategy_map[market_condition]["allowed_timeframes"]:
            logger.warning(f"Time frame {time_frame} not supported for {market_condition}. Defaulting to 1m.")
            time_frame = strategy_map[market_condition]["allowed_timeframes"][0]

        strategies = strategy_map[market_condition]["strategies"]
        return strategies


    def execute_strategy(self, strategy_name, asset_identifier, asset_data, time_frame):
        """
        Execute a specific strategy for a given asset or pair.

        :param strategy_name: Name of the strategy to execute.
        :param asset_identifier: Identifier for the asset (symbol or pair name).
        :param asset_data: Market data for the asset (or tuple for pairs).
        :param time_frame: Time frame for strategy execution (e.g., "1m", "1h").
        :return: Strategy result.
        """
        if strategy_name not in self.strategies:
            logger.error(f"Strategy {strategy_name} not found.")
            return None

        strategy = self.strategies[strategy_name]
        try:
            result = strategy(asset_data, time_frame)
            logger.info(f"Executed {strategy_name} for {asset_identifier}. Result: {result}")
            return result
        except Exception as e:
            logger.error(f"Error executing {strategy_name} for {asset_identifier}: {e}")
            return None

    def run_concurrent_strategies(self, asset_identifier, asset_data, selected_strategies, time_frame):
        """
        Execute multiple strategies concurrently for an asset.

        :param asset_identifier: Identifier for the asset being traded.
        :param asset_data: Market data for the asset.
        :param selected_strategies: List of strategy names to execute.
        :param time_frame: Time frame for execution.
        :return: Results for all executed strategies.
        """
        results = {}
        with ThreadPoolExecutor(max_workers=self.max_concurrent_strategies) as executor:
            futures = {
                executor.submit(self.execute_strategy, strategy, asset_identifier, asset_data, time_frame): strategy
                for strategy in selected_strategies
            }
            for future in as_completed(futures):
                strategy = futures[future]
                try:
                    results[strategy] = future.result()
                except Exception as e:
                    logger.error(f"Error executing strategy {strategy}: {e}")
                    results[strategy] = None
        return results

    def run_multiple_assets(self, assets_data, market_conditions, time_frames, pairwise_data=None):
        """
        Run strategies for multiple assets and asset pairs concurrently.

        :param assets_data: Dictionary of market data for each asset.
        :param market_conditions: Dictionary of market conditions for each asset.
        :param time_frames: Dictionary of time frames for each asset.
        :param pairwise_data: Dictionary of paired data for pairwise strategies.
        :return: Nested results for each asset/pair and strategy.
        """
        overall_results = {}

        with ThreadPoolExecutor(max_workers=self.max_concurrent_strategies) as executor:
            single_asset_futures = {
                executor.submit(
                    self.run_concurrent_strategies,
                    asset,
                    data,
                    self.select_strategy(market_conditions.get(asset, "trend"), time_frames.get(asset, "1h")),
                    time_frames.get(asset, "1h")
                ): asset
                for asset, data in assets_data.items()
            }
            for future in as_completed(single_asset_futures):
                asset = single_asset_futures[future]
                try:
                    overall_results[asset] = future.result()
                except Exception as e:
                    logger.error(f"Error processing asset {asset}: {e}")
                    overall_results[asset] = None

        if pairwise_data:
            pairwise_futures = {
                executor.submit(
                    self.run_concurrent_strategies,
                    pair_name,
                    (series1, series2),
                    self.select_strategy("volatility", "1h", pairwise=True, series1=series1, series2=series2),
                    "1h"
                ): pair_name
                for pair_name, (series1, series2) in pairwise_data.items()
            }
            for future in as_completed(pairwise_futures):
                pair_name = pairwise_futures[future]
                try:
                    overall_results[pair_name] = future.result()
                except Exception as e:
                    logger.error(f"Error processing pair {pair_name}: {e}")
                    overall_results[pair_name] = None

        return overall_results
