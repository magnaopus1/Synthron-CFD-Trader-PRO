import MetaTrader5 as mt5
import pandas as pd
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import List, Dict, Optional
from models.sentiment_analysis import VaderAnalyzer, BERTSentimentAnalyzer

# Configure logging
logging.basicConfig(
    format="%(asctime)s [%(levelname)s]: %(message)s",
    level=logging.INFO,
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

class DataLoader:
    def __init__(self, max_workers: int = 5):
        """
        Initialize DataLoader and connect to MT5.
        :param max_workers: Maximum number of threads for concurrent data fetching.
        """
        self.max_workers = max_workers
        self.vader_analyzer = VaderAnalyzer()
        self.bert_analyzer = BERTSentimentAnalyzer()

        if not mt5.initialize():
            logger.error(f"Failed to initialize MT5: {mt5.last_error()}")
            raise RuntimeError("MT5 Initialization failed")
        logger.info("MT5 initialized successfully.")

    def fetch_symbol_info(self, symbol: str) -> Optional[Dict]:
        """Fetch symbol information."""
        if not symbol:
            logger.error("Symbol cannot be empty.")
            return None
        info = mt5.symbol_info(symbol)
        if info is None:
            logger.warning(f"Symbol {symbol} information not found: {mt5.last_error()}")
            return None
        return info._asdict()

    def fetch_symbol_tick(self, symbol: str) -> Optional[Dict]:
        """Fetch the latest tick data for a symbol."""
        if not symbol:
            logger.error("Symbol cannot be empty.")
            return None
        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            logger.warning(f"Tick data for symbol {symbol} not found: {mt5.last_error()}")
            return None
        return tick._asdict()

    def fetch_historical_data(
        self, symbol: str, timeframe: int, start: datetime, end: datetime
    ) -> Optional[pd.DataFrame]:
        """Fetch historical bars for a symbol."""
        if not symbol or not start or not end:
            logger.error("Symbol, start date, and end date must be provided.")
            return None
        if start >= end:
            logger.error("Start date must be earlier than the end date.")
            return None
        rates = mt5.copy_rates_range(symbol, timeframe, start, end)
        if rates is None:
            logger.error(f"Failed to fetch historical data for {symbol}: {mt5.last_error()}")
            return None
        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        return df

    def perform_sentiment_analysis(self, text_data: List[str]) -> Dict[str, List[float]]:
        """
        Perform sentiment analysis using both Vader and BERT models.
        
        :param text_data: List of text data to analyze.
        :return: Dictionary containing sentiment scores from both models.
        """
        if not text_data:
            logger.warning("No text data provided for sentiment analysis.")
            return {"vader": [], "bert": []}

        vader_scores = []
        bert_scores = []

        for text in text_data:
            try:
                vader_score = self.vader_analyzer.analyze(text)
                bert_score = self.bert_analyzer.analyze(text)
                vader_scores.append(vader_score)
                bert_scores.append(bert_score)
            except Exception as e:
                logger.error(f"Error during sentiment analysis: {e}")
                vader_scores.append(None)
                bert_scores.append(None)

        return {"vader": vader_scores, "bert": bert_scores}

    def analyze_live_sentiment(self, symbols: List[str]) -> Dict[str, Dict[str, List[float]]]:
        """
        Analyze sentiment for live market data for given symbols.

        :param symbols: List of symbols to analyze.
        :return: Sentiment scores for each symbol.
        """
        if not symbols:
            logger.error("Symbol list cannot be empty.")
            return {}

        sentiment_results = {}
        live_data = self.fetch_live_data(symbols)

        for tick in live_data:
            symbol = tick['symbol']
            text_data = [f"Price for {symbol} is {tick['bid']}."]  # Example text data
            sentiment_results[symbol] = self.perform_sentiment_analysis(text_data)

        return sentiment_results

    def fetch_open_positions(self) -> List[Dict]:
        """Fetch all open positions."""
        positions = mt5.positions_get()
        if positions is None:
            logger.error(f"Failed to fetch open positions: {mt5.last_error()}")
            return []
        return [pos._asdict() for pos in positions]

    def fetch_live_data(self, symbols: List[str]) -> List[Dict]:
        """Fetch live data for multiple symbols concurrently."""
        if not symbols:
            logger.error("Symbol list cannot be empty.")
            return []
        results = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {executor.submit(self.fetch_symbol_tick, symbol): symbol for symbol in symbols}
            for future in as_completed(futures):
                try:
                    result = future.result()
                    if result:
                        results.append(result)
                except Exception as e:
                    logger.error(f"Error fetching live data: {e}")
        return results

    def shutdown(self):
        """Shutdown MT5 connection."""
        if mt5.shutdown():
            logger.info("MT5 shutdown completed.")
        else:
            logger.error(f"Failed to shutdown MT5: {mt5.last_error()}")
