import logging
import numpy as np
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import blpapi
from utils.exception_handler import log_exception
from models.config.ml_settings import SENTIMENT_ANALYSIS_SETTINGS

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VaderAnalyzer:
    """
    Sentiment Analysis using Vader (Valence Aware Dictionary and sEntiment Reasoner).
    This model is used for analyzing sentiment of market-related news from Bloomberg.
    """

    def __init__(self):
        """
        Initialize the Vader Sentiment Analyzer using configuration settings and Bloomberg API for fetching news.
        """
        try:
            # Load configuration parameters from settings
            self.sentiment_sources = SENTIMENT_ANALYSIS_SETTINGS["sentiment_sources"]
            self.bloomberg_api_key = SENTIMENT_ANALYSIS_SETTINGS["bloomberg_api_key"]  # Assuming API key in config
            
            # Initialize Vader Sentiment Analyzer
            self.analyzer = SentimentIntensityAnalyzer()

            logger.info("VaderAnalyzer model initialized with sentiment sources: %s", self.sentiment_sources)
        except KeyError as e:
            logger.error(f"Configuration key missing: {e}")
            raise

    def fetch_bloomberg_news(self):
        """
        Fetches the latest news from Bloomberg using their API.
        :return: List of fetched news headlines/articles.
        """
        try:
            # Create a Bloomberg API session
            session = blpapi.Session()

            if not session.start():
                logger.error("Failed to start Bloomberg API session.")
                raise ConnectionError("Failed to connect to Bloomberg API.")

            # Define the request for fetching news
            news_request = session.getRequest('News', 'news headlines and articles request parameters here')
            news_request.set("query", "market")  # Assuming we want market-related news
            
            logger.info("Fetching Bloomberg news...")
            session.sendRequest(news_request)

            # Assuming we get news data here in 'news_data'
            news_data = []  # Replace with actual data fetch from Bloomberg API

            logger.info("Bloomberg news fetched successfully.")
            return news_data

        except Exception as e:
            logger.error("Failed to fetch Bloomberg news.")
            log_exception(e)
            raise

    def analyze_sentiment(self, text):
        """
        Analyzes the sentiment of a given text using Vader Sentiment Analysis.
        :param text: Text string to be analyzed.
        :return: Sentiment polarity score.
        """
        try:
            sentiment_score = self.analyzer.polarity_scores(text)
            logger.info("Sentiment analysis completed with score: %s", sentiment_score)
            return sentiment_score
        except Exception as e:
            logger.error("Sentiment analysis failed.")
            log_exception(e)
            raise

    def classify_sentiment(self, text):
        """
        Classify sentiment as Positive, Neutral, or Negative.
        :param text: Text string to be analyzed.
        :return: Sentiment classification (Positive/Neutral/Negative).
        """
        try:
            sentiment_score = self.analyze_sentiment(text)
            if sentiment_score['compound'] >= 0.05:
                sentiment_class = "Positive"
            elif sentiment_score['compound'] <= -0.05:
                sentiment_class = "Negative"
            else:
                sentiment_class = "Neutral"
            
            logger.info("Sentiment classified as: %s", sentiment_class)
            return sentiment_class
        except Exception as e:
            logger.error("Sentiment classification failed.")
            log_exception(e)
            raise

    def process_bloomberg_news(self):
        """
        Fetch news from Bloomberg and classify sentiment for each article.
        :return: A list of articles with sentiment classification.
        """
        try:
            news_data = self.fetch_bloomberg_news()
            news_with_sentiment = []

            for article in news_data:
                headline = article.get('headline')  # Assuming each article has a headline key
                sentiment = self.classify_sentiment(headline)
                news_with_sentiment.append({
                    'headline': headline,
                    'sentiment': sentiment
                })

            logger.info("Processed sentiment for Bloomberg news.")
            return news_with_sentiment
        except Exception as e:
            logger.error("Failed to process Bloomberg news.")
            log_exception(e)
            raise

    def save_model(self, path):
        """
        Save the current sentiment model (optional, for future deployment).
        :param path: Path to save the model.
        """
        try:
            logger.info("Vader Sentiment model does not require saving, as it is rule-based.")
        except Exception as e:
            logger.error("Failed to save Vader Sentiment model.")
            log_exception(e)
            raise

    def load_model(self, path):
        """
        Load a saved sentiment model (if applicable).
        :param path: Path to load the model.
        """
        try:
            logger.info("Vader Sentiment model does not require loading, as it is rule-based.")
        except Exception as e:
            logger.error("Failed to load Vader Sentiment model.")
            log_exception(e)
            raise
