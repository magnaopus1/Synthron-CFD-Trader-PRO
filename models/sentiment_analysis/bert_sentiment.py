import logging
import numpy as np
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from newsapi import NewsApiClient
import blpapi
from utils.exception_handler import log_exception
from models.config.ml_settings import SENTIMENT_ANALYSIS_SETTINGS

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BERTSentiment:
    """
    Sentiment Analysis using BERT (Bidirectional Encoder Representations from Transformers).
    This model is used for analyzing sentiment of market-related news articles or tweets.
    """

    def __init__(self):
        """
        Initialize the BERT Sentiment Analyzer using configuration settings and external APIs (NewsAPI, Bloomberg).
        """
        try:
            # Load configuration parameters from settings
            self.sentiment_sources = SENTIMENT_ANALYSIS_SETTINGS["sentiment_sources"]
            self.bloomberg_api_key = SENTIMENT_ANALYSIS_SETTINGS.get("bloomberg_api_key", None)
            self.news_api_key = SENTIMENT_ANALYSIS_SETTINGS.get("news_api_key", None)
            
            # Load pre-trained BERT model for sentiment analysis
            self.tokenizer = BertTokenizer.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')
            self.model = BertForSequenceClassification.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')

            logger.info("BERT Sentiment model initialized with sentiment sources: %s", self.sentiment_sources)
        except KeyError as e:
            logger.error(f"Configuration key missing: {e}")
            raise
        except Exception as e:
            logger.error("Failed to initialize BERT Sentiment model.")
            log_exception(e)
            raise

    def analyze_sentiment(self, text):
        """
        Analyzes the sentiment of a given text using BERT-based sentiment analysis.
        :param text: Text string to be analyzed.
        :return: Sentiment polarity score.
        """
        try:
            inputs = self.tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                sentiment_score = torch.softmax(logits, dim=-1).numpy()
                sentiment_class = np.argmax(sentiment_score)

            logger.info("BERT Sentiment analysis completed with sentiment class: %s", sentiment_class)
            return sentiment_class  # 0-4 scale for sentiment polarity
        except Exception as e:
            logger.error("Sentiment analysis failed.")
            log_exception(e)
            raise

    def classify_sentiment(self, text):
        """
        Classify sentiment as Positive, Neutral, or Negative using BERT.
        :param text: Text string to be analyzed.
        :return: Sentiment classification (Positive/Neutral/Negative).
        """
        try:
            sentiment_score = self.analyze_sentiment(text)
            if sentiment_score > 2:
                sentiment_class = "Positive"
            elif sentiment_score == 2:
                sentiment_class = "Neutral"
            else:
                sentiment_class = "Negative"

            logger.info("Sentiment classified as: %s", sentiment_class)
            return sentiment_class
        except Exception as e:
            logger.error("Sentiment classification failed.")
            log_exception(e)
            raise

    def fetch_news_data(self, source="newsapi"):
        """
        Fetch news data from an external source (NewsAPI or Bloomberg).
        :param source: Data source (NewsAPI or Bloomberg).
        :return: List of news articles.
        """
        try:
            if source == "newsapi":
                news_api = NewsApiClient(api_key=self.news_api_key)
                all_articles = news_api.get_everything(q="market", language="en", sort_by="relevancy")
                articles = [{"headline": article['title'], "content": article['description']} for article in all_articles['articles']]
                logger.info("News data fetched from NewsAPI.")
                return articles
            
            elif source == "bloomberg":
                # Replace with actual Bloomberg API call using blpapi (see previous examples)
                # Placeholder for Bloomberg API call:
                articles = []  # Replace with real Bloomberg API data fetch
                logger.info("News data fetched from Bloomberg.")
                return articles

            else:
                raise ValueError("Invalid news source specified.")

        except Exception as e:
            logger.error("Failed to fetch news data.")
            log_exception(e)
            raise

    def process_sentiment_from_news(self):
        """
        Fetch news and classify sentiment for each article using BERT.
        :return: List of news with sentiment classification.
        """
        try:
            all_articles = self.fetch_news_data(source="newsapi")  # Example source, can be changed to "bloomberg"
            news_with_sentiment = []

            for article in all_articles:
                headline = article.get('headline')
                sentiment = self.classify_sentiment(headline)
                news_with_sentiment.append({
                    'headline': headline,
                    'sentiment': sentiment
                })

            logger.info("Processed sentiment for news articles.")
            return news_with_sentiment

        except Exception as e:
            logger.error("Failed to process news articles for sentiment.")
            log_exception(e)
            raise

    def save_model(self, path):
        """
        Save the BERT sentiment model (optional).
        :param path: Path to save the model.
        """
        try:
            logger.info("BERT model does not require saving, as it is pre-trained.")
        except Exception as e:
            logger.error("Failed to save BERT sentiment model.")
            log_exception(e)
            raise

    def load_model(self, path):
        """
        Load the BERT sentiment model (optional).
        :param path: Path to load the model.
        """
        try:
            logger.info("BERT model does not require loading, as it is pre-trained.")
        except Exception as e:
            logger.error("Failed to load BERT sentiment model.")
            log_exception(e)
            raise
