# Import the sentiment analysis models for easy access
from .vader_analyzer import VaderAnalyzer
from .bert_sentiment import BERTSentimentAnalyzer

# Ensure that only the necessary classes are accessible when the module is imported
__all__ = [
    "VaderAnalyzer",         # Sentiment analysis using Vader (rule-based)
    "BERTSentimentAnalyzer"  # Sentiment analysis using BERT (deep learning)
]

# Log the successful initialization of sentiment analysis models
import logging
logger = logging.getLogger(__name__)

def initialize_sentiment_models():
    """
    Initialize sentiment analysis models at the module level to be ready for use.
    This ensures that any necessary resources or API connections are established.
    """
    try:
        # Example initialization of models (if applicable)
        vader_model = VaderAnalyzer()
        bert_model = BERTSentimentAnalyzer()

        logger.info("Sentiment analysis models (Vader and BERT) initialized successfully.")

        return vader_model, bert_model
    except Exception as e:
        logger.error("Failed to initialize sentiment analysis models.", exc_info=True)
        raise e

# Initialize models when this module is imported (optional)
initialize_sentiment_models()
