import logging
import traceback
from typing import Optional, Dict, List
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from models.anomaly_detection import Autoencoder, IsolationForestModel, OneClassSVMModel

# Configure logging
logger = logging.getLogger("trading_system")
logger.setLevel(logging.DEBUG)

file_handler = logging.FileHandler("trading_system.log")
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logger.addHandler(file_handler)

console_handler = logging.StreamHandler()
console_handler.setFormatter(logging.Formatter('%(levelname)s: %(message)s'))
logger.addHandler(console_handler)


class NotificationHandler:
    """Handles sending notifications for critical exceptions."""
    SMTP_SERVER = "smtp.gmail.com"
    SMTP_PORT = 587
    EMAIL_ADDRESS = "your_email@example.com"  # Update with your email
    EMAIL_PASSWORD = "your_email_password"    # Update with your email password
    RECIPIENTS = ["admin@example.com"]        # Update with recipient email(s)

    @staticmethod
    def send_email(subject: str, body: str):
        """Send an email notification."""
        try:
            msg = MIMEMultipart()
            msg['From'] = NotificationHandler.EMAIL_ADDRESS
            msg['To'] = ", ".join(NotificationHandler.RECIPIENTS)
            msg['Subject'] = subject

            msg.attach(MIMEText(body, 'plain'))
            with smtplib.SMTP(NotificationHandler.SMTP_SERVER, NotificationHandler.SMTP_PORT) as server:
                server.starttls()
                server.login(NotificationHandler.EMAIL_ADDRESS, NotificationHandler.EMAIL_PASSWORD)
                server.sendmail(NotificationHandler.EMAIL_ADDRESS, NotificationHandler.RECIPIENTS, msg.as_string())
            logger.info("Email notification sent successfully.")
        except Exception as email_exc:
            logger.error("Failed to send email notification: %s", email_exc)


class ExceptionHandler:
    """Handles exceptions gracefully to prevent system crashes."""

    @staticmethod
    def log_and_handle_exception(exc: Exception, context: str = "Unknown"):
        """Log and handle an exception."""
        logger.error("Exception occurred in context '%s': %s", context, exc)
        logger.error("Traceback: %s", traceback.format_exc())

    @staticmethod
    def handle_critical_exception(exc: Exception, context: str = "Critical"):
        """Handle a critical exception by logging, notifying, and halting execution."""
        logger.critical("Critical exception in '%s': %s", context, exc)
        logger.critical("Traceback: %s", traceback.format_exc())
        NotificationHandler.send_email(
            f"Critical Exception in {context}",
            f"Exception: {exc}\nTraceback:\n{traceback.format_exc()}"
        )
        raise SystemExit("Critical exception encountered. System terminating.")

    @staticmethod
    def suppress_exceptions(exc: Exception, context: str = "Suppressed"):
        """Log suppressed exceptions."""
        logger.warning("Suppressed exception in '%s': %s", context, exc)


class AnomalyDetectionManager:
    """Manages anomaly detection using multiple models."""

    def __init__(self):
        """Initialize anomaly detection models."""
        try:
            self.autoencoder = Autoencoder()
            self.isolation_forest = IsolationForestModel()
            self.one_class_svm = OneClassSVMModel()
            logger.info("Anomaly detection models initialized.")
        except Exception as exc:
            ExceptionHandler.handle_critical_exception(exc, "AnomalyDetectionManager Initialization")

    def detect_anomalies(self, feature_vector: List[float]) -> Dict[str, float]:
        """Detect anomalies in a feature vector using all models."""
        try:
            autoencoder_score = self.autoencoder.detect_anomaly(feature_vector)
            isolation_forest_score = self.isolation_forest.detect_anomaly(feature_vector)
            one_class_svm_score = self.one_class_svm.detect_anomaly(feature_vector)

            logger.debug(f"Anomaly Scores: Autoencoder={autoencoder_score}, "
                         f"IsolationForest={isolation_forest_score}, OneClassSVM={one_class_svm_score}")

            return {
                "autoencoder": autoencoder_score,
                "isolation_forest": isolation_forest_score,
                "one_class_svm": one_class_svm_score
            }
        except Exception as exc:
            ExceptionHandler.log_and_handle_exception(exc, "AnomalyDetectionManager.detect_anomalies")
            return {"autoencoder": 0.0, "isolation_forest": 0.0, "one_class_svm": 0.0}

    def assess_risk(self, feature_vector: List[float], threshold: float = 0.5) -> bool:
        """Assess risk based on anomaly scores and return a decision."""
        try:
            anomalies = self.detect_anomalies(feature_vector)
            avg_score = sum(anomalies.values()) / len(anomalies)

            if avg_score > threshold:
                logger.warning(f"High risk detected. Average anomaly score: {avg_score}, Threshold: {threshold}.")
                return False

            logger.info(f"Risk acceptable. Average anomaly score: {avg_score}, Threshold: {threshold}.")
            return True
        except Exception as exc:
            ExceptionHandler.log_and_handle_exception(exc, "AnomalyDetectionManager.assess_risk")
            return False
