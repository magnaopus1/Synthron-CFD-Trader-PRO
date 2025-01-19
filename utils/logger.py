import logging
import os
from logging.handlers import RotatingFileHandler, SMTPHandler
from typing import Optional

class Logger:
    """
    Centralized logging class for debugging and monitoring.
    Supports console output, file logging, and email notifications for critical issues.
    """

    LOG_FILE = "trading_system.log"
    LOG_DIR = "./logs"
    MAX_LOG_FILE_SIZE = 10 * 1024 * 1024  # 10 MB
    BACKUP_COUNT = 7  # Keep 7 backups
    EMAIL_NOTIFICATIONS_ENABLED = True  # Enable email notifications
    SMTP_CONFIG = {
        "mailhost": ("smtp.gmail.com", 587),
        "fromaddr": "your_email@example.com",
        "toaddrs": ["admin@example.com"],
        "subject": "Critical Error in Trading System",
        "credentials": ("your_email@example.com", "your_email_password"),
        "secure": (),
    }

    @staticmethod
    def get_logger(name: Optional[str] = None) -> logging.Logger:
        """
        Configures and returns a logger instance.

        Args:
            name (Optional[str]): Name of the logger. Defaults to None (root logger).

        Returns:
            logging.Logger: Configured logger instance.
        """
        logger = logging.getLogger(name)
        logger.setLevel(logging.DEBUG)

        # Avoid adding duplicate handlers
        if logger.hasHandlers():
            return logger

        # Create log directory if it doesn't exist
        if not os.path.exists(Logger.LOG_DIR):
            os.makedirs(Logger.LOG_DIR)

        # File handler with rotation
        log_file_path = os.path.join(Logger.LOG_DIR, Logger.LOG_FILE)
        file_handler = RotatingFileHandler(
            log_file_path, maxBytes=Logger.MAX_LOG_FILE_SIZE, backupCount=Logger.BACKUP_COUNT
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(
            logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        )

        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(
            logging.Formatter('%(levelname)s - %(message)s')
        )

        # Add handlers to logger
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

        # Add email handler for critical logs
        if Logger.EMAIL_NOTIFICATIONS_ENABLED:
            Logger._add_email_handler(logger)

        return logger

    @staticmethod
    def _add_email_handler(logger: logging.Logger):
        """
        Adds an email handler to the logger for critical errors.

        Args:
            logger (logging.Logger): The logger instance to which the email handler is added.
        """
        try:
            mailhost, port = Logger.SMTP_CONFIG["mailhost"]
            fromaddr = Logger.SMTP_CONFIG["fromaddr"]
            toaddrs = Logger.SMTP_CONFIG["toaddrs"]
            subject = Logger.SMTP_CONFIG["subject"]
            credentials = Logger.SMTP_CONFIG["credentials"]
            secure = Logger.SMTP_CONFIG["secure"]

            email_handler = SMTPHandler(
                mailhost=(mailhost, port),
                fromaddr=fromaddr,
                toaddrs=toaddrs,
                subject=subject,
                credentials=credentials,
                secure=secure,
            )
            email_handler.setLevel(logging.CRITICAL)
            email_handler.setFormatter(
                logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            )
            logger.addHandler(email_handler)
            logger.info("Email notifications enabled for critical errors.")
        except Exception as e:
            logger.error("Failed to configure email notifications: %s", e)

    @staticmethod
    def test_logger():
        """
        A simple test to verify the logging configuration.
        """
        logger = Logger.get_logger("TestLogger")
        logger.debug("This is a debug message.")
        logger.info("This is an info message.")
        logger.warning("This is a warning message.")
        logger.error("This is an error message.")
        logger.critical("This is a critical error message.")

