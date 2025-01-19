# Utils Package Documentation

The `utils` package provides utility classes and functions to support the development of a robust and scalable trading system. It includes modules for logging, exception handling, and helper methods to handle repeated tasks, ensuring modularity and maintainability.

---

## Table of Contents

1. [Exception Handler (`exception_handler.py`)](#exception-handler)
2. [Helpers (`helpers.py`)](#helpers)
3. [Logger (`logger.py`)](#logger)

---

## Exception Handler

### File: `exception_handler.py`

The `exception_handler` module ensures graceful handling of exceptions to prevent system crashes. It also supports critical notifications via email for immediate alerting.

### Classes and Methods

#### 1. `NotificationHandler`

Handles email notifications for critical exceptions.

- **Attributes**:
  - `SMTP_SERVER`: SMTP server address.
  - `SMTP_PORT`: SMTP server port.
  - `EMAIL_ADDRESS`: Sender's email address.
  - `EMAIL_PASSWORD`: Sender's email password.
  - `RECIPIENTS`: List of recipients.

- **Methods**:
  - `send_email(subject: str, body: str)`: Sends an email notification.

#### 2. `ExceptionHandler`

Handles exceptions gracefully to ensure system stability.

- **Methods**:
  - `log_and_handle_exception(exc: Exception, context: str = "Unknown")`:
    Logs exception details and handles the error gracefully.
  - `handle_critical_exception(exc: Exception, context: str = "Critical")`:
    Handles critical exceptions by logging, sending email notifications, and halting execution.
  - `suppress_exceptions(exc: Exception, context: str = "Suppressed")`:
    Logs suppressed exceptions without interrupting execution.
  - `capture_and_reraise(exc: Exception, context: Optional[str] = None)`:
    Captures and re-raises exceptions with detailed logging.

#### 3. `AnomalyDetectionManager`

Manages anomaly detection using multiple models.

- **Methods**:
  - `detect_anomalies(feature_vector: List[float]) -> Dict[str, float]`:
    Detects anomalies using Autoencoder, Isolation Forest, and One-Class SVM models.
  - `assess_risk(feature_vector: List[float], threshold: float = 0.5) -> bool`:
    Assesses risk based on anomaly scores and returns a decision.

---

## Helpers

### File: `helpers.py`

The `helpers` module provides utility methods for common tasks such as timestamp conversions, rounding values, formatting currencies, and constructing API endpoints.

### Classes and Methods

#### 1. `Helpers`

A utility class for repeated tasks.

- **Methods**:
  - `convert_to_utc(timestamp: Union[str, int, datetime.datetime], tz: str = "UTC") -> datetime.datetime`:
    Converts a timestamp (string, epoch, or datetime object) to UTC.
  - `round_value(value: float, precision: int = 2) -> float`:
    Rounds a float to the specified number of decimal places.
  - `format_currency(value: float, currency_symbol: str = "$", precision: int = 2) -> str`:
    Formats a float as a currency string with a specified symbol and precision.
  - `build_api_endpoint(base_url: str, endpoint: str, params: Optional[Dict[str, Any]] = None) -> str`:
    Constructs a complete API endpoint with optional query parameters.
  - `is_market_open(current_time: datetime.datetime, open_time: str, close_time: str, tz: str = "UTC") -> bool`:
    Checks if the market is open based on specified open and close times.
  - `safely_cast(value: Any, target_type: type, default: Optional[Any] = None) -> Any`:
    Safely casts a value to a specified type, returning a default if the cast fails.

---

## Logger

### File: `logger.py`

The `logger` module provides centralized logging for debugging and monitoring. It supports console output, file logging, and email notifications for critical issues.

### Classes and Methods

#### 1. `Logger`

Manages logging across the trading system.

- **Attributes**:
  - `LOG_FILE`: Name of the log file.
  - `LOG_DIR`: Directory for log files.
  - `MAX_LOG_FILE_SIZE`: Maximum size of a log file before rotation (default: 10 MB).
  - `BACKUP_COUNT`: Number of backup log files to retain.
  - `EMAIL_NOTIFICATIONS_ENABLED`: Flag to enable email notifications for critical logs.
  - `SMTP_CONFIG`: Configuration for the email handler.

- **Methods**:
  - `get_logger(name: Optional[str] = None) -> logging.Logger`:
    Configures and returns a logger instance.
  - `_add_email_handler(logger: logging.Logger)`:
    Adds an email handler to the logger for critical errors.
  - `test_logger()`:
    Tests the logger by emitting logs at different levels (`DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`).

### Features

- **Log Rotation**: Automatically rotates log files to prevent excessive disk usage.
- **Email Notifications**: Sends critical logs via email for real-time monitoring.
- **Console and File Logging**: Logs are streamed to both the console and a file for visibility and persistence.

---

## Conclusion

The `utils` package is an essential component of the Synthron CFD Trader Pro system, providing reusable utilities for exception handling, logging, and other supportive tasks. Its modular design ensures easy integration and scalability across different components of the trading system.
