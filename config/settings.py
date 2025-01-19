import os
from dotenv import load_dotenv

# Load environment variables from a .env file (if present)
load_dotenv()

# ===== Account Details =====
ACCOUNT_ID = int(os.getenv("ACCOUNT_ID", "0"))  # Ensure ACCOUNT_ID is an integer
ACCOUNT_PASSWORD = os.getenv("ACCOUNT_PASSWORD", "DefaultPassword")
BROKER_NAME = os.getenv("BROKER_NAME", "ICM")

# ===== Leverage and Trading Limits =====
LEVERAGE = int(os.getenv("LEVERAGE", 100))  # Default: 1:100 leverage
MAX_DRAWDOWN = float(os.getenv("MAX_DRAWDOWN", 0.20))  # Default: 20% drawdown
RISK_PER_TRADE = float(os.getenv("RISK_PER_TRADE", 0.005))  # Default: 0.5% risk per trade

# ===== API Credentials =====
MT5_API_USERNAME = os.getenv("MT5_API_USERNAME", "DefaultMT5Username")
MT5_API_PASSWORD = os.getenv("MT5_API_PASSWORD", "DefaultMT5Password")
MT5_SERVER = os.getenv("MT5_SERVER", "YourMT5Server")

# ===== Trading Pairs =====
TRADING_PAIRS = os.getenv("TRADING_PAIRS", "").split(",") or ["EURUSD", "GBPUSD", "USDJPY"]

# ===== Pairwise Pairs (for strategies that require pair data) =====
PAIRWISE_PAIRS = os.getenv("PAIRWISE_PAIRS", "").split(",") or ["EURUSD_GBPUSD", "USDJPY_EURUSD"]

# ===== Time Frames for Strategies =====
TIME_FRAMES = {
    "trend_following": os.getenv("TIME_FRAMES_trend_following", "1h").split(","),
    "mean_reversion": os.getenv("TIME_FRAMES_mean_reversion", "5m,15m").split(","),
    "scalping_strategy": os.getenv("TIME_FRAMES_scalping_strategy", "1m,5m").split(","),
    "momentum_strategy": os.getenv("TIME_FRAMES_momentum_strategy", "15m,30m").split(","),
    "breakout_strategy": os.getenv("TIME_FRAMES_breakout_strategy", "1h,4h").split(","),
    "cointegration_strategy": os.getenv("TIME_FRAMES_cointegration_strategy", "1h,4h").split(",")
}

# ===== Global Thresholds =====
TRADE_THRESHOLD = float(os.getenv("TRADE_THRESHOLD", 0.02))  # Minimum expected return
STOP_LOSS_BUFFER = float(os.getenv("STOP_LOSS_BUFFER", 0.001))  # Stop-loss buffer
TAKE_PROFIT_BUFFER = float(os.getenv("TAKE_PROFIT_BUFFER", 0.002))  # Take-profit buffer
MIN_SPREAD_THRESHOLD = float(os.getenv("MIN_SPREAD_THRESHOLD", 1.5))  # Min spread in pips
MAX_SPREAD_THRESHOLD = float(os.getenv("MAX_SPREAD_THRESHOLD", 3.0))  # Max spread in pips

# ===== Trading Hours =====
TRADING_START_HOUR = int(os.getenv("TRADING_START_HOUR", 9))  # Start at 9 AM
TRADING_END_HOUR = int(os.getenv("TRADING_END_HOUR", 17))  # End at 5 PM

# ===== Logging and Debug Settings =====
LOGGING_LEVEL = os.getenv("LOGGING_LEVEL", "INFO")  # Default log level
LOG_FILE = os.getenv("LOG_FILE", "trading_system.log")  # Log file path

# ===== Alerts and Notifications =====
ENABLE_NOTIFICATIONS = os.getenv("ENABLE_NOTIFICATIONS", "True").lower() == "true"
EMAIL_ALERTS = os.getenv("EMAIL_ALERTS", "False").lower() == "true"
SLACK_WEBHOOK_URL = os.getenv("SLACK_WEBHOOK_URL", "")  # Slack webhook URL

# ===== Miscellaneous =====
DEFAULT_CURRENCY = os.getenv("DEFAULT_CURRENCY", "USD")  # Default account currency
DEFAULT_LOT_SIZE = float(os.getenv("DEFAULT_LOT_SIZE", 0.1))  # Default trade lot size
ALLOW_HEDGING = os.getenv("ALLOW_HEDGING", "False").lower() == "true"
MAX_OPEN_TRADES = int(os.getenv("MAX_OPEN_TRADES", 20))  # Max open trades

# ===== Validation for Critical Settings =====
def validate_settings():
    """
    Validates critical settings to ensure they are within acceptable ranges.
    """
    if ACCOUNT_ID <= 0:
        raise ValueError("ACCOUNT_ID must be a positive integer.")

    if not ACCOUNT_PASSWORD:
        raise ValueError("ACCOUNT_PASSWORD must be provided.")

    if not BROKER_NAME:
        raise ValueError("BROKER_NAME must be provided.")

    if not (0 < RISK_PER_TRADE <= 1):
        raise ValueError("RISK_PER_TRADE must be between 0 and 1 (as a percentage).")

    if not (0 < MAX_DRAWDOWN <= 1):
        raise ValueError("MAX_DRAWDOWN must be between 0 and 1 (as a percentage).")

    if LEVERAGE <= 0:
        raise ValueError("LEVERAGE must be a positive integer.")

    if TRADING_START_HOUR < 0 or TRADING_START_HOUR > 23:
        raise ValueError("TRADING_START_HOUR must be between 0 and 23.")

    if TRADING_END_HOUR < 0 or TRADING_END_HOUR > 23:
        raise ValueError("TRADING_END_HOUR must be between 0 and 23.")

    if TRADING_START_HOUR >= TRADING_END_HOUR:
        raise ValueError("TRADING_START_HOUR must be earlier than TRADING_END_HOUR.")

    if not TRADING_PAIRS:
        raise ValueError("At least one trading pair must be specified.")

    if not TIME_FRAMES:
        raise ValueError("TIME_FRAMES must be specified for each strategy.")

    for strategy, time_frames in TIME_FRAMES.items():
        if not time_frames:
            raise ValueError(f"Time frames must be specified for {strategy} strategy.")

validate_settings()
