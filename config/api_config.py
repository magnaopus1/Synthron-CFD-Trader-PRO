"""
Production-Level MetaTrader 5 API Configuration and Initialization.

This script sets up and authenticates the MetaTrader 5 API connection,
handles API requests, monitors connection health, and provides utility functions
for managing the API in a robust and scalable manner.
"""

import MetaTrader5 as mt5
import logging
import time
from config.settings import ACCOUNT_ID, ACCOUNT_PASSWORD, MT5_SERVER

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("MT5_API")

# Constants
RECONNECT_ATTEMPTS = 3  # Number of reconnection attempts
RECONNECT_DELAY = 5  # Delay in seconds between reconnection attempts


def initialize_mt5():
    """
    Initialize and authenticate the MetaTrader 5 API.
    
    Returns:
        bool: True if initialized and authenticated successfully, False otherwise.
    """
    logger.info("Initializing MetaTrader 5 API...")
    
    # Initialize the MetaTrader 5 platform
    if not mt5.initialize():
        logger.error(f"MetaTrader 5 initialization failed: {mt5.last_error()}")
        return False
    
    logger.info("MetaTrader 5 initialized successfully.")
    
    # Authenticate using account credentials
    try:
        authorized = mt5.login(login=ACCOUNT_ID, password=ACCOUNT_PASSWORD, server=MT5_SERVER)
        if not authorized:
            raise Exception(f"Authentication failed: {mt5.last_error()}")
        logger.info(f"Successfully connected to account {ACCOUNT_ID} on server {MT5_SERVER}.")
        return True
    except Exception as e:
        logger.error(f"Error during authentication: {e}")
        mt5.shutdown()
        return False


def shutdown_mt5():
    """
    Shut down the MetaTrader 5 API connection.
    """
    logger.info("Shutting down MetaTrader 5 API...")
    try:
        mt5.shutdown()
        logger.info("MetaTrader 5 API connection closed successfully.")
    except Exception as e:
        logger.error(f"Error during shutdown: {e}")


def is_connected():
    """
    Check if the MetaTrader 5 API is connected and authenticated.
    
    Returns:
        bool: True if connected, False otherwise.
    """
    try:
        state = mt5.connection_state()
        if state == mt5.CONNECTION_STATUS_CONNECTED:
            logger.info("MetaTrader 5 API is connected.")
            return True
        else:
            logger.warning("MetaTrader 5 API is not connected. Connection state: %s", state)
            return False
    except Exception as e:
        logger.error(f"Error checking connection status: {e}")
        return False


def reconnect_mt5():
    """
    Attempt to reconnect to the MetaTrader 5 API in case of disconnection.
    
    Returns:
        bool: True if reconnected successfully, False otherwise.
    """
    logger.info("Attempting to reconnect to MetaTrader 5 API...")
    for attempt in range(1, RECONNECT_ATTEMPTS + 1):
        logger.info(f"Reconnection attempt {attempt}/{RECONNECT_ATTEMPTS}...")
        shutdown_mt5()
        time.sleep(RECONNECT_DELAY)
        if initialize_mt5():
            logger.info("Reconnection successful.")
            return True
        logger.warning(f"Reconnection attempt {attempt} failed.")
    
    logger.error("Failed to reconnect after multiple attempts.")
    return False


def get_account_info():
    """
    Retrieve and log account information.
    
    Returns:
        dict: Account information dictionary or None if an error occurs.
    """
    logger.info("Fetching account information...")
    try:
        account_info = mt5.account_info()
        if account_info:
            account_dict = account_info._asdict()
            logger.info("Account information retrieved successfully.")
            return account_dict
        else:
            logger.warning("Failed to fetch account information.")
            return None
    except Exception as e:
        logger.error(f"Error retrieving account information: {e}")
        return None


def log_account_info():
    """
    Log the details of the connected account.
    """
    account_info = get_account_info()
    if account_info:
        logger.info("Account Details:")
        for key, value in account_info.items():
            logger.info(f"{key}: {value}")


def safe_mt5_operation(func, *args, **kwargs):
    """
    Safely perform an MT5 operation with error handling.
    
    Args:
        func: The MetaTrader 5 function to execute.
        *args: Positional arguments for the function.
        **kwargs: Keyword arguments for the function.
    
    Returns:
        The result of the function, or None if an error occurs.
    """
    try:
        result = func(*args, **kwargs)
        return result
    except Exception as e:
        logger.error(f"Error during MetaTrader 5 operation '{func.__name__}': {e}")
        return None


