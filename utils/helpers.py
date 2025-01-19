import datetime
from typing import Union, Optional, Any, Dict
from urllib.parse import urlencode, urljoin
import pytz
import logging

# Configure logging for the helpers module
logger = logging.getLogger("helpers")
logger.setLevel(logging.INFO)
file_handler = logging.FileHandler("helpers.log")
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logger.addHandler(file_handler)

class Helpers:
    """
    A utility class for common repeated tasks, such as timestamp conversion, rounding, or API helpers.
    """

    @staticmethod
    def convert_to_utc(timestamp: Union[str, int, datetime.datetime], tz: str = "UTC") -> datetime.datetime:
        """
        Converts a timestamp to a UTC datetime object.

        Args:
            timestamp (Union[str, int, datetime.datetime]): The input timestamp (ISO format, epoch, or datetime object).
            tz (str): The timezone of the input timestamp (default is UTC).

        Returns:
            datetime.datetime: The UTC datetime object.
        """
        try:
            if isinstance(timestamp, datetime.datetime):
                dt = timestamp
            elif isinstance(timestamp, int):  # Epoch timestamp
                dt = datetime.datetime.fromtimestamp(timestamp, pytz.timezone(tz))
            elif isinstance(timestamp, str):  # ISO format
                dt = datetime.datetime.fromisoformat(timestamp)
            else:
                raise ValueError("Unsupported timestamp format.")
            
            if not dt.tzinfo:
                dt = pytz.timezone(tz).localize(dt)

            utc_time = dt.astimezone(pytz.UTC)
            logger.info("Converted timestamp to UTC: %s", utc_time)
            return utc_time
        except Exception as e:
            logger.error("Failed to convert timestamp to UTC: %s", e)
            raise ValueError(f"Failed to convert timestamp to UTC: {e}")

    @staticmethod
    def round_value(value: float, precision: int = 2) -> float:
        """
        Rounds a value to the specified precision.

        Args:
            value (float): The value to round.
            precision (int): Number of decimal places to round to.

        Returns:
            float: The rounded value.
        """
        try:
            rounded_value = round(value, precision)
            logger.info("Rounded value: %s to %d decimal places -> %s", value, precision, rounded_value)
            return rounded_value
        except Exception as e:
            logger.error("Failed to round value: %s", e)
            raise ValueError(f"Failed to round value: {e}")

    @staticmethod
    def format_currency(value: float, currency_symbol: str = "$", precision: int = 2) -> str:
        """
        Formats a value as a currency string.

        Args:
            value (float): The value to format.
            currency_symbol (str): The currency symbol to prepend.
            precision (int): Number of decimal places to display.

        Returns:
            str: The formatted currency string.
        """
        try:
            formatted_value = f"{currency_symbol}{value:,.{precision}f}"
            logger.info("Formatted value as currency: %s", formatted_value)
            return formatted_value
        except Exception as e:
            logger.error("Failed to format currency: %s", e)
            raise ValueError(f"Failed to format currency: {e}")

    @staticmethod
    def build_api_endpoint(base_url: str, endpoint: str, params: Optional[Dict[str, Any]] = None) -> str:
        """
        Builds a complete API endpoint with query parameters.

        Args:
            base_url (str): The base URL of the API.
            endpoint (str): The specific endpoint path.
            params (Optional[Dict[str, Any]]): Dictionary of query parameters.

        Returns:
            str: The complete API endpoint URL.
        """
        try:
            full_url = urljoin(base_url, endpoint)
            if params:
                query_string = urlencode(params)
                full_url += f"?{query_string}"
            logger.info("Built API endpoint: %s", full_url)
            return full_url
        except Exception as e:
            logger.error("Failed to build API endpoint: %s", e)
            raise ValueError(f"Failed to build API endpoint: {e}")

    @staticmethod
    def is_market_open(current_time: datetime.datetime, open_time: str, close_time: str, tz: str = "UTC") -> bool:
        """
        Checks if the market is open based on the provided times.

        Args:
            current_time (datetime.datetime): The current time.
            open_time (str): Market open time in HH:MM format.
            close_time (str): Market close time in HH:MM format.
            tz (str): Timezone of the market.

        Returns:
            bool: True if the market is open, False otherwise.
        """
        try:
            market_tz = pytz.timezone(tz)
            localized_current_time = current_time.astimezone(market_tz)

            open_dt = datetime.datetime.strptime(open_time, "%H:%M").replace(
                year=localized_current_time.year,
                month=localized_current_time.month,
                day=localized_current_time.day
            )
            close_dt = datetime.datetime.strptime(close_time, "%H:%M").replace(
                year=localized_current_time.year,
                month=localized_current_time.month,
                day=localized_current_time.day
            )

            open_dt = market_tz.localize(open_dt)
            close_dt = market_tz.localize(close_dt)

            is_open = open_dt <= localized_current_time <= close_dt
            logger.info("Market open status at %s: %s", localized_current_time, is_open)
            return is_open
        except Exception as e:
            logger.error("Failed to check market hours: %s", e)
            raise ValueError(f"Failed to check market hours: {e}")

    @staticmethod
    def safely_cast(value: Any, target_type: type, default: Optional[Any] = None) -> Any:
        """
        Safely casts a value to a specified type, returning a default value if casting fails.

        Args:
            value (Any): The value to cast.
            target_type (type): The target type to cast to.
            default (Optional[Any]): The default value to return if casting fails.

        Returns:
            Any: The cast value, or the default value if casting fails.
        """
        try:
            cast_value = target_type(value)
            logger.info("Safely cast value: %s to type: %s -> %s", value, target_type, cast_value)
            return cast_value
        except (ValueError, TypeError) as e:
            logger.warning("Failed to cast value: %s to type: %s. Returning default: %s", value, target_type, default)
            return default
