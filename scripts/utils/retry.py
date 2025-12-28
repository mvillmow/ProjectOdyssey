"""Retry decorator with exponential backoff for network/git operations.

Provides automatic retry logic with configurable parameters:
- Exponential backoff (2^attempt seconds)
- Network error detection
- Max retries limit
- Logging integration
"""

import functools
import time
from typing import Any, Callable, Optional, TypeVar, cast

# Type variable for generic function decoration
F = TypeVar("F", bound=Callable[..., Any])

# Network error keywords to detect transient failures
NETWORK_ERROR_KEYWORDS = [
    "connection",
    "network",
    "timeout",
    "timed out",
    "temporary failure",
    "could not resolve",
    "name resolution",
    "rate limit",
    "throttle",
    "503",
    "502",
    "504",
]


def is_network_error(error: Exception) -> bool:
    """Check if error is likely a transient network issue.

    Args:
        error: Exception to check

    Returns:
        True if error message contains network error keywords
    """
    error_str = str(error).lower()
    return any(keyword in error_str for keyword in NETWORK_ERROR_KEYWORDS)


def retry_with_backoff(
    max_retries: int = 3,
    initial_delay: float = 1.0,
    backoff_factor: int = 2,
    retry_on: tuple[type[Exception], ...] = (Exception,),
    logger: Optional[Callable[[str], None]] = None,
) -> Callable[[F], F]:
    """Decorator to retry function with exponential backoff.

    Args:
        max_retries: Maximum number of retry attempts (default: 3)
        initial_delay: Initial delay in seconds before first retry (default: 1.0)
        backoff_factor: Multiplier for delay between retries (default: 2)
        retry_on: Tuple of exception types to retry on (default: all exceptions)
        logger: Optional logging function for retry attempts

    Returns:
        Decorated function with retry logic

    Example:
        @retry_with_backoff(max_retries=3, initial_delay=2.0)
        def unstable_network_call():
            # May fail transiently
            response = requests.get("https://api.github.com")
            return response.json()
    """

    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            last_exception = None

            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except retry_on as e:
                    last_exception = e

                    # Don't retry on last attempt
                    if attempt == max_retries:
                        break

                    # Calculate delay with exponential backoff
                    delay = initial_delay * (backoff_factor**attempt)

                    # Log retry attempt
                    if logger:
                        error_type = type(e).__name__
                        is_network = is_network_error(e)
                        network_tag = " [NETWORK]" if is_network else ""
                        logger(
                            f"Retry {attempt + 1}/{max_retries} after {error_type}{network_tag}: {e} (waiting {delay}s)"
                        )

                    # Wait before retry
                    time.sleep(delay)

            # All retries exhausted, raise last exception
            if last_exception:
                raise last_exception

            # Should never reach here, but satisfy type checker
            return None

        return cast(F, wrapper)

    return decorator


def retry_on_network_error(max_retries: int = 3, logger: Optional[Callable[[str], None]] = None) -> Callable[[F], F]:
    """Convenience decorator for retrying on network errors only.

    Args:
        max_retries: Maximum number of retry attempts
        logger: Optional logging function

    Returns:
        Decorated function with network error retry logic
    """
    return retry_with_backoff(
        max_retries=max_retries,
        initial_delay=2.0,
        backoff_factor=2,
        retry_on=(ConnectionError, TimeoutError, OSError),
        logger=logger,
    )
