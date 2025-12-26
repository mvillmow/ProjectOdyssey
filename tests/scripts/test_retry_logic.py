#!/usr/bin/env python3
"""Tests for retry logic with exponential backoff.

Tests:
- Successful operation (no retries)
- Transient failure then success
- All retries exhausted
- Exponential backoff timing
- Network error detection
- Logger integration
"""

import pathlib
import sys
import time
import unittest
from unittest.mock import Mock

# Add scripts directory to path
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent.parent / "scripts"))

from utils.retry import (
    is_network_error,
    retry_on_network_error,
    retry_with_backoff,
)


class TestNetworkErrorDetection(unittest.TestCase):
    """Test network error keyword detection."""

    def test_detects_connection_errors(self):
        """Verify connection errors are detected."""
        error = ConnectionError("connection refused")
        self.assertTrue(is_network_error(error))

    def test_detects_timeout_errors(self):
        """Verify timeout errors are detected."""
        error = TimeoutError("operation timed out")
        self.assertTrue(is_network_error(error))

    def test_detects_dns_errors(self):
        """Verify DNS resolution errors are detected."""
        error = OSError("could not resolve hostname")
        self.assertTrue(is_network_error(error))

    def test_detects_rate_limit_errors(self):
        """Verify rate limit errors are detected."""
        error = Exception("rate limit exceeded")
        self.assertTrue(is_network_error(error))

    def test_detects_http_errors(self):
        """Verify HTTP 5xx errors are detected."""
        for code in ["502", "503", "504"]:
            error = Exception(f"HTTP {code} Bad Gateway")
            self.assertTrue(is_network_error(error), f"Failed to detect {code}")

    def test_ignores_non_network_errors(self):
        """Verify non-network errors are not detected."""
        error = ValueError("invalid input")
        self.assertFalse(is_network_error(error))


class TestRetryWithBackoff(unittest.TestCase):
    """Test retry_with_backoff decorator."""

    def test_success_no_retry(self):
        """Verify successful operation doesn't retry."""
        mock_func = Mock(return_value="success")
        decorated = retry_with_backoff(max_retries=3)(mock_func)

        result = decorated()

        self.assertEqual(result, "success")
        self.assertEqual(mock_func.call_count, 1)

    def test_transient_failure_then_success(self):
        """Verify retry on transient failure."""
        mock_func = Mock(side_effect=[ConnectionError("network error"), "success"])
        decorated = retry_with_backoff(max_retries=3, initial_delay=0.01)(mock_func)

        result = decorated()

        self.assertEqual(result, "success")
        self.assertEqual(mock_func.call_count, 2)

    def test_all_retries_exhausted(self):
        """Verify exception raised after max retries."""
        mock_func = Mock(side_effect=ConnectionError("permanent failure"))
        decorated = retry_with_backoff(max_retries=3, initial_delay=0.01)(mock_func)

        with self.assertRaises(ConnectionError) as ctx:
            decorated()

        self.assertIn("permanent failure", str(ctx.exception))
        self.assertEqual(mock_func.call_count, 4)  # Initial + 3 retries

    def test_exponential_backoff_timing(self):
        """Verify exponential backoff delays."""
        call_times = []

        def failing_func():
            call_times.append(time.time())
            raise ConnectionError("fail")

        decorated = retry_with_backoff(max_retries=3, initial_delay=0.1, backoff_factor=2)(failing_func)

        with self.assertRaises(ConnectionError):
            decorated()

        # Verify delays: ~0.1s, ~0.2s, ~0.4s
        self.assertEqual(len(call_times), 4)  # Initial + 3 retries

        # Check approximate delays (allow 50ms tolerance)
        delay1 = call_times[1] - call_times[0]
        delay2 = call_times[2] - call_times[1]
        delay3 = call_times[3] - call_times[2]

        self.assertAlmostEqual(delay1, 0.1, delta=0.05)  # 0.1 * 2^0
        self.assertAlmostEqual(delay2, 0.2, delta=0.05)  # 0.1 * 2^1
        self.assertAlmostEqual(delay3, 0.4, delta=0.05)  # 0.1 * 2^2

    def test_logger_integration(self):
        """Verify logger is called with retry information."""
        mock_logger = Mock()
        mock_func = Mock(side_effect=[ConnectionError("network error"), "success"])

        decorated = retry_with_backoff(max_retries=3, initial_delay=0.01, logger=mock_logger)(mock_func)

        result = decorated()

        self.assertEqual(result, "success")
        self.assertEqual(mock_logger.call_count, 1)

        # Verify log message contains retry info
        log_msg = mock_logger.call_args[0][0]
        self.assertIn("Retry 1/3", log_msg)
        self.assertIn("ConnectionError", log_msg)
        self.assertIn("[NETWORK]", log_msg)  # Network error detected

    def test_retry_on_specific_exceptions(self):
        """Verify retry_on parameter filters exceptions."""

        @retry_with_backoff(max_retries=2, initial_delay=0.01, retry_on=(ConnectionError,))
        def func():
            raise ValueError("not retryable")

        # ValueError not in retry_on, should raise immediately
        with self.assertRaises(ValueError):
            func()

    def test_preserves_function_metadata(self):
        """Verify decorator preserves function name and docstring."""

        @retry_with_backoff(max_retries=3)
        def my_function():
            """My docstring."""
            return "result"

        self.assertEqual(my_function.__name__, "my_function")
        self.assertEqual(my_function.__doc__, "My docstring.")


class TestRetryOnNetworkError(unittest.TestCase):
    """Test retry_on_network_error convenience decorator."""

    def test_retries_connection_errors(self):
        """Verify retries ConnectionError."""
        mock_func = Mock(side_effect=[ConnectionError("fail"), "success"])
        decorated = retry_on_network_error(max_retries=3)(mock_func)

        result = decorated()

        self.assertEqual(result, "success")
        self.assertEqual(mock_func.call_count, 2)

    def test_retries_timeout_errors(self):
        """Verify retries TimeoutError."""
        mock_func = Mock(side_effect=[TimeoutError("fail"), "success"])
        decorated = retry_on_network_error(max_retries=3)(mock_func)

        result = decorated()

        self.assertEqual(result, "success")
        self.assertEqual(mock_func.call_count, 2)

    def test_retries_os_errors(self):
        """Verify retries OSError."""
        mock_func = Mock(side_effect=[OSError("fail"), "success"])
        decorated = retry_on_network_error(max_retries=3)(mock_func)

        result = decorated()

        self.assertEqual(result, "success")
        self.assertEqual(mock_func.call_count, 2)

    def test_does_not_retry_other_errors(self):
        """Verify does not retry non-network errors."""
        mock_func = Mock(side_effect=ValueError("not network"))
        decorated = retry_on_network_error(max_retries=3)(mock_func)

        with self.assertRaises(ValueError):
            decorated()

        self.assertEqual(mock_func.call_count, 1)  # No retry


class TestRetryWithArguments(unittest.TestCase):
    """Test retry decorator with function arguments."""

    def test_preserves_args(self):
        """Verify decorator preserves positional arguments."""
        mock_func = Mock(return_value="success")
        decorated = retry_with_backoff(max_retries=3)(mock_func)

        result = decorated("arg1", "arg2")

        self.assertEqual(result, "success")
        mock_func.assert_called_once_with("arg1", "arg2")

    def test_preserves_kwargs(self):
        """Verify decorator preserves keyword arguments."""
        mock_func = Mock(return_value="success")
        decorated = retry_with_backoff(max_retries=3)(mock_func)

        result = decorated(key1="value1", key2="value2")

        self.assertEqual(result, "success")
        mock_func.assert_called_once_with(key1="value1", key2="value2")

    def test_preserves_mixed_args(self):
        """Verify decorator preserves mixed arguments."""
        mock_func = Mock(return_value="success")
        decorated = retry_with_backoff(max_retries=3)(mock_func)

        result = decorated("arg1", key="value")

        self.assertEqual(result, "success")
        mock_func.assert_called_once_with("arg1", key="value")


if __name__ == "__main__":
    unittest.main()
