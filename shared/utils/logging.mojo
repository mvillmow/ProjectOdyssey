"""Logging utilities for ML Odyssey training and debugging.

This module provides structured logging capabilities with multiple handlers
and formatters. It supports file and console output with configurable
log levels.

Example:
    from shared.utils import Logger, StreamHandler, FileHandler

    var logger = Logger("training")
    logger.add_handler(StreamHandler())
    logger.add_handler(FileHandler("training.log"))

    logger.info("Training started")
    logger.warning("High loss detected")
    logger.error("Validation failed")
"""


# ============================================================================
# Log Level Enumeration
# ============================================================================


@value
struct LogLevel:
    """Log level enumeration with numeric values for comparison.

    Levels are ordered from least (DEBUG) to most (CRITICAL) severe.
    Log filtering uses numeric comparison: logger only outputs
    messages with level >= logger's configured level.
    """

    alias DEBUG = 10
    alias INFO = 20
    alias WARNING = 30
    alias ERROR = 40
    alias CRITICAL = 50


# ============================================================================
# Log Record
# ============================================================================


struct LogRecord:
    """Record of a single log message.

    Contains all information needed by handlers to format and output
    a log message including the logger name, level, message, and timestamp.
    """

    var logger_name: String
    var level: Int
    var message: String
    var timestamp: String

    fn __init__(
        out self,
        logger_name: String,
        level: Int,
        message: String,
        timestamp: String = "",
    ):
        """Initialize a log record.

        Args:
            logger_name: Name of the logger that created this record
            level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            message: The log message
            timestamp: Optional timestamp (auto-generated if empty)
        """
        self.logger_name = logger_name
        self.level = level
        self.message = message
        self.timestamp = timestamp if timestamp else self._get_timestamp()

    fn _get_timestamp(self) -> String:
        """Generate current timestamp string.

        Uses time.now() to get current Unix timestamp and formats it
        as ISO 8601 basic format. Mojo v0.25.7 lacks full datetime
        support, so this is a simplified implementation.

        Returns:
            Timestamp string in format "YYYY-MM-DD HH:MM:SS"
        """
        # NOTE: This is a simplified implementation until Mojo
        # stdlib includes full datetime support
        from time import now

        # Get Unix timestamp in nanoseconds
        var timestamp_ns = now()
        var timestamp_s = timestamp_ns // 1_000_000_000

        # Calculate time components
        var seconds_per_day = 86400
        var days_since_epoch = timestamp_s // seconds_per_day
        var seconds_today = timestamp_s % seconds_per_day

        var hours = seconds_today // 3600
        var minutes = (seconds_today % 3600) // 60
        var seconds = seconds_today % 60

        # Approximate date calculation (starting from 1970-01-01)
        # This is simplified and doesn't account for leap years
        var year = 1970 + (days_since_epoch // 365)
        var day_of_year = days_since_epoch % 365
        var month = (day_of_year // 30) + 1  # Rough approximation
        var day = (day_of_year % 30) + 1

        # Format as YYYY-MM-DD HH:MM:SS using string concatenation
        # (Mojo may not support all Python string formatting features)
        return (
            str(year)
            + "-"
            + str(month).zfill(2)
            + "-"
            + str(day).zfill(2)
            + " "
            + str(hours).zfill(2)
            + ":"
            + str(minutes).zfill(2)
            + ":"
            + str(seconds).zfill(2)
        )

    fn level_name(self) -> String:
        """Get human-readable level name."""
        if self.level == LogLevel.DEBUG:
            return "DEBUG"
        elif self.level == LogLevel.INFO:
            return "INFO"
        elif self.level == LogLevel.WARNING:
            return "WARNING"
        elif self.level == LogLevel.ERROR:
            return "ERROR"
        elif self.level == LogLevel.CRITICAL:
            return "CRITICAL"
        else:
            return "UNKNOWN"


# ============================================================================
# Formatter Trait and Implementations
# ============================================================================


trait Formatter:
    """Base formatter interface for log messages."""

    fn format(self, record: LogRecord) -> String:
        """Format a log record into a string."""
        ...


struct SimpleFormatter(Formatter):
    """Simple formatter: [LEVEL] message"""

    fn format(self, record: LogRecord) -> String:
        """Format log record as: [LEVEL] message"""
        return f"[{record.level_name()}] {record.message}"


struct TimestampFormatter(Formatter):
    """Timestamp formatter: YYYY-MM-DD HH:MM:SS [LEVEL] message"""

    fn format(self, record: LogRecord) -> String:
        """Format log record with timestamp."""
        return f"{record.timestamp} [{record.level_name()}] {record.message}"


struct DetailedFormatter(Formatter):
    """Detailed formatter: [LEVEL] logger_name - message"""

    fn format(self, record: LogRecord) -> String:
        """Format log record with logger name."""
        return (
            f"[{record.level_name()}] {record.logger_name} - {record.message}"
        )


struct ColoredFormatter(Formatter):
    """Colored formatter using ANSI escape codes."""

    # ANSI color codes
    alias RED = "\033[91m"
    alias YELLOW = "\033[93m"
    alias GREEN = "\033[92m"
    alias BLUE = "\033[94m"
    alias RESET = "\033[0m"

    fn format(self, record: LogRecord) -> String:
        """Format log record with ANSI color codes."""
        var color = self._get_color(record.level)
        return f"{color}[{record.level_name()}]{self.RESET} {record.message}"

    fn _get_color(self, level: Int) -> String:
        """Get ANSI color for level."""
        if level == LogLevel.ERROR or level == LogLevel.CRITICAL:
            return self.RED
        elif level == LogLevel.WARNING:
            return self.YELLOW
        elif level == LogLevel.INFO:
            return self.GREEN
        else:  # DEBUG
            return self.BLUE


# ============================================================================
# Handler Trait and Implementations
# ============================================================================


trait Handler:
    """Base handler interface for log output."""

    fn emit(self, record: LogRecord):
        """Output a log record."""
        ...


struct StreamHandler(Handler):
    """Write log messages to stdout/stderr."""

    var formatter: SimpleFormatter

    fn __init__(out self):
        """Create stream handler with default formatter."""
        self.formatter = SimpleFormatter()

    fn emit(self, record: LogRecord):
        """Write formatted log record to stdout."""
        var formatted = self.formatter.format(record)
        print(formatted)


struct FileHandler(Handler):
    """Write log messages to a file."""

    var filepath: String
    var formatter: TimestampFormatter

    fn __init__(out self, filepath: String):
        """Create file handler that writes to given file.

        Args:
            filepath: Path to log file to write to
        """
        self.filepath = filepath
        self.formatter = TimestampFormatter()

    fn emit(self, record: LogRecord):
        """Write formatted log record to file."""
        var formatted = self.formatter.format(record)
        self._write_to_file(formatted)

    fn _write_to_file(self, message: String):
        """Write message to log file (append mode).

        Opens file in append mode, writes the message with newline,
        and closes the file. If file cannot be opened, falls back
        to stderr printing.

        Args:
            message: Formatted log message to write
        """
        try:
            # Open file in append mode
            with open(self.filepath, "a") as file:
                _ = file.write(message + "\n")
        except:
            # Fallback to stderr if file write fails
            print("[LOG ERROR] Failed to write to", self.filepath, file=2)
            print(message, file=2)


# ============================================================================
# Logger Class
# ============================================================================


struct Logger:
    """Structured logger with multiple output handlers.

    Supports configurable log levels, multiple handlers, and various
    formatters for different output formats. Log messages are filtered
    by the configured level threshold.
    """

    var name: String
    var level: Int
    var handlers: List[StreamHandler]  # For now just StreamHandler

    fn __init__(out self, name: String, level: Int = LogLevel.INFO):
        """Create logger with name and optional level.

        Args:
            name: Logger name (e.g., "training", "evaluation")
            level: Minimum log level to output (default: INFO)
        """
        self.name = name
        self.level = level
        self.handlers = List[StreamHandler]()

    fn add_handler(mut self, handler: StreamHandler):
        """Add an output handler to this logger.

        Handlers receive all log records that pass the level filter.

        Args:
            handler: Handler to add
        """
        self.handlers.append(handler)

    fn debug(self, message: String):
        """Log a debug message (lowest priority).

        Args:
            message: Message to log
        """
        if self.level <= LogLevel.DEBUG:
            self._log(LogLevel.DEBUG, message)

    fn info(self, message: String):
        """Log an info message (normal priority).

        Args:
            message: Message to log
        """
        if self.level <= LogLevel.INFO:
            self._log(LogLevel.INFO, message)

    fn warning(self, message: String):
        """Log a warning message (medium priority).

        Args:
            message: Message to log
        """
        if self.level <= LogLevel.WARNING:
            self._log(LogLevel.WARNING, message)

    fn error(self, message: String):
        """Log an error message (high priority).

        Args:
            message: Message to log
        """
        if self.level <= LogLevel.ERROR:
            self._log(LogLevel.ERROR, message)

    fn critical(self, message: String):
        """Log a critical message (highest priority).

        Args:
            message: Message to log
        """
        if self.level <= LogLevel.CRITICAL:
            self._log(LogLevel.CRITICAL, message)

    fn _log(self, level: Int, message: String):
        """Internal method to create and emit log record.

        Args:
            level: Log level for this message
            message: Message to log
        """
        var record = LogRecord(self.name, level, message)
        for handler in self.handlers:
            handler.emit(record)

    fn set_level(mut self, level: Int):
        """Change the log level for this logger.

        Args:
            level: New log level threshold
        """
        self.level = level


# ============================================================================
# Module-level Functions
# ============================================================================


var _default_logger: Logger


fn _init_default_logger():
    """Initialize the default logger (module initialization)."""
    _default_logger = Logger("ml-odyssey")


fn get_logger(name: String, level: Int = LogLevel.INFO) -> Logger:
    """Get or create a named logger.

    Args:
        name: Logger name
        level: Log level threshold (default: INFO)

    Returns:
        Logger with specified name and level

    Example:
        var logger = get_logger("training")
        logger.info("Training started")
    """
    return Logger(name, level)
