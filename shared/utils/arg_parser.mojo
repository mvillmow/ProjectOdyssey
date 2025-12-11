"""Argument parser for training and utility scripts.

This module provides a simple argument parser for command-line applications,
with support for typed arguments (int, float, string, bool) and flags.

Example:
    from shared.utils import ArgumentParser

    var parser = ArgumentParser()
    parser.add_argument("epochs", "int", "100")
    parser.add_argument("batch-size", "int", "32")
    parser.add_argument("lr", "float", "0.001")
    parser.add_argument("output", "string", "model.weights")
    parser.add_flag("verbose")

    var args = parser.parse()
    var epochs = args.get_int("epochs")
    var batch_size = args.get_int("batch-size")
    var learning_rate = args.get_float("lr")
    var output_path = args.get_string("output")
    var verbose = args.get_bool("verbose")
    ```
"""

from sys import argv
from collections import Dict


# ============================================================================
# Argument Specification
# ============================================================================


@fieldwise_init
struct ArgumentSpec(Copyable, Movable):
    """Specification for a single command-line argument.

    Attributes:
        name: Argument name (e.g., "epochs", "batch-size")
        arg_type: Type string ("int", "float", "string", "bool")
        default_value: Default value as string (parsed based on arg_type)
        is_flag: Whether this is a boolean flag (--flag with no value)
    """

    var name: String
    var arg_type: String
    var default_value: String
    var is_flag: Bool


# ============================================================================
# Parsed Arguments Container
# ============================================================================


struct ParsedArgs(Copyable, Movable):
    """Container for parsed command-line arguments.

    Stores argument values as strings internally, with typed getters
    to retrieve values in the appropriate type
    """

    var values: Dict[String, String]

    fn __init__(out self):
        """Initialize empty parsed arguments."""
        self.values = Dict[String, String]()

    fn set(mut self, name: String, value: String):
        """Set an argument value.

        Args:
            name: Argument name
            value: Value as string
        """
        self.values[name] = value

    fn has(self, name: String) -> Bool:
        """Check if an argument was provided.

        Args:
            name: Argument name

        Returns:
            True if argument exists, False otherwise
        """
        return name in self.values

    fn get_string(self, name: String, default: String = "") raises -> String:
        """Get argument value as string.

        Args:
            name: Argument name
            default: Default value if not provided

        Returns:
            String value or default
        """
        if name in self.values:
            return self.values[name]
        return default

    fn get_int(self, name: String, default: Int = 0) raises -> Int:
        """Get argument value as integer.

        Args:
            name: Argument name
            default: Default value if not provided

        Returns:
            Integer value or default

        Raises:
            Error if value cannot be parsed as integer
        """
        if name not in self.values:
            return default
        var value = self.values[name]
        try:
            return Int(value)
        except:
            raise Error(
                "Cannot parse '"
                + value
                + "' as integer for argument '"
                + name
                + "'"
            )

    fn get_float(self, name: String, default: Float64 = 0.0) raises -> Float64:
        """Get argument value as float.

        Args:
            name: Argument name
            default: Default value if not provided

        Returns:
            Float64 value or default

        Raises:
            Error if value cannot be parsed as float
        """
        if name not in self.values:
            return default
        var value = self.values[name]
        try:
            return Float64(value)
        except:
            raise Error(
                "Cannot parse '"
                + value
                + "' as float for argument '"
                + name
                + "'"
            )

    fn get_bool(self, name: String) -> Bool:
        """Get boolean flag status.

        Args:
            name: Argument name

        Returns:
            True if flag was provided, False otherwise
        """
        return self.has(name)


# ============================================================================
# Argument Parser
# ============================================================================


struct ArgumentParser(Copyable, Movable):
    """Simple command-line argument parser.

    Supports typed arguments with defaults and boolean flags
    Arguments are specified with add_argument() or add_flag(),
    then parsed from sys.argv with parse()

    Example:
        ```mojo
        var parser = ArgumentParser()
        parser.add_argument("epochs", "int", "100")
        parser.add_flag("verbose")
        var args = parser.parse()
        var epochs = args.get_int("epochs")
        var verbose = args.get_bool("verbose")
        ```
    """

    var arguments: Dict[String, ArgumentSpec]

    fn __init__(out self):
        """Initialize empty argument parser."""
        self.arguments = Dict[String, ArgumentSpec]()

    fn add_argument(
        mut self,
        name: String,
        arg_type: String,
        default: String = "",
    ) raises:
        """Add a typed argument specification.

        Args:
            name: Argument name (e.g., "epochs", "batch-size")
            arg_type: Type string ("int", "float", "string", "bool")
            default: Default value as string

        Raises:
            Error if arg_type is not recognized
        """
        # Validate type
        if (
            arg_type != "int"
            and arg_type != "float"
            and arg_type != "string"
            and arg_type != "bool"
        ):
            raise Error("Unknown argument type: " + arg_type)

        var spec = ArgumentSpec(
            name=name, arg_type=arg_type, default_value=default, is_flag=False
        )
        self.arguments[name] = spec^

    fn add_flag(mut self, name: String):
        """Add a boolean flag argument.

        Flags are provided as --name without a value

        Args:
            name: Flag name (e.g., "verbose", "debug")
        """
        var spec = ArgumentSpec(
            name=name, arg_type="bool", default_value="", is_flag=True
        )
        self.arguments[name] = spec^

    fn parse(self) raises -> ParsedArgs:
        """Parse command-line arguments from sys.argv

        Returns:
            ParsedArgs container with parsed values

        Raises:
            Error if argument parsing fails
        """
        var result = ParsedArgs()

        # Initialize with defaults (using ref for non-copyable dict entries)
        for ref item in self.arguments.items():
            var name = item.key
            # Access value fields directly to avoid implicit copy
            if not item.value.is_flag and len(item.value.default_value) > 0:
                result.set(name, item.value.default_value)

        # Parse sys.argv
        var args = argv()
        var i = 1  # Skip program name
        while i < len(args):
            var arg = args[i]

            # Check if it's a flag starting with --
            if arg.startswith("--"):
                var arg_name = arg[2:]  # Remove -- prefix

                # Check if this is a known argument
                if arg_name not in self.arguments:
                    raise Error("Unknown argument: --" + arg_name)

                # Access spec attributes directly
                var is_flag = self.arguments[arg_name].is_flag

                if is_flag:
                    # Boolean flag - no value needed
                    result.set(arg_name, "true")
                    i += 1
                else:
                    # Typed argument - expect a value
                    if i + 1 >= len(args):
                        raise Error("Missing value for argument: --" + arg_name)
                    var value = args[i + 1]
                    result.set(arg_name, value)
                    i += 2
            else:
                raise Error("Invalid argument format: " + arg)

        return result^


# ============================================================================
# Helper Functions
# ============================================================================


fn create_parser() raises -> ArgumentParser:
    """Create a new argument parser.

    Returns:
            New ArgumentParser instance
    """
    return ArgumentParser()


# ============================================================================
# ML Training Argument Helpers
# ============================================================================


fn create_training_parser() raises -> ArgumentParser:
    """Create argument parser with common ML training arguments.

        Provides predefined arguments for:
            - epochs (int, default 100)
            - batch-size (int, default 32)
            - lr/learning-rate (float, default 0.001)
            - momentum (float, default 0.9)
            - weight-decay (float, default 0.0)
            - model-path (string, default "model.weights")
            - data-dir (string, default "datasets")
            - seed (int, default 42)
            - verbose (flag)

    Returns:
            ArgumentParser configured with ML training arguments
    """
    var parser = ArgumentParser()

    parser.add_argument("epochs", "int", "100")
    parser.add_argument("batch-size", "int", "32")
    parser.add_argument("lr", "float", "0.001")
    parser.add_argument("learning-rate", "float", "0.001")
    parser.add_argument("momentum", "float", "0.9")
    parser.add_argument("weight-decay", "float", "0.0")
    parser.add_argument("model-path", "string", "model.weights")
    parser.add_argument("data-dir", "string", "datasets")
    parser.add_argument("seed", "int", "42")
    parser.add_flag("verbose")

    return parser^


fn validate_positive_int(value: Int, name: String) raises:
    """Validate that an integer argument is positive.

    Args:
            value: Value to validate
            name: Argument name for error messages

    Raises:
            Error if value is not positive
    """
    if value <= 0:
        raise Error(name + " must be positive, got: " + String(value))


fn validate_positive_float(value: Float64, name: String) raises:
    """Validate that a float argument is positive.

    Args:
            value: Value to validate
            name: Argument name for error messages

    Raises:
            Error if value is not positive
    """
    if value <= 0.0:
        raise Error(name + " must be positive, got: " + String(value))


fn validate_range_float(
    value: Float64, min_val: Float64, max_val: Float64, name: String
) raises:
    """Validate that a float argument is within a range.

    Args:
            value: Value to validate
            min_val: Minimum allowed value (inclusive)
            max_val: Maximum allowed value (inclusive)
            name: Argument name for error messages

    Raises:
            Error if value is outside the range
    """
    if value < min_val or value > max_val:
        raise Error(
            name
            + " must be between "
            + String(min_val)
            + " and "
            + String(max_val)
            + ", got: "
            + String(value)
        )
