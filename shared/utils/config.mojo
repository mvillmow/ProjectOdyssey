"""Configuration management utilities for ML Odyssey.

This module provides configuration loading, validation, and merging
from YAML and JSON files. Supports nested configurations, environment
variable substitution, and validation rules.

Example:
    from shared.utils import Config

    var config = Config.from_yaml("config.yaml")
    var lr = config.get_float("learning_rate")
    var batch_size = config.get_int("batch_size")
"""

from python import Python


# ============================================================================
# Configuration Value Union Type
# ============================================================================


@value
struct ConfigValue:
    """Union type to hold different configuration value types.

    Supports common types needed for ML configurations: integers, floats,
    strings, booleans, and lists.
    """

    var value_type: String  # "int", "float", "string", "bool", "list"
    var int_val: Int
    var float_val: Float64
    var str_val: String
    var bool_val: Bool
    var list_val: List[String]

    fn __init__(inoutself, value: Int):
        """Create ConfigValue from Int."""
        self.value_type = "int"
        self.int_val = value
        self.float_val = 0.0
        self.str_val = ""
        self.bool_val = False
        self.list_val = List[String]()

    fn __init__(inoutself, value: Float64):
        """Create ConfigValue from Float64."""
        self.value_type = "float"
        self.int_val = 0
        self.float_val = value
        self.str_val = ""
        self.bool_val = False
        self.list_val = List[String]()

    fn __init__(inoutself, value: String):
        """Create ConfigValue from String."""
        self.value_type = "string"
        self.int_val = 0
        self.float_val = 0.0
        self.str_val = value
        self.bool_val = False
        self.list_val = List[String]()

    fn __init__(inoutself, value: Bool):
        """Create ConfigValue from Bool."""
        self.value_type = "bool"
        self.int_val = 0
        self.float_val = 0.0
        self.str_val = ""
        self.bool_val = value
        self.list_val = List[String]()

    fn __init__(inoutself, value: List[String]):
        """Create ConfigValue from List[String]."""
        self.value_type = "list"
        self.int_val = 0
        self.float_val = 0.0
        self.str_val = ""
        self.bool_val = False
        self.list_val = value

    fn __init__(inoutself, value: List[Int]):
        """Create ConfigValue from List[Int]."""
        self.value_type = "list"
        self.int_val = 0
        self.float_val = 0.0
        self.str_val = ""
        self.bool_val = False
        self.list_val = List[String]()
        # Convert int list to string list for storage
        for i in range(len(value)):
            self.list_val.append(str(value[i]))


# ============================================================================
# Config Struct
# ============================================================================


@value
struct Config:
    """Configuration container with nested access and validation.

    Stores configuration as key-value pairs with support for nested
    access using dot notation (e.g., "model.learning_rate").
    """

    var data: Dict[String, ConfigValue]

    fn __init__(inoutself):
        """Create empty configuration."""
        self.data = Dict[String, ConfigValue]()

    fn set(inoutself, key: String, value: Int):
        """Set integer configuration value."""
        self.data[key] = ConfigValue(value)

    fn set(inoutself, key: String, value: Float64):
        """Set float configuration value."""
        self.data[key] = ConfigValue(value)

    fn set(inoutself, key: String, value: String):
        """Set string configuration value."""
        self.data[key] = ConfigValue(value)

    fn set(inoutself, key: String, value: Bool):
        """Set boolean configuration value."""
        self.data[key] = ConfigValue(value)

    fn set(inoutself, key: String, value: List[Int]):
        """Set list of integers configuration value."""
        self.data[key] = ConfigValue(value)

    fn set(inoutself, key: String, value: List[String]):
        """Set list of strings configuration value."""
        self.data[key] = ConfigValue(value)

    fn has(self, key: String) -> Bool:
        """Check if configuration key exists.

        Args:
            key: Configuration key

        Returns:
            True if key exists, False otherwise
        """
        return key in self.data

    fn has_key(self, key: String) -> Bool:
        """Check if configuration key exists (alias for has).

        Args:
            key: Configuration key

        Returns:
            True if key exists, False otherwise
        """
        return self.has(key)

    fn get_string(self, key: String, default: String = "") raises -> String:
        """Get string value with optional default.

        Args:
            key: Configuration key
            default: Value to return if key not found

        Returns:
            String value or default

        Raises:
            Error: If key exists but type is not string
        """
        if key not in self.data:
            return default

        var val = self.data[key]
        if val.value_type != "string":
            raise Error(
                "Type mismatch for key '"
                + key
                + "': expected string but got "
                + val.value_type
            )
        return val.str_val

    fn get_int(self, key: String, default: Int = 0) raises -> Int:
        """Get integer value with optional default.

        Args:
            key: Configuration key
            default: Value to return if key not found

        Returns:
            Integer value or default

        Raises:
            Error: If key exists but type is not int
        """
        if key not in self.data:
            return default

        var val = self.data[key]
        if val.value_type != "int":
            raise Error(
                "Type mismatch for key '"
                + key
                + "': expected int but got "
                + val.value_type
            )
        return val.int_val

    fn get_float(self, key: String, default: Float64 = 0.0) raises -> Float64:
        """Get float value with optional default.

        Args:
            key: Configuration key
            default: Value to return if key not found

        Returns:
            Float value or default

        Raises:
            Error: If key exists but type is not float
        """
        if key not in self.data:
            return default

        var val = self.data[key]
        if val.value_type != "float":
            raise Error(
                "Type mismatch for key '"
                + key
                + "': expected float but got "
                + val.value_type
            )
        return val.float_val

    fn get_bool(self, key: String, default: Bool = False) raises -> Bool:
        """Get boolean value with optional default.

        Args:
            key: Configuration key
            default: Value to return if key not found

        Returns:
            Boolean value or default

        Raises:
            Error: If key exists but type is not bool
        """
        if key not in self.data:
            return default

        var val = self.data[key]
        if val.value_type != "bool":
            raise Error(
                "Type mismatch for key '"
                + key
                + "': expected bool but got "
                + val.value_type
            )
        return val.bool_val

    fn get_list(self, key: String) raises -> List[String]:
        """Get list value.

        Args:
            key: Configuration key

        Returns:
            List value or empty list if not found

        Raises:
            Error: If key exists but type is not list
        """
        if key not in self.data:
            return List[String]()

        var val = self.data[key]
        if val.value_type != "list":
            raise Error(
                "Type mismatch for key '"
                + key
                + "': expected list but got "
                + val.value_type
            )
        return val.list_val

    fn get(self, key: String) -> ConfigValue:
        """Get raw configuration value by key.

        Args:
            key: Configuration key

        Returns:
            Configuration value or default ConfigValue if not found
        """
        if key in self.data:
            return self.data[key]
        return ConfigValue("")  # Default to empty string

    fn get_with_default[T: AnyType](self, key: String, default: T) -> T:
        """Get configuration value with default.

        Args:
            key: Configuration key
            default: Default value to return if key not found

        Returns:
            Value or default
        """
        if not self.has(key):
            return default

        # Type-specific retrieval
        @parameter
        if _type_is_eq[T, Int]():
            return self.get_int(key, default)
        elif _type_is_eq[T, Float64]():
            return self.get_float(key, default)
        elif _type_is_eq[T, String]():
            return self.get_string(key, default)
        elif _type_is_eq[T, Bool]():
            return self.get_bool(key, default)
        else:
            return default

    fn merge(self, other: Config) -> Config:
        """Merge with another config, other takes precedence.

        Args:
            other: Config to merge (overrides self)

        Returns:
            New merged configuration
        """
        var result = Config()

        # Copy all from self
        for item in self.data.items():
            result.data[item[].key] = item[].value

        # Override with other
        for item in other.data.items():
            result.data[item[].key] = item[].value

        return result

    fn validate(self, required_keys: List[String]) raises:
        """Validate that all required keys are present.

        Args:
            required_keys: List of required configuration keys

        Raises:
            Error if any required key is missing
        """
        for i in range(len(required_keys)):
            var key = required_keys[i]
            if not self.has(key):
                raise Error("Missing required configuration key: " + key)

    fn validate_type(self, key: String, type_name: String) raises:
        """Validate that a key has the expected type.

        Args:
            key: Configuration key
            type_name: Expected type name ("int", "float", "string", "bool", "list")

        Raises:
            Error if type doesn't match
        """
        if not self.has(key):
            raise Error("Key not found: " + key)

        var val = self.data[key]
        if val.value_type != type_name:
            raise Error(
                "Type mismatch for key '"
                + key
                + "': expected "
                + type_name
                + " but got "
                + val.value_type
            )

    fn validate_range(
        self, key: String, min_val: Float64, max_val: Float64
    ) raises:
        """Validate that a numeric value is in range.

        Args:
            key: Configuration key
            min_val: Minimum allowed value
            max_val: Maximum allowed value

        Raises:
            Error if value is out of range
        """
        if not self.has(key):
            raise Error("Key not found: " + key)

        var val = self.data[key]
        var num_val: Float64

        if val.value_type == "float":
            num_val = val.float_val
        elif val.value_type == "int":
            num_val = Float64(val.int_val)
        else:
            raise Error("Cannot validate range for non-numeric type")

        if num_val < min_val or num_val > max_val:
            raise Error(
                "Value for key '"
                + key
                + "' is out of range ["
                + str(min_val)
                + ", "
                + str(max_val)
                + "]"
            )

    fn validate_enum(self, key: String, valid_values: List[String]) raises:
        """Validate that a value is one of allowed enum values.

        Args:
            key: Configuration key
            valid_values: List of allowed values

        Raises:
            Error if value is not in valid_values
        """
        if not self.has(key):
            raise Error("Key not found: " + key)

        var val = self.get_string(key)
        var found = False
        for i in range(len(valid_values)):
            if val == valid_values[i]:
                found = True
                break

        if not found:
            raise Error("Invalid value for key '" + key + "': " + val)

    fn validate_exclusive(self, keys: List[String]) raises:
        """Validate that at most one of the mutually exclusive keys is set.

        Args:
            keys: List of mutually exclusive keys

        Raises:
            Error if more than one key is present
        """
        var count = 0
        for i in range(len(keys)):
            if self.has(keys[i]):
                count += 1

        if count > 1:
            raise Error("Mutually exclusive keys found - only one allowed")

    @staticmethod
    fn from_yaml(filepath: String) raises -> Config:
        """Load configuration from YAML file with validation.

        NOTE: Current implementation only supports flat key-value pairs.
        Nested objects and arrays are not yet supported. For complex configs,
        use flattened keys (e.g., "model.learning_rate" instead of nested
        "model: {learning_rate: 0.001}") or consider using Python's PyYAML
        for parsing and converting to Config.

        Args:
            filepath: Path to YAML file

        Returns:
            Loaded configuration

        Raises:
            Error if file not found, empty, or invalid YAML
        """
        # NOTE: Current implementation supports flat key-value pairs.
        # Full nested YAML parsing can be added as needed. Using basic parsing.
        var config = Config()

        try:
            with open(filepath, "r") as f:
                var content = f.read()

                # Validate: Check for empty file
                if len(content.strip()) == 0:
                    raise Error("Config file is empty: " + filepath)

                var lines = content.split("\n")

                for i in range(len(lines)):
                    var line = lines[i].strip()
                    if len(line) == 0 or line.startswith("#"):
                        continue

                    if ":" in line:
                        var parts = line.split(":")
                        if len(parts) >= 2:
                            var key = parts[0].strip()
                            var value_str = parts[1].strip()

                            # Try to parse as number
                            if "." in value_str:
                                try:
                                    var float_val = Float64(atof(value_str))
                                    config.set(key, float_val)
                                except:
                                    config.set(key, value_str)
                            else:
                                try:
                                    var int_val = atol(value_str)
                                    config.set(key, int_val)
                                except:
                                    config.set(key, value_str)
        except e:
            raise Error("Failed to load YAML file: " + str(e))

        return config

    @staticmethod
    fn from_json(filepath: String) raises -> Config:
        """Load configuration from JSON file with validation.

        NOTE: Current implementation only supports flat key-value pairs.
        Nested objects and arrays are not yet supported. For complex configs,
        use flattened keys (e.g., "model.learning_rate" instead of nested
        {"model": {"learning_rate": 0.001}}) or consider using Python's json
        module for parsing and converting to Config.

        Args:
            filepath: Path to JSON file

        Returns:
            Loaded configuration

        Raises:
            Error if file not found, empty, or invalid JSON
        """
        # NOTE: Current implementation supports flat key-value pairs.
        # Full nested JSON parsing can be added as needed. Using basic parsing.
        var config = Config()

        try:
            with open(filepath, "r") as f:
                var content = f.read()

                # Validate: Check for empty file
                if len(content.strip()) == 0:
                    raise Error("Config file is empty: " + filepath)

                # Remove braces and parse key:value pairs
                var clean = (
                    content.replace("{", "").replace("}", "").replace('"', "")
                )
                var pairs = clean.split(",")

                for i in range(len(pairs)):
                    var pair = pairs[i].strip()
                    if ":" in pair:
                        var parts = pair.split(":")
                        if len(parts) >= 2:
                            var key = parts[0].strip()
                            var value_str = parts[1].strip()

                            # Try to parse as number
                            if "." in value_str:
                                try:
                                    var float_val = Float64(atof(value_str))
                                    config.set(key, float_val)
                                except:
                                    config.set(key, value_str)
                            else:
                                try:
                                    var int_val = atol(value_str)
                                    config.set(key, int_val)
                                except:
                                    config.set(key, value_str)
        except e:
            raise Error("Failed to load JSON file: " + str(e))

        return config

    fn to_yaml(self, filepath: String) raises:
        """Save configuration to YAML file.

        Args:
            filepath: Output file path

        Raises:
            Error if file cannot be written
        """
        try:
            with open(filepath, "w") as f:
                for item in self.data.items():
                    var key = item[].key
                    var val = item[].value

                    if val.value_type == "int":
                        _ = f.write(key + ": " + str(val.int_val) + "\n")
                    elif val.value_type == "float":
                        _ = f.write(key + ": " + str(val.float_val) + "\n")
                    elif val.value_type == "string":
                        _ = f.write(key + ': "' + val.str_val + '"\n')
                    elif val.value_type == "bool":
                        _ = f.write(key + ": " + str(val.bool_val) + "\n")
                    elif val.value_type == "list":
                        _ = f.write(key + ": [")
                        for i in range(len(val.list_val)):
                            _ = f.write('"' + val.list_val[i] + '"')
                            if i < len(val.list_val) - 1:
                                _ = f.write(", ")
                        _ = f.write("]\n")
        except e:
            raise Error("Failed to save YAML file: " + str(e))

    fn to_json(self, filepath: String) raises:
        """Save configuration to JSON file.

        Args:
            filepath: Output file path

        Raises:
            Error if file cannot be written
        """
        try:
            with open(filepath, "w") as f:
                _ = f.write("{\n")

                var count = 0
                var total = len(self.data)

                for item in self.data.items():
                    var key = item[].key
                    var val = item[].value

                    _ = f.write('  "' + key + '": ')

                    if val.value_type == "int":
                        _ = f.write(str(val.int_val))
                    elif val.value_type == "float":
                        _ = f.write(str(val.float_val))
                    elif val.value_type == "string":
                        _ = f.write('"' + val.str_val + '"')
                    elif val.value_type == "bool":
                        _ = f.write(str(val.bool_val))
                    elif val.value_type == "list":
                        _ = f.write("[")
                        for i in range(len(val.list_val)):
                            _ = f.write('"' + val.list_val[i] + '"')
                            if i < len(val.list_val) - 1:
                                _ = f.write(", ")
                        _ = f.write("]")

                    count += 1
                    if count < total:
                        _ = f.write(",\n")
                    else:
                        _ = f.write("\n")

                _ = f.write("}\n")
        except e:
            raise Error("Failed to save JSON file: " + str(e))

    fn substitute_env_vars(self) -> Config:
        """Substitute environment variables in config values.

        Replaces ${VAR_NAME} with environment variable values.
        Supports default syntax: ${VAR_NAME:-default}

        Returns:
            New config with substituted values
        """
        var result = Config()

        for item in self.data.items():
            var key = item[].key
            var val = item[].value

            if val.value_type == "string":
                var str_val = val.str_val
                var new_val = self._substitute_env_in_string(str_val)
                result.set(key, new_val)
            else:
                result.data[key] = val

        return result

    fn _substitute_env_in_string(self, value: String) -> String:
        """Helper to substitute environment variables in a string.

        Replaces ${VAR} or ${VAR:-default} patterns with environment values.

        Args:
            value: String that may contain ${VAR} patterns

        Returns:
            String with substituted values
        """
        var result = value

        # Find all ${...} patterns and replace them
        # Simple implementation without regex - iterate and find patterns
        var start_pos = 0
        while True:
            var dollar_pos = result.find("${", start_pos)
            if dollar_pos == -1:
                break

            var close_pos = result.find("}", dollar_pos)
            if close_pos == -1:
                break  # Malformed pattern, skip

            # Extract variable spec: VAR or VAR:-default
            var var_spec = result[dollar_pos + 2 : close_pos]

            # Check for default value syntax: VAR:-default
            var default_value = ""
            var var_name = var_spec
            var colon_pos = var_spec.find(":-")
            if colon_pos != -1:
                var_name = var_spec[:colon_pos]
                default_value = var_spec[colon_pos + 2 :]

            # Get environment variable value using Python
            var env_value = default_value
            try:
                var python = Python.import_module("os")
                var py_value = python.getenv(var_name, default_value)
                env_value = str(py_value)
            except:
                # If Python interop fails, use default value
                pass

            # Replace ${...} with environment value
            var before = result[:dollar_pos]
            var after = result[close_pos + 1 :]
            result = before + env_value + after

            # Move start position past the replaced value
            start_pos = len(before) + len(env_value)

        return result

    @staticmethod
    fn load_template(name: String) -> Config:
        """Load a predefined configuration template.

        Args:
            name: Template name (e.g., "training_default", "lenet5")

        Returns:
            Template configuration
        """
        var config = Config()

        if name == "training_default":
            config.set("learning_rate", 0.001)
            config.set("batch_size", 32)
            config.set("epochs", 10)
            config.set("optimizer", "sgd")
        elif name == "lenet5":
            config.set("layers", 3)
            config.set("activation", "relu")
            config.set("input_shape", 28)

        return config


# ============================================================================
# Configuration Loading (Legacy Functions)
# ============================================================================


fn load_config(filepath: String) raises -> Config:
    """Load configuration from YAML or JSON file.

    Automatically detects file format based on extension (.yaml/.yml
    for YAML, .json for JSON).

    Args:
        filepath: Path to configuration file

    Returns:
        Loaded configuration

    Raises:
        Error if file doesn't exist or invalid format

    Example:
        var config = load_config("configs/lenet5.yaml")
        var lr = config.get_float("learning_rate")
    """
    if filepath.endswith(".yaml") or filepath.endswith(".yml"):
        return Config.from_yaml(filepath)
    elif filepath.endswith(".json"):
        return Config.from_json(filepath)
    else:
        raise Error("Unknown file format - use .yaml, .yml, or .json")


fn save_config(config: Config, filepath: String) raises:
    """Save configuration to YAML or JSON file.

    Automatically determines output format from file extension.

    Args:
        config: Configuration to save
        filepath: Output file path

    Raises:
        Error if file cannot be written

    Example:
        var config = Config()
        config.set("learning_rate", 0.001)
        save_config(config, "config.yaml")
    """
    if filepath.endswith(".yaml") or filepath.endswith(".yml"):
        config.to_yaml(filepath)
    elif filepath.endswith(".json"):
        config.to_json(filepath)
    else:
        raise Error("Unknown file format - use .yaml, .yml, or .json")


# ============================================================================
# Configuration Merging
# ============================================================================


fn merge_configs(base: Config, override: Config) -> Config:
    """Merge two configurations with override taking precedence.

    Creates a new configuration that combines both inputs, with values
    from override taking precedence over base. Useful for creating
    experiment-specific configs that override defaults.

    Args:
        base: Base configuration (defaults)
        override: Override configuration (experiment-specific)

    Returns:
        Merged configuration

    Example:
        var defaults = load_config("config.yaml")
        var experiment = load_config("config.lenet5.yaml")
        var config = merge_configs(defaults, experiment)
    """
    return base.merge(override)


# ============================================================================
# Configuration Validation
# ============================================================================


struct ConfigValidator:
    """Validator for configuration values."""

    var required_keys: List[String]
    var allowed_keys: Dict[String, String]  # key -> type name

    fn __init__(inoutself):
        """Create empty validator."""
        self.required_keys = List[String]()
        self.allowed_keys = Dict[String, String]()

    fn require(inoutself, key: String) -> Self:
        """Mark key as required.

        Args:
            key: Configuration key

        Returns:
            Self for method chaining
        """
        self.required_keys.append(key)
        return self

    fn allow(inoutself, key: String, type_name: String) -> Self:
        """Mark key as allowed with specific type.

        Args:
            key: Configuration key
            type_name: Expected type name

        Returns:
            Self for method chaining
        """
        self.allowed_keys[key] = type_name
        return self

    fn validate(self, config: Config) -> Bool:
        """Validate configuration against rules.

        Args:
            config: Configuration to validate

        Returns:
            True if valid, False otherwise
        """
        # Check required keys
        for i in range(len(self.required_keys)):
            if not config.has(self.required_keys[i]):
                return False

        return True


fn create_validator() -> ConfigValidator:
    """Create new configuration validator.

    Returns:
        Empty validator ready for configuration
    """
    return ConfigValidator()
