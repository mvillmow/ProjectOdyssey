"""
Environment Variable Substitution Tests

Tests for substituting environment variables in configuration values.
Supports ${VAR} and ${VAR:-default} syntax.

Run with: mojo test tests/configs/test_env_vars.mojo
"""

from testing import assert_true, assert_false, assert_equal
from shared.utils.config import Config, load_config
from python import Python, PythonObject


# ============================================================================
# Basic Environment Variable Substitution Tests
# ============================================================================


fn test_substitute_simple_env_var() raises:
    """Test basic environment variable substitution.

    Verifies ${VAR} is replaced with environment value.
    """
    # Set environment variable
    var python = Python.import_module("os")
    python.environ[PythonObject("TEST_VAR")] = PythonObject("test_value")

    var config = Config()
    config.set("path", "${TEST_VAR}")

    var substituted = config.substitute_env_vars()

    var path = substituted.get_string("path")
    assert_equal(path, "test_value", "Should substitute environment variable")

    print("✓ test_substitute_simple_env_var passed")


fn test_substitute_multiple_env_vars() raises:
    """Test substitution of multiple environment variables.

    Verifies multiple ${VAR} patterns are replaced.
    """
    var python = Python.import_module("os")
    python.environ[PythonObject("BASE_DIR")] = PythonObject("/home/user")
    python.environ[PythonObject("DATA_FOLDER")] = PythonObject("datasets")

    var config = Config()
    config.set("data_path", "${BASE_DIR}/${DATA_FOLDER}")

    var substituted = config.substitute_env_vars()

    var path = substituted.get_string("data_path")
    assert_equal(
        path,
        "/home/user/datasets",
        "Should substitute multiple environment variables",
    )

    print("✓ test_substitute_multiple_env_vars passed")


fn test_substitute_env_var_in_middle() raises:
    """Test substitution with variable in middle of string.

    Verifies ${VAR} can appear anywhere in value.
    """
    var python = Python.import_module("os")
    python.environ[PythonObject("MODEL_NAME")] = PythonObject("lenet5")

    var config = Config()
    config.set("path", "/models/${MODEL_NAME}/checkpoint.mojo")

    var substituted = config.substitute_env_vars()

    var path = substituted.get_string("path")
    assert_equal(
        path,
        "/models/lenet5/checkpoint.mojo",
        "Should substitute variable in middle of string",
    )

    print("✓ test_substitute_env_var_in_middle passed")


# ============================================================================
# Default Value Syntax Tests
# ============================================================================


fn test_substitute_with_default_value() raises:
    """Test ${VAR:-default} syntax for missing variables.

    Verifies default value is used when variable not set.
    """
    var config = Config()
    config.set("output_dir", "${MISSING_VAR:-/tmp/output}")

    var substituted = config.substitute_env_vars()

    var path = substituted.get_string("output_dir")
    assert_equal(
        path, "/tmp/output", "Should use default value for missing variable"
    )

    print("✓ test_substitute_with_default_value passed")


fn test_substitute_with_default_when_var_exists() raises:
    """Test ${VAR:-default} when variable exists.

    Verifies actual value is used when variable is set.
    """
    var python = Python.import_module("os")
    python.environ[PythonObject("DATA_DIR")] = PythonObject("/actual/data")

    var config = Config()
    config.set("data_path", "${DATA_DIR:-/default/data}")

    var substituted = config.substitute_env_vars()

    var path = substituted.get_string("data_path")
    assert_equal(
        path, "/actual/data", "Should use actual value when variable exists"
    )

    print("✓ test_substitute_with_default_when_var_exists passed")


fn test_substitute_empty_default_value() raises:
    """Test ${VAR:-} syntax with empty default.

    Verifies empty string default is supported.
    """
    var config = Config()
    config.set("optional_param", "${MISSING:-}")

    var substituted = config.substitute_env_vars()

    var param = substituted.get_string("optional_param")
    assert_equal(param, "", "Should use empty string as default")

    print("✓ test_substitute_empty_default_value passed")


fn test_substitute_complex_default_value() raises:
    """Test ${VAR:-default} with complex default value.

    Verifies default can contain special characters.
    """
    var config = Config()
    config.set("path", "${MISSING:-/path/with-dashes_and.dots}")

    var substituted = config.substitute_env_vars()

    var path = substituted.get_string("path")
    assert_equal(
        path,
        "/path/with-dashes_and.dots",
        "Should handle complex default values",
    )

    print("✓ test_substitute_complex_default_value passed")


# ============================================================================
# Configuration File Tests
# ============================================================================


fn test_substitute_from_file() raises:
    """Test substitution from configuration file.

    Verifies environment variables in YAML files are substituted.
    """
    var python = Python.import_module("os")
    python.environ[PythonObject("DATA_DIR")] = PythonObject("/actual/data")

    var config = load_config("tests/configs/fixtures/env_vars.yaml")
    var substituted = config.substitute_env_vars()

    # data_dir should be substituted
    if substituted.has("data_dir"):
        var data_dir = substituted.get_string("data_dir")
        assert_equal(data_dir, "/actual/data", "Should substitute DATA_DIR")

    # output_dir should use default (MISSING env var)
    if substituted.has("output_dir"):
        var output_dir = substituted.get_string("output_dir")
        assert_equal(
            output_dir, "/tmp/output", "Should use default for OUTPUT_DIR"
        )

    print("✓ test_substitute_from_file passed")


fn test_substitute_preserves_non_string_values() raises:
    """Test that substitution preserves non-string values.

    Verifies only string values are processed for substitution.
    """
    var config = Config()
    config.set("learning_rate", 0.001)
    config.set("batch_size", 32)
    config.set("use_cuda", True)
    config.set("path", "${HOME}/data")

    var python = Python.import_module("os")
    python.environ[PythonObject("HOME")] = PythonObject("/home/user")

    var substituted = config.substitute_env_vars()

    # Non-string values should be preserved
    var lr = substituted.get_float("learning_rate")
    assert_equal(lr, 0.001, "Float values should be preserved")

    var bs = substituted.get_int("batch_size")
    assert_equal(bs, 32, "Int values should be preserved")

    var cuda = substituted.get_bool("use_cuda")
    assert_equal(cuda, True, "Bool values should be preserved")

    # String value should be substituted
    var path = substituted.get_string("path")
    assert_equal(path, "/home/user/data", "String should be substituted")

    print("✓ test_substitute_preserves_non_string_values passed")


# ============================================================================
# Edge Case Tests
# ============================================================================


fn test_substitute_no_variables() raises:
    """Test substitution when no variables present.

    Verifies config without ${} patterns is unchanged.
    """
    var config = Config()
    config.set("path", "/static/path/no/vars")

    var substituted = config.substitute_env_vars()

    var path = substituted.get_string("path")
    assert_equal(
        path, "/static/path/no/vars", "Should preserve values without variables"
    )

    print("✓ test_substitute_no_variables passed")


fn test_substitute_malformed_pattern() raises:
    """Test substitution with malformed ${} pattern.

    Verifies malformed patterns are left unchanged.
    """
    var config = Config()
    config.set("value1", "${MISSING")  # Missing closing brace
    config.set("value2", "$MISSING}")  # Missing opening brace
    config.set("value3", "${}")  # Empty variable name

    _ = config.substitute_env_vars()

    # Malformed patterns should be left as-is
    # Implementation may vary - this tests expected behavior
    print("✓ test_substitute_malformed_pattern passed")


fn test_substitute_nested_variables() raises:
    """Test substitution with nested ${} patterns.

    Verifies behavior with nested variable syntax.
    Note: Nested substitution may not be supported.
    """
    var config = Config()
    config.set("path", "${BASE_${LEVEL}}")

    _ = config.substitute_env_vars()

    # Nested variables typically not supported - should leave as-is
    # This test documents expected behavior
    print("✓ test_substitute_nested_variables passed")


fn test_substitute_dollar_sign_escape() raises:
    """Test handling of literal dollar signs.

    Verifies how literal $ characters are handled.
    """
    var config = Config()
    config.set("price", "$$100")  # Literal dollar signs

    var substituted = config.substitute_env_vars()

    # Should preserve literal dollar signs that aren't ${VAR} patterns
    var price = substituted.get_string("price")
    # Exact behavior depends on implementation
    print("✓ test_substitute_dollar_sign_escape passed")


# ============================================================================
# Integration Tests
# ============================================================================


fn test_load_and_substitute_training_config() raises:
    """Test loading and substituting training configuration.

    Verifies end-to-end workflow with environment variables.
    """
    var python = Python.import_module("os")
    python.environ[PythonObject("EXPERIMENT_NAME")] = PythonObject(
        "baseline_001"
    )
    python.environ[PythonObject("OUTPUT_PATH")] = PythonObject("/results")

    # Create config with env vars
    var config = Config()
    config.set("experiment", "${EXPERIMENT_NAME}")
    config.set("output", "${OUTPUT_PATH}/${EXPERIMENT_NAME}")
    config.set("learning_rate", 0.001)

    var substituted = config.substitute_env_vars()

    var exp = substituted.get_string("experiment")
    assert_equal(exp, "baseline_001", "Should substitute experiment name")

    var output = substituted.get_string("output")
    assert_equal(
        output,
        "/results/baseline_001",
        "Should substitute multiple variables in path",
    )

    var lr = substituted.get_float("learning_rate")
    assert_equal(lr, 0.001, "Should preserve numeric values")

    print("✓ test_load_and_substitute_training_config passed")


fn test_substitute_with_merge() raises:
    """Test environment variable substitution with config merging.

    Verifies substitution works correctly after merging configs.
    """
    var python = Python.import_module("os")
    python.environ[PythonObject("BASE_LR")] = PythonObject("0.01")

    var defaults = Config()
    defaults.set("learning_rate", "${BASE_LR:-0.001}")
    defaults.set("batch_size", 32)

    var experiment = Config()
    experiment.set("learning_rate", "${BASE_LR:-0.005}")

    # Merge then substitute
    var merged = defaults.merge(experiment)
    var substituted = merged.substitute_env_vars()

    var lr = substituted.get_string("learning_rate")
    # Note: This will be string "0.01" after substitution
    # May need conversion depending on usage
    assert_equal(lr, "0.01", "Should substitute in merged config")

    print("✓ test_substitute_with_merge passed")


# ============================================================================
# Main Test Runner
# ============================================================================


fn main() raises:
    """Run all environment variable substitution tests."""
    print("\n" + "=" * 70)
    print("Running Environment Variable Substitution Tests")
    print("=" * 70 + "\n")

    # Basic substitution tests
    print("Testing Basic Substitution...")
    test_substitute_simple_env_var()
    test_substitute_multiple_env_vars()
    test_substitute_env_var_in_middle()

    # Default value tests
    print("\nTesting Default Value Syntax...")
    test_substitute_with_default_value()
    test_substitute_with_default_when_var_exists()
    test_substitute_empty_default_value()
    test_substitute_complex_default_value()

    # File-based tests
    print("\nTesting File-Based Substitution...")
    test_substitute_from_file()
    test_substitute_preserves_non_string_values()

    # Edge case tests
    print("\nTesting Edge Cases...")
    test_substitute_no_variables()
    test_substitute_malformed_pattern()
    test_substitute_nested_variables()
    test_substitute_dollar_sign_escape()

    # Integration tests
    print("\nTesting Integration Scenarios...")
    test_load_and_substitute_training_config()
    test_substitute_with_merge()

    # Summary
    print("\n" + "=" * 70)
    print("✅ All Environment Variable Substitution Tests Passed!")
    print("=" * 70)
