"""
Configuration Loading Tests

Tests for loading configurations from YAML and JSON files.
These tests validate the configuration loading system for the configs/ directory.

Run with: mojo test tests/configs/test_loading.mojo
"""

from testing import assert_true, assert_false, assert_equal
from shared.utils.config import Config, load_config


# ============================================================================
# Default Configuration Loading Tests
# ============================================================================


fn test_load_default_training_config() raises:
    """Test loading default training configuration.

    Verifies that the default training config loads successfully and
    contains expected optimizer settings.
    """
    # This will fail until Issue #74 creates the configs
    var config = load_config("configs/defaults/training.yaml")

    # Verify optimizer section exists
    assert_true(config.has("optimizer.name"), "Should have optimizer.name")
    var optimizer = config.get_string("optimizer.name")
    assert_equal(optimizer, "sgd", "Default optimizer should be SGD")

    # Verify learning rate exists and is reasonable
    assert_true(config.has("optimizer.learning_rate"), "Should have learning_rate")
    var lr = config.get_float("optimizer.learning_rate")
    assert_true(lr > 0.0, "Learning rate should be positive")

    print("✓ test_load_default_training_config passed")


fn test_load_default_model_config() raises:
    """Test loading default model configuration.

    Verifies that default model settings load correctly.
    """
    var config = load_config("configs/defaults/model.yaml")

    # Verify model config has expected fields
    # Exact fields will depend on Issue #74 implementation
    assert_true(len(config.data) > 0, "Config should not be empty")

    print("✓ test_load_default_model_config passed")


fn test_load_default_data_config() raises:
    """Test loading default data configuration.

    Verifies that default data processing settings load correctly.
    """
    var config = load_config("configs/defaults/data.yaml")

    assert_true(len(config.data) > 0, "Config should not be empty")

    print("✓ test_load_default_data_config passed")


# ============================================================================
# Paper-Specific Configuration Loading Tests
# ============================================================================


fn test_load_lenet5_model_config() raises:
    """Test loading LeNet-5 model configuration.

    Verifies that paper-specific model config loads with correct architecture.
    """
    var config = load_config("configs/papers/lenet5/model.yaml")

    # Verify model name
    assert_true(config.has("name"), "Should have model name")
    var name = config.get_string("name")
    assert_equal(name, "LeNet-5", "Model name should be LeNet-5")

    # Verify architecture details
    assert_true(config.has("num_classes"), "Should have num_classes")
    var num_classes = config.get_int("num_classes")
    assert_equal(num_classes, 10, "LeNet-5 should have 10 classes")

    print("✓ test_load_lenet5_model_config passed")


fn test_load_lenet5_training_config() raises:
    """Test loading LeNet-5 training configuration.

    Verifies that paper-specific training config loads correctly.
    """
    var config = load_config("configs/papers/lenet5/training.yaml")

    # Should have training parameters
    assert_true(len(config.data) > 0, "Training config should not be empty")

    # Check for learning rate (common parameter)
    assert_true(
        config.has("learning_rate") or config.has("optimizer.learning_rate"),
        "Should have learning rate configuration"
    )

    print("✓ test_load_lenet5_training_config passed")


# ============================================================================
# Experiment Configuration Loading Tests
# ============================================================================


fn test_load_experiment_baseline_config() raises:
    """Test loading baseline experiment configuration.

    Verifies that experiment configs can reference base configs.
    """
    var config = load_config("configs/experiments/lenet5/baseline.yaml")

    # Experiment configs may have an "extends" field
    # The exact structure depends on Issue #74
    assert_true(len(config.data) > 0, "Experiment config should not be empty")

    print("✓ test_load_experiment_baseline_config passed")


fn test_load_experiment_augmented_config() raises:
    """Test loading augmented experiment configuration.

    Verifies experiment config with data augmentation settings.
    """
    var config = load_config("configs/experiments/lenet5/augmented.yaml")

    assert_true(len(config.data) > 0, "Experiment config should not be empty")

    print("✓ test_load_experiment_augmented_config passed")


# ============================================================================
# File Format Tests
# ============================================================================


fn test_load_yaml_config() raises:
    """Test loading YAML configuration file.

    Verifies YAML parsing works correctly.
    """
    var config = load_config("tests/configs/fixtures/minimal.yaml")

    assert_true(config.has("learning_rate"), "Should have learning_rate")
    assert_true(config.has("batch_size"), "Should have batch_size")

    var lr = config.get_float("learning_rate")
    assert_equal(lr, 0.001, "Learning rate should be 0.001")

    var bs = config.get_int("batch_size")
    assert_equal(bs, 32, "Batch size should be 32")

    print("✓ test_load_yaml_config passed")


fn test_load_json_config() raises:
    """Test loading JSON configuration file.

    Verifies JSON parsing works correctly.
    """
    # Create a simple JSON config for testing
    var config = Config()
    config.set("learning_rate", 0.001)
    config.set("batch_size", 32)
    config.to_json("tests/configs/fixtures/test_output.json")

    # Load it back
    var loaded = load_config("tests/configs/fixtures/test_output.json")

    assert_true(loaded.has("learning_rate"), "Should have learning_rate")
    assert_true(loaded.has("batch_size"), "Should have batch_size")

    print("✓ test_load_json_config passed")


# ============================================================================
# Error Handling Tests
# ============================================================================


fn test_load_missing_file() raises:
    """Test loading non-existent configuration file.

    Verifies proper error handling for missing files.
    """
    var error_raised = False

    try:
        var config = load_config("configs/nonexistent/file.yaml")
    except:
        error_raised = True

    assert_true(error_raised, "Should raise error for missing file")

    print("✓ test_load_missing_file passed")


fn test_load_empty_file() raises:
    """Test loading empty configuration file.

    Verifies proper error handling for empty files.
    """
    # Create empty file
    with open("tests/configs/fixtures/empty.yaml", "w") as f:
        _ = f.write("")

    var error_raised = False
    try:
        var config = load_config("tests/configs/fixtures/empty.yaml")
    except:
        error_raised = True

    assert_true(error_raised, "Should raise error for empty file")

    print("✓ test_load_empty_file passed")


fn test_load_invalid_format() raises:
    """Test loading file with invalid format.

    Verifies proper error handling for unsupported formats.
    """
    var error_raised = False

    try:
        var config = load_config("configs/test.txt")
    except:
        error_raised = True

    assert_true(error_raised, "Should raise error for unsupported format")

    print("✓ test_load_invalid_format passed")


# ============================================================================
# Complex Configuration Tests
# ============================================================================


fn test_load_complex_nested_config() raises:
    """Test loading complex nested configuration.

    Verifies that nested structures are properly parsed.
    Note: Current implementation has limitations with nested structures.
    """
    var config = load_config("tests/configs/fixtures/complex.yaml")

    # Should load successfully even with nested structure
    # Nested access may be limited until full YAML parsing implemented
    assert_true(len(config.data) > 0, "Complex config should not be empty")

    print("✓ test_load_complex_nested_config passed")


# ============================================================================
# Main Test Runner
# ============================================================================


fn main() raises:
    """Run all configuration loading tests."""
    print("\n" + "=" * 70)
    print("Running Configuration Loading Tests")
    print("=" * 70 + "\n")

    # Default config tests
    print("Testing Default Configurations...")
    test_load_default_training_config()
    test_load_default_model_config()
    test_load_default_data_config()

    # Paper config tests
    print("\nTesting Paper-Specific Configurations...")
    test_load_lenet5_model_config()
    test_load_lenet5_training_config()

    # Experiment config tests
    print("\nTesting Experiment Configurations...")
    test_load_experiment_baseline_config()
    test_load_experiment_augmented_config()

    # File format tests
    print("\nTesting File Format Support...")
    test_load_yaml_config()
    test_load_json_config()

    # Error handling tests
    print("\nTesting Error Handling...")
    test_load_missing_file()
    test_load_empty_file()
    test_load_invalid_format()

    # Complex config tests
    print("\nTesting Complex Configurations...")
    test_load_complex_nested_config()

    # Summary
    print("\n" + "=" * 70)
    print("✅ All Configuration Loading Tests Passed!")
    print("=" * 70)
    print("\nNote: Some tests will fail until Issue #74 creates config files")
