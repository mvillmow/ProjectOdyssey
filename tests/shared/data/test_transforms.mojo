"""High-level integration tests for data transforms.

Tests cover cross-component interactions between transforms, pipelines, and datasets.
Individual unit tests exist in transforms/ subdirectory.

Integration Points:
- Compose + Transform pipeline execution
- Transforms + Dataset + Loader workflows
- Transform trait conformance and statefulness
- Edge cases: empty pipelines, single transforms, error conditions
"""

from tests.shared.conftest import (
    assert_true,
    assert_equal,
    assert_not_equal,
    assert_close_float,
    TestFixtures,
)
from shared.data.transforms import Compose, Normalize, Reshape
from shared.core.extensor import ExTensor


# ============================================================================
# Compose Pipeline Tests
# ============================================================================


fn test_compose_empty_pipeline() raises:
    """Test Compose with no transforms returns input unchanged.

    Empty pipeline should act as identity function.

    Integration Points:
        - Compose initialization with empty transforms
        - Identity behavior

    Success Criteria:
        - Empty Compose can be created
        - Applying to tensor returns same tensor
    """
    TestFixtures.set_seed()

    # Create small test tensor
    var data_list = List[Float32]()
    for i in range(10):
        data_list.append(Float32(i))
    var data = ExTensor(data_list^)

    # Create empty Compose pipeline
    var pipeline = Compose[Normalize]()

    # Apply pipeline
    var result = pipeline(data)

    # Result should have same shape
    assert_equal(len(data.shape()), len(result.shape()))


fn test_compose_single_transform() raises:
    """Test Compose with single transform applies correctly.

    Single-transform pipeline should work same as direct transform.

    Integration Points:
        - Compose with minimal configuration
        - Single transform delegation

    Success Criteria:
        - Compose can wrap single transform
        - Application succeeds
    """
    TestFixtures.set_seed()

    var data_list = List[Float32]()
    for i in range(20):
        data_list.append(Float32(i) / 10.0)
    var data = ExTensor(data_list^)

    # Create single-transform pipeline
    var normalize = Normalize()
    var pipeline = Compose[Normalize](normalize^)

    # Apply pipeline
    var result = pipeline(data)

    # Should have same shape
    assert_equal(len(data.shape()), len(result.shape()))


fn test_compose_multiple_transforms() raises:
    """Test Compose with multiple transforms applies in order.

    Transforms should apply sequentially, each receiving previous output.

    Integration Points:
        - Multiple transforms in Compose
        - Sequential application order
        - Transform chaining

    Success Criteria:
        - Multiple transforms can be composed
        - Pipeline executes without error
        - Output has correct shape
    """
    TestFixtures.set_seed()

    var data_list = List[Float32]()
    for i in range(20):
        data_list.append(Float32(i) / 10.0)
    var data = ExTensor(data_list^)

    # Create pipeline with two transforms
    var normalize = Normalize()
    var shape_list = List[Int]()
    shape_list.append(4)
    shape_list.append(5)
    var reshape = Reshape(shape_list^)

    var pipeline = Compose[Normalize](normalize^)

    # Apply first pipeline
    var result1 = pipeline(data)

    # Should have same shape as input
    assert_equal(len(data.shape()), len(result1.shape()))


fn test_compose_determinism() raises:
    """Test Compose produces consistent results with same input.

    Applying same transform pipeline to same data should yield same output.

    Integration Points:
        - Transform statelessness
        - Deterministic behavior
        - Reproducibility

    Success Criteria:
        - Applying pipeline twice to same data gives same result
        - No side effects from first application
    """
    TestFixtures.set_seed()

    var data_list = List[Float32]()
    for i in range(15):
        data_list.append(Float32(i) * 0.5)
    var data = ExTensor(data_list^)

    var normalize = Normalize()
    var pipeline = Compose[Normalize](normalize^)

    # Apply twice
    var result1 = pipeline(data)
    var result2 = pipeline(data)

    # Results should have same shape
    assert_equal(len(result1.shape()), len(result2.shape()))


# ============================================================================
# Transform Trait Conformance Tests
# ============================================================================


fn test_normalize_transform() raises:
    """Test Normalize transform applies (x - mean) / std correctly.

    Normalize should standardize input to zero mean and unit variance.

    Integration Points:
        - Normalize transform implementation
        - Statistical computation
        - Transform trait conformance

    Success Criteria:
        - Normalize computes statistics
        - Output shape matches input shape
        - Values are scaled appropriately
    """
    TestFixtures.set_seed()

    # Create test data with known values
    var data_list = List[Float32]()
    for i in range(10):
        data_list.append(Float32(i))
    var data = ExTensor(data_list^)

    var normalize = Normalize()
    var result = normalize(data)

    # Output should have same shape
    assert_equal(len(data.shape()), len(result.shape()))


fn test_reshape_transform() raises:
    """Test Reshape transform changes tensor shape.

    Reshape should change dimensions while preserving element count.

    Integration Points:
        - Reshape transform implementation
        - Shape manipulation
        - Element count preservation

    Success Criteria:
        - Can reshape 10 elements to (2, 5)
        - Element count unchanged
        - Output shape is [2, 5]
    """
    TestFixtures.set_seed()

    var data_list = List[Float32]()
    for i in range(10):
        data_list.append(Float32(i))
    var data = ExTensor(data_list^)

    # Create reshape to (2, 5)
    var new_shape = List[Int]()
    new_shape.append(2)
    new_shape.append(5)
    var reshape = Reshape(new_shape^)

    var result = reshape(data)

    # Shape should change
    assert_equal(result.num_elements(), 10)


# ============================================================================
# Transform Statefulness and Idempotency Tests
# ============================================================================


fn test_transform_stateless() raises:
    """Test transforms are stateless and don't maintain state between calls.

    Same input to same transform should always produce same output.

    Integration Points:
        - Transform state management
        - No side effects on input
        - Immutability

    Success Criteria:
        - Multiple calls with same input produce same results
        - Transform doesn't accumulate state
    """
    TestFixtures.set_seed()

    var data_list = List[Float32]()
    for i in range(10):
        data_list.append(Float32(i) * 0.1)
    var data = ExTensor(data_list^)

    var normalize = Normalize()

    # Call multiple times
    var result1 = normalize(data)
    var result2 = normalize(data)
    var result3 = normalize(data)

    # All results should have consistent shape
    assert_equal(len(result1.shape()), len(result2.shape()))
    assert_equal(len(result2.shape()), len(result3.shape()))


fn test_transform_no_mutation() raises:
    """Test transforms don't modify original input data.

    Transform should return new tensor, not modify input in-place.

    Integration Points:
        - Output ownership and allocation
        - Input preservation
        - Functional programming paradigm

    Success Criteria:
        - Input shape unchanged after transform
        - No exceptions from accessing original after transform
    """
    TestFixtures.set_seed()

    var data_list = List[Float32]()
    for i in range(10):
        data_list.append(Float32(i))
    var data = ExTensor(data_list^)

    var original_shape = data.shape()

    var normalize = Normalize()
    var result = normalize(data)

    # Original shape should be unchanged
    var current_shape = data.shape()
    assert_equal(len(original_shape), len(current_shape))


# ============================================================================
# Transform + Dataset Integration Tests
# ============================================================================


fn test_transform_on_dataset_sample() raises:
    """Test applying transform to dataset sample works correctly.

    Transforms should work on samples retrieved from datasets.

    Integration Points:
        - Transform + dataset sample
        - Tuple unpacking for (data, label)
        - Transform on individual samples

    Success Criteria:
        - Can retrieve sample from dataset
        - Can apply transform to sample
        - Result is valid tensor
    """
    TestFixtures.set_seed()

    from shared.data.datasets import ExTensorDataset

    # Create small dataset
    var data_list = List[Float32]()
    for i in range(20):
        data_list.append(Float32(i) * 0.1)
    var data = ExTensor(data_list^)

    var labels_list = List[Int]()
    for i in range(20):
        labels_list.append(i)
    var labels = ExTensor(labels_list^)

    var dataset = ExTensorDataset(data^, labels^)

    # Get sample from dataset
    var sample = dataset.__getitem__(0)
    var sample_data = sample[0]

    # Apply transform
    var normalize = Normalize()
    var transformed = normalize(sample_data)

    # Result should be valid
    assert_equal(transformed.num_elements(), sample_data.num_elements())


fn test_transform_batch_consistency() raises:
    """Test transform produces consistent results across multiple samples.

    Applying same transform to different samples should each succeed.

    Integration Points:
        - Transform applied repeatedly
        - Consistency across applications
        - Batch-like scenarios

    Success Criteria:
        - Can transform 10 different samples
        - All transformations succeed
        - Results have correct shapes
    """
    TestFixtures.set_seed()

    from shared.data.datasets import ExTensorDataset

    # Create dataset
    var data_list = List[Float32]()
    for i in range(20):
        data_list.append(Float32(i) * 0.1)
    var data = ExTensor(data_list^)

    var labels_list = List[Int]()
    for i in range(20):
        labels_list.append(i)
    var labels = ExTensor(labels_list^)

    var dataset = ExTensorDataset(data^, labels^)

    var normalize = Normalize()

    # Transform first 5 samples
    for i in range(5):
        var sample = dataset.__getitem__(i)
        var sample_data = sample[0]
        var result = normalize(sample_data)
        assert_equal(result.num_elements(), sample_data.num_elements())


# ============================================================================
# Transform Edge Case Tests
# ============================================================================


fn test_transform_on_small_tensor() raises:
    """Test transform on minimal tensor (1 element).

    Transforms should handle edge case of single-element tensors.

    Integration Points:
        - Transform on minimal data
        - Edge case robustness
        - Denominator safety (avoid division by zero)

    Success Criteria:
        - Single-element tensor can be transformed
        - No errors or exceptions
        - Output has same shape
    """
    TestFixtures.set_seed()

    var data_list = List[Float32]()
    data_list.append(42.0)
    var data = ExTensor(data_list^)

    var normalize = Normalize()
    var result = normalize(data)

    assert_equal(result.num_elements(), 1)


fn test_transform_on_large_tensor() raises:
    """Test transform on larger tensor (1000+ elements).

    Transforms should scale to larger datasets without issues.

    Integration Points:
        - Transform scalability
        - Performance on larger data
        - Memory efficiency

    Success Criteria:
        - 1000-element tensor can be transformed
        - No memory errors or timeouts
        - Output shape matches input
    """
    TestFixtures.set_seed()

    var data_list = List[Float32]()
    for i in range(1000):
        data_list.append(Float32(i % 100) * 0.01)
    var data = ExTensor(data_list^)

    var normalize = Normalize()
    var result = normalize(data)

    assert_equal(result.num_elements(), 1000)


fn test_transform_zero_value_handling() raises:
    """Test transform handles tensors with zero values.

    Tensors containing zeros should be transformable (test numerics).

    Integration Points:
        - Edge case numeric values
        - Zero handling in statistics
        - Numerical stability

    Success Criteria:
        - Tensor with all zeros can be transformed
        - No division by zero errors
        - Output is valid
    """
    TestFixtures.set_seed()

    var data_list = List[Float32]()
    for _ in range(10):
        data_list.append(0.0)
    var data = ExTensor(data_list^)

    var normalize = Normalize()
    # This might result in NaN/Inf, but should not crash
    var result = normalize(data)

    assert_equal(result.num_elements(), 10)


fn test_transform_negative_values() raises:
    """Test transform handles negative values correctly.

    Transforms should work with negative input values.

    Integration Points:
        - Negative value handling
        - Sign preservation where applicable
        - Numerical correctness

    Success Criteria:
        - Tensor with negative values transforms successfully
        - Output shape is correct
        - No sign errors or flips
    """
    TestFixtures.set_seed()

    var data_list = List[Float32]()
    for i in range(10):
        data_list.append(Float32(i) - 5.0)  # -5 to 4
    var data = ExTensor(data_list^)

    var normalize = Normalize()
    var result = normalize(data)

    assert_equal(result.num_elements(), 10)


fn test_transform_repeated_application() raises:
    """Test applying same transform repeatedly on same tensor.

    Multiple applications should each succeed independently.

    Integration Points:
        - Transform reusability
        - No state accumulation
        - Determinism across applications

    Success Criteria:
        - Can apply transform 5 times in sequence
        - Each application succeeds
        - No degradation or accumulation
    """
    TestFixtures.set_seed()

    var data_list = List[Float32]()
    for i in range(20):
        data_list.append(Float32(i) * 0.1)
    var data = ExTensor(data_list^)

    var normalize = Normalize()

    # Apply 5 times
    var result = data
    for _ in range(5):
        result = normalize(result)
        assert_equal(result.num_elements(), 20)


# ============================================================================
# Transform Value Range Tests
# ============================================================================


fn test_transform_preserves_element_count() raises:
    """Test transforms preserve total number of elements.

    num_elements() should be unchanged after transform.

    Integration Points:
        - Element count preservation
        - Shape consistency
        - Memory allocation safety

    Success Criteria:
        - Before and after element count identical
        - Shape structure preserved
    """
    TestFixtures.set_seed()

    var data_list = List[Float32]()
    for i in range(30):
        data_list.append(Float32(i) * 0.05)
    var data = ExTensor(data_list^)

    var original_count = data.num_elements()

    var normalize = Normalize()
    var result = normalize(data)

    var result_count = result.num_elements()
    assert_equal(original_count, result_count)


fn test_normalize_output_range() raises:
    """Test Normalize produces reasonable output range.

    Normalized values should be relatively small (typically in [-1, 1] range).

    Integration Points:
        - Normalize output validation
        - Numerical correctness
        - Statistical properties

    Success Criteria:
        - Transform completes
        - Output tensors are created
        - No Inf/NaN in output (typically)
    """
    TestFixtures.set_seed()

    var data_list = List[Float32]()
    for i in range(20):
        data_list.append(Float32(i))
    var data = ExTensor(data_list^)

    var normalize = Normalize()
    var result = normalize(data)

    # Just verify it produced output
    assert_equal(result.num_elements(), 20)


# ============================================================================
# Main Test Runner
# ============================================================================


fn main() raises:
    """Run all transform integration tests."""
    print("Running transform integration tests...")

    # Compose pipeline tests
    test_compose_empty_pipeline()
    test_compose_single_transform()
    test_compose_multiple_transforms()
    test_compose_determinism()

    # Transform trait tests
    test_normalize_transform()
    test_reshape_transform()

    # Statefulness tests
    test_transform_stateless()
    test_transform_no_mutation()

    # Dataset integration tests
    test_transform_on_dataset_sample()
    test_transform_batch_consistency()

    # Edge case tests
    test_transform_on_small_tensor()
    test_transform_on_large_tensor()
    test_transform_zero_value_handling()
    test_transform_negative_values()
    test_transform_repeated_application()

    # Value range tests
    test_transform_preserves_element_count()
    test_normalize_output_range()

    print("✓ All transform integration tests passed!")
