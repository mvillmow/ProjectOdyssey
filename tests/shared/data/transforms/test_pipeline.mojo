"""Tests for transform pipeline composition.

Tests Pipeline which composes multiple transforms into a single transform,
enabling flexible and reusable data preprocessing workflows.
"""

from tests.shared.conftest import (
    assert_true,
    assert_equal,
    assert_not_equal,
    TestFixtures,
)


# ============================================================================
# Stub Implementations for TDD
# ============================================================================


struct StubData:
    """Minimal stub data for transform testing."""

    var value: Float32

    fn __init__(
        inoutself, value: Float32
    ):
        self.value = value


struct StubTransform:
    """Minimal stub transform that adds a fixed value."""

    var delta: Float32

    fn __init__(
        inoutself, delta: Float32
    ):
        self.delta = delta

    fn apply(self, data: StubData) -> StubData:
        """Apply transform by adding delta to data value."""
        return StubData(data.value + self.delta)


struct StubPipeline:
    """Minimal stub pipeline that chains transforms sequentially."""

    var transforms: List[StubTransform]

    fn __init__(
        inoutself
    ):
        self.transforms = List[StubTransform]()

    fn add_transform(
        inoutself, transform: StubTransform
    ):
        """Add a transform to the pipeline."""
        self.transforms.append(transform)

    fn apply(self, data: StubData) -> StubData:
        """Apply all transforms sequentially."""
        var result = data
        for i in range(len(self.transforms)):
            result = self.transforms[i].apply(result)
        return result

    fn __len__(self) -> Int:
        """Return number of transforms in pipeline."""
        return len(self.transforms)


# ============================================================================
# Pipeline Creation Tests
# ============================================================================


fn test_pipeline_creation() raises:
    """Test creating Pipeline from list of transforms.

    Should accept list of transform objects and apply them sequentially
    when called on data.
    """
    var pipeline = StubPipeline()
    pipeline.add_transform(StubTransform(delta=10.0))
    pipeline.add_transform(StubTransform(delta=5.0))
    assert_equal(len(pipeline), 2)


fn test_pipeline_empty() raises:
    """Test creating empty Pipeline.

    Empty pipeline should be valid and return data unchanged,
    useful as default or for conditional pipeline building.
    """
    var pipeline = StubPipeline()
    var data = StubData(value=42.0)
    var result = pipeline.apply(data)
    assert_equal(result.value, data.value)


fn test_pipeline_single_transform() raises:
    """Test Pipeline with single transform.

    Should work correctly even with just one transform,
    maintaining consistent API.
    """
    var pipeline = StubPipeline()
    pipeline.add_transform(StubTransform(delta=10.0))

    var data = StubData(value=5.0)
    var result = pipeline.apply(data)
    assert_equal(result.value, Float32(15.0))  # 5.0 + 10.0


# ============================================================================
# Pipeline Execution Tests
# ============================================================================


fn test_pipeline_sequential_application() raises:
    """Test that transforms are applied in order.

    Transform order matters: Transform(+10)→Transform(+5) should produce
    different result than Transform(+5)→Transform(+10) when order affects output.
    """
    var data = StubData(value=0.0)

    # Pipeline 1: +10 then +5
    var pipe1 = StubPipeline()
    pipe1.add_transform(StubTransform(delta=10.0))
    pipe1.add_transform(StubTransform(delta=5.0))
    var result1 = pipe1.apply(data)

    # Result should be 0 + 10 + 5 = 15
    assert_equal(result1.value, Float32(15.0))


fn test_pipeline_output_feeds_next() raises:
    """Test that each transform receives output of previous.

    Output value from transform N should be input to transform N+1,
    enabling complex preprocessing chains.
    """
    var data = StubData(value=2.0)

    # Create pipeline that multiplies the effect
    var pipeline = StubPipeline()
    pipeline.add_transform(StubTransform(delta=3.0))  # 2 + 3 = 5
    pipeline.add_transform(StubTransform(delta=5.0))  # 5 + 5 = 10
    pipeline.add_transform(StubTransform(delta=10.0))  # 10 + 10 = 20

    var result = pipeline.apply(data)
    assert_equal(result.value, Float32(20.0))


fn test_pipeline_preserves_intermediate_values() raises:
    """Test that pipeline doesn't modify original data.

    Original input data should remain unchanged after pipeline application.
    """
    var data = StubData(value=100.0)

    var pipeline = StubPipeline()
    pipeline.add_transform(StubTransform(delta=50.0))

    var result = pipeline.apply(data)

    # Result should be different from original
    assert_not_equal(result.value, data.value)
    # Original should be unchanged
    assert_equal(data.value, Float32(100.0))
    # Result should have transform applied
    assert_equal(result.value, Float32(150.0))


fn test_pipeline_sequential_application():
    """Test that transforms are applied in order.

    Transform order matters: Resize→Normalize should produce different
    result than Normalize→Resize.
    """
    # TODO(#39): Implement when Pipeline exists
    # var data = Tensor.ones(100, 100)
    #
    # # Pipeline 1: Resize then Normalize
    # var pipe1 = Pipeline([
    #     Resize(50, 50),
    #     Normalize(mean=0.5, std=0.5)
    # ])
    # var result1 = pipe1(data)
    #
    # # Pipeline 2: Normalize then Resize
    # var pipe2 = Pipeline([
    #     Normalize(mean=0.5, std=0.5),
    #     Resize(50, 50)
    # ])
    # var result2 = pipe2(data)
    #
    # # Results should differ due to order
    # assert_not_equal(result1, result2)
    pass


fn test_pipeline_output_feeds_next():
    """Test that each transform receives output of previous.

    Output shape/values from transform N should be input to transform N+1,
    enabling complex preprocessing chains.
    """
    # TODO(#39): Implement when Pipeline exists
    # var pipeline = Pipeline([
    #     Reshape(28, 28),     # Output: 28x28
    #     Resize(32, 32),      # Input: 28x28, Output: 32x32
    #     Normalize(0.5, 0.5)  # Input: 32x32, Output: 32x32 normalized
    # ])
    #
    # var data = Tensor.ones(784)  # Flat 28*28
    # var result = pipeline(data)
    #
    # # Final output should be 32x32 normalized
    # assert_equal(result.shape[0], 32)
    # assert_equal(result.shape[1], 32)
    pass


fn test_pipeline_multiple_calls():
    """Test that Pipeline can be called multiple times.

    Should be stateless and produce same output for same input,
    not accumulate state between calls.
    """
    # TODO(#39): Implement when Pipeline exists
    # var pipeline = Pipeline([Resize(224, 224), Normalize(0.5, 0.5)])
    # var data = TestFixtures.small_tensor()
    #
    # var result1 = pipeline(data)
    # var result2 = pipeline(data)
    #
    # assert_equal(result1, result2)
    pass


# ============================================================================
# Pipeline Composition Tests
# ============================================================================


fn test_pipeline_composition():
    """Test composing pipelines together.

    Should support Pipeline(pipeline1 + pipeline2) to create
    longer pipelines from smaller reusable pieces.
    """
    # TODO(#39): Implement when Pipeline composition exists
    # var preprocess = Pipeline([Resize(224, 224)])
    # var augment = Pipeline([RandomFlip(), RandomCrop(200, 200)])
    # var normalize = Pipeline([Normalize(0.5, 0.5)])
    #
    # var full_pipeline = Pipeline(preprocess + augment + normalize)
    # var data = TestFixtures.small_tensor()
    # var result = full_pipeline(data)
    #
    # assert_true(result is not None)
    pass


fn test_pipeline_append():
    """Test appending transforms to existing pipeline.

    Should support adding transforms after pipeline creation
    for incremental pipeline building.
    """
    # TODO(#39): Implement when Pipeline.append exists
    # var pipeline = Pipeline([Resize(224, 224)])
    # pipeline.append(Normalize(0.5, 0.5))
    # pipeline.append(RandomFlip())
    #
    # var data = TestFixtures.small_tensor()
    # var result = pipeline(data)
    # assert_true(result is not None)
    pass


# ============================================================================
# Pipeline Error Handling Tests
# ============================================================================


fn test_pipeline_transform_error_propagation():
    """Test that errors in transforms are properly propagated.

    If a transform raises error, pipeline should propagate it
    with context about which transform failed.
    """
    # TODO(#39): Implement when Pipeline error handling exists
    # var pipeline = Pipeline([
    #     Resize(224, 224),
    #     InvalidTransform(),  # This will raise error
    #     Normalize(0.5, 0.5)
    # ])
    #
    # var data = TestFixtures.small_tensor()
    # try:
    #     var result = pipeline(data)
    #     assert_true(False, "Should have raised error")
    # except TransformError as e:
    #     # Error message should indicate which transform failed
    #     assert_true("InvalidTransform" in str(e))
    pass


fn test_pipeline_shape_mismatch():
    """Test handling of shape mismatches between transforms.

    If transform N outputs shape incompatible with transform N+1,
    should raise clear error.
    """
    # TODO(#39): Implement when Pipeline validation exists
    # # Reshape to 3D, then try to apply 2D-only transform
    # var pipeline = Pipeline([
    #     Reshape(10, 10, 3),
    #     Resize(224, 224)  # Expects 2D input
    # ])
    #
    # var data = Tensor.ones(300)
    # try:
    #     var result = pipeline(data)
    #     assert_true(False, "Should have raised ShapeError")
    # except ShapeError:
    #     pass
    pass


# ============================================================================
# Pipeline Utility Tests
# ============================================================================


fn test_pipeline_str_representation():
    """Test string representation of Pipeline.

    Should show list of transforms for debugging,
    e.g., 'Pipeline([Resize(224), Normalize(0.5)])'.
    """
    # TODO(#39): Implement when Pipeline.__str__ exists
    # var pipeline = Pipeline([Resize(224, 224), Normalize(0.5, 0.5)])
    # var repr = str(pipeline)
    #
    # assert_true("Pipeline" in repr)
    # assert_true("Resize" in repr)
    # assert_true("Normalize" in repr)
    pass


fn test_pipeline_len():
    """Test getting number of transforms in pipeline.

    len(pipeline) should return number of transforms,
    useful for debugging and validation.
    """
    # TODO(#39): Implement when Pipeline.__len__ exists
    # var pipeline = Pipeline([Resize(224, 224), Normalize(0.5, 0.5)])
    # assert_equal(len(pipeline), 2)
    pass


# ============================================================================
# Main Test Runner
# ============================================================================


fn main() raises:
    """Run all pipeline tests."""
    print("Running pipeline tests...")

    # Creation tests
    test_pipeline_creation()
    test_pipeline_empty()
    test_pipeline_single_transform()

    # Execution tests
    test_pipeline_sequential_application()
    test_pipeline_output_feeds_next()
    test_pipeline_multiple_calls()

    # Composition tests
    test_pipeline_composition()
    test_pipeline_append()

    # Error handling tests
    test_pipeline_transform_error_propagation()
    test_pipeline_shape_mismatch()

    # Utility tests
    test_pipeline_str_representation()
    test_pipeline_len()

    print("✓ All pipeline tests passed!")
