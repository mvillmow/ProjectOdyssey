"""Tests for ExTensor shape manipulation operations.

Tests shape manipulation including reshape, squeeze, unsqueeze, expand_dims,
flatten, ravel, concatenate, stack, split, tile, repeat, broadcast_to, permute.
"""

from sys import DType

# Import ExTensor and operations
from extensor import ExTensor, zeros, ones, full, arange, reshape, squeeze, unsqueeze, expand_dims, flatten, ravel, concatenate, stack

# Import test helpers
from ..helpers.assertions import (
    assert_dtype,
    assert_numel,
    assert_dim,
    assert_value_at,
    assert_all_values,
)


# ============================================================================
# Test reshape()
# ============================================================================

fn test_reshape_valid() raises:
    """Test reshaping to compatible size."""
    var shape_orig = DynamicVector[Int](1)
    shape_orig[0] = 12
    let a = arange(0.0, 12.0, 1.0, DType.float32)  # 12 elements
    var new_shape = DynamicVector[Int](2)
    new_shape[0] = 3
    new_shape[1] = 4
    let b = reshape(a, new_shape)

    assert_dim(b, 2, "Reshaped tensor should be 2D")
    assert_numel(b, 12, "Reshaped tensor should have same number of elements")


fn test_reshape_invalid_size() raises:
    """Test that reshape with incompatible size raises error."""
    var shape = DynamicVector[Int](1)
    shape[0] = 12
    let a = arange(0.0, 12.0, 1.0, DType.float32)
    # var new_shape = DynamicVector[Int](2)
    # new_shape[0] = 3
    # new_shape[1] = 5  # 15 elements, incompatible with 12
    # let b = reshape(a, new_shape)  # Should raise error

    # TODO: Verify error handling
    pass  # Placeholder


fn test_reshape_infer_dimension() raises:
    """Test reshape with inferred dimension (-1)."""
    var shape = DynamicVector[Int](1)
    shape[0] = 12
    let a = arange(0.0, 12.0, 1.0, DType.float32)
    var new_shape = DynamicVector[Int](2)
    new_shape[0] = 3
    new_shape[1] = -1  # Infer: should be 4
    let b = reshape(a, new_shape)

    assert_dim(b, 2, "Should be 2D")
    assert_numel(b, 12, "Should have 12 elements")


# ============================================================================
# Test squeeze()
# ============================================================================

fn test_squeeze_all_dims() raises:
    """Test removing all size-1 dimensions."""
    var shape = DynamicVector[Int](4)
    shape[0] = 1
    shape[1] = 3
    shape[2] = 1
    shape[3] = 4
    let a = ones(shape, DType.float32)  # Shape (1, 3, 1, 4)
    let b = squeeze(a)

    # Result should be (3, 4)
    assert_dim(b, 2, "Should remove all size-1 dims")
    assert_numel(b, 12, "Should have 12 elements")


fn test_squeeze_specific_dim() raises:
    """Test removing specific size-1 dimension."""
    var shape = DynamicVector[Int](3)
    shape[0] = 1
    shape[1] = 3
    shape[2] = 4
    let a = ones(shape, DType.float32)  # Shape (1, 3, 4)
    let b = squeeze(a, dim=0)

    # Result should be (3, 4)
    assert_dim(b, 2, "Should remove dim 0")


# ============================================================================
# Test unsqueeze() / expand_dims()
# ============================================================================

fn test_unsqueeze_add_dim() raises:
    """Test adding a size-1 dimension."""
    var shape = DynamicVector[Int](2)
    shape[0] = 3
    shape[1] = 4
    let a = ones(shape, DType.float32)  # Shape (3, 4)
    let b = unsqueeze(a, dim=0)

    # Result should be (1, 3, 4)
    assert_dim(b, 3, "Should add dimension")
    assert_numel(b, 12, "Should have same elements")


fn test_expand_dims_at_end() raises:
    """Test adding dimension at end."""
    var shape = DynamicVector[Int](2)
    shape[0] = 3
    shape[1] = 4
    let a = ones(shape, DType.float32)
    let b = expand_dims(a, dim=-1)

    # Result should be (3, 4, 1)
    assert_dim(b, 3, "Should add trailing dimension")


# ============================================================================
# Test flatten() / ravel()
# ============================================================================

fn test_flatten_c_order() raises:
    """Test flattening tensor to 1D (C order)."""
    var shape = DynamicVector[Int](2)
    shape[0] = 3
    shape[1] = 4
    let a = arange(0.0, 12.0, 1.0, DType.float32)
    let b = flatten(a)

    assert_dim(b, 1, "Flattened tensor should be 1D")
    assert_numel(b, 12, "Should have 12 elements")


fn test_ravel_view() raises:
    """Test ravel (should return view if possible)."""
    var shape = DynamicVector[Int](2)
    shape[0] = 3
    shape[1] = 4
    let a = ones(shape, DType.float32)
    let b = ravel(a)

    # Should be 1D view of same data (currently copies, TODO: implement views)
    assert_dim(b, 1, "Ravel should be 1D")


# ============================================================================
# Test concatenate()
# ============================================================================

fn test_concatenate_axis_0() raises:
    """Test concatenating along axis 0."""
    var shape_a = DynamicVector[Int](2)
    shape_a[0] = 2
    shape_a[1] = 3
    var shape_b = DynamicVector[Int](2)
    shape_b[0] = 3
    shape_b[1] = 3

    let a = ones(shape_a, DType.float32)  # 2x3
    let b = full(shape_b, 2.0, DType.float32)  # 3x3

    var tensors = DynamicVector[ExTensor](2)
    tensors[0] = a
    tensors[1] = b
    let c = concatenate(tensors, axis=0)

    # Result should be 5x3 (2+3 rows, 3 cols)
    assert_dim(c, 2, "Concatenated tensor should be 2D")
    assert_numel(c, 15, "Should have 15 elements (5*3)")


fn test_concatenate_axis_1() raises:
    """Test concatenating along axis 1."""
    var shape_a = DynamicVector[Int](2)
    shape_a[0] = 3
    shape_a[1] = 2
    var shape_b = DynamicVector[Int](2)
    shape_b[0] = 3
    shape_b[1] = 4

    let a = ones(shape_a, DType.float32)  # 3x2
    let b = full(shape_b, 2.0, DType.float32)  # 3x4

    var tensors = DynamicVector[ExTensor](2)
    tensors[0] = a
    tensors[1] = b
    let c = concatenate(tensors, axis=1)

    # Result should be 3x6 (3 rows, 2+4 cols)
    assert_numel(c, 18, "Should have 18 elements (3*6)")


# ============================================================================
# Test stack()
# ============================================================================

fn test_stack_new_axis() raises:
    """Test stacking tensors along new axis."""
    var shape = DynamicVector[Int](2)
    shape[0] = 2
    shape[1] = 3

    let a = ones(shape, DType.float32)  # 2x3
    let b = full(shape, 2.0, DType.float32)  # 2x3

    var tensors = DynamicVector[ExTensor](2)
    tensors[0] = a
    tensors[1] = b
    let c = stack(tensors, axis=0)

    # Result should be 2x2x3 (stacked along new axis 0)
    assert_dim(c, 3, "Stacked tensor should be 3D")
    assert_numel(c, 12, "Should have 12 elements (2*2*3)")


fn test_stack_axis_1() raises:
    """Test stacking along axis 1."""
    var shape = DynamicVector[Int](2)
    shape[0] = 2
    shape[1] = 3

    let a = ones(shape, DType.float32)
    let b = full(shape, 2.0, DType.float32)

    var tensors = DynamicVector[ExTensor](2)
    tensors[0] = a
    tensors[1] = b
    let c = stack(tensors, axis=1)

    # Result should be 2x2x3 (stacked along axis 1)
    assert_dim(c, 3, "Should be 3D")


# ============================================================================
# Test split()
# ============================================================================

fn test_split_equal() raises:
    """Test splitting into equal parts."""
    var shape = DynamicVector[Int](1)
    shape[0] = 12
    let a = arange(0.0, 12.0, 1.0, DType.float32)
    # let parts = split(a, 3)  # TODO: Implement split()

    # Should give 3 tensors of size 4 each
    # assert_equal_int(len(parts), 3, "Should split into 3 parts")
    # for i in range(3):
    #     assert_numel(parts[i], 4, "Each part should have 4 elements")
    pass  # Placeholder


fn test_split_unequal() raises:
    """Test splitting into unequal parts."""
    var shape = DynamicVector[Int](1)
    shape[0] = 10
    let a = arange(0.0, 10.0, 1.0, DType.float32)
    # let parts = split(a, [3, 5, 10])  # TODO: Implement split with indices

    # Should give 3 tensors of sizes 3, 2, 5
    # assert_numel(parts[0], 3, "First part should have 3 elements")
    # assert_numel(parts[1], 2, "Second part should have 2 elements")
    # assert_numel(parts[2], 5, "Third part should have 5 elements")
    pass  # Placeholder


# ============================================================================
# Test tile()
# ============================================================================

fn test_tile_1d() raises:
    """Test tiling 1D tensor."""
    var shape = DynamicVector[Int](1)
    shape[0] = 3
    let a = arange(0.0, 3.0, 1.0, DType.float32)  # [0, 1, 2]
    # let b = tile(a, 3)  # TODO: Implement tile()

    # Result: [0, 1, 2, 0, 1, 2, 0, 1, 2] (9 elements)
    # assert_numel(b, 9, "Tiled tensor should have 9 elements")
    pass  # Placeholder


fn test_tile_multidim() raises:
    """Test tiling with multi-dimensional repetitions."""
    var shape = DynamicVector[Int](2)
    shape[0] = 2
    shape[1] = 3
    let a = ones(shape, DType.float32)  # 2x3
    # let b = tile(a, (2, 3))  # TODO: Implement tile() with tuple

    # Result should be 4x9 (2*2 rows, 3*3 cols)
    # assert_numel(b, 36, "Should have 36 elements (4*9)")
    pass  # Placeholder


# ============================================================================
# Test repeat()
# ============================================================================

fn test_repeat_elements() raises:
    """Test repeating each element."""
    var shape = DynamicVector[Int](1)
    shape[0] = 3
    let a = arange(0.0, 3.0, 1.0, DType.float32)  # [0, 1, 2]
    # let b = repeat(a, 2)  # TODO: Implement repeat()

    # Result: [0, 0, 1, 1, 2, 2] (6 elements)
    # assert_numel(b, 6, "Repeated tensor should have 6 elements")
    pass  # Placeholder


fn test_repeat_axis() raises:
    """Test repeating along specific axis."""
    var shape = DynamicVector[Int](2)
    shape[0] = 2
    shape[1] = 3
    let a = ones(shape, DType.float32)  # 2x3
    # let b = repeat(a, 2, axis=0)  # TODO: Implement repeat() with axis

    # Result should be 4x3 (each row repeated twice)
    # assert_numel(b, 12, "Should have 12 elements (4*3)")
    pass  # Placeholder


# ============================================================================
# Test broadcast_to()
# ============================================================================

fn test_broadcast_to_compatible() raises:
    """Test broadcasting to compatible shape."""
    var shape_orig = DynamicVector[Int](1)
    shape_orig[0] = 3
    let a = arange(0.0, 3.0, 1.0, DType.float32)  # Shape (3,)
    # var target_shape = DynamicVector[Int](2)
    # target_shape[0] = 4
    # target_shape[1] = 3
    # let b = broadcast_to(a, target_shape)  # TODO: Implement broadcast_to()

    # Result should be 4x3 (broadcasting (3,) to (4,3))
    # assert_dim(b, 2, "Broadcasted tensor should be 2D")
    # assert_numel(b, 12, "Should have 12 elements")
    pass  # Placeholder


fn test_broadcast_to_incompatible() raises:
    """Test that broadcasting to incompatible shape raises error."""
    var shape_orig = DynamicVector[Int](1)
    shape_orig[0] = 3
    let a = arange(0.0, 3.0, 1.0, DType.float32)
    # var target_shape = DynamicVector[Int](1)
    # target_shape[0] = 5  # Incompatible: 3 != 5
    # let b = broadcast_to(a, target_shape)  # Should raise error

    # TODO: Verify error handling
    pass  # Placeholder


# ============================================================================
# Test permute()
# ============================================================================

fn test_permute_axes() raises:
    """Test permuting axes (similar to transpose with axes)."""
    var shape = DynamicVector[Int](3)
    shape[0] = 2
    shape[1] = 3
    shape[2] = 4
    let a = ones(shape, DType.float32)  # Shape (2, 3, 4)
    # let b = permute(a, (2, 0, 1))  # TODO: Implement permute()

    # Result should be (4, 2, 3)
    # assert_dim(b, 3, "Should still be 3D")
    # assert_numel(b, 24, "Should have same elements")
    pass  # Placeholder


# ============================================================================
# Test dtype preservation
# ============================================================================

fn test_reshape_preserves_dtype() raises:
    """Test that reshape preserves dtype."""
    var shape = DynamicVector[Int](1)
    shape[0] = 12
    let a = arange(0.0, 12.0, 1.0, DType.float64)
    # var new_shape = DynamicVector[Int](2)
    # new_shape[0] = 3
    # new_shape[1] = 4
    # let b = reshape(a, new_shape)

    # assert_dtype(b, DType.float64, "Reshape should preserve dtype")
    pass  # Placeholder


# ============================================================================
# Main test runner
# ============================================================================

fn main() raises:
    """Run all shape manipulation tests."""
    print("Running ExTensor shape manipulation tests...")

    # reshape() tests
    print("  Testing reshape()...")
    test_reshape_valid()
    test_reshape_invalid_size()
    test_reshape_infer_dimension()

    # squeeze() tests
    print("  Testing squeeze()...")
    test_squeeze_all_dims()
    test_squeeze_specific_dim()

    # unsqueeze() / expand_dims() tests
    print("  Testing unsqueeze() / expand_dims()...")
    test_unsqueeze_add_dim()
    test_expand_dims_at_end()

    # flatten() / ravel() tests
    print("  Testing flatten() / ravel()...")
    test_flatten_c_order()
    test_ravel_view()

    # concatenate() tests
    print("  Testing concatenate()...")
    test_concatenate_axis_0()
    test_concatenate_axis_1()

    # stack() tests
    print("  Testing stack()...")
    test_stack_new_axis()
    test_stack_axis_1()

    # split() tests
    print("  Testing split()...")
    test_split_equal()
    test_split_unequal()

    # tile() tests
    print("  Testing tile()...")
    test_tile_1d()
    test_tile_multidim()

    # repeat() tests
    print("  Testing repeat()...")
    test_repeat_elements()
    test_repeat_axis()

    # broadcast_to() tests
    print("  Testing broadcast_to()...")
    test_broadcast_to_compatible()
    test_broadcast_to_incompatible()

    # permute() tests
    print("  Testing permute()...")
    test_permute_axes()

    # Dtype preservation
    print("  Testing dtype preservation...")
    test_reshape_preserves_dtype()

    print("All shape manipulation tests completed!")
