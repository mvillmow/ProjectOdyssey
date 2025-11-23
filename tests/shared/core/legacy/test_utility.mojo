"""Tests for ExTensor utility operations.

Tests utility functions including copy, clone, properties, conversions,
and helper methods like numel, dim, size, stride, is_contiguous.
"""

from sys import DType

# Import ExTensor and operations
from shared.core import ExTensor, zeros, ones, full, arange

# Import test helpers
from ..helpers.assertions import (
    assert_dtype,
    assert_numel,
    assert_dim,
    assert_value_at,
    assert_equal_int,
)


# ============================================================================
# Test copy() and clone()
# ============================================================================

fn test_copy_independence() raises:
    """Test that copy creates independent tensor."""
    var shape = List[Int]()
    shape.append(5)
    let a = full(shape, 3.0, DType.float32)
    # let b = copy(a)  # TODO: Implement copy()

    # Modify a, b should not change
    # a[0] = 99.0  # TODO: Implement __setitem__
    # assert_value_at(b, 0, 3.0, 1e-6, "Copy should be independent")
    pass  # Placeholder


fn test_clone_identical() raises:
    """Test that clone creates identical tensor."""
    var shape = List[Int]()
    shape.append(3)
    shape.append(4)
    let a = arange(0.0, 12.0, 1.0, DType.float32)
    # Reshape to 3x4 first
    # let b = clone(a)  # TODO: Implement clone()

    # Should have same values
    # for i in range(12):
    #     assert_value_at(b, i, Float64(i), 1e-6, "Clone should have same values")
    pass  # Placeholder


# ============================================================================
# Test property accessors
# ============================================================================

fn test_numel_total_elements() raises:
    """Test numel() returns total number of elements."""
    var shape = List[Int]()
    shape.append(2)
    shape.append(3)
    shape.append(4)
    let t = ones(shape, DType.float32)

    assert_numel(t, 24, "numel should return 24 for (2,3,4)")


fn test_dim_num_dimensions() raises:
    """Test that dim is correct."""
    var shape_1d = List[Int]()
    shape_1d.append(10)
    let t1 = ones(shape_1d, DType.float32)
    assert_dim(t1, 1, "1D tensor should have dim=1")

    var shape_3d = List[Int]()
    shape_3d.append(2)
    shape_3d.append(3)
    shape_3d.append(4)
    let t3 = ones(shape_3d, DType.float32)
    assert_dim(t3, 3, "3D tensor should have dim=3")


fn test_shape_property() raises:
    """Test shape() returns correct shape."""
    var shape = List[Int]()
    shape.append(3)
    shape.append(4)
    let t = ones(shape, DType.float32)

    let s = t.shape()
    assert_equal_int(len(s), 2, "Shape should have 2 dimensions")
    assert_equal_int(s[0], 3, "First dimension should be 3")
    assert_equal_int(s[1], 4, "Second dimension should be 4")


fn test_dtype_property() raises:
    """Test dtype() returns correct data type."""
    var shape = List[Int]()
    shape.append(5)

    let t32 = ones(shape, DType.float32)
    assert_dtype(t32, DType.float32, "Should be float32")

    let t64 = ones(shape, DType.float64)
    assert_dtype(t64, DType.float64, "Should be float64")


# ============================================================================
# Test stride calculations
# ============================================================================

fn test_stride_row_major() raises:
    """Test stride calculation for row-major (C-order)."""
    var shape = List[Int]()
    shape.append(2)
    shape.append(3)
    shape.append(4)
    let t = ones(shape, DType.float32)  # Shape (2, 3, 4)

    # Row-major strides for (2,3,4): [12, 4, 1]
    let strides = t._strides
    assert_equal_int(strides[0], 12, "Stride for dim 0 should be 12")
    assert_equal_int(strides[1], 4, "Stride for dim 1 should be 4")
    assert_equal_int(strides[2], 1, "Stride for dim 2 should be 1")


# ============================================================================
# Test contiguity
# ============================================================================

fn test_is_contiguous_true() raises:
    """Test that newly created tensors are contiguous."""
    var shape = List[Int]()
    shape.append(3)
    shape.append(4)
    let t = ones(shape, DType.float32)

    # TODO: assert_contiguous(t)
    # For now, just test the method exists
    let contig = t.is_contiguous()
    # Should be True for newly created tensor
    pass  # Placeholder


fn test_is_contiguous_after_transpose() raises:
    """Test that transposed tensor is not contiguous."""
    var shape = List[Int]()
    shape.append(3)
    shape.append(4)
    let a = ones(shape, DType.float32)
    # let b = transpose(a)  # TODO: Implement transpose()

    # Transposed tensor is typically not contiguous
    # let contig = b.is_contiguous()
    # assert_false(contig, "Transposed tensor should not be contiguous")
    pass  # Placeholder


# ============================================================================
# Test contiguous() - make contiguous copy
# ============================================================================

fn test_contiguous_on_noncontiguous() raises:
    """Test making non-contiguous tensor contiguous."""
    var shape = List[Int]()
    shape.append(3)
    shape.append(4)
    let a = ones(shape, DType.float32)
    # let b = transpose(a)  # Not contiguous
    # let c = contiguous(b)  # TODO: Implement contiguous()

    # c should now be contiguous
    # let contig = c.is_contiguous()
    # assert_true(contig, "contiguous() should make tensor contiguous")
    pass  # Placeholder


# ============================================================================
# Test item() - scalar extraction
# ============================================================================

fn test_item_single_element() raises:
    """Test extracting value from single-element tensor."""
    var shape = List[Int]()
    let t = full(shape, 42.0, DType.float32)
    # let val = item(t)  # TODO: Implement item()

    # assert_equal_float(val, 42.0, 1e-6, "item() should extract scalar value")
    pass  # Placeholder


fn test_item_requires_single_element() raises:
    """Test that item() requires single-element tensor."""
    var shape = List[Int]()
    shape.append(5)
    let t = ones(shape, DType.float32)
    # let val = item(t)  # Should raise error

    # TODO: Verify error handling
    pass  # Placeholder


# ============================================================================
# Test tolist() - convert to nested list
# ============================================================================

fn test_tolist_1d() raises:
    """Test converting 1D tensor to list."""
    let t = arange(0.0, 5.0, 1.0, DType.float32)
    # let lst = tolist(t)  # TODO: Implement tolist()

    # Should return [0, 1, 2, 3, 4]
    # assert_equal_int(len(lst), 5, "List should have 5 elements")
    pass  # Placeholder


fn test_tolist_nested() raises:
    """Test converting multi-dimensional tensor to nested list."""
    var shape = List[Int]()
    shape.append(2)
    shape.append(3)
    let t = arange(0.0, 6.0, 1.0, DType.float32)
    # Reshape to 2x3 first
    # let lst = tolist(t)  # TODO: Implement tolist()

    # Should return [[0, 1, 2], [3, 4, 5]]
    # assert_equal_int(len(lst), 2, "Outer list should have 2 elements")
    # assert_equal_int(len(lst[0]), 3, "Inner list should have 3 elements")
    pass  # Placeholder


# ============================================================================
# Test __len__
# ============================================================================

fn test_len_first_dim() raises:
    """Test __len__ returns size of first dimension."""
    var shape = List[Int]()
    shape.append(5)
    shape.append(3)
    let t = ones(shape, DType.float32)

    # let length = len(t)  # TODO: Implement __len__
    # assert_equal_int(length, 5, "__len__ should return first dimension")
    pass  # Placeholder


fn test_len_1d() raises:
    """Test __len__ on 1D tensor."""
    var shape = List[Int]()
    shape.append(10)
    let t = ones(shape, DType.float32)

    # let length = len(t)
    # assert_equal_int(length, 10, "__len__ should return size for 1D")
    pass  # Placeholder


# ============================================================================
# Test __bool__
# ============================================================================

fn test_bool_single_element() raises:
    """Test __bool__ on single-element tensor."""
    var shape = List[Int]()
    let t_zero = full(shape, 0.0, DType.float32)
    let t_nonzero = full(shape, 5.0, DType.float32)

    # if t_zero:  # Should be False
    #     raise Error("Zero tensor should be falsy")
    # if not t_nonzero:  # Should be True
    #     raise Error("Non-zero tensor should be truthy")
    pass  # Placeholder


fn test_bool_requires_single_element() raises:
    """Test that __bool__ requires single-element tensor."""
    var shape = List[Int]()
    shape.append(5)
    let t = ones(shape, DType.float32)

    # if t:  # Should raise error for multi-element tensor
    #     pass

    # TODO: Verify error handling
    pass  # Placeholder


# ============================================================================
# Test type conversions
# ============================================================================

fn test_int_conversion() raises:
    """Test __int__ conversion."""
    var shape = List[Int]()
    let t = full(shape, 42.5, DType.float32)

    # let val = int(t)  # TODO: Implement __int__
    # assert_equal_int(val, 42, "__int__ should convert to int")
    pass  # Placeholder


fn test_float_conversion() raises:
    """Test __float__ conversion."""
    var shape = List[Int]()
    let t = full(shape, 42.0, DType.int32)

    # let val = float(t)  # TODO: Implement __float__
    # assert_equal_float(val, 42.0, 1e-6, "__float__ should convert to float")
    pass  # Placeholder


# ============================================================================
# Test __str__ and __repr__
# ============================================================================

fn test_str_readable() raises:
    """Test __str__ produces readable output."""
    var shape = List[Int]()
    shape.append(3)
    let t = arange(0.0, 3.0, 1.0, DType.float32)

    # let s = str(t)  # TODO: Implement __str__
    # Should produce something like "ExTensor([0, 1, 2], dtype=float32)"
    pass  # Placeholder


fn test_repr_complete() raises:
    """Test __repr__ produces complete representation."""
    var shape = List[Int]()
    shape.append(2)
    shape.append(2)
    let t = ones(shape, DType.float32)

    # let r = repr(t)  # TODO: Implement __repr__
    # Should produce detailed representation
    pass  # Placeholder


# ============================================================================
# Test __hash__
# ============================================================================

fn test_hash_immutable() raises:
    """Test __hash__ for immutable tensors."""
    var shape = List[Int]()
    shape.append(3)
    let a = arange(0.0, 3.0, 1.0, DType.float32)
    let b = arange(0.0, 3.0, 1.0, DType.float32)

    # let hash_a = hash(a)  # TODO: Implement __hash__
    # let hash_b = hash(b)
    # Equal tensors should have same hash
    # assert_equal_int(hash_a, hash_b, "Equal tensors should have same hash")
    pass  # Placeholder


# ============================================================================
# Test diff() - consecutive differences
# ============================================================================

fn test_diff_1d() raises:
    """Test computing consecutive differences."""
    let t = arange(0.0, 5.0, 1.0, DType.float32)  # [0, 1, 2, 3, 4]
    # let d = diff(t)  # TODO: Implement diff()

    # Result: [1, 1, 1, 1] (4 elements)
    # assert_numel(d, 4, "diff should have n-1 elements")
    # assert_all_values(d, 1.0, 1e-6, "Consecutive differences should be 1")
    pass  # Placeholder


fn test_diff_higher_order() raises:
    """Test higher-order differences."""
    let t = arange(0.0, 5.0, 1.0, DType.float32)
    # let d = diff(t, n=2)  # TODO: Implement n parameter

    # Second differences of [0,1,2,3,4] -> [0,0,0]
    # assert_numel(d, 3, "Second diff should have n-2 elements")
    pass  # Placeholder


# ============================================================================
# Main test runner
# ============================================================================

fn main() raises:
    """Run all utility operation tests."""
    print("Running ExTensor utility operation tests...")

    # copy() and clone() tests
    print("  Testing copy() and clone()...")
    test_copy_independence()
    test_clone_identical()

    # Property accessors
    print("  Testing property accessors...")
    test_numel_total_elements()
    test_dim_num_dimensions()
    test_shape_property()
    test_dtype_property()

    # Stride calculations
    print("  Testing stride calculations...")
    test_stride_row_major()

    # Contiguity
    print("  Testing contiguity...")
    test_is_contiguous_true()
    test_is_contiguous_after_transpose()
    test_contiguous_on_noncontiguous()

    # item() extraction
    print("  Testing item()...")
    test_item_single_element()
    test_item_requires_single_element()

    # tolist() conversion
    print("  Testing tolist()...")
    test_tolist_1d()
    test_tolist_nested()

    # __len__
    print("  Testing __len__...")
    test_len_first_dim()
    test_len_1d()

    # __bool__
    print("  Testing __bool__...")
    test_bool_single_element()
    test_bool_requires_single_element()

    # Type conversions
    print("  Testing type conversions...")
    test_int_conversion()
    test_float_conversion()

    # __str__ and __repr__
    print("  Testing __str__ and __repr__...")
    test_str_readable()
    test_repr_complete()

    # __hash__
    print("  Testing __hash__...")
    test_hash_immutable()

    # diff()
    print("  Testing diff()...")
    test_diff_1d()
    test_diff_higher_order()

    print("All utility operation tests completed!")
