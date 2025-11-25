# Issue #1973: Fix dtype comparison helper for test assertions

## Objective

Fix test failures in the Core: Advanced Layers test group by creating a proper dtype comparison helper function and updating tests to use it correctly.

## Problem

Tests in `test_initializers_validation.mojo` attempted to compare DType values using the generic `assert_equal[T]` function:

```mojo
assert_equal(tensor.dtype, expected_dtype, message)
```

This fails with "failed to infer parameter 'T'" error because DType cannot be used as a template parameter for the generic comparison function. The test file also imported a non-existent `assert_equal_float` function.

## Solution

Two changes were made:

### 1. Fixed Import Statement (tests/shared/core/test_initializers_validation.mojo)

Removed the non-existent `assert_equal_float` import and added `assert_dtype_equal`:

```mojo
from tests.shared.conftest import (
    assert_close_float,
    assert_dtype_equal,      # Added this
    assert_equal,
    assert_equal_int,
    assert_false,
    assert_true,
)
```

### 2. Updated All dtype Comparisons (tests/shared/core/test_initializers_validation.mojo)

Replaced all 7 dtype assertions (lines 350-356) to use the specialized `assert_dtype_equal` function:

```mojo
# Before
assert_equal(xu.dtype, dt, "Xavier uniform dtype: " + name)

# After
assert_dtype_equal(xu.dtype, dt, "Xavier uniform dtype: " + name)
```

## Implementation Details

The `assert_dtype_equal` function already existed in `tests/shared/conftest.mojo` (lines 127-140):

```mojo
fn assert_dtype_equal(a: DType, b: DType, message: String = "") raises:
    """Assert exact equality of DType values.

    Args:
        a: First DType.
        b: Second DType.
        message: Optional error message.

    Raises:
        Error if a != b.
    """
    if a != b:
        var error_msg = message if message else "DTypes are not equal"
        raise Error(error_msg)
```

This specialized function handles DType comparison correctly without requiring template parameter inference.

## Files Changed

1. **tests/shared/core/test_initializers_validation.mojo**
   - Updated import statement: removed `assert_equal_float`, added `assert_dtype_equal`
   - Updated 7 dtype assertions to use `assert_dtype_equal` instead of `assert_equal`

## Success Criteria

- [x] Fixed import statement in test_initializers_validation.mojo
- [x] Updated all dtype comparisons to use assert_dtype_equal
- [x] Minimal changes - only touched the problematic assertions
- [x] No changes to test_conv.mojo (it didn't have dtype comparison issues)
- [x] Preserved all test logic and structure
- [x] Tests should now pass without "failed to infer parameter 'T'" errors

## References

- Existing assertion function: `tests/shared/conftest.mojo` lines 127-140 (`assert_dtype_equal`)
- Generic assertion function: `tests/shared/conftest.mojo` lines 49-62 (`assert_equal[T]`)
- Type system constraints: DType cannot be used as template parameter in generic functions

## Notes

- The fix leverages the existing `assert_dtype_equal` function that was already available but not imported
- This is consistent with other specialized assertion functions like `assert_close_float` and `assert_equal_int`
- The solution follows Mojo's principle of explicit type-specific functions rather than relying on template inference for special types
