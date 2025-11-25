# Issue #1969: Add assert_greater overload for Int type

## Objective

Add Int type overload for the `assert_greater` function in `tests/shared/conftest.mojo` to support integer comparisons in unit tests.

## Deliverables

- Added `assert_greater(a: Int, b: Int, message: String = "") raises` function in `tests/shared/conftest.mojo`

## Success Criteria

- ✅ Int overload added after existing Float32 and Float64 overloads (lines 249-262)
- ✅ Follows same pattern as existing overloads with proper docstring
- ✅ Raises Error when a <= b (assertion fails when condition is not met)
- ✅ Supports optional custom error message parameter

## References

- Existing Float32 overload: lines 217-230
- Existing Float64 overload: lines 233-246
- Test framework: `tests/shared/conftest.mojo`

## Implementation Notes

The Int overload was added with the following implementation:

```mojo
fn assert_greater(a: Int, b: Int, message: String = "") raises:
    """Assert a > b for Int values.

    Args:
        a: First value.
        b: Second value.
        message: Optional error message.

    Raises:
        Error if a <= b.
    """
    if a <= b:
        var error_msg = message if message else String(a) + " <= " + String(b)
        raise Error(error_msg)
```

This overload enables tests that need to assert integer values are in proper order, fixing the "ambiguous call" errors that occurred when using assert_greater with Int parameters.
