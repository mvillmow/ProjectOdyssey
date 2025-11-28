## Error
```
value of type 'List[Int]' cannot be implicitly copied
```

## Affected Lines in test_properties.mojo
- Line 184: `var strides = t._strides` (in `test_strides_1d`)
- Line 196: `var strides = t._strides` (in `test_strides_2d_row_major`)
- Line 210: `var strides = t._strides` (in `test_strides_3d_row_major`)

## Root Cause
`List[Int]` is NOT `ImplicitlyCopyable` in Mojo v0.25.7+. When accessing `t._strides` (which is `List[Int]`), the assignment to a local variable attempts an implicit copy, causing compilation failure.

**From Mojo docs:**
> Collections like `List`, `Dict`, `String` are NOT implicitly copyable. Use explicit copy constructors or transfer operators (`^`).

## Solution
Replace implicit copies with **explicit copy constructor**:

### Before (fails compilation)
```mojo
var strides = t._strides  # ❌ Implicit copy - compilation error
```

### After (compiles successfully)
```mojo
var strides = List[Int](t._strides)  # ✅ Explicit copy - works
```

## Files Modified
- `tests/shared/core/test_properties.mojo` (3 lines)

## Part of
- Issue #2057 - Phase 3 comprehensive test fixes
- Comprehensive test fix plan (Fix #10)

## References
- [Mojo v0.25.7+ Ownership Semantics](https://docs.modular.com/mojo/manual/values/ownership/)
- [Internal: Mojo Test Failure Learnings](notes/review/mojo-test-failure-learnings.md#12-implicitlycopyable-trait-violations)

## Acceptance Criteria
- [ ] All 3 implicit copy violations fixed
- [ ] test_properties.mojo compiles without errors
- [ ] Tests execute successfully
- [ ] Pattern documented for future reference
