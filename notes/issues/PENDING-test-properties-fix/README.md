# [Test Fix] test_properties.mojo - Fix List ImplicitlyCopyable violations

## Status
Pending issue creation

## Error Description
```
value of type 'List[Int]' cannot be implicitly copied
```

## Affected Lines
- Line 184: `var strides = t._strides`
- Line 196: `var strides = t._strides`
- Line 210: `var strides = t._strides`

## Root Cause
`List[Int]` is not `ImplicitlyCopyable` in Mojo v0.25.7+. Direct assignment from `t._strides` (which returns `List[Int]`) to a local variable attempts an implicit copy, which fails compilation.

## Solution
Replace implicit copies with explicit copy constructor:

```mojo
# BEFORE (implicit copy - compilation error)
var strides = t._strides

# AFTER (explicit copy - compiles successfully)
var strides = List[Int](t._strides)
```

## Files to Modify
- `tests/shared/core/test_properties.mojo` (3 occurrences at lines 184, 196, 210)

## Part of
Comprehensive test fix plan (Fix #10)

## References
- [Mojo Test Failure Learnings - ImplicitlyCopyable Violations](../../review/mojo-test-failure-learnings.md#12-implicitlycopyable-trait-violations)
- Issue #2057 - Phase 3 comprehensive test fixes

## GitHub Issue Command
```bash
gh issue create \
  --title "[Test Fix] test_properties.mojo - Fix List ImplicitlyCopyable violations" \
  --body "$(cat notes/issues/PENDING-test-properties-fix/issue_body.md)" \
  --label "testing,bug,mojo"
```

## Implementation Steps
1. ✅ Create issue documentation
2. ⬜ Create GitHub issue
3. ⬜ Create worktree: `git worktree add ../worktree-<issue>-properties -b <issue>-properties`
4. ⬜ Read test file and verify line numbers
5. ⬜ Apply fixes (explicit copy constructor)
6. ⬜ Verify compilation
7. ⬜ Commit with detailed message
8. ⬜ Push to remote
9. ⬜ Create PR linked to issue
