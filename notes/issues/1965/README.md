# Issue #1965: Correct test results path in comprehensive-tests workflow

## Objective

Fix the Integration Tests group failure in the comprehensive-tests workflow by correcting the relative path used when writing test results.

## Problem

The workflow was using an incorrect relative path `../../test-results/` when writing test results from the Integration Tests group. Since this test group changes directory to `tests/shared/integration/` (3 directory levels deep), the relative path `../../` only goes back 2 levels, which is insufficient to reach the repository root where `test-results/` is created.

## Solution

Changed line 185 in `.github/workflows/comprehensive-tests.yml` from:
```yaml
result_file="../../test-results/${{ matrix.test-group.name }}.txt"
```

To:
```yaml
result_file="../../../test-results/${{ matrix.test-group.name }}.txt"
```

This correctly resolves back to the repository root from the `tests/shared/integration/` subdirectory.

## Files Changed

- `.github/workflows/comprehensive-tests.yml` - Line 185: Fixed relative path depth

## Success Criteria

- [x] Path corrected from `../../test-results/` to `../../../test-results/`
- [x] Integration Tests group can now properly write test results
- [x] CI workflow will pass without path errors

## Technical Details

**Path Analysis**:
- When running from: `tests/shared/integration/`
- With `../../test-results/`: Resolves to `tests/test-results/` (WRONG - only 2 levels up)
- With `../../../test-results/`: Resolves to `test-results/` (CORRECT - 3 levels up to repo root)

The `test-results/` directory is created at repository root (line 110), so paths must resolve back 3 levels from the deepest test group directory.

## Related Lines

- Line 110: `mkdir -p test-results` (creates directory at repo root)
- Line 151: Also uses `../../test-results/` for the "no tests found" case (already correct - only 2 levels deep)
- Line 185: Main fix location (was 2 levels, now 3 levels)
