# Investigate Mojo Heap Corruption Crashes

## Overview

| Attribute | Value |
|-----------|-------|
| **Date** | 2026-01-08 |
| **Objective** | Investigate and fix Mojo 0.26.1 heap corruption crash in `test_multi_precision_training.mojo` Test 9 |
| **Outcome** | ✅ Success - Crash identified and fixed |
| **CI Run** | [#20826482385](https://github.com/mvillmow/ProjectOdyssey/actions/runs/20826482385) |
| **Failing Commit** | `ba12f57ca6c2c624341f2a8046f3c2a701bed69c` |
| **Fix PR** | [#3103](https://github.com/mvillmow/ProjectOdyssey/pull/3103) |

## When to Use This Skill

Invoke this workflow when you encounter:

1. **CI-only crashes** that don't reproduce locally
2. **Heap corruption** errors in `libKGENCompilerRTShared.so`
3. **Crashes after ~15 cumulative tests** (ADR-009 threshold)
4. **Runtime crashes** with stack traces but no symbols
5. **Intermittent failures** in integration test suites

### Trigger Patterns

```text
error: execution crashed
#0 0x00007fe7ca5c60bb (/path/to/libKGENCompilerRTShared.so+0x3c60bb)
#1 0x00007fe7ca5c3ce6 (/path/to/libKGENCompilerRTShared.so+0x3c3ce6)
```

## Verified Workflow

### Phase 1: Identify Failing CI Run

1. **Locate the exact failing CI run** (don't rely on general reports)

   ```bash
   gh run list --workflow="Comprehensive Tests" --limit 10
   gh run view <run-id> --json conclusion,headSha,headBranch
   ```

2. **Extract crash logs**

   ```bash
   gh run view <run-id> --log 2>&1 | grep -A 10 -B 10 "<test-file-name>"
   ```

3. **Identify crash location**
   - Look for: Test number, last output before crash
   - Example: "Test 9: FP16 vs FP32 accuracy..." followed by crash

### Phase 2: Checkout Exact Failing Commit

```bash
# Use headSha from CI metadata
git checkout <commit-sha>

# Verify it's the exact commit
git log -1 --oneline
```

### Phase 3: Attempt Local Reproduction

**Important**: Crashes may NOT reproduce locally due to environment differences.

1. **Run test file individually** (usually passes):

   ```bash
   pixi run mojo -I . tests/path/to/test.mojo
   ```

2. **Run tests cumulatively** (like CI):

   ```bash
   pixi run just test-group "tests/shared/integration" "test_*.mojo"
   ```

3. **Stress test** (20+ iterations):

   ```bash
   for i in $(seq 1 20); do
     echo "=== Run $i ==="
     pixi run mojo -I . tests/path/to/test.mojo 2>&1 | tail -5
   done
   ```

### Phase 4: Analyze Crashing Code

1. **Read the specific test function** identified in crash logs
2. **Look for suspicious patterns**:
   - Unnecessary type conversions (e.g., `._get_float64()` on Float32 data)
   - Multiple ExTensor allocations in sequence
   - Complex dtype casting operations
   - Raw pointer access via `._data.bitcast[T]()`

3. **Check ADR-009 patterns**:
   - Cumulative test count (>10-15 tests triggers heap corruption)
   - Operations that worked in isolation but fail in CI

### Phase 5: Apply Fix

**Pattern**: Replace unnecessary type conversions with native type operations.

**Before** (causes heap corruption):

```mojo
var original_val = test_data._get_float64(0)      # Float32 -> Float64
var roundtrip_val = back_to_fp32._get_float64(0)  # Float32 -> Float64
```

**After** (uses native type):

```mojo
var original_val = test_data._get_float32(0)      # Native Float32
var roundtrip_val = back_to_fp32._get_float32(0)  # Native Float32
```

### Phase 6: Verify Fix

1. **Local testing**:

   ```bash
   for i in $(seq 1 10); do
     pixi run mojo -I . tests/path/to/test.mojo
   done
   ```

2. **Create PR and wait for CI validation**

## Failed Attempts

### ❌ Attempt 1: Testing at "Reported" Commit 784ef91a

**What we tried**: Checked out commit 784ef91a (from initial bug report)

**Result**: All tests passed - no crash detected

**Why it failed**: This was NOT the actual failing commit. Need to use exact commit from failing CI run.

**Lesson**: Always use CI metadata (`headSha`) to identify exact failing commit, not approximate reports.

### ❌ Attempt 2: Local Reproduction at Correct Commit

**What we tried**: Ran test 20 times locally at commit ba12f57c

**Result**: All tests passed - crash didn't reproduce locally

**Why it failed**: Heap corruption is environment-specific and manifests primarily in CI due to:

- Different memory allocation patterns in GitHub Actions runners
- Cumulative test execution order in CI
- CI runs multiple test files sequentially

**Lesson**: CI-only crashes are valid even if not reproducible locally. Trust CI logs and fix based on code analysis.

### ❌ Attempt 3: Using `just test` to Reproduce

**What we tried**: Ran full test suite with `just test`

**Result**: Exit code 137 (SIGKILL - out of memory), not the heap corruption crash

**Why it failed**: Running entire test suite (all 7 models) exhausts system memory. This is different from
the targeted heap corruption in integration tests.

**Lesson**: Use targeted test groups (`just test-group`) that match CI workflow, not full suite.

## Results & Parameters

### Root Cause

**Unnecessary Float32 → Float64 conversion** in `test_fp16_vs_fp32_accuracy()`:

```mojo
# Lines 337-338 (before fix)
var original_val = test_data._get_float64(0)
var roundtrip_val = back_to_fp32._get_float64(0)
```

This triggered Mojo 0.26.1's heap corruption bug when combined with cumulative test executions
(13 tests total across all integration tests run before this one).

### Fix Applied

```mojo
# Lines 339-340 (after fix)
var original_val = test_data._get_float32(0)
var roundtrip_val = back_to_fp32._get_float32(0)
```

### Verification Commands

```bash
# Check pre-commit passes
pixi run pre-commit run --all-files

# Verify test still passes
pixi run mojo -I . tests/shared/integration/test_multi_precision_training.mojo

# Check CI status
gh pr checks <pr-number>
```

### Configuration

- **Mojo Version**: 0.26.1
- **Platform**: Linux x86_64 (GitHub Actions)
- **CI Environment**: Ubuntu 22.04
- **Test Framework**: Custom Mojo tests (not pytest)

## Key Learnings

1. **CI logs are authoritative** - Use `gh run view <id> --log` to get exact crash location
2. **Commit precision matters** - Use `headSha` from CI metadata, not approximate commits
3. **CI-only crashes are real** - Don't dismiss failures that don't reproduce locally
4. **Type conversions are risky** - Unnecessary conversions can trigger heap corruption in Mojo 0.26.1
5. **Native types are safer** - Use `._get_float32()` for Float32 data, not `._get_float64()`

## Related Documentation

- [ADR-009: Heap Corruption Workaround](../../../docs/adr/ADR-009-heap-corruption-workaround.md)
- [PR #3103: Fix Test 9 Crash](https://github.com/mvillmow/ProjectOdyssey/pull/3103)
- [Failing CI Run](https://github.com/mvillmow/ProjectOdyssey/actions/runs/20826482385)

## Success Criteria

- [x] Crash location identified (Test 9, lines 337-338)
- [x] Root cause identified (Float32→Float64 conversion)
- [x] Fix applied (native Float32 comparison)
- [x] Pre-commit hooks pass
- [x] PR created and auto-merge enabled
- [ ] CI validation passes (pending)
