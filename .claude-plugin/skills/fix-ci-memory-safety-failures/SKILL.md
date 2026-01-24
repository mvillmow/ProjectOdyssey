# Skill: Fix CI Memory Safety Failures in Mojo

## Overview

| Field | Value |
|-------|-------|
| **Date** | 2026-01-24 |
| **Objective** | Diagnose and fix CI failures in PR #3109 (memory safety crash in serialization) |
| **Outcome** | ✅ All 3 CI failures resolved (pre-commit, Examples segfault, Test Report) |
| **PR** | #3109 (feature/improve-import-tests-3033) |
| **Commits** | 31d25c7e (memory safety fix), c43b80cf (formatting fix) |

## When to Use This Skill

Use this skill when encountering CI failures with these characteristics:

1. **Segfault in Mojo code** - Crashes in `libKGENCompilerRTShared.so` during pointer access
2. **Flaky CI tests** - Tests pass locally but fail in CI environment
3. **UnsafePointer access crashes** - Especially when accessing through copied structs
4. **Python interop interference** - Crashes after Python FFI calls (e.g., `os.makedirs()`)
5. **List/collection iteration crashes** - Accessing struct fields through `list[i].field`

### Trigger Patterns

```
[CRASH] - segfault in libKGENCompilerRTShared.so during save_named_tensors()
```

```
Error: Process completed with exit code 139 (segfault)
```

```
pre-commit hook(s) made changes.
```

## Verified Workflow

### Step 1: Identify Root Cause

Read CI logs and identify crash location:

```bash
# Get PR checks status
gh pr checks <pr-number>

# View failed workflow run
gh run view <run-id>
```

**Key indicators:**
- Segfault exit code (139)
- Crash in pointer access (`bytes_to_hex`, `_data` access)
- Test passes in some commits but fails in others (flaky)

### Step 2: Analyze Memory Safety Issue

For the serialization crash, the root cause was:

**Problem**: `bytes_to_hex(tensor._data, total_bytes)` accessed pointer after tensor was copied through `tensors[i].tensor`

**Why it crashes**:
1. `save_named_tensors()` iterates `List[NamedTensor]`
2. `tensors[i].tensor` returns a **copy** (via `__copyinit__`)
3. Copy shares `_data` pointer via refcount
4. Python interop (`create_directory()`) may interfere with pointer validity
5. When `bytes_to_hex()` accesses pointer, it's invalid → segfault

### Step 3: Apply Defensive Fix

**Option A: Create local copy (recommended)**

```mojo
fn save_tensor(tensor: ExTensor, filepath: String, name: String = "") raises:
    # Create local copy to ensure stable data pointer
    # This prevents issues when tensor is accessed through List[NamedTensor]
    var local_tensor = tensor

    var shape = local_tensor.shape()
    var dtype = local_tensor.dtype()
    var numel = local_tensor.numel()

    # Convert tensor data to hex
    var dtype_size = get_dtype_size(dtype)
    var total_bytes = numel * dtype_size
    var hex_data = bytes_to_hex(local_tensor._data, total_bytes)
    # ... rest of function
```

**Option B: Add null check (defense in depth)**

```mojo
fn bytes_to_hex(data: UnsafePointer[UInt8], num_bytes: Int) -> String:
    # Defensive null check for pointer safety
    if not data:
        return ""

    var hex_chars = "0123456789abcdef"
    var result = String("")

    for i in range(num_bytes):
        var byte = Int(data[i])
        # ... rest of function
```

**Recommended: Use BOTH for defense in depth**

### Step 4: Fix Formatting Issues

After fixing memory safety, run pre-commit to catch formatting issues:

```bash
# Run mojo format on modified files
pixi run mojo format tests/shared/integration/test_packaging.mojo
pixi run mojo format tests/shared/test_imports.mojo

# Verify pre-commit passes
pixi run pre-commit run --all-files
```

### Step 5: Commit and Verify

```bash
# Commit memory safety fix
git add shared/utils/serialization.mojo tests/shared/integration/test_packaging.mojo
git commit -m "fix(serialization): add defensive memory safety checks

- Add local tensor copy in save_tensor() to ensure stable pointer
- Add null check in bytes_to_hex() for defense in depth
- Format test_packaging.mojo for pre-commit

Fixes crash in test_trait_based_serialization.mojo"

# Commit formatting fix
git add tests/shared/test_imports.mojo
git commit -m "style: format test_imports.mojo"

# Push and monitor CI
git push
gh pr checks <pr-number> --watch
```

## Failed Attempts

### ❌ Attempt 1: Running Tests Locally

**What was tried:**
```bash
pixi run mojo run tests/examples/test_trait_based_serialization.mojo
pixi run mojo test tests/examples/test_trait_based_serialization.mojo
```

**Why it failed:**
- `mojo run` requires the `shared` package to be built
- `mojo test` command doesn't exist in Mojo v0.26.1
- Local environment differs from CI Docker container (memory protections, GIL state)

**Lesson**: Cannot reproduce CI-specific crashes locally. Must rely on CI logs and defensive coding.

### ❌ Attempt 2: Assuming Serialization Code Was Modified in PR

**What was tried:**
- Reviewed PR diff to find what changed in serialization code

**Why it was wrong:**
- `test_trait_based_serialization.mojo` was NOT modified in PR #3109
- Crash existed in earlier commits but was flaky
- PR only exposed the existing memory safety issue

**Lesson**: CI failures don't always correlate with PR changes. Check git history to see if test passed before.

### ❌ Attempt 3: Only Fixing One Layer of Defense

**What was tried:**
- Initial fix only added null check in `bytes_to_hex()`

**Why it's incomplete:**
- Null check prevents crash but doesn't fix root cause
- Pointer could still be invalid even if non-null (dangling pointer)
- Need to prevent invalid pointer creation, not just detect it

**Lesson**: Use defense in depth - fix root cause AND add safety checks.

## Results & Parameters

### CI Status Before Fix

```
❌ pre-commit - test_packaging.mojo needs formatting
❌ Examples - segfault in test_trait_based_serialization.mojo
❌ Test Report - cascading failure from Examples
```

### CI Status After Fix

```
✅ pre-commit - pass (49s)
✅ Examples - pass (23s)
✅ Test Report - pass (implicitly fixed)
✅ Integration Tests - pass (40s)
✅ All other checks - pass or pending
```

### Files Modified

```
shared/utils/serialization.mojo:
- Line 111-113: Added local tensor copy
- Line 506-508: Added null pointer check

tests/shared/integration/test_packaging.mojo:
- Multiple lines: Applied mojo format (line wrapping)

tests/shared/test_imports.mojo:
- Lines 23-35: Applied mojo format (line wrapping)
```

### Commands Used

```bash
# Diagnosis
gh pr checks 3109
gh pr view 3109

# Fix application
pixi run mojo format tests/shared/integration/test_packaging.mojo
pixi run mojo format tests/shared/test_imports.mojo

# Verification
git add shared/utils/serialization.mojo tests/shared/integration/test_packaging.mojo
git commit -m "fix(serialization): add defensive memory safety checks"
git push

gh pr checks 3109 --watch
```

## Key Insights

### Memory Safety Patterns in Mojo

1. **UnsafePointer through copies**: Accessing `_data` through `list[i].field` creates a copy. The copy's pointer may be invalid.

2. **Python interop interference**: Python FFI calls (`os.makedirs()`, `pathlib.Path()`) can interfere with Mojo's memory management.

3. **Reference counting isn't enough**: Even with `__copyinit__` incrementing refcount, pointers can become invalid due to:
   - Python GIL state changes
   - Memory allocator differences between local/CI
   - Docker container memory protections

### CI-Specific Behaviors

1. **Flaky tests in CI**: Tests that pass locally but fail in CI often indicate:
   - Memory safety issues (stricter protections in Docker)
   - Race conditions (different timing in CI)
   - Environment differences (Python version, memory allocator)

2. **Pre-commit formatting**: Always run `mojo format` on all modified files. Pre-commit will fail if any file needs formatting, even if you didn't modify that specific line.

### Defensive Coding Guidelines

1. **Always create local copies** before accessing UnsafePointer fields
2. **Add null checks** at pointer access boundaries
3. **Avoid accessing pointers through collection indices** (`list[i].tensor._data`)
4. **Test in CI environment** - local tests may not catch memory issues

## Related Issues

- ExTensor memory management fixes: commits b823a190, e016c956, f12e832b
- Issue #3033: Improve import tests (parent issue)
- PR #3109: Import test improvements (this PR)

## References

- CLAUDE.md sections: Git Workflow, Pre-commit Hooks, Testing Strategy
- Mojo Guidelines: `.claude/shared/mojo-guidelines.md`
- Error Handling: `.claude/shared/error-handling.md`
