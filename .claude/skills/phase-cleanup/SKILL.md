---
name: phase-cleanup
description: Refactor and finalize code after parallel phases complete, addressing technical debt and ensuring consistency. Use in cleanup phase to polish implementation before merge.
---

# Cleanup Phase Coordination Skill

This skill coordinates the cleanup phase to refactor and finalize code.

## When to Use

- User asks to clean up code (e.g., "run cleanup phase")
- Cleanup phase of 5-phase workflow (runs after parallel phases)
- Addressing technical debt
- Finalizing before merge

## Cleanup Workflow

### 1. Collect Issues

Gather issues discovered during parallel phases:

```bash
# Review implementation notes
grep "TODO\|FIXME\|HACK" -r src/

# Check test feedback
cat notes/issues/<number>/test-feedback.md

# Review package integration issues
cat notes/issues/<number>/package-issues.md
```

### 2. Refactor Code

Address code quality issues:

**Remove duplication:**

```bash
./scripts/detect_duplication.sh
# Refactor duplicated code into shared functions
```

**Improve naming:**

```bash
./scripts/check_naming.sh
# Rename unclear variables/functions
```

**Simplify complexity:**

```bash
./scripts/check_complexity.sh
# Break down complex functions
```

### 3. Update Documentation

Ensure documentation is accurate and complete:

```bash
# Update README if needed
# Add/update docstrings
# Create/update ADRs
# Update examples
```

### 4. Final Quality Checks

```bash
# Format all code
mojo format src/**/*.mojo

# Run all tests
mojo test tests/

# Run linters
pre-commit run --all-files

# Check coverage
./scripts/check_coverage.sh

# Verify no TODOs remain
grep -r "TODO" src/ || echo "✅ No TODOs"
```

## Refactoring Guidelines

### KISS - Keep It Simple

```mojo
# Before: Overly complex
fn process(data: Tensor) -> Tensor:
    let intermediate1 = transform1(data)
    let intermediate2 = transform2(intermediate1)
    let intermediate3 = transform3(intermediate2)
    return finalize(intermediate3)

# After: Simplified
fn process(data: Tensor) -> Tensor:
    return pipeline(data, [transform1, transform2, transform3])
```

### DRY - Don't Repeat Yourself

```mojo
# Before: Duplication
fn add_f32(a: Float32, b: Float32) -> Float32:
    return a + b

fn add_f64(a: Float64, b: Float64) -> Float64:
    return a + b

# After: Generic
fn add[dtype: DType](a: Scalar[dtype], b: Scalar[dtype]) -> Scalar[dtype]:
    return a + b
```

### Single Responsibility

```mojo
# Before: Multiple responsibilities
fn load_and_process_data(path: String) -> Tensor:
    let data = load_file(path)
    let cleaned = remove_outliers(data)
    let normalized = normalize(cleaned)
    return normalized

# After: Separate responsibilities
fn load_data(path: String) -> RawData:
    return load_file(path)

fn preprocess_data(data: RawData) -> Tensor:
    let cleaned = remove_outliers(data)
    return normalize(cleaned)
```

## Cleanup Checklist

- [ ] All TODOs/FIXMEs addressed or documented
- [ ] Code duplication removed
- [ ] Complex functions simplified
- [ ] Naming is clear and consistent
- [ ] Documentation updated
- [ ] All tests passing
- [ ] Code formatted
- [ ] No linting errors
- [ ] Performance requirements met
- [ ] Ready for review

## Common Cleanup Tasks

### 1. Remove Dead Code

```bash
# Find unused functions
./scripts/find_unused_code.sh

# Remove after verification
```

### 2. Consolidate Imports

```mojo
# Before: Scattered imports
from module1 import func1
from module2 import func2
from module1 import func3

# After: Organized
from module1 import func1, func3
from module2 import func2
```

### 3. Standardize Error Handling

```mojo
# Ensure consistent error handling patterns
fn safe_operation() raises -> Result:
    # Proper error handling
    pass
```

### 4. Add Missing Tests

```bash
# Check coverage
./scripts/check_coverage.sh

# Add tests for uncovered code
./scripts/generate_missing_tests.sh
```

## Integration with Workflow

**Cleanup runs after:**

- Test phase completes
- Implementation phase completes
- Package phase completes

**Cleanup produces:**

- Refactored, clean code
- Updated documentation
- Passing quality checks
- Merge-ready state

## Success Criteria

- [ ] No critical code smells
- [ ] Test coverage > 80%
- [ ] All quality checks pass
- [ ] Documentation complete
- [ ] Code reviewed and approved
- [ ] Ready to merge

See CLAUDE.md for development principles (KISS, DRY, SOLID) and cleanup guidelines.
