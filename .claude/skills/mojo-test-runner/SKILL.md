---
name: mojo-test-runner
description: Run Mojo test files and test suites using mojo test command with reporting and filtering capabilities. Use when executing Mojo tests or verifying test coverage.
---

# Mojo Test Runner Skill

This skill runs Mojo tests using the `mojo test` command with various filtering and reporting options.

## When to Use

- User asks to run tests (e.g., "run the Mojo tests")
- Verifying implementation correctness
- Running TDD red-green-refactor cycle
- Checking test coverage
- CI/CD test execution

## Usage

### Run All Tests

```bash
# Run all tests in tests/ directory
mojo test tests/

# Run specific test file
mojo test tests/test_tensor.mojo

# Run with verbose output
mojo test -v tests/
```

### Run Specific Tests

```bash
# Run single test file
./scripts/run_tests.sh test_tensor

# Run tests matching pattern
./scripts/run_tests.sh tensor

# Run unit tests only
./scripts/run_tests.sh --unit

# Run integration tests only
./scripts/run_tests.sh --integration
```

### Test Output

```text
Running tests from: tests/test_tensor.mojo
test_tensor_creation ... ok
test_tensor_addition ... ok
test_tensor_multiplication ... ok
test_edge_case_empty ... ok

4 tests, 4 passed, 0 failed
```

## Test Types

### Unit Tests

Location: `tests/unit/`

Fast, isolated tests for individual functions/classes:
```bash
mojo test tests/unit/
```

### Integration Tests

Location: `tests/integration/`

Tests for component interactions:
```bash
mojo test tests/integration/
```

### Performance Tests

Location: `tests/performance/`

Benchmarks and performance validation:
```bash
mojo test tests/performance/
```

## Test Discovery

Mojo discovers tests by:
- Files matching `test_*.mojo` or `*_test.mojo`
- Functions starting with `test_`
- In specified directory or file

```mojo
# This function will be discovered and run
fn test_my_feature() raises:
    assert_equal(result, expected)

# This won't be discovered (no test_ prefix)
fn my_helper_function():
    pass
```

## Error Handling

### Test Failures

```text
Running tests from: tests/test_tensor.mojo
test_tensor_creation ... ok
test_tensor_addition ... FAILED

Failures:
test_tensor_addition
  Expected: 5
  Got: 4

1 test failed, 1 passed
```

**Actions:**
1. Review failure message
2. Fix code or test
3. Re-run tests
4. Verify passing

### Common Issues

- **Import errors**: Check module paths and dependencies
- **Syntax errors**: Fix before running tests
- **Memory errors**: Check ownership and lifetimes
- **Timeout**: Optimize or increase timeout

## Test Reporting

### Basic Report

```bash
mojo test tests/
# Shows pass/fail summary
```

### Verbose Report

```bash
mojo test -v tests/
# Shows detailed output for each test
```

### Coverage Report (Future)

```bash
# When coverage tooling available
./scripts/run_tests.sh --coverage
```

## CI Integration

Tests run automatically in CI:

```yaml
- name: Run Mojo Tests
  run: mojo test tests/
```

## TDD Workflow

1. **Write failing test** (Red)
   ```bash
   mojo test tests/test_feature.mojo  # Fails
   ```

2. **Implement minimal code** (Green)
   ```bash
   mojo test tests/test_feature.mojo  # Passes
   ```

3. **Refactor** (Refactor)
   ```bash
   mojo test tests/test_feature.mojo  # Still passes
   ```

## Examples

**Run all tests:**
```bash
./scripts/run_tests.sh
```

**Run specific file:**
```bash
./scripts/run_tests.sh test_tensor
```

**Run unit tests only:**
```bash
./scripts/run_tests.sh --unit
```

**Watch mode (re-run on changes):**
```bash
./scripts/run_tests.sh --watch
```

## Scripts Available

- `scripts/run_tests.sh` - Run tests with filtering
- `scripts/test_watch.sh` - Watch mode for continuous testing
- `scripts/test_coverage.sh` - Coverage reporting (future)

## Best Practices

1. **Run tests frequently** - After each small change
2. **Test first** - Write tests before implementation (TDD)
3. **Fast tests** - Keep unit tests fast (< 1s each)
4. **Isolated tests** - No dependencies between tests
5. **Clear names** - Test names describe what they test
6. **Edge cases** - Test boundaries and error conditions

## Performance Testing

### Benchmark Tests

```mojo
from benchmark import Benchmark

fn test_performance_simd() raises:
    """Benchmark SIMD operation performance."""
    let b = Benchmark()

    fn work():
        # SIMD operation to benchmark
        pass

    let result = b.run(work)
    # Assert performance requirements
    assert_true(result.mean < 1000)  # Must be < 1000ns
```

## Integration with Other Skills

- **phase-test-tdd** - Generate test files
- **mojo-format** - Format test files
- **gh-check-ci-status** - Verify tests pass in CI

See testing documentation in `/notes/review/` for comprehensive testing strategy.
