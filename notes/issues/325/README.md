# Issue #325: [Test] Step Scheduler

## Phase

Testing (TDD - Test-Driven Development)

## Component

StepLR - Step Decay Learning Rate Scheduler

## Objective

Implement comprehensive test suite for StepLR scheduler following TDD principles. Tests define the expected behavior and API before/during implementation.

## Test Strategy

### Test Categories

1. **Core Functionality Tests**
   - Initialization with valid parameters
   - Step decay at correct epochs
   - Multiple decay steps over training
   - Formula correctness

2. **Parameter Variation Tests**
   - Different gamma values (0.1, 0.5, 0.9)
   - Different step sizes (1, 10, 100)
   - Edge case gamma=1.0 (no decay)

3. **Edge Case Tests**
   - Zero step size (should error)
   - Negative gamma (should error)
   - Very small LR (numerical precision)

4. **Integration Tests**
   - Optimizer LR updates
   - Training loop integration
   - State save/load

5. **Property-Based Tests**
   - Monotonic decrease property
   - Reproducibility

## Test Specifications

### 1. Core Functionality Tests

#### test_step_scheduler_initialization()
```mojo
"""Test StepLR initializes with correct parameters."""

# Setup
var scheduler = StepLR(base_lr=0.1, step_size=10, gamma=0.1)

# Verify
assert_equal(scheduler.base_lr, 0.1)
assert_equal(scheduler.step_size, 10)
assert_almost_equal(scheduler.gamma, 0.1)
```

**Expected**: Scheduler stores parameters correctly

#### test_step_scheduler_reduces_lr_at_step()
```mojo
"""Test StepLR reduces LR at step boundary."""

# Setup
var scheduler = StepLR(base_lr=1.0, step_size=5, gamma=0.1)

# Test epochs 0-4 (before step)
for epoch in range(5):
    var lr = scheduler.get_lr(epoch)
    assert_almost_equal(lr, 1.0)

# Test epoch 5 (at step)
var lr5 = scheduler.get_lr(epoch=5)
assert_almost_equal(lr5, 0.1)
```

**Expected**: LR unchanged until step_size, then reduced by gamma

#### test_step_scheduler_multiple_steps()
```mojo
"""Test StepLR continues reducing at each step."""

# Setup
var scheduler = StepLR(base_lr=1.0, step_size=5, gamma=0.1)

# Test
assert_almost_equal(scheduler.get_lr(epoch=0), 1.0)
assert_almost_equal(scheduler.get_lr(epoch=5), 0.1)
assert_almost_equal(scheduler.get_lr(epoch=10), 0.01)
assert_almost_equal(scheduler.get_lr(epoch=15), 0.001)
```

**Expected**: LR = base_lr × gamma^⌊epoch/step_size⌋

### 2. Parameter Variation Tests

#### test_step_scheduler_different_gamma_values()
```mojo
"""Test StepLR with various gamma values."""

# Test gamma=0.5 (reduce to 50%)
var sched1 = StepLR(base_lr=1.0, step_size=1, gamma=0.5)
assert_almost_equal(sched1.get_lr(1), 0.5)
assert_almost_equal(sched1.get_lr(2), 0.25)

# Test gamma=0.9 (slower decay)
var sched2 = StepLR(base_lr=1.0, step_size=1, gamma=0.9)
assert_almost_equal(sched2.get_lr(1), 0.9)
assert_almost_equal(sched2.get_lr(2), 0.81)
```

**Expected**: Different gamma produces different decay rates

#### test_step_scheduler_gamma_one()
```mojo
"""Test StepLR with gamma=1.0 (no decay)."""

# Setup
var scheduler = StepLR(base_lr=1.0, step_size=1, gamma=1.0)

# Test - LR should stay constant
for epoch in range(10):
    assert_almost_equal(scheduler.get_lr(epoch), 1.0)
```

**Expected**: gamma=1.0 results in constant LR (no decay)

#### test_step_scheduler_different_step_sizes()
```mojo
"""Test StepLR with various step_size values."""

# Test step_size=1 (decay every epoch)
var sched1 = StepLR(base_lr=1.0, step_size=1, gamma=0.5)
assert_almost_equal(sched1.get_lr(0), 1.0)
assert_almost_equal(sched1.get_lr(1), 0.5)
assert_almost_equal(sched1.get_lr(2), 0.25)

# Test step_size=10 (decay every 10 epochs)
var sched2 = StepLR(base_lr=1.0, step_size=10, gamma=0.5)
for epoch in range(10):
    assert_almost_equal(sched2.get_lr(epoch), 1.0)
assert_almost_equal(sched2.get_lr(10), 0.5)
```

**Expected**: step_size controls decay frequency

### 3. Edge Case Tests

#### test_step_scheduler_zero_step_size()
```mojo
"""Test StepLR with step_size=0 raises error."""

# Should raise error
try:
    var scheduler = StepLR(base_lr=1.0, step_size=0, gamma=0.1)
    assert_true(False, "Expected error for step_size=0")
except Error:
    pass  # Expected
```

**Expected**: step_size=0 raises error (division by zero)

#### test_step_scheduler_negative_gamma()
```mojo
"""Test StepLR with negative gamma raises error."""

# Should raise error
try:
    var scheduler = StepLR(base_lr=1.0, step_size=5, gamma=-0.1)
    assert_true(False, "Expected error for negative gamma")
except Error:
    pass  # Expected
```

**Expected**: Negative gamma raises error (invalid parameter)

#### test_step_scheduler_very_small_lr()
```mojo
"""Test StepLR with very small LR values."""

# Setup
var scheduler = StepLR(base_lr=1.0, step_size=1, gamma=0.1)

# Decay many times
for epoch in range(10):
    var lr = scheduler.get_lr(epoch)

# After 10 steps: LR = 1.0 × 0.1^10 = 1e-10
var final_lr = scheduler.get_lr(10)
assert_almost_equal(final_lr, 1e-10, tolerance=1e-15)
```

**Expected**: LR can become very small without numerical issues

### 4. Integration Tests

#### test_step_scheduler_updates_optimizer_lr()
```mojo
"""Test StepLR integrates with optimizer."""

# Setup
var model = create_simple_model()
var optimizer = SGD(learning_rate=1.0)
var scheduler = StepLR(base_lr=1.0, step_size=5, gamma=0.1)

# Training loop
for epoch in range(10):
    # Update LR
    var new_lr = scheduler.get_lr(epoch)
    optimizer.set_lr(new_lr)

    # Verify LR
    var expected_lr = 1.0 if epoch < 5 else 0.1
    assert_almost_equal(optimizer.get_lr(), expected_lr)
```

**Expected**: Scheduler correctly updates optimizer LR

### 5. Property-Based Tests

#### test_step_scheduler_property_monotonic_decrease()
```mojo
"""Property: LR should never increase."""

# Setup
var scheduler = StepLR(base_lr=1.0, step_size=5, gamma=0.5)

# Test monotonic decrease
var prev_lr = scheduler.get_lr(0)
for epoch in range(1, 51):
    var curr_lr = scheduler.get_lr(epoch)
    assert_less_or_equal(curr_lr, prev_lr)
    prev_lr = curr_lr
```

**Expected**: LR monotonically decreases (or stays constant)

## Test Coverage Goals

- **Line coverage**: 100% of StepLR implementation
- **Branch coverage**: All conditional paths tested
- **Edge cases**: All boundary conditions covered
- **Integration**: Works with optimizer and training loop

## Test File Structure

```
tests/shared/training/test_step_scheduler.mojo
│
├── Core Functionality Tests (3 tests)
│   ├── test_step_scheduler_initialization
│   ├── test_step_scheduler_reduces_lr_at_step
│   └── test_step_scheduler_multiple_steps
│
├── Parameter Variation Tests (3 tests)
│   ├── test_step_scheduler_different_gamma_values
│   ├── test_step_scheduler_gamma_one
│   └── test_step_scheduler_different_step_sizes
│
├── Edge Case Tests (3 tests)
│   ├── test_step_scheduler_zero_step_size
│   ├── test_step_scheduler_negative_gamma
│   └── test_step_scheduler_very_small_lr
│
├── Integration Tests (2 tests)
│   ├── test_step_scheduler_updates_optimizer_lr
│   └── test_step_scheduler_works_with_multiple_param_groups
│
└── Property-Based Tests (1 test)
    └── test_step_scheduler_property_monotonic_decrease
```

## Test Execution

```bash
# Run all StepLR tests
mojo test tests/shared/training/test_step_scheduler.mojo

# Expected output
Running StepLR core tests... ✓
Running gamma factor tests... ✓
Running step size tests... ✓
Running optimizer integration tests... ✓
Running edge cases and error handling... ✓
Running property-based tests... ✓

All StepLR scheduler tests passed! ✓
```

## Success Criteria

- [ ] All test functions implemented
- [ ] 100% line coverage of StepLR code
- [ ] All tests pass consistently
- [ ] Edge cases properly handled
- [ ] Integration tests verify real-world usage
- [ ] Test code is readable and well-documented

## Files

**Test Implementation**:
- `tests/shared/training/test_step_scheduler.mojo` - 354 lines of comprehensive tests

**Test Utilities**:
- `tests/shared/conftest.mojo` - Shared test fixtures and assertions

## Implementation Status

✅ **COMPLETED** - Test suite exists in `tests/shared/training/test_step_scheduler.mojo`

This test specification was created retrospectively to document the existing test suite.

## Notes

- Tests follow TDD principle: define behavior before implementation
- Use mock objects where optimizer integration is tested
- Property-based tests verify mathematical properties
- Tests serve as executable documentation
