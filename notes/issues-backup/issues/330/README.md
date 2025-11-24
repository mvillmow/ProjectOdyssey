# Issue #330: [Test] Cosine Scheduler

## Phase

Testing (TDD - Test-Driven Development)

## Component

CosineAnnealingLR - Cosine Annealing Learning Rate Scheduler

## Objective

Implement comprehensive test suite for CosineAnnealingLR scheduler following TDD principles. Tests define the expected behavior and API before/during implementation, with focus on validating the smooth cosine decay curve.

## Test Strategy

### Test Categories

1. **Core Functionality Tests**
   - Initialization with valid parameters
   - Cosine curve correctness
   - Smooth continuous decay
   - Boundary values (epoch=0, T_max)

1. **Parameter Variation Tests**
   - Different T_max values
   - Different eta_min values
   - Edge case eta_min = base_lr

1. **Mathematical Property Tests**
   - Cosine formula accuracy
   - Monotonic decrease property
   - Symmetry around midpoint

1. **Edge Case Tests**
   - Zero/negative T_max
   - eta_min > base_lr (invalid)
   - epoch > T_max (beyond period)
   - Very small/large T_max

1. **Integration Tests**
   - Optimizer LR updates
   - Training loop integration
   - Warmup combination

1. **Numerical Stability Tests**
   - Very small eta_min (near 0)
   - Very large T_max
   - Precision at boundaries

## Test Specifications

### 1. Core Functionality Tests

#### test_cosine_scheduler_initialization()

```mojo
"""Test CosineAnnealingLR initializes with correct parameters."""

# Setup
var scheduler = CosineAnnealingLR(base_lr=0.1, T_max=100, eta_min=0.0)

# Verify
assert_almost_equal(scheduler.base_lr, 0.1)
assert_equal(scheduler.T_max, 100)
assert_almost_equal(scheduler.eta_min, 0.0)
```text

**Expected**: Scheduler stores parameters correctly

#### test_cosine_scheduler_follows_cosine_curve()

```mojo
"""Test CosineAnnealingLR follows cosine annealing formula.

This is the CRITICAL test for cosine scheduler correctness.
"""

# Setup
var scheduler = CosineAnnealingLR(base_lr=1.0, T_max=100, eta_min=0.0)

# Test key points on cosine curve
# Epoch 0: cos(0) = 1, lr = 0 + (1-0)*(1+1)/2 = 1.0
var lr0 = scheduler.get_lr(epoch=0)
assert_almost_equal(lr0, 1.0, tolerance=1e-6)

# Epoch 50: cos(π) = -1, lr = 0 + (1-0)*(1-1)/2 = 0.0
var lr50 = scheduler.get_lr(epoch=50)
assert_almost_equal(lr50, 0.0, tolerance=1e-6)

# Epoch 100: cos(2π) = 1, but we clamp, so lr = eta_min = 0.0
var lr100 = scheduler.get_lr(epoch=100)
assert_almost_equal(lr100, 0.0, tolerance=1e-6)

# Epoch 25: cos(π/2) = 0, lr = 0 + (1-0)*(1+0)/2 = 0.5
var lr25 = scheduler.get_lr(epoch=25)
assert_almost_equal(lr25, 0.5, tolerance=1e-6)
```text

**Expected**: LR follows exact cosine formula at all test points

#### test_cosine_scheduler_smooth_decay()

```mojo
"""Test CosineAnnealingLR provides smooth continuous decay."""

# Setup
var scheduler = CosineAnnealingLR(base_lr=1.0, T_max=100, eta_min=0.0)

# Verify smooth decay in first half
var lr0 = scheduler.get_lr(epoch=0)   # Maximum
var lr25 = scheduler.get_lr(epoch=25)  # Partway
var lr50 = scheduler.get_lr(epoch=50)  # Midpoint

# Should decrease smoothly
assert_greater(lr0, lr25)  # 1.0 > ~0.85
assert_greater(lr25, lr50)  # ~0.85 > 0.5

# Verify adjacent epochs change smoothly (no jumps)
for epoch in range(100):
    var lr_curr = scheduler.get_lr(epoch)
    var lr_next = scheduler.get_lr(epoch + 1)
    var change = abs(lr_curr - lr_next)

    # Change should be small and smooth
    assert_less(change, 0.1)  # No sudden jumps
```text

**Expected**: LR changes smoothly without discrete jumps

#### test_cosine_scheduler_boundary_values()

```mojo
"""Test CosineAnnealingLR at boundary epochs."""

# Setup
var scheduler = CosineAnnealingLR(base_lr=0.1, T_max=100, eta_min=0.0)

# Epoch 0: Should be base_lr
var lr_start = scheduler.get_lr(epoch=0)
assert_almost_equal(lr_start, 0.1)

# Epoch T_max: Should be eta_min
var lr_end = scheduler.get_lr(epoch=100)
assert_almost_equal(lr_end, 0.0)

# Beyond T_max: Should stay at eta_min
var lr_beyond = scheduler.get_lr(epoch=150)
assert_almost_equal(lr_beyond, 0.0)
```text

**Expected**: Correct LR at start, end, and beyond period

### 2. Parameter Variation Tests

#### test_cosine_scheduler_different_T_max()

```mojo
"""Test CosineAnnealingLR with various T_max values."""

# Test T_max = 10 (short period)
var sched1 = CosineAnnealingLR(base_lr=1.0, T_max=10, eta_min=0.0)
assert_almost_equal(sched1.get_lr(0), 1.0)
assert_almost_equal(sched1.get_lr(5), 0.0, tolerance=1e-6)  # Midpoint
assert_almost_equal(sched1.get_lr(10), 0.0)

# Test T_max = 1000 (long period)
var sched2 = CosineAnnealingLR(base_lr=1.0, T_max=1000, eta_min=0.0)
assert_almost_equal(sched2.get_lr(0), 1.0)
assert_almost_equal(sched2.get_lr(500), 0.0, tolerance=1e-6)
assert_almost_equal(sched2.get_lr(1000), 0.0)
```text

**Expected**: T_max controls period length correctly

#### test_cosine_scheduler_different_eta_min()

```mojo
"""Test CosineAnnealingLR with various eta_min values."""

# Test eta_min = 0.0 (decay to zero)
var sched1 = CosineAnnealingLR(base_lr=1.0, T_max=100, eta_min=0.0)
assert_almost_equal(sched1.get_lr(100), 0.0)

# Test eta_min = 0.01 (decay to 1%)
var sched2 = CosineAnnealingLR(base_lr=1.0, T_max=100, eta_min=0.01)
assert_almost_equal(sched2.get_lr(0), 1.0)
assert_almost_equal(sched2.get_lr(50), 0.505, tolerance=1e-3)  # Midpoint
assert_almost_equal(sched2.get_lr(100), 0.01)

# Test eta_min = 0.1 (decay to 10%)
var sched3 = CosineAnnealingLR(base_lr=1.0, T_max=100, eta_min=0.1)
assert_almost_equal(sched3.get_lr(100), 0.1)
```text

**Expected**: eta_min sets minimum LR correctly

#### test_cosine_scheduler_eta_min_equals_base_lr()

```mojo
"""Test CosineAnnealingLR with eta_min = base_lr (no annealing)."""

# Setup: No annealing (constant LR)
var scheduler = CosineAnnealingLR(base_lr=0.1, T_max=100, eta_min=0.1)

# LR should stay constant at base_lr
for epoch in range(150):
    var lr = scheduler.get_lr(epoch)
    assert_almost_equal(lr, 0.1, tolerance=1e-6)
```text

**Expected**: Constant LR when eta_min = base_lr (degenerate case)

### 3. Mathematical Property Tests

#### test_cosine_scheduler_formula_accuracy()

```mojo
"""Test CosineAnnealingLR matches exact cosine formula."""

# Setup
var scheduler = CosineAnnealingLR(base_lr=1.0, T_max=100, eta_min=0.0)

# Test formula at multiple points
import math

for epoch in range(101):
    var actual_lr = scheduler.get_lr(epoch)

    # Compute expected LR using formula
    var progress = Float64(epoch) / Float64(100)
    var cos_val = math.cos(math.pi * progress)
    var expected_lr = 0.0 + (1.0 - 0.0) * (1.0 + cos_val) / 2.0

    # Clamp at T_max
    if epoch >= 100:
        expected_lr = 0.0

    assert_almost_equal(actual_lr, expected_lr, tolerance=1e-6)
```text

**Expected**: Exact match with mathematical formula

#### test_cosine_scheduler_monotonic_decrease()

```mojo
"""Property: LR should monotonically decrease from 0 to T_max."""

# Setup
var scheduler = CosineAnnealingLR(base_lr=1.0, T_max=100, eta_min=0.0)

# Test monotonic decrease
var prev_lr = scheduler.get_lr(0)
for epoch in range(1, 101):
    var curr_lr = scheduler.get_lr(epoch)

    # LR should decrease or stay same (at T_max)
    assert_less_or_equal(curr_lr, prev_lr)
    prev_lr = curr_lr
```text

**Expected**: LR never increases from epoch 0 to T_max

#### test_cosine_scheduler_symmetry()

```mojo
"""Test cosine annealing is symmetric around midpoint."""

# Setup
var scheduler = CosineAnnealingLR(base_lr=1.0, T_max=100, eta_min=0.0)

# Test symmetry: lr(25) should equal lr(75) reflected around midpoint
var lr25 = scheduler.get_lr(25)
var lr75 = scheduler.get_lr(75)

# Both should be equidistant from midpoint lr
var lr_mid = scheduler.get_lr(50)  # 0.0
var dist25 = lr25 - lr_mid  # ~0.85
var dist75 = lr_mid - lr75  # Should also be ~0.85

assert_almost_equal(dist25, -dist75, tolerance=1e-6)
```text

**Expected**: Decay is symmetric around T_max/2

### 4. Edge Case Tests

#### test_cosine_scheduler_zero_T_max()

```mojo
"""Test CosineAnnealingLR with T_max=0 raises error."""

# Should raise error
try:
    var scheduler = CosineAnnealingLR(base_lr=1.0, T_max=0, eta_min=0.0)
    assert_true(False, "Expected error for T_max=0")
except Error:
    pass  # Expected
```text

**Expected**: T_max=0 raises error (undefined period)

#### test_cosine_scheduler_negative_T_max()

```mojo
"""Test CosineAnnealingLR with negative T_max raises error."""

# Should raise error
try:
    var scheduler = CosineAnnealingLR(base_lr=1.0, T_max=-10, eta_min=0.0)
    assert_true(False, "Expected error for negative T_max")
except Error:
    pass  # Expected
```text

**Expected**: Negative T_max raises error

#### test_cosine_scheduler_eta_min_greater_than_base_lr()

```mojo
"""Test CosineAnnealingLR with eta_min > base_lr raises error."""

# Should raise error
try:
    var scheduler = CosineAnnealingLR(base_lr=0.1, T_max=100, eta_min=0.5)
    assert_true(False, "Expected error for eta_min > base_lr")
except Error:
    pass  # Expected
```text

**Expected**: eta_min > base_lr raises error (invalid configuration)

#### test_cosine_scheduler_beyond_T_max()

```mojo
"""Test CosineAnnealingLR behavior beyond T_max."""

# Setup
var scheduler = CosineAnnealingLR(base_lr=1.0, T_max=100, eta_min=0.0)

# Beyond T_max, LR should stay at eta_min
for epoch in range(100, 200):
    var lr = scheduler.get_lr(epoch)
    assert_almost_equal(lr, 0.0, tolerance=1e-6)
```text

**Expected**: LR stays at eta_min for epochs > T_max

#### test_cosine_scheduler_very_large_T_max()

```mojo
"""Test CosineAnnealingLR with very large T_max."""

# Setup
var scheduler = CosineAnnealingLR(base_lr=1.0, T_max=1000000, eta_min=0.0)

# Should not overflow or have numerical issues
var lr_start = scheduler.get_lr(0)
var lr_mid = scheduler.get_lr(500000)
var lr_end = scheduler.get_lr(1000000)

assert_almost_equal(lr_start, 1.0)
assert_almost_equal(lr_mid, 0.0, tolerance=1e-6)
assert_almost_equal(lr_end, 0.0)
```text

**Expected**: No numerical issues with large T_max

### 5. Integration Tests

#### test_cosine_scheduler_updates_optimizer_lr()

```mojo
"""Test CosineAnnealingLR integrates with optimizer."""

# Setup
var optimizer = SGD(learning_rate=1.0)
var scheduler = CosineAnnealingLR(base_lr=1.0, T_max=100, eta_min=0.0)

# Training loop
for epoch in range(100):
    # Update LR
    var new_lr = scheduler.get_lr(epoch)
    optimizer.set_lr(new_lr)

    # Verify optimizer has new LR
    assert_almost_equal(optimizer.get_lr(), new_lr, tolerance=1e-6)
```text

**Expected**: Scheduler correctly updates optimizer LR

#### test_cosine_scheduler_with_warmup()

```mojo
"""Test CosineAnnealingLR combined with warmup."""

# Setup
var warmup_epochs = 10
var total_epochs = 100
var warmup = WarmupLR(base_lr=0.1, warmup_epochs=warmup_epochs)
var cosine = CosineAnnealingLR(
    base_lr=0.1,
    T_max=total_epochs - warmup_epochs,
    eta_min=0.0
)

# Verify combined behavior
for epoch in range(total_epochs):
    var lr: Float64
    if epoch < warmup_epochs:
        lr = warmup.get_lr(epoch)
        # Should increase during warmup
        if epoch > 0:
            var prev_lr = warmup.get_lr(epoch - 1)
            assert_greater_or_equal(lr, prev_lr)
    else:
        lr = cosine.get_lr(epoch - warmup_epochs)
        # Should decrease during cosine annealing
        if epoch > warmup_epochs:
            var prev_lr = cosine.get_lr(epoch - warmup_epochs - 1)
            assert_less_or_equal(lr, prev_lr)
```text

**Expected**: Warmup + cosine annealing combination works correctly

### 6. Numerical Stability Tests

#### test_cosine_scheduler_very_small_eta_min()

```mojo
"""Test CosineAnnealingLR with very small eta_min."""

# Setup
var scheduler = CosineAnnealingLR(base_lr=1.0, T_max=100, eta_min=1e-10)

# Should handle very small values without underflow
var lr_end = scheduler.get_lr(100)
assert_almost_equal(lr_end, 1e-10, tolerance=1e-15)
```text

**Expected**: No numerical underflow with very small eta_min

#### test_cosine_scheduler_precision_at_midpoint()

```mojo
"""Test CosineAnnealingLR precision at T_max/2."""

# Setup
var scheduler = CosineAnnealingLR(base_lr=1.0, T_max=100, eta_min=0.0)

# At exactly T_max/2, cos(π) = -1, so LR should be exactly eta_min
var lr_mid = scheduler.get_lr(50)
assert_almost_equal(lr_mid, 0.0, tolerance=1e-10)  # Very tight tolerance
```text

**Expected**: High precision at critical points

## Test Coverage Goals

- **Line coverage**: 100% of CosineAnnealingLR implementation
- **Branch coverage**: All conditional paths tested
- **Edge cases**: All boundary conditions covered
- **Mathematical properties**: Cosine formula verified
- **Integration**: Works with optimizer and training loop

## Test File Structure

```text
tests/shared/training/test_cosine_scheduler.mojo
│
├── Core Functionality Tests (4 tests)
│   ├── test_cosine_scheduler_initialization
│   ├── test_cosine_scheduler_follows_cosine_curve
│   ├── test_cosine_scheduler_smooth_decay
│   └── test_cosine_scheduler_boundary_values
│
├── Parameter Variation Tests (3 tests)
│   ├── test_cosine_scheduler_different_T_max
│   ├── test_cosine_scheduler_different_eta_min
│   └── test_cosine_scheduler_eta_min_equals_base_lr
│
├── Mathematical Property Tests (3 tests)
│   ├── test_cosine_scheduler_formula_accuracy
│   ├── test_cosine_scheduler_monotonic_decrease
│   └── test_cosine_scheduler_symmetry
│
├── Edge Case Tests (5 tests)
│   ├── test_cosine_scheduler_zero_T_max
│   ├── test_cosine_scheduler_negative_T_max
│   ├── test_cosine_scheduler_eta_min_greater_than_base_lr
│   ├── test_cosine_scheduler_beyond_T_max
│   └── test_cosine_scheduler_very_large_T_max
│
├── Integration Tests (2 tests)
│   ├── test_cosine_scheduler_updates_optimizer_lr
│   └── test_cosine_scheduler_with_warmup
│
└── Numerical Stability Tests (2 tests)
    ├── test_cosine_scheduler_very_small_eta_min
    └── test_cosine_scheduler_precision_at_midpoint
```text

**Total**: 19 test functions

## Test Execution

```bash
# Run all CosineAnnealingLR tests
mojo test tests/shared/training/test_cosine_scheduler.mojo

# Expected output
Running Core Functionality Tests... ✓
Running Parameter Variation Tests... ✓
Running Mathematical Property Tests... ✓
Running Edge Case Tests... ✓
Running Integration Tests... ✓
Running Numerical Stability Tests... ✓

All CosineAnnealingLR scheduler tests passed! ✓
```text

## Success Criteria

- [ ] All test functions implemented
- [ ] 100% line coverage of CosineAnnealingLR code
- [ ] All tests pass consistently
- [ ] Cosine formula verified mathematically
- [ ] Edge cases properly handled
- [ ] Integration tests verify real-world usage
- [ ] Numerical stability verified
- [ ] Test code is readable and well-documented

## Files

### Test Implementation

- `tests/shared/training/test_cosine_scheduler.mojo` - 350 lines of comprehensive tests

### Test Utilities

- `tests/shared/conftest.mojo` - Shared test fixtures and assertions

## Implementation Status

✅ **COMPLETED** - Test suite exists in `tests/shared/training/test_cosine_scheduler.mojo`

This test specification was created retrospectively to document the existing test suite.

## Notes

- Tests validate the smooth cosine decay curve (critical property)
- Mathematical property tests ensure formula correctness
- Symmetry test verifies cosine curve shape
- Integration with warmup is common pattern (tested)
- Numerical stability important for very small eta_min values
- Tests serve as executable specification of behavior
