# Issue #458: [Plan] Test Training - Design and Documentation

## Objective

Design comprehensive unit tests for training utilities including base trainer, learning rate schedulers, and callbacks.
These tests verify training workflow correctness, scheduler behavior, and callback integration without requiring full
training runs.

## Deliverables

- Tests for trainer interface and loops
- Tests for all LR schedulers
- Tests for all callbacks
- Mock-based integration tests
- Training workflow verification

## Success Criteria

- [ ] Trainer tests verify training and validation
- [ ] Scheduler tests verify rate adjustments
- [ ] Callback tests verify hook invocations
- [ ] Integration tests verify component interaction

## Design Decisions

### Test Architecture

#### 1. Test Organization

- **Trainer Tests**: Separate test files for base trainer interface and training/validation loops
- **Scheduler Tests**: Individual test files for each scheduler type (StepLR, ExponentialLR, CosineAnnealing, etc.)
- **Callback Tests**: Tests for each callback type and hook invocation mechanism
- **Integration Tests**: Mock-based tests verifying component interaction without full training runs

#### 2. Testing Strategy

- **Use Toy Models**: Simple models (e.g., single-layer networks) for fast execution
- **Mock Expensive Operations**: Mock actual training computations to focus on workflow correctness
- **Mathematical Verification**: Verify scheduler outputs against mathematical formulas
- **State Tracking**: Use counters and state tracking to verify callback invocations

#### 3. Test Fixtures

- **Simple Datasets**: Small synthetic datasets (e.g., 100 samples) for speed
- **Known Outputs**: Pre-computed expected values for validation
- **Configurable Parameters**: Parameterized tests for different hyperparameter combinations

### Component Test Specifications

#### Trainer Tests

### Base Trainer Interface

- Test initialization with valid/invalid configurations
- Test training loop execution (epoch iteration, batch processing)
- Test validation loop execution
- Test state management (epoch counter, loss history)
- Test error handling for invalid inputs

### Training/Validation Workflows

- Test epoch loop completes specified number of iterations
- Test batch loop processes all batches correctly
- Test validation runs at correct intervals
- Test loss computation and accumulation
- Test metric tracking and reporting

#### Learning Rate Scheduler Tests

### Common Tests (All Schedulers)

- Test initialization with valid/invalid parameters
- Test learning rate update at each step
- Test scheduler state persistence and restoration

### Scheduler-Specific Tests

- **StepLR**: Verify rate drops by gamma every step_size epochs
- **ExponentialLR**: Verify exponential decay with formula: lr = lr0 * gamma^epoch
- **CosineAnnealing**: Verify cosine schedule with formula:
  `lr = lr_min + 0.5 * (lr_max - lr_min) * (1 + cos(epoch/T_max * π))`
- **ReduceLROnPlateau**: Verify rate reduction when metric plateaus (requires metric tracking mock)

### Mathematical Verification

- Pre-compute expected learning rates for first 10-20 steps
- Assert actual rates match expected values within tolerance (e.g., 1e-6)
- Test edge cases (epoch 0, final epoch, boundary conditions)

#### Callback Tests

### Callback Interface

- Test callback registration with trainer
- Test hook invocation order (on_train_begin, on_epoch_start, on_batch_end, etc.)
- Test callback state sharing between hooks
- Test callback removal and modification

### Hook Invocation Tests

- **on_train_begin**: Called once before training starts
- **on_train_end**: Called once after training completes
- **on_epoch_start**: Called at start of each epoch
- **on_epoch_end**: Called at end of each epoch
- **on_batch_start**: Called before each batch
- **on_batch_end**: Called after each batch

### State Tracking

- Use counters to verify each hook is called correct number of times
- Track parameters passed to hooks (epoch number, batch number, loss values)
- Verify callbacks can modify trainer state (e.g., early stopping)

#### Integration Tests

### Component Interaction

- Test trainer + scheduler: Verify learning rate updates during training
- Test trainer + callbacks: Verify callback hooks are called at correct points
- Test trainer + scheduler + callbacks: Verify all components work together
- Test early stopping workflow: Callback stops training when condition met
- Test checkpointing workflow: Callback saves model at intervals

### Mock Strategy

- Mock forward pass (return fixed loss value)
- Mock backward pass (skip gradient computation)
- Mock data loading (return synthetic batches)
- Focus on workflow correctness, not numerical accuracy

### Test Performance Requirements

- **Total Test Time**: < 30 seconds for all training tests
- **Individual Test Time**: < 1 second per test
- **Dataset Size**: ≤ 100 samples for unit tests
- **Model Complexity**: Single-layer or two-layer networks maximum
- **Epoch Count**: ≤ 10 epochs for workflow tests

### Edge Cases and Error Conditions

### Trainer Edge Cases

- Empty dataset
- Single batch dataset
- Validation without training data
- Resume from checkpoint
- Training with zero epochs

### Scheduler Edge Cases

- Learning rate at boundaries (min/max)
- Zero learning rate
- Invalid gamma values (negative, > 1 for exponential)
- Plateau with constant metrics

### Callback Edge Cases

- No callbacks registered
- Multiple callbacks of same type
- Callback raises exception
- Callback modifies training state

### Testing Tools and Frameworks

**Test Framework**: Mojo's built-in testing framework (when available) or pytest for Python-based tests

### Assertion Libraries

- Numerical assertions with tolerance (for floating-point comparisons)
- State assertions (verify expected state transitions)
- Mock verification (verify mock calls and arguments)

### Test Utilities

- Fixture generators for toy models
- Fixture generators for synthetic datasets
- Mock builders for training components
- Assertion helpers for scheduler formulas

## References

- **Source Plan**: [notes/plan/04-first-paper/05-testing/01-unit-tests/02-test-training/plan.md](../../../plan/04-first-paper/05-testing/01-unit-tests/02-test-training/plan.md)
- **Parent Plan**: [notes/plan/04-first-paper/05-testing/01-unit-tests/plan.md](../../../plan/04-first-paper/05-testing/01-unit-tests/plan.md)
- **Training Pipeline Plan**: [notes/plan/04-first-paper/03-training-pipeline/plan.md](../../../plan/04-first-paper/03-training-pipeline/plan.md)
- **Related Issues**:
  - Issue #459: [Test] Test Training - Write Tests
  - Issue #460: [Impl] Test Training - Implementation
  - Issue #461: [Package] Test Training - Integration and Packaging
  - Issue #462: [Cleanup] Test Training - Refactor and Finalize

## Implementation Notes

### Current Test Coverage

**Existing Test Files** (in `/home/user/ml-odyssey/tests/shared/training/`):

- `test_optimizers.mojo` - TDD stubs with 16 TODOs, well-defined API contracts
- `test_schedulers.mojo` - Test file exists (needs verification)
- `test_warmup_scheduler.mojo` - 14 test functions defined
- `test_cosine_scheduler.mojo` - Test file exists
- `test_step_scheduler.mojo` - Test file exists
- `test_training_loop.mojo` - 14 TODOs, workflow tests defined
- `test_validation_loop.mojo` - 15 TODOs, validation tests defined
- `test_trainer_interface.mojo` - 13+ test functions, 9 TODOs
- `test_callbacks.mojo` - Empty (1 line only)
- `test_early_stopping.mojo` - 12 test functions defined
- `test_checkpointing.mojo` - Test file exists
- `test_logging_callback.mojo` - Test file exists
- `test_metrics.mojo` - Test file exists
- `test_numerical_safety.mojo` - 11 TODOs for safety checks
- `test_loops.mojo` - Test file exists

### Test Status Analysis

- **Total test functions**: 100+ defined across training test files
- **Implementation status**: Mix of TDD stubs and implemented tests
- **TODOs count**: ~95 TODO markers across 15 files
- **Test runner**: Not yet created for training tests

### Comparison with Data Tests

- Data tests have comprehensive test runner (`run_all_tests.mojo`)
- Training tests have more organizational structure
- Both follow TDD principles with clear API contracts

### Gap Analysis

### What Exists

1. **Well-organized test structure** - Separate files for each component
1. **Clear API contracts** - Test stubs document expected interfaces
1. **Comprehensive coverage** - Tests for all training components
1. **Numerical safety tests** - Dedicated file for stability checks

### What's Missing

1. **Test implementations** - Most tests are stubs waiting for implementation
1. **Callback tests** - test_callbacks.mojo is empty
1. **Mock frameworks** - Need mocking strategy for training workflows
1. **Mathematical verification** - Scheduler formulas need reference implementations
1. **Integration tests** - Full training workflow verification
1. **Test runner** - Unified runner for all training tests

### Recommendations

1. **Prioritize Schedulers First**:
   - StepLR (simplest - rate drops at intervals)
   - CosineAnnealing (mathematical formula verification)
   - WarmupScheduler (combines multiple schedulers)

1. **Mock Strategy for Workflows**:
   - Mock forward/backward passes
   - Mock optimizer steps
   - Focus on workflow correctness, not numerical accuracy

1. **Statistical Verification**:
   - Pre-compute expected LR values for 10-20 steps
   - Compare actual vs expected within tolerance (1e-6)
   - Test boundary conditions (epoch 0, final epoch)

1. **Create Test Runner**:
   - Similar to `tests/shared/data/run_all_tests.mojo`
   - Group tests by component (schedulers, callbacks, loops)
   - Provide summary statistics
