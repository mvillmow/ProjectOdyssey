# Issue #440: [Impl] Test Utilities

## Objective

Implement missing test utilities and enhance existing ones based on findings from the Test phase (#439), focusing on utilities that are actually needed and will reduce test code duplication.

## Deliverables

- Implemented or enhanced assertion helpers
- Completed tensor comparison utilities
- Implemented test data generators
- Mock objects (if needed)
- Updated timing/profiling utilities

## Success Criteria

- [ ] All gaps identified in Test phase are addressed
- [ ] Utilities reduce test code duplication measurably
- [ ] Floating-point comparisons handle edge cases correctly
- [ ] Data generators are deterministic and reliable
- [ ] All utilities are well-documented
- [ ] Error messages are clear and actionable

## Current State Analysis

### Existing Implementations

**Fully Implemented** (conftest.mojo):
- 7 assertion functions (assert_true, assert_false, assert_equal, etc.)
- TestFixtures with deterministic seeding
- BenchmarkResult struct for performance tracking
- 3 data generators (create_test_vector, create_test_matrix, create_sequential_vector)

**Placeholder/TODO** (conftest.mojo):
- `measure_time[func]()` - Returns 0.0, needs implementation
- `measure_throughput[func](n_iterations)` - Uses measure_time, needs implementation
- Tensor fixture methods (commented out, lines 160-181)

**Fully Implemented** (helpers/assertions.mojo):
- 4 basic assertions (assert_true, assert_equal_int, assert_equal_float, assert_close_float)
- 8 ExTensor-specific assertions (assert_shape, assert_dtype, assert_all_close, etc.)

**Placeholder** (helpers/):
- `fixtures.mojo` - All TODOs, no implementation
- `utils.mojo` - All TODOs, no implementation

### Implementation Priorities

Based on **YAGNI** and **Minimal Changes** principles, prioritize:

1. **High Priority**: Fill gaps in actively used utilities
2. **Medium Priority**: Implement TODO items that tests actually need
3. **Low Priority**: Remove unused placeholders

## Implementation Strategy

### 1. Timing Utilities

**Current**: Placeholders returning 0.0

**Decision Point**: Are these actually used in tests?

**If Used** - Implement:
```mojo
fn measure_time[func: fn () raises -> None]() raises -> Float64:
    """Measure execution time of a function.

    Parameters:
        func: Function to measure (must be fn, not def).

    Returns:
        Execution time in milliseconds.
    """
    # Check if Mojo's time module is available
    from time import now  # If available in stdlib

    var start = now()
    func()
    var end = now()

    return Float64((end - start).total_seconds() * 1000.0)
```

**If Not Used** - Remove placeholders to reduce clutter

### 2. Tensor Fixture Methods

**Current**: Commented out (lines 160-181 in conftest.mojo)

**Decision Point**: Are these needed now, or wait for Tensor implementation?

**Options**:
1. **Implement now** if Tensor is available:
   ```mojo
   @staticmethod
   fn small_tensor() -> Tensor:
       """Create small 3x3 tensor for unit tests."""
       return Tensor.ones((3, 3))

   @staticmethod
   fn random_tensor(rows: Int, cols: Int) -> Tensor:
       """Create random tensor with deterministic seed."""
       Self.set_seed()
       return Tensor.randn((rows, cols))
   ```

2. **Keep commented** if Tensor not ready (current approach is fine)

3. **Remove entirely** if not needed soon

### 3. Enhanced Data Generators

**Current**: Basic generators exist (uniform, sequential)

**Potential Additions** (if tests need them):
```mojo
fn create_random_vector(size: Int, seed: Int = 42) -> List[Float32]:
    """Create random vector with deterministic seed.

    Args:
        size: Vector size.
        seed: Random seed for reproducibility.

    Returns:
        List of random Float32 values.
    """
    random.seed(seed)
    var vec = List[Float32](capacity=size)
    for _ in range(size):
        vec.append(randn())
    return vec

fn create_ones_vector(size: Int) -> List[Float32]:
    """Create vector filled with ones.

    Args:
        size: Vector size.

    Returns:
        List of 1.0 values.
    """
    return create_test_vector(size, 1.0)

fn create_zeros_vector(size: Int) -> List[Float32]:
    """Create vector filled with zeros.

    Args:
        size: Vector size.

    Returns:
        List of 0.0 values.
    """
    return create_test_vector(size, 0.0)
```

**Decision**: Only add if tests actually need them (check test suite)

### 4. Mock Objects

**Current**: No mock implementations

**Potential Needs**:
- Mock file system for I/O tests
- Mock data loaders for training tests
- Mock timers for performance tests

**Decision**: Implement ONLY if identified as needed in Test phase

**Example** (if needed):
```mojo
struct MockFileSystem:
    """Mock file system for testing file operations."""
    var files: Dict[String, String]  # path -> content

    fn __init__(inout self):
        self.files = Dict[String, String]()

    fn write(inout self, path: String, content: String):
        """Mock file write."""
        self.files[path] = content

    fn read(self, path: String) -> String:
        """Mock file read."""
        return self.files.get(path, "")

    fn exists(self, path: String) -> Bool:
        """Mock file existence check."""
        return path in self.files
```

### 5. Enhanced Assertion Messages

**Current**: Basic error messages

**Enhancement** (if needed):
```mojo
fn assert_equal[T: Comparable](a: T, b: T, message: String = "") raises:
    """Assert exact equality of two values.

    Args:
        a: First value.
        b: Second value.
        message: Optional error message.

    Raises:
        Error if a != b.
    """
    if a != b:
        # Enhanced error message with values
        var error_msg = "Assertion failed: " + str(a) + " != " + str(b)
        if message:
            error_msg = message + " | " + error_msg
        raise Error(error_msg)
```

### 6. Placeholder Resolution (helpers/)

**fixtures.mojo** (17 lines, all TODOs):
- **Option 1**: Implement if ExTensor fixtures are needed now
- **Option 2**: Remove file if not needed yet
- **Option 3**: Keep placeholders if implementation is near-term

**utils.mojo** (14 lines, all TODOs):
- **Option 1**: Implement if utilities are needed now
- **Option 2**: Remove file if not needed yet
- **Option 3**: Keep placeholders if implementation is near-term

**Decision Matrix**:
| File | Used in Tests? | ExTensor Ready? | Action |
|------|---------------|-----------------|--------|
| fixtures.mojo | No | No | Remove or keep TODO |
| fixtures.mojo | Yes | No | Keep TODO, track in issue |
| fixtures.mojo | Yes | Yes | Implement now |
| utils.mojo | No | - | Remove or keep TODO |
| utils.mojo | Yes | - | Implement now |

## Implementation Plan

### Phase 1: Assess Actual Needs (from Test Phase #439)

1. Review Test phase findings
2. Identify which utilities are actually needed
3. Determine which TODOs to implement vs remove
4. Prioritize based on impact and effort

### Phase 2: Implement High-Priority Items

1. **Timing utilities** (if needed and Mojo time module available)
2. **Enhanced error messages** (if identified as pain point)
3. **Additional generators** (if tests need them)
4. **Mock objects** (if identified as necessary)

### Phase 3: Resolve Placeholders

1. Implement needed placeholders
2. Remove unnecessary placeholders
3. Document TODOs that depend on external factors (Tensor, time module)

### Phase 4: Documentation

1. Update all docstrings
2. Add usage examples
3. Document any new utilities
4. Update README if needed

## References

- **Source Plan**: [notes/plan/02-shared-library/04-testing/01-test-framework/02-test-utilities/plan.md](../../../plan/02-shared-library/04-testing/01-test-framework/02-test-utilities/plan.md)
- **Related Issues**:
  - Issue #438: [Plan] Test Utilities
  - Issue #439: [Test] Test Utilities (source of requirements)
  - Issue #441: [Package] Test Utilities
  - Issue #442: [Cleanup] Test Utilities
- **Existing Code**:
  - `/tests/shared/conftest.mojo` (337 lines)
  - `/tests/helpers/assertions.mojo` (365 lines)
  - `/tests/helpers/fixtures.mojo` (17 lines, mostly TODO)
  - `/tests/helpers/utils.mojo` (14 lines, mostly TODO)

## Implementation Notes

### Findings from Test Phase

(To be filled based on Issue #439 results)

**Utilities Actually Needed**:
- TBD

**TODOs to Implement**:
- TBD

**Placeholders to Remove**:
- TBD

### Implementation Decisions

**Decision Log**:

| Date | Decision | Rationale |
|------|----------|-----------|
| TBD | Implement/Skip timing utilities | TBD |
| TBD | Implement/Skip mock objects | TBD |
| TBD | Remove/Keep fixtures.mojo placeholders | TBD |
| TBD | Remove/Keep utils.mojo placeholders | TBD |

### Code Changes

**Files Modified**:
- TBD based on actual needs

**Files Created**:
- TBD based on actual needs

**Files Removed**:
- TBD based on actual needs

### Testing Validation

After implementation:
1. Run meta-tests from Issue #439
2. Verify all new utilities pass their tests
3. Run full test suite to ensure no regressions
4. Update documentation to reflect changes
