# Issue #447: [Cleanup] Test Fixtures

## Objective

Refactor and finalize test fixtures based on learnings from Test, Implementation, and Packaging phases, ensuring production-ready fixture code with no technical debt.

## Deliverables

- Cleaned up fixture implementations
- Resolved or removed placeholder TODOs
- Optimized fixture performance
- Updated and accurate documentation
- No redundant or dead code

## Success Criteria

- [ ] All fixture code meets quality standards
- [ ] No unused or placeholder fixtures remain (unless properly tracked)
- [ ] Documentation matches implementations
- [ ] All TODOs are resolved or tracked in issues
- [ ] Fixtures are maintainable and extensible
- [ ] No performance issues in fixture generation

## Current State Analysis

### Fixture Code Quality

**conftest.mojo TestFixtures** (lines 139-181):

### Implemented

```mojo
struct TestFixtures:
    @staticmethod
    fn deterministic_seed() -> Int:
        """Get deterministic random seed for reproducible tests."""
        return 42

    @staticmethod
    fn set_seed():
        """Set random seed for deterministic test execution."""
        seed(Self.deterministic_seed())
```text

### Strengths

- Simple, focused implementation
- Well-documented
- Follows Mojo patterns

**TODO/Placeholder** (lines 160-181):

```mojo
# TODO(#1538): Add tensor fixture methods when Tensor type is implemented
# @staticmethod
# fn small_tensor() -> Tensor
#     """Create small 3x3 tensor for unit tests."""
#     pass
```text

**Decision Needed**: Keep, implement, or remove commented code

**Data Generators** (lines 288-337):

### Implemented

- `create_test_vector()`
- `create_test_matrix()`
- `create_sequential_vector()`

### Strengths

- Simple implementations
- Good docstrings
- Efficient (pre-allocated capacity)

**Potential Optimizations**: Check if any can be improved

**helpers/fixtures.mojo** (17 lines):

**Status**: All placeholders

```mojo
# TODO: Implement fixture functions once ExTensor is fully functional
# These will include
# - random_tensor(shape, dtype) -> ExTensor
# - sequential_tensor(shape, dtype) -> ExTensor
# - nan_tensor(shape) -> ExTensor
# - inf_tensor(shape) -> ExTensor
# - ones_like(tensor) -> ExTensor
# - zeros_like(tensor) -> ExTensor
```text

**Decision Needed**: Implement, remove file, or keep as tracked TODO

## Cleanup Strategy

### 1. TODO Resolution

**Commented Tensor Fixtures** (conftest.mojo, lines 160-181):

### Options

1. **Implement** if Tensor is now available and fixtures are needed
1. **Remove** commented code (confusing to have commented code)
1. **Convert to docstring TODO** with issue tracking

### Recommended Action

```mojo
// Instead of commented code:
# TODO(#1538): Add tensor fixture methods when Tensor type is implemented
# @staticmethod
# fn small_tensor() -> Tensor
#     pass

// Do this:
struct TestFixtures:
    # ... existing methods ...

    # Future fixtures (pending Tensor implementation):
    # - small_tensor() -> Tensor  (Issue #1538)
    # - random_tensor(rows, cols) -> Tensor  (Issue #1538)
    # - simple_linear_model() -> Linear  (Issue #1538)
    # - synthetic_dataset(n_samples) -> TensorDataset  (Issue #1538)
```text

**Rationale**: Cleaner than commented code, still documents intentions

**helpers/fixtures.mojo Placeholders**:

### Assessment

```bash
# Check if ExTensor is available
ls shared/core/tensor*.mojo
ls extensor/*.mojo

# Check if fixtures are used
grep -r "from tests.helpers.fixtures import" tests/
```text

### Decision Matrix

| ExTensor Available? | Fixtures Used? | Action |
|-------------------|----------------|--------|
| Yes | Yes | Implement now |
| Yes | No | Implement if beneficial |
| No | - | Remove file or keep tracked TODO |

### Recommended Actions

- **If ExTensor ready AND fixtures needed**: Implement (see Issue #445)
- **If ExTensor ready BUT fixtures not needed**: Remove placeholders
- **If ExTensor not ready**: Remove file, add TODO to conftest.mojo docstring

### 2. Code Quality Improvements

### Data Generator Optimization

**Current** (create_test_vector):

```mojo
fn create_test_vector(size: Int, value: Float32 = 1.0) -> List[Float32]:
    var vec = List[Float32](capacity=size)  // Good: pre-allocate
    for _ in range(size):
        vec.append(value)
    return vec
```text

**Assessment**: Already efficient (pre-allocated)

**Potential Enhancement** (if needed):

```mojo
fn create_test_vector(size: Int, value: Float32 = 1.0) -> List[Float32]:
    """Create test vector filled with specific value.

    Args:
        size: Vector size.
        value: Fill value (default 1.0).

    Returns:
        List of Float32 values.

    Performance: O(n) time, O(n) space.
    """
    var vec = List[Float32](capacity=size)
    for _ in range(size):
        vec.append(value)
    return vec
```text

### Add performance notes to docstrings if helpful

### Error Handling

### Consider edge cases

```mojo
fn create_test_vector(size: Int, value: Float32 = 1.0) -> List[Float32]:
    """Create test vector filled with specific value.

    Args:
        size: Vector size (must be >= 0).
        value: Fill value.

    Returns:
        List of Float32 values.

    Raises:
        Error if size < 0.
    """
    if size < 0:
        raise Error("Vector size must be non-negative, got " + str(size))

    var vec = List[Float32](capacity=size)
    for _ in range(size):
        vec.append(value)
    return vec
```text

**Decision**: Only add if tests actually pass invalid sizes

### 3. Documentation Accuracy

### Review Checklist

- [ ] All docstrings match current implementations
- [ ] Parameter descriptions are accurate
- [ ] Return value descriptions are accurate
- [ ] Usage examples work correctly
- [ ] Performance characteristics documented (if relevant)

### Actions

- Verify all docstrings
- Update any outdated information
- Add missing details
- Remove incorrect statements

### 4. Naming Consistency

### Review Current Names

- `TestFixtures` - Good (struct for fixtures)
- `deterministic_seed()` - Good (clear purpose)
- `set_seed()` - Good (action verb)
- `create_test_vector()` - Good (verb + noun pattern)
- `create_test_matrix()` - Good (consistent with vector)
- `create_sequential_vector()` - Good (descriptive adjective)

### Ensure Future Fixtures Follow Patterns

- Generators: `create_*` or `*_fixture`
- Models: `simple_*_model` or `test_*_model`
- Datasets: `toy_*_dataset` or `test_*_dataset`

### 5. Redundancy Check

### Questions

- Are any fixtures duplicated?
- Are any fixtures unused?
- Can any fixtures be consolidated?

### Analysis

```bash
# Check which fixtures are actually used
grep -r "create_test_vector" tests/
grep -r "create_test_matrix" tests/
grep -r "create_sequential_vector" tests/
grep -r "TestFixtures.set_seed" tests/
grep -r "TestFixtures.deterministic_seed" tests/
```text

### Actions

- Keep used fixtures
- Consider removing unused fixtures (or document why they're kept)
- Consolidate similar fixtures if beneficial

### 6. Performance Validation

### Questions

- Are fixtures fast enough?
- Do fixtures allocate unnecessarily?
- Are there performance bottlenecks?

**Benchmarking** (if needed):

```mojo
fn benchmark_vector_creation() raises:
    """Benchmark vector fixture creation."""
    var iterations = 1000

    var start = time.now()  // If available
    for _ in range(iterations):
        var vec = create_test_vector(1000, 1.0)
    var end = time.now()

    var duration = (end - start).milliseconds()
    print("Created", iterations, "vectors in", duration, "ms")
    print("Average:", duration / iterations, "ms per vector")
```text

**Optimization** (if needed):

- Profile slow fixtures
- Optimize hot paths
- Consider caching if appropriate

## Cleanup Tasks

### Phase 1: Assessment

1. **Identify TODOs**:
   - List all TODOs in fixture code
   - Categorize: implement now, track, or remove

1. **Check Usage**:
   - Find which fixtures are actually used
   - Identify unused fixtures

1. **Performance Check**:
   - Benchmark fixture creation (if concerns exist)
   - Identify any slow fixtures

1. **Documentation Review**:
   - Check all docstrings for accuracy
   - Verify examples work

### Phase 2: Resolution

1. **Resolve TODOs**:
   - Implement if dependencies ready and needed
   - Remove if not needed
   - Track in issues if deferred

1. **Remove Unused Code**:
   - Delete unused fixtures (with justification)
   - Clean up commented code

1. **Fix Issues**:
   - Correct inaccurate documentation
   - Fix any performance issues
   - Resolve any code quality issues

### Phase 3: Verification

1. **Test Suite**:
   - Run all tests to ensure no regressions
   - Verify fixtures still work correctly

1. **Documentation**:
   - Verify all docstrings are accurate
   - Check examples work

1. **CI**:
   - Ensure CI passes
   - No new warnings or errors

### Phase 4: Final Documentation

1. **Update Documentation**:
   - Update fixture catalog
   - Update usage guide
   - Document cleanup decisions

1. **Track Deferred Work**:
   - Create issues for deferred TODOs
   - Document dependencies for future fixtures

## References

- **Source Plan**: [notes/plan/02-shared-library/04-testing/01-test-framework/03-test-fixtures/plan.md](../../../plan/02-shared-library/04-testing/01-test-framework/03-test-fixtures/plan.md)
- **Related Issues**:
  - Issue #443: [Plan] Test Fixtures
  - Issue #444: [Test] Test Fixtures
  - Issue #445: [Impl] Test Fixtures
  - Issue #446: [Package] Test Fixtures
  - Issue #1538: Tensor implementation (dependency for tensor fixtures)
- **Existing Code**:
  - `/tests/shared/conftest.mojo` (TestFixtures struct, lines 139-181)
  - `/tests/shared/conftest.mojo` (Data generators, lines 288-337)
  - `/tests/helpers/fixtures.mojo` (17 lines, placeholders)

## Implementation Notes

### Cleanup Findings

(To be filled during cleanup phase)

### TODOs Resolved

- TBD

**TODOs Deferred** (with tracking):

- TBD

### Code Removed

- TBD

### Code Optimized

- TBD

### Documentation Updated

- TBD

### Lessons Learned

### What Worked Well

- TBD

### What Could Be Improved

- TBD

### Recommendations for Future Fixtures

- TBD

### Final State

After cleanup:

- All fixtures are production-ready
- No placeholder code without tracking
- Documentation is accurate and complete
- Code quality meets project standards
- Fixtures are maintainable and extensible
- All tests pass without regressions
