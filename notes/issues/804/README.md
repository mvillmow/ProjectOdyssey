# Issue #804: [Plan] Test Runner - Design and Documentation

## Objective

Create a unified test runner that discovers, executes, and reports on all tests in the repository. The runner provides a single command to run all tests with clear output showing successes, failures, and performance metrics.

## Deliverables

- Comprehensive test runner architecture specification
- API contracts for test discovery, execution, and reporting interfaces
- Test isolation and parallel execution strategy
- Design documentation covering both Mojo and Python test support
- Report format specification (console, file, JSON outputs)
- CI/CD integration guidelines

## Success Criteria

- [ ] All tests are discovered automatically using standard naming conventions
- [ ] Tests run in isolation without interference between test cases
- [ ] Results are clearly reported with successes, failures, and performance metrics
- [ ] Failed tests show helpful error information including file paths and line numbers
- [ ] Parallel execution strategy defined for performance optimization
- [ ] Both Mojo and Python test support specifications completed
- [ ] Filtering options documented (by paper, tag, pattern)
- [ ] Exit codes defined for CI/CD integration
- [ ] All child plans are reviewed and validated

## Design Decisions

### Architecture

**Three-Component Pipeline**: The test runner follows a clear pipeline architecture with three distinct components:

1. **Test Discovery** - Finds all test files in repository
2. **Test Execution** - Runs tests with proper isolation
3. **Result Reporting** - Formats and displays test results

**Rationale**: Separating concerns allows each component to be developed, tested, and maintained independently. This modularity also enables future extensions (custom reporters, different execution strategies) without modifying the core pipeline.

**Alternatives Considered**:

- Monolithic runner: Rejected due to tight coupling and difficulty maintaining
- Plugin-based architecture: Deferred to future enhancement (YAGNI principle)

### Test Discovery Strategy

**Pattern-Based Discovery**: Use filesystem walking with pattern matching:

- Mojo tests: `test_*.mojo`, `*_test.mojo`
- Python tests: `test_*.py`, `*_test.py`
- Exclusions: hidden directories (`.*/`), build artifacts (`build/`, `dist/`)

**Metadata Collection**: Associate tests with papers based on directory structure:

- Path: `papers/<paper-name>/tests/test_*.mojo` → Paper: `<paper-name>`
- Path: `src/tests/test_*.mojo` → Paper: `core`

**Caching Strategy**: Cache discovery results to avoid repeated filesystem traversal:

- Cache key: Repository root + modification time
- Invalidation: When new test files added or existing tests modified
- Storage: In-memory for single runs, file-based for repeated runs

**Rationale**: Standard naming conventions align with pytest and other Python testing tools, making the system familiar to developers. Caching improves performance for large repositories with many tests.

**Alternatives Considered**:

- Explicit test registration: Rejected as too manual and error-prone
- Git-based discovery: Rejected due to complexity and git dependency

### Test Execution Strategy

**Isolation Requirements**:

- Each test runs in separate process to prevent state contamination
- Clean environment variables for each test
- Temporary directories for test artifacts
- Proper cleanup after test completion

**Parallel Execution**:

- Default: Run tests in parallel using worker pool
- Worker count: CPU count or user-specified limit
- Ordering: Deterministic within parallel batches for reproducibility
- Fallback: Sequential execution when parallelism disabled

**Timeout Handling**:

- Default timeout: 30 seconds per test
- User-configurable via command-line flag
- Timeout results in test failure with clear error message

**Error Recovery**:

- Failed tests don't stop execution (continue-on-error by default)
- Option to stop on first failure (--fail-fast flag)
- Capture and preserve all error output for debugging

**Rationale**: Process isolation prevents test interference and matches industry best practices (pytest-xdist, Go test runner). Parallel execution improves developer productivity by reducing wait times. Continue-on-error provides complete test coverage in CI/CD.

**Alternatives Considered**:

- Thread-based parallelism: Rejected due to GIL limitations in Python components
- Container-based isolation: Deferred to future enhancement (added complexity)

### Reporting Strategy

**Output Formats**:

1. **Console** (default): Human-readable with colors
   - Green checkmarks for passes
   - Red X marks for failures
   - Yellow warnings for skipped tests
   - Summary statistics at end

2. **File**: Plain text report for CI/CD logs
   - No ANSI color codes
   - Same content as console output

3. **JSON**: Machine-readable for tooling integration
   - Structured test results
   - Execution timing data
   - Error details with stack traces

**Information Priority**:

1. Summary statistics (pass/fail counts, percentages)
2. Failed test details (file, error, stack trace)
3. Performance metrics (slowest tests, total time)
4. Full test list (on verbose flag)

**Exit Codes**:

- 0: All tests passed
- 1: One or more tests failed
- 2: Test discovery failed
- 3: Configuration error

**Rationale**: Multiple output formats serve different audiences (developers, CI/CD, tooling). Prioritizing failures helps developers quickly identify and fix issues. Standard exit codes integrate cleanly with CI/CD pipelines.

**Alternatives Considered**:

- XML output (JUnit format): Deferred to future enhancement (low priority)
- HTML reports: Deferred to future enhancement (requires web server or file viewing)

### Language Selection

**Python for Test Runner Implementation**:

The test runner will be implemented in Python rather than Mojo based on the following justification:

**Technical Reasons**:

1. **Subprocess Output Capture**: Test execution requires capturing stdout/stderr from both Mojo and Python test processes. Mojo v0.25.7 lacks subprocess output capture capabilities (see ADR-001).

2. **Regex Pattern Matching**: Test discovery uses regex patterns to match test files. Mojo lacks production-ready regex support (mojo-regex is alpha stage).

3. **Process Management**: Managing parallel test execution requires robust process pool management, which Python's `multiprocessing` provides maturely.

4. **Cross-Language Test Support**: Runner must execute both Mojo and Python tests, making Python a natural choice as the orchestrator.

**Alignment with ADR-001**: This decision follows the "Python for Automation" guideline in ADR-001. Test running is automation infrastructure, not ML/AI implementation. The runner automates test discovery, execution, and reporting - classic automation tasks.

**Documentation Header** (to be added to implementation files):

```python
# Language Selection: Python
#
# Justification:
# - Subprocess output capture required for test execution (Mojo v0.25.7 limitation)
# - Regex pattern matching needed for test discovery (no Mojo stdlib support)
# - Process pool management for parallel execution (Python multiprocessing)
# - Cross-language test orchestration (Mojo + Python tests)
#
# See: ADR-001 (Language Selection - Tooling and Automation)
```

### Filtering and Selection

**Filter Types**:

1. **By Paper**: Run only tests for specific paper (`--paper lenet5`)
2. **By Pattern**: Match test names with glob pattern (`--pattern "test_conv*"`)
3. **By Tag**: Run tests with specific tags (`--tag slow`)
4. **By Directory**: Run tests in specific directory (`--dir papers/lenet5`)

**Tag Support**: Tests can declare tags via docstrings or decorators:

```python
def test_something():
    """Test something important.

    Tags: slow, integration
    """
    pass
```

**Rationale**: Flexible filtering enables targeted test runs during development (run just my paper's tests) and comprehensive runs in CI/CD (run all tests). Tag-based filtering allows logical grouping beyond directory structure.

### Performance Considerations

**Optimization Targets**:

- Fast discovery: < 1 second for repositories with < 1000 tests
- Parallel execution: Near-linear speedup with CPU count
- Minimal overhead: Test runner adds < 100ms per test
- Efficient reporting: Report generation in < 100ms

**Profiling Strategy**:

- Measure discovery time separately from execution
- Track per-test overhead vs. actual test time
- Identify slow tests and report in summary
- Provide --profile flag for detailed timing breakdown

## References

### Source Plans

- [Test Runner Plan](../../../plan/03-tooling/02-testing-tools/01-test-runner/plan.md)
- [Testing Tools Parent Plan](../../../plan/03-tooling/02-testing-tools/plan.md)

### Child Components

- [Discover Tests Plan](../../../plan/03-tooling/02-testing-tools/01-test-runner/01-discover-tests/plan.md)
- [Run Tests Plan](../../../plan/03-tooling/02-testing-tools/01-test-runner/02-run-tests/plan.md)
- [Report Results Plan](../../../plan/03-tooling/02-testing-tools/01-test-runner/03-report-results/plan.md)

### Related Issues

- Issue #805: [Test] Test Runner - Test Cases
- Issue #806: [Impl] Test Runner - Implementation
- Issue #807: [Package] Test Runner - Integration
- Issue #808: [Cleanup] Test Runner - Finalization

### Comprehensive Documentation

- [ADR-001: Language Selection - Tooling and Automation](../../review/adr/ADR-001-language-selection-tooling.md)
- [Agent Hierarchy](../../../agents/hierarchy.md)
- [5-Phase Workflow](../../review/README.md)

## Implementation Notes

This section will be populated during the implementation phase with:

- Discoveries and insights from development
- Challenges encountered and solutions applied
- API refinements and design adjustments
- Performance measurements and optimizations
- Integration issues and resolutions
