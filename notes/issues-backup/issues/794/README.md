# Issue #794: [Plan] Run Tests - Design and Documentation

## Objective

Design and document the test execution engine that runs discovered tests with proper isolation, environment setup, and error handling. This component is responsible for executing tests individually, capturing their output and results, and handling failures gracefully without stopping the entire test suite.

## Deliverables

- Detailed architecture specification for test execution engine
- API design for test runner interface
- Test isolation strategy documentation
- Output capture mechanism design
- Error handling and recovery patterns
- Configuration schema for test execution options
- Performance and parallelization strategy
- Support for both Mojo and Python test execution

## Success Criteria

- [ ] Tests run in isolation without interference
- [ ] All output is captured correctly (stdout, stderr, and return codes)
- [ ] Failed tests don't stop execution (continue-on-failure behavior)
- [ ] Execution time is tracked accurately for each test
- [ ] Architecture supports both Mojo and Python tests
- [ ] Design enables parallel test execution
- [ ] Timeout handling is specified
- [ ] Stop-on-first-failure option is documented

## Design Decisions

### 1. Test Isolation Strategy

**Decision**: Each test runs in a separate subprocess with its own environment.

### Rationale

- Prevents side effects between tests (environment variables, global state, file system changes)
- Enables true parallel execution without shared state concerns
- Allows per-test resource limits (memory, CPU time)
- Captures complete output without cross-contamination

### Alternatives Considered

- In-process execution: Rejected due to state contamination risks
- Docker containers: Rejected as too heavyweight for unit tests
- Virtual environments: Rejected as insufficient for isolation (shared process space)

### 2. Output Capture Mechanism

**Decision**: Use subprocess pipes to capture stdout and stderr separately.

### Rationale

- Preserves distinction between normal output and error messages
- Enables real-time streaming for long-running tests
- Compatible with both Mojo and Python subprocess execution
- Standard approach with robust library support

### Alternatives Considered

- File-based capture: Rejected due to filesystem I/O overhead
- Combined stdout/stderr: Rejected as loses diagnostic information
- TTY emulation: Rejected as unnecessary complexity for test output

### 3. Error Handling and Recovery

**Decision**: Implement graceful degradation with continue-on-failure as default behavior.

### Rationale

- Maximizes test coverage visibility (see all failures, not just the first)
- Aligns with CI/CD best practices
- Provides option for fail-fast when needed (debugging specific issues)
- Captures and reports errors without losing subsequent test results

### Error Categories

- Test failures (assertions): Record and continue
- Test errors (exceptions): Record and continue
- Timeout errors: Kill process, record, continue
- Setup/teardown errors: Record and skip test

### 4. Execution Time Tracking

**Decision**: Use high-resolution monotonic clock for timing measurements.

### Rationale

- Immune to system clock adjustments
- Provides microsecond precision for performance analysis
- Available in both Python (time.monotonic) and system calls for Mojo
- Standard approach for benchmark timing

### Metrics to Track

- Per-test execution time
- Setup/teardown time (if applicable)
- Total suite execution time
- Parallel execution efficiency (wall time vs CPU time)

### 5. Language Support (Mojo and Python)

**Decision**: Implement test runner in Python with subprocess-based execution for both languages.

### Rationale

- Python's subprocess module provides robust output capture (Mojo v0.25.7 limitation)
- Enables consistent handling of both test types
- Documented in ADR-001 as acceptable use case for Python
- Test runner is automation tooling, not ML/AI implementation

### Execution Modes

- Mojo tests: Execute with `mojo test <file>` or `mojo run <file>`
- Python tests: Execute with `python -m pytest <file>` or `python <file>`
- Auto-detect based on file extension (.mojo, .🔥, .py)

### 6. Parallel Execution Strategy

**Decision**: Use process-based parallelism with configurable worker pool size.

### Rationale

- Maximizes CPU utilization on multi-core systems
- Test isolation naturally supports parallel execution
- Python's multiprocessing or concurrent.futures provides robust implementation
- Configurable workers allow tuning for different environments

### Configuration Options

- `--workers N`: Number of parallel workers (default: CPU count)
- `--workers 1`: Sequential execution (for debugging)
- `--workers auto`: Auto-detect based on CPU count

### 7. Timeout Handling

**Decision**: Implement per-test timeout with configurable default and per-test overrides.

### Rationale

- Prevents hanging tests from blocking suite execution
- Enables early detection of infinite loops or deadlocks
- Configurable to accommodate legitimately slow tests

### Implementation

- Default timeout: 60 seconds (configurable)
- Per-test override via test metadata or naming convention
- Timeout kills test process and reports as "TIMEOUT" status
- Captured output before timeout is preserved

### 8. Stop-on-First-Failure Option

**Decision**: Provide `-x` / `--exitfirst` flag (pytest-style) for fail-fast behavior.

### Rationale

- Useful for debugging (stop immediately when issue found)
- Reduces wasted CI time when early test fails
- Aligns with common testing tool conventions (pytest, unittest)

### Behavior

- Default: Run all tests, report all failures
- With `-x`: Stop execution after first failure, report partial results
- Exit code reflects failure status in both modes

## Architecture

### Component Structure

```text
test_runner/
├── executor.py          # Main execution engine
├── isolation.py         # Test isolation and subprocess management
├── capture.py          # Output capture and streaming
├── timer.py            # Execution time tracking
├── parallel.py         # Parallel execution coordination
└── config.py           # Configuration schema and parsing
```text

### Execution Flow

1. **Input**: Receive list of discovered tests from test discovery phase
1. **Configuration**: Parse execution options (parallelism, timeout, fail-fast)
1. **Setup**: Initialize worker pool (if parallel execution enabled)
1. **Execution Loop**:
   - For each test:
     - Spawn isolated subprocess
     - Start timer
     - Capture stdout/stderr
     - Monitor for timeout
     - Collect result (pass/fail/error/timeout)
     - Record execution time
   - If fail-fast enabled and test fails: Break loop
1. **Aggregation**: Collect all test results
1. **Output**: Return structured results for reporting phase

### API Design

```python
class TestExecutor:
    """Execute tests with isolation and output capture."""

    def __init__(self, config: ExecutionConfig):
        """Initialize executor with configuration."""
        pass

    def execute_tests(self, tests: List[TestInfo]) -> ExecutionResults:
        """Execute all tests and return results."""
        pass

    def execute_single_test(self, test: TestInfo) -> TestResult:
        """Execute a single test in isolation."""
        pass

class ExecutionConfig:
    """Configuration for test execution."""
    workers: int = None  # None = auto-detect
    timeout: float = 60.0  # seconds
    exit_first: bool = False
    capture_output: bool = True

class TestResult:
    """Result of a single test execution."""
    test_name: str
    status: str  # "PASSED", "FAILED", "ERROR", "TIMEOUT"
    stdout: str
    stderr: str
    execution_time: float
    error_message: Optional[str]

class ExecutionResults:
    """Aggregated results from test execution."""
    total_tests: int
    passed: int
    failed: int
    errors: int
    timeouts: int
    total_time: float
    results: List[TestResult]
```text

## References

- **Source Plan**: [notes/plan/03-tooling/02-testing-tools/01-test-runner/02-run-tests/plan.md](notes/plan/03-tooling/02-testing-tools/01-test-runner/02-run-tests/plan.md)
- **Parent Component**: Test Runner ([notes/plan/03-tooling/02-testing-tools/01-test-runner/plan.md](notes/plan/03-tooling/02-testing-tools/01-test-runner/plan.md))
- **Related Issues**:
  - Issue #795: [Test] Run Tests - Write Tests
  - Issue #796: [Impl] Run Tests - Implementation
  - Issue #797: [Package] Run Tests - Integration and Packaging
  - Issue #798: [Cleanup] Run Tests - Refactor and Finalize
- **Language Selection**: [ADR-001: Language Selection for Tooling](../../review/adr/ADR-001-language-selection-tooling.md)
- **Agent Hierarchy**: [agents/hierarchy.md](agents/hierarchy.md)

## Implementation Notes

(This section will be populated during implementation with findings, challenges, and decisions made during development)
