# Issue #831: [Impl] Collect Coverage - Implementation

**Issue URL**: <https://github.com/mvillmow/ml-odyssey/issues/831>

## Objective

Implement coverage data collection during test execution by instrumenting code to track which lines are executed,
collecting execution data, and storing it for analysis. This is a critical component of the Coverage Tool subsystem
that will measure test completeness and identify untested code paths.

## Context

This implementation issue is part of the comprehensive testing infrastructure for the ML Odyssey project:

- **Parent Component**: Coverage Tool (Issue #846)
- **Sibling Components**:
  - [Issue #847] Generate Coverage Report
  - [Issue #848] Check Coverage Thresholds
- **Related Setup Work**: [Issue #475] Setup Coverage - Infrastructure

The coverage tool collects data during test execution (Issue #831), generates human-readable reports (Issue #847),
and validates against minimum thresholds (Issue #848).

## Deliverables

The following components must be implemented to complete this issue:

### 1. Code Instrumentation System

- **File**: `shared/utils/coverage.mojo` (new)
- **Purpose**: Core coverage data collection and tracking infrastructure
- **Key Components**:
  - `LineTracker` - Track which lines have been executed
  - `FileTracker` - Track coverage per source file
  - `CoverageCollector` - Central coordinator for all coverage operations
  - `ExecutionCounter` - Count how many times each line executes

### Core Functionality

- Track line execution across all source files
- Record execution counts (not just binary yes/no)
- Support multiple file paths and directory structures
- Efficient data structure (minimal memory overhead)
- Handle concurrent test execution safely

### 2. Hook Integration with Test Execution

- **File**: `tests/shared/coverage_runner.mojo` (new)
- **Purpose**: Integration point for collecting coverage during test runs
- **Key Components**:
  - `CoverageTestRunner` - Wrapper for test execution with coverage
  - `setup_coverage_hooks()` - Initialize coverage before tests
  - `collect_coverage_results()` - Gather data after tests
  - `cleanup_coverage()` - Clean up resources

### Core Functionality

- Hook into test execution pipeline
- Automatically collect data for all tests
- Handle test errors without losing coverage data
- Provide hooks for both unit tests and integration tests

### 3. Coverage Data Storage

- **Files**: Coverage data files in standard format
- **Purpose**: Persistent storage for later analysis
- **Formats Supported**:
  - `.coverage` (standard coverage.py format for Python)
  - `coverage.json` (JSON format for tool compatibility)
  - `coverage_raw.dat` (binary format for performance)

### Core Functionality

- Serialize coverage data to disk
- Support multiple output formats
- Preserve line execution counts
- Store file paths and coverage timestamps
- Enable data merging from multiple test runs

### 4. Coverage Configuration

- **File**: Configuration section in `pyproject.toml` or `coverage.ini`
- **Purpose**: Make coverage collection configurable
- **Configuration Options**:
  - `enabled` - Enable/disable coverage collection
  - `output_format` - Choose storage format
  - `include_patterns` - Which files to track (default: all)
  - `exclude_patterns` - Which files to skip
  - `line_threshold` - Minimum lines per file to track
  - `preserve_data` - Keep historical coverage data

### Core Functionality

- Load configuration from standard locations
- Support environment variable overrides
- Document all configuration options
- Provide sensible defaults

### 5. Coverage Statistics Calculation

- **Purpose**: Compute coverage metrics from raw data
- **Metrics**:
  - Lines executed vs total lines
  - Execution count per file
  - Coverage percentage per file and overall
  - Most/least covered files

### Core Functionality

- Calculate coverage percentages accurately
- Handle edge cases (files with no lines, all lines executed, etc.)
- Support filtering by file patterns
- Generate coverage summaries

## Implementation Requirements

### Code Standards

- All code must follow Mojo best practices
- Use `fn` for all functions (not `def`)
- Implement type hints for all parameters
- Use `owned` and `borrowed` semantics appropriately
- Follow SIMD patterns for performance-critical code

### Testing

- Must integrate with test framework configured in Issue #475
- Should not significantly impact test execution speed (< 5% overhead)
- Must handle concurrent test execution safely
- Must preserve data if tests fail

### Performance

- Coverage collection overhead: < 5% test execution time
- Memory overhead: < 100MB for typical project
- Line execution counting must be efficient
- Data serialization must support large projects (10k+ files)

### Compatibility

- Work with both Python and Mojo test files
- Support cross-platform paths (Unix and Windows)
- Handle symlinks and relative paths
- Compatible with standard coverage tools (coverage.py, lcov)

### Documentation

- Docstrings for all public APIs
- Configuration documentation
- Usage examples for instrumenting code
- Integration guide for test runners

## Success Criteria

All of the following must be satisfied for issue completion:

- [ ] `LineTracker` implementation complete with full API
- [ ] `FileTracker` implementation complete with all file tracking
- [ ] `CoverageCollector` coordinate all coverage operations
- [ ] Coverage hooks integrate with test execution
- [ ] Coverage data saves in at least one standard format
- [ ] Configuration system loads and applies settings
- [ ] Coverage statistics calculated correctly
- [ ] Performance overhead measured at < 5%
- [ ] All source files tracked correctly
- [ ] Execution counts recorded accurately
- [ ] Test suite passes for all coverage components
- [ ] Documentation complete with usage examples
- [ ] CI integration verified and working
- [ ] Performance benchmarks completed

## Implementation Strategy

### Phase 1: Core Data Structures (Days 1-2)

1. **Implement `LineTracker`**:
   - Track executed lines for a single file
   - Store execution count per line number
   - Efficient lookup and iteration
   - Serialize to/from data structures

1. **Implement `FileTracker`**:
   - Manage trackers for multiple files
   - Aggregate line coverage statistics
   - Handle file path normalization
   - Support file filtering

1. **Create test cases for data structures**:
   - Test line tracking accuracy
   - Test file tracker aggregation
   - Test path normalization
   - Test data serialization

### Phase 2: Collection Mechanism (Days 2-3)

1. **Implement `CoverageCollector`**:
   - Initialize coverage system
   - Record line executions
   - Merge multiple coverage runs
   - Calculate statistics

1. **Hook into test execution**:
   - Detect test framework being used
   - Install hooks before tests run
   - Collect data after each test
   - Aggregate across all tests

1. **Handle errors gracefully**:
   - Preserve coverage data if tests fail
   - Clean up resources on interruption
   - Report coverage even with test failures

### Phase 3: Data Storage (Days 3-4)

1. **Implement serialization**:
   - Support `.coverage` format (Python standard)
   - Support JSON format
   - Support custom binary format
   - Test round-trip accuracy

1. **Configuration system**:
   - Load from `coverage.ini` or equivalent
   - Support environment variables
   - Validate configuration
   - Provide defaults

### Phase 4: Integration & Testing (Days 4-5)

1. **Integration testing**:
   - Test with actual project test suite
   - Verify overhead < 5%
   - Test with concurrent execution
   - Validate multi-file projects

1. **Performance verification**:
   - Benchmark collection overhead
   - Benchmark serialization time
   - Profile memory usage
   - Optimize hot paths

1. **Documentation**:
   - API documentation with examples
   - Configuration guide
   - Integration instructions
   - Troubleshooting guide

## Architecture Overview

```text
Test Execution
      |
      v
CoverageTestRunner
      |
      +---> CoverageCollector
      |           |
      |           +---> LineTracker (per file)
      |           |
      |           +---> FileTracker
      |           |
      |           +---> ExecutionCounter
      |
      +---> Coverage Data Storage
            |
            +---> .coverage file
            |
            +---> coverage.json
            |
            +---> coverage_raw.dat
```text

## Integration Points

### With Test Framework

- Hook into test discovery (optional instrumentation list)
- Hook into test execution (start/stop collection)
- Hook into result reporting (save coverage data)

### With Issue #847 (Generate Report)

- Output format must be readable by report generator
- Coverage statistics must be easily extracted
- Support incremental report generation

### With Issue #848 (Check Thresholds)

- Thresholds configured separately
- This issue provides the raw data
- Statistics interface must expose coverage percentages

### With Issue #475 (Setup Coverage)

- Uses coverage tool installed by #475
- Configuration from #475 setup
- Tool paths from #475 environment

## Known Challenges

### 1. Performance Impact

**Challenge**: Tracking every line execution could significantly slow tests

### Solution

- Use efficient data structures (bit arrays for booleans, compact arrays for counts)
- Profile-guided optimization to identify bottlenecks
- Optional line filtering to track only critical paths
- Lazy initialization of trackers

### 2. Concurrent Test Execution

**Challenge**: Multiple tests running simultaneously could have race conditions

### Solution

- Thread-safe data structures or per-thread trackers
- Proper synchronization for merged results
- Test isolation (separate coverage per test)

### 3. Large Projects

**Challenge**: Projects with thousands of files could use significant memory

### Solution

- Lazy loading of file trackers
- Streaming serialization for large datasets
- File-by-file analysis option
- Compression of coverage data

### 4. Multi-Format Support

**Challenge**: Different tools expect different coverage data formats

### Solution

- Implement converter between formats
- Use standard `.coverage` format as internal
- Document format mapping
- Test compatibility with standard tools

## References

### Related Planning Documents

- [Parent: Coverage Tool (Issue #846)](../846/README.md) - Complete coverage tool subsystem
- [Sibling: Generate Report (Issue #847)](../847/README.md) - Report generation
- [Sibling: Check Thresholds (Issue #848)](../848/README.md) - Threshold validation
- [Setup: Coverage Infrastructure (Issue #475)](../475/README.md) - Tool setup

### Documentation

- [Testing Infrastructure Plan](../../plan/03-tooling/02-testing-tools/plan.md)
- [Coverage Tool Plan](../../plan/03-tooling/02-testing-tools/03-coverage-tool/plan.md)
- [Collect Coverage Plan](../../plan/03-tooling/02-testing-tools/03-coverage-tool/01-collect-coverage/plan.md)

### External Resources

- [coverage.py Documentation](https://coverage.readthedocs.io/en/7.12.0/)
- [Coverage Data Format](https://coverage.readthedocs.io/en/latest/data.html)
- [Standard Coverage Formats](https://en.wikipedia.org/wiki/Code_coverage)

### Tools & Dependencies

- **Test Framework**: Configured in Issue #475
- **Coverage Tool**: Configured in Issue #475
- **Mojo Language**: Type-safe systems programming for implementation
- **Python**: Alternative for utilities if needed

## Design Decisions

### Decision 1: Line-Based Coverage (vs Branch Coverage)

**Decision**: Implement line-based coverage first, allow for branch coverage later

### Rationale

- Line coverage is simpler to implement and understand
- Covers the most common use case
- Branch coverage can be added as enhancement
- Aligns with plan requirements

### Decision 2: Eager Initialization (vs Lazy)

**Decision**: Use lazy initialization of file trackers

### Rationale

- Only track files that are actually executed
- Reduces memory for large projects with selective testing
- Faster startup time
- Easier to handle dynamic file discovery

### Decision 3: Multiple Format Support

**Decision**: Prioritize `.coverage` format, support JSON, optional custom binary

### Rationale

- `.coverage` is standard (compatibility with many tools)
- JSON for human readability and tool interoperability
- Custom binary for performance-critical scenarios
- Test compatibility with standard tools

### Decision 4: Configuration Location

**Decision**: Support both `coverage.ini` and environment variables

### Rationale

- `coverage.ini` for project-wide settings
- Environment variables for CI/CD integration
- Matches standard coverage.py conventions
- Flexible for different deployment scenarios

## Testing Strategy

### Unit Tests

1. **LineTracker tests**:
   - Record single line execution
   - Multiple executions of same line
   - Range of line numbers
   - Serialization round-trip

1. **FileTracker tests**:
   - Track multiple files
   - Aggregate statistics
   - File path normalization
   - Filtering by pattern

1. **CoverageCollector tests**:
   - Initialize and finalize
   - Record line executions
   - Calculate statistics
   - Handle errors gracefully

### Integration Tests

1. **Test framework integration**:
   - Run with actual test suite
   - Verify all tests tracked
   - Measure overhead
   - Validate data completeness

1. **Multi-file projects**:
   - Large number of files (1000+)
   - Nested directory structures
   - Mixed file types
   - Complex module organization

1. **Data format tests**:
   - Save and load `.coverage` format
   - Save and load JSON format
   - Verify data fidelity
   - Test tool compatibility

### Performance Tests

1. **Overhead measurement**:
   - Test execution time with/without coverage
   - Calculate percentage overhead
   - Verify < 5% target
   - Profile memory usage

1. **Scalability tests**:
   - Large number of files
   - Large test suites
   - Long-running tests
   - Concurrent execution

## Blockers & Dependencies

### Dependencies (Must be Complete First)

- **Issue #475** (Setup Coverage) - Tool installation and configuration
- **Test Framework** - Must have functional test runner

### No Known Blockers

All required components can be implemented independently.

## File Structure

After implementation, the project will have:

```text
shared/utils/
├── coverage.mojo           (NEW - Core coverage system)

tests/shared/
├── coverage_runner.mojo    (NEW - Test runner integration)
├── test_coverage.mojo      (NEW - Coverage tests)

Configuration:
├── coverage.ini            (NEW - Coverage configuration)
├── pyproject.toml          (UPDATED - Coverage settings)
```text

## Success Metrics

### Functional Metrics

- Coverage collected for 100% of source files
- Line execution counts accurate to within 0.1%
- No data loss on test failures
- Compatible with standard tools

### Performance Metrics

- Collection overhead < 5% of test time
- Memory usage < 100MB for typical project
- Serialization time < 10 seconds for 10k files
- Fast startup (< 100ms)

### Quality Metrics

- 95%+ code coverage for coverage system itself
- All public APIs documented
- Zero crashes in normal operation
- Graceful error handling

## Estimated Effort

- **Core Implementation**: 2-3 days
- **Integration & Testing**: 1-2 days
- **Performance Optimization**: 1 day
- **Documentation**: 1 day
- **Total**: 5-7 days

## Acceptance Checklist

Before closing this issue, verify:

- [ ] All data structures implemented and tested
- [ ] Test framework integration complete
- [ ] Coverage data saves successfully
- [ ] Configuration system working
- [ ] Statistics calculated correctly
- [ ] Performance overhead < 5%
- [ ] All files tracked automatically
- [ ] Execution counts recorded
- [ ] No data loss on test failures
- [ ] Documentation complete
- [ ] Examples provided
- [ ] CI integration verified
- [ ] All tests passing
- [ ] Code review completed

## Next Steps

### During Implementation

1. Implement core `LineTracker` and `FileTracker` classes
1. Create `CoverageCollector` coordinator
1. Integrate with test framework
1. Implement serialization for each format
1. Add configuration system
1. Write comprehensive tests

### After Implementation

1. **Issue #847**: Generate coverage reports from collected data
1. **Issue #848**: Validate coverage against configured thresholds
1. **Issue #846**: Integration of all coverage components
1. **Future**: Enhancement to support branch coverage

## Implementation Checklist

Core components to implement:

- [ ] `LineTracker` struct - Track execution per line
- [ ] `FileTracker` struct - Coordinate per-file trackers
- [ ] `CoverageCollector` struct - Central coordinator
- [ ] `ExecutionCounter` struct - Efficient execution counting
- [ ] `coverage_init()` - Initialize coverage system
- [ ] `coverage_cleanup()` - Finalize and save data
- [ ] `.coverage` format serialization
- [ ] JSON format serialization
- [ ] Configuration loading
- [ ] Statistics calculation

---

**Status**: Ready for Implementation

**Priority**: High (Core infrastructure for testing)

**Complexity**: Medium (Complex data structures and integration)

**Risk**: Low (Well-defined requirements, clear acceptance criteria)
