# Issue #824: [Plan] Paper Test Script - Design and Documentation

## Objective

Design and document a specialized script for testing individual paper implementations. This tool will validate
paper directory structure, execute paper-specific tests, and generate comprehensive health reports showing what's
missing or broken. The planning phase will define detailed specifications, architecture, and implementation approach.

## Deliverables

- Complete specification for paper test script functionality
- Architecture and design documentation
- API contract definition for the test runner
- Integration strategy with main test suite
- Validation and health check specification
- Usage examples and integration guide

## Success Criteria

- [ ] Script can test any paper by name or path
- [ ] Structure validation catches common issues
- [ ] All paper tests are executed with clear results
- [ ] Health report shows pass/fail status and recommendations
- [ ] Script works standalone and integrates with main test runner
- [ ] Design documentation is comprehensive and actionable
- [ ] Performance requirements are defined and validated

## References

- **Testing Strategy**: `/docs/core/testing-strategy.md` - ML Odyssey testing philosophy and patterns
- **Paper Template**: `/papers/_template/` - Standard paper directory structure
- **Test Framework**: `/tests/` - Existing test organization and patterns
- **Configuration System**: `/notes/review/configs-architecture.md` - Paper configuration management
- **Related Issues**:
  - #825 [Test] Paper Test Script - Test Suite
  - #826 [Implementation] Paper Test Script - Implementation
  - #827 [Packaging] Paper Test Script - Integration
  - #828 [Cleanup] Paper Test Script - Finalization

## Implementation Notes

To be filled during implementation

## Design Specification

### 1. Overview and Purpose

The Paper Test Script (PTS) is a specialized utility for developers working on individual paper implementations
within the ML Odyssey project. Its primary purposes are:

1. **Validation**: Verify paper directory structure matches repository standards
1. **Testing**: Execute all tests associated with a specific paper
1. **Reporting**: Generate clear pass/fail status and health metrics
1. **Development**: Provide quick feedback during active development
1. **Integration**: Work both standalone and as part of the main test suite

### 2. Core Functionality

#### 2.1 Paper Discovery

The script must support multiple ways to specify which paper to test:

```bash
# Test by paper name (from papers/ directory)
python scripts/test_paper.py lenet5

# Test by paper path
python scripts/test_paper.py /path/to/papers/lenet5

# Test current directory if it's a paper
python scripts/test_paper.py

# Test multiple papers
python scripts/test_paper.py lenet5 vgg16 resnet50
```text

### Requirements

- Accept paper name and resolve to `papers/<name>` directory
- Accept absolute or relative paths to paper directories
- Support "current directory" mode (detect if cwd is a paper)
- Support multiple paper arguments for batch testing
- Validate that specified path is actually a valid paper directory

#### 2.2 Structure Validation

Validate that paper directory structure matches the template defined in `/papers/_template/`:

**Required Directories** (must exist):

- `src/` - Model and implementation code
- `tests/` - Test files
- `data/` - Data directory structure
- `configs/` - Configuration files
- `examples/` - Example usage scripts

**Required Files** (must exist):

- `README.md` - Paper documentation
- `src/__init__.mojo` - Package initialization
- `tests/__init__.mojo` - Test package initialization
- `configs/config.yaml` - Paper configuration

### Validation Output

- [PASS] if all required items exist
- [WARN] if optional items are missing
- [FAIL] if required items are missing
- [ERROR] if directory is not a valid paper

### Example Output

```text
Paper Structure Validation: lenet5
────────────────────────────────────
[✓] src/ directory exists
[✓] tests/ directory exists
[✓] README.md exists
[✓] configs/config.yaml exists
[✓] tests/__init__.mojo exists
[✓] src/__init__.mojo exists
[✓] data/ directory exists
[!] examples/ directory not found (optional)

Structure Status: PASS (7/8 items valid)
```text

#### 2.3 Test Discovery and Execution

Discover and execute all tests associated with a paper:

### Test Types to Support

1. **Mojo Tests** (`tests/test_*.mojo`)
   - Use Mojo's built-in testing framework
   - Execute with `mojo test <test_file.mojo>`
   - Parse results for pass/fail/error status

1. **Python Tests** (`tests/test_*.py`)
   - Use pytest framework
   - Execute with `pytest tests/test_*.py`
   - Collect coverage metrics if available

1. **Integration Tests** (`tests/integration/`)
   - Execute training/model tests
   - Validate end-to-end workflows
   - Report execution time

### Test Discovery Rules

- All `test_*.mojo` files in `tests/` directory
- All `test_*.py` files in `tests/` directory
- All subdirectories in `tests/` are searched
- Respect `.testignore` file if present (for skipped tests)

### Execution Strategy

- Run tests in dependency order (unit → integration)
- Capture stdout/stderr for each test
- Record execution time for performance monitoring
- Continue on failures to collect all results
- Report summary statistics (passed, failed, skipped, errors)

#### 2.4 Health Report Generation

Generate a comprehensive health report for the paper:

### Report Sections

1. **Summary** - Quick overview of paper health
   - Overall status (Green/Yellow/Red)
   - Key metrics (test count, pass rate, etc.)
   - Quick recommendations

1. **Structure Validation** - Directory and file compliance
   - Required/optional item status
   - Missing critical files
   - Structure recommendations

1. **Test Results** - Test execution details
   - Pass/fail/error counts
   - Failed test names and error messages
   - Performance metrics (slowest tests)

1. **Coverage Analysis** (if available)
   - Code coverage percentage
   - Uncovered critical paths
   - Coverage recommendations

1. **Configuration Validation**
   - Config file syntax validity
   - Required configuration keys present
   - Config merge validation

1. **Recommendations** - Actionable suggestions
   - Missing tests to add
   - Structure improvements needed
   - Performance optimizations
   - Documentation gaps

### Example Health Report

```text
====================================================
    PAPER HEALTH REPORT: lenet5
====================================================

STATUS: ⚠ WARNING (Mostly Healthy, Some Issues)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

QUICK SUMMARY
─────────────
Tests Passed:     42/45 (93%)
Structure Valid:  7/8 items (Good)
Config Valid:     ✓ Yes
Documentation:    ✓ Complete

STRUCTURE VALIDATION
────────────────────
[✓] src/          - Model implementation
[✓] tests/        - Test suite
[✓] README.md     - Documentation
[✓] configs/      - Configuration
[!] examples/     - Missing (optional)

TEST RESULTS
────────────
Unit Tests:       24/24 passed
Integration:      18/20 failed
  └─ test_training_convergence.mojo: timeout after 60s
  └─ test_inference_speed.mojo: accuracy below threshold

COVERAGE ANALYSIS
─────────────────
Code Coverage:    87% (good)
  └─ src/layers.mojo:    94%
  └─ src/trainer.mojo:   72% (needs work)

CONFIGURATION
──────────────
Config File:      ✓ Valid YAML
Required Keys:    ✓ All present
Merge Result:     ✓ Successful

RECOMMENDATIONS
────────────────
1. Fix integration tests (2 failures)
   - test_training_convergence: May need longer timeout
   - test_inference_speed: Check optimization settings

2. Improve trainer coverage (72% → target 85%)
   - Add tests for learning rate scheduling
   - Test edge cases in batch normalization

3. Add examples/ directory
   - Create notebook with usage walkthrough
   - Add scripts/ for reproducible experiments

4. Documentation improvements
   - Add experiment results section
   - Document hyperparameter tuning

TIMING
──────
Total Test Time:  23.4 seconds
Slowest Tests:
  1. test_training_epoch (18.2s)
  2. test_inference_batch (3.5s)
  3. test_model_save_load (1.1s)

Generated: 2025-11-16 10:30:45 UTC
====================================================
```text

#### 2.5 Output Formats

Support multiple output formats for integration with different tools:

**Human-readable** (default):

- Formatted text report with colors/symbols
- Easy to read at terminal
- Shows all details

**JSON** (for CI/CD integration):

```json
{
  "paper": "lenet5",
  "timestamp": "2025-11-16T10:30:45Z",
  "status": "warning",
  "structure": {
    "valid": true,
    "items": { "valid": 7, "total": 8 }
  },
  "tests": {
    "passed": 42,
    "failed": 3,
    "skipped": 0,
    "total": 45
  },
  "coverage": { "percentage": 87 }
}
```text

**JUnit XML** (for CI integration):

- Standard format for test reporting
- Compatible with GitHub Actions, Jenkins, etc.
- Includes timing and failure details

### 3. Architecture

#### 3.1 Module Structure

```text
test_paper.py
├── cli.py              # Command-line interface
├── paper.py            # Paper directory handling
├── validator.py        # Structure validation
├── test_runner.py      # Test execution
│   ├── mojo_runner.py  # Mojo test execution
│   └── python_runner.py # Python test execution
├── reporter.py         # Report generation
│   ├── text_reporter.py
│   ├── json_reporter.py
│   └── junit_reporter.py
└── utils.py            # Helper utilities
```text

#### 3.2 Key Classes

Paper:

```python
class Paper:
    """Represents a paper in the papers/ directory."""
    def __init__(self, name_or_path: str) -> None
    def is_valid(self) -> bool
    def validate_structure(self) -> ValidationResult
    def get_tests(self) -> List[TestFile]
    def get_config(self) -> Dict[str, Any]
    def __str__(self) -> str
```text

TestRunner:

```python
class TestRunner:
    """Executes tests for a paper."""
    def __init__(self, paper: Paper) -> None
    def run_all_tests(self) -> TestResults
    def run_specific_test(self, test_file: str) -> TestResult
    def get_test_files(self) -> List[TestFile]
```text

HealthReporter:

```python
class HealthReporter:
    """Generates health reports."""
    def __init__(self, paper: Paper) -> None
    def generate_report(self) -> HealthReport
    def save_report(self, format: str, path: str) -> None
```text

#### 3.3 Execution Flow

```text
1. Parse command-line arguments
   ↓
2. Resolve paper directory
   ↓
3. Validate paper structure
   │
   ├─ If invalid: Report issues and exit
   │
4. Discover tests
   │
   ├─ Mojo tests (test_*.mojo)
   ├─ Python tests (test_*.py)
   │
5. Execute tests in order
   │
   ├─ Unit tests first
   ├─ Integration tests second
   ├─ Capture output and timing
   │
6. Collect results and metrics
   │
   ├─ Pass/fail counts
   ├─ Coverage if available
   ├─ Timing data
   │
7. Generate health report
   │
   ├─ Validate configuration
   ├─ Analyze results
   ├─ Generate recommendations
   │
8. Output results (text/JSON/JUnit)
   ↓
9. Exit with appropriate code
   - 0: All tests passed
   - 1: Tests failed
   - 2: Validation failed
```text

### 4. Integration Points

#### 4.1 With Main Test Suite

The paper test script should integrate with the main CI/CD pipeline:

### In `.github/workflows/test-papers.yml`

```yaml
- name: Test Paper Implementations
  run: |
    for paper in papers/*/; do
      if [ -d "$paper" ]; then
        python scripts/test_paper.py "$(basename $paper)" \
          --format json \
          --output-dir results/
      fi
    done

- name: Collect Results
  run: |
    python scripts/collect_paper_results.py results/ \
      --format junit \
      --output test-results.xml
```text

#### 4.2 With Configuration System

Use the paper configuration system to inform testing:

```python
# In test_runner.py
config = paper.get_config()
timeout = config.get('testing.timeout', 60)
parallel = config.get('testing.parallel', False)
skip_integration = config.get('testing.skip_integration', False)
```text

#### 4.3 With Development Workflow

Support developer workflow during paper implementation:

```bash
# Quick check during development
python scripts/test_paper.py lenet5 --fast

# Full test with coverage
python scripts/test_paper.py lenet5 --coverage --verbose

# Watch mode for TDD
python scripts/test_paper.py lenet5 --watch

# Run specific test
python scripts/test_paper.py lenet5 tests/test_model.mojo
```text

### 5. Command-Line Interface

#### 5.1 Basic Usage

```bash
# Test a paper by name
python scripts/test_paper.py lenet5

# Test by path
python scripts/test_paper.py papers/lenet5

# Test current directory
cd papers/lenet5
python scripts/test_paper.py

# Test multiple papers
python scripts/test_paper.py lenet5 vgg16 resnet50
```text

#### 5.2 Options

```bash
# Output format
--format {text,json,junit}   # Default: text

# Output destination
--output-dir DIR             # Write results to directory
--output-file FILE           # Write results to specific file

# Execution options
--fast                       # Skip slow integration tests
--parallel N                 # Run tests in parallel (default: 1)
--timeout SECONDS            # Test timeout (default: 60)
--verbose                    # Detailed output
--quiet                      # Minimal output

# Analysis options
--coverage                   # Generate coverage report
--profile                    # Profile test execution
--warnings-as-errors         # Treat warnings as failures

# Development options
--watch                      # Watch mode (re-run on changes)
--keep-going                 # Continue after first failure
--only-failed                # Re-run only failed tests
```text

### 6. Error Handling

### Exit Codes

- `0`: Success (all tests passed)
- `1`: Test failures detected
- `2`: Paper structure invalid
- `3`: Configuration error
- `4`: Test execution error (timeout, crash)
- `5`: Command-line argument error

### Error Messages

Example for invalid paper:

```text
Error: Paper 'unknown_paper' not found

Looking in:
  - papers/unknown_paper/
  - unknown_paper/

Available papers:
  - lenet5
  - vgg16
  - resnet50

Usage: python scripts/test_paper.py <paper_name_or_path>
```text

### 7. Performance Considerations

- **Test Discovery**: Should complete in < 1 second
- **Structure Validation**: Should complete in < 500ms
- **Test Execution**: Depends on paper tests (typically 10-60 seconds)
- **Report Generation**: Should complete in < 1 second
- **Total Time**: Should not exceed 2 minutes for typical paper

### 8. Extensibility

Design for future enhancements:

1. **Custom validators** - Allow papers to define custom validation rules
1. **Test hooks** - Before/after test execution hooks
1. **Plugins** - Custom reporters and analyzers
1. **Benchmarking** - Integration with performance tracking
1. **Cloud testing** - Run tests on remote infrastructure

## Implementation Strategy

### Phase 1: Core Infrastructure

1. Implement Paper class for directory handling
1. Implement structure validator
1. Create basic test runner (Mojo support)
1. Basic text reporting

### Phase 2: Full Test Support

1. Add Python test support
1. Implement parallel test execution
1. Add coverage integration
1. Enhance error handling

### Phase 3: Advanced Reporting

1. JSON and JUnit output formats
1. Configuration validation
1. Detailed health report generation
1. Performance profiling

### Phase 4: Integration

1. CI/CD pipeline integration
1. Development workflow support
1. Watch mode implementation
1. Parallel paper testing

## Testing Approach

The paper test script itself must be thoroughly tested:

1. **Unit Tests**: Paper class, validators, runners
1. **Integration Tests**: Full workflow with real papers
1. **End-to-End Tests**: CLI and output validation
1. **Performance Tests**: Execution time benchmarks
1. **Error Tests**: Edge cases and error handling

## Next Steps

After this planning phase is complete:

1. **Issue #825**: Create test suite to validate the script
1. **Issue #826**: Implement the actual paper test script
1. **Issue #827**: Integrate with CI/CD pipeline and packaging
1. **Issue #828**: Cleanup and finalization
