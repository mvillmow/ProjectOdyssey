# Issue #479: [Test] Coverage Reports - Write Tests

## Objective

Write comprehensive tests for coverage reporting functionality, validating that reports accurately display coverage data in multiple formats (console, HTML) and provide clear visualization of uncovered code.

## Deliverables

- Tests for console report generation
- Tests for HTML report generation
- Tests for coverage statistics calculation
- Tests for historical tracking
- Tests for report accuracy and formatting
- Test fixtures and mock coverage data

## Success Criteria

- [ ] Console reports display correctly
- [ ] HTML reports render properly
- [ ] Coverage statistics are accurate
- [ ] Historical tracking works
- [ ] Reports highlight uncovered code clearly
- [ ] All tests pass and are documented

## References

### Parent Issue

- [Issue #478: [Plan] Coverage Reports](../478/README.md) - Design and architecture

### Related Issues

- [Issue #480: [Impl] Coverage Reports](../480/README.md) - Implementation
- [Issue #481: [Package] Coverage Reports](../481/README.md) - Packaging
- [Issue #482: [Cleanup] Coverage Reports](../482/README.md) - Cleanup

### Dependencies

- [Issue #473-477: Setup Coverage](../473/README.md) - Coverage data collection must work first

### Comprehensive Documentation

- [5-Phase Workflow](../../../../../../../home/user/ml-odyssey/notes/review/README.md)
- [Agent Hierarchy](../../../../../../../home/user/ml-odyssey/agents/hierarchy.md)

## Implementation Notes

### Testing Strategy

Since coverage reporting builds on Python coverage.py (from Issue #475), tests should focus on:

**1. Console Report Testing**

Test console output formatting and statistics:

```python
def test_console_report_shows_total_coverage():
    """Test console report displays total coverage percentage."""
    # Given: Coverage data with known coverage
    # When: Generate console report
    # Then: Report shows correct percentage

def test_console_report_lists_low_coverage_files():
    """Test console report highlights files below threshold."""
    # Given: Coverage data with files at different percentages
    # When: Generate console report
    # Then: Files < 80% are highlighted
```text

**2. HTML Report Testing**

Test HTML generation and visualization:

```python
def test_html_report_generates_index():
    """Test HTML report creates index page."""
    # Given: Coverage data
    # When: Generate HTML report
    # Then: index.html exists with correct structure

def test_html_report_highlights_uncovered_lines():
    """Test HTML report shows uncovered lines in red."""
    # Given: File with partial coverage
    # When: Generate HTML file page
    # Then: Uncovered lines have red background
```text

**3. Statistics Calculation Testing**

Test accuracy of coverage metrics:

```python
def test_line_coverage_calculation():
    """Test line coverage percentage is accurate."""
    # Given: 80 covered lines, 100 total lines
    # When: Calculate line coverage
    # Then: Returns 80.0%

def test_branch_coverage_calculation():
    """Test branch coverage includes all branches."""
    # Given: Coverage data with branch information
    # When: Calculate branch coverage
    # Then: Accounts for both taken and missed branches
```text

**4. Historical Tracking Testing**

Test coverage trend tracking:

```python
def test_historical_tracking_stores_snapshots():
    """Test coverage snapshots are stored correctly."""
    # Given: Coverage data for multiple commits
    # When: Store historical snapshots
    # Then: Each snapshot persists with metadata

def test_historical_tracking_calculates_delta():
    """Test coverage delta between commits."""
    # Given: Two coverage snapshots
    # When: Calculate delta
    # Then: Returns difference in percentage points
```text

### Test Fixtures

### Mock Coverage Data

```python
# conftest.py or fixtures file
@pytest.fixture
def sample_coverage_data():
    """Provide sample coverage data for testing."""
    return {
        "files": {
            "src/core.py": {
                "total_lines": 100,
                "covered_lines": 90,
                "missing_lines": [23, 45, 67, 89, 90, 91, 92, 93, 94, 95]
            },
            "src/utils.py": {
                "total_lines": 50,
                "covered_lines": 40,
                "missing_lines": list(range(41, 51))
            }
        },
        "totals": {
            "total_lines": 150,
            "covered_lines": 130,
            "coverage_percent": 86.67
        }
    }
```text

### Key Test Scenarios

### Report Generation

- [ ] Console report generates for valid coverage data
- [ ] HTML report creates all necessary files
- [ ] Reports handle empty coverage data gracefully
- [ ] Reports handle 100% coverage correctly
- [ ] Reports handle 0% coverage correctly

### Accuracy

- [ ] Coverage percentages match input data
- [ ] Uncovered lines are correctly identified
- [ ] File-level statistics aggregate correctly
- [ ] Module-level rollups are accurate

### Formatting

- [ ] Console report is readable in terminal
- [ ] HTML report is valid HTML5
- [ ] Colors are accessible (colorblind-friendly)
- [ ] Reports work in different browsers

### Performance

- [ ] Report generation completes in < 10 seconds (Issue #478)
- [ ] HTML report size is reasonable
- [ ] Large codebases don't cause timeouts

### Integration Testing

Test integration with coverage.py:

```python
def test_integration_with_coverage_py():
    """Test report generation from actual coverage.py data."""
    # Given: Real .coverage file from pytest-cov
    # When: Generate reports
    # Then: Reports accurately reflect coverage.py data
```text

### Open Questions to Address

- [ ] What report formats are required? (Console, HTML, both?)
- [ ] Should we support JSON/XML output for CI integration?
- [ ] How detailed should HTML reports be? (Line-by-line vs summary?)
- [ ] What historical tracking storage format? (JSON, SQLite, CSV?)

### Status

Created: 2025-11-19
Status: Pending implementation
Dependencies: Issue #478 (Plan) and #473-477 (Setup Coverage) must be completed first
