# Issue #835: [Test] Generate Report - Write Tests

## Objective

Write comprehensive test cases for the coverage report generation module, following TDD principles to validate
HTML report generation, text summary creation, per-file statistics calculation, and overall coverage percentage
computation. Tests will ensure reports accurately reflect coverage data and are formatted correctly for user
consumption.

## Deliverables

- Unit tests for HTML report generation (`tests/coverage/test_html_report.mojo`)
- Unit tests for text summary generation (`tests/coverage/test_text_summary.mojo`)
- Unit tests for coverage statistics calculation (`tests/coverage/test_statistics.mojo`)
- Unit tests for color coding and formatting (`tests/coverage/test_formatting.mojo`)
- Integration tests for complete report workflows (`tests/coverage/test_integration.mojo`)
- Test fixtures with sample coverage data
- Test utilities for report validation

## Success Criteria

- [ ] HTML report generation tested for all expected outputs
- [ ] Text summary generation tested with various coverage levels
- [ ] Coverage statistics calculation tested (per-file and overall)
- [ ] Color coding (green/red) applied correctly to covered/uncovered lines
- [ ] File sorting by coverage percentage tested
- [ ] Line numbers and code context included in reports
- [ ] Edge cases tested (empty reports, 0% coverage, 100% coverage)
- [ ] Report navigation tested (links, sections, etc.)
- [ ] Tests pass with 100% coverage of report module
- [ ] Test fixtures created for reusable sample data

## References

- **Planning Phase**: [Issue #834: Plan Generate Report](../834/README.md) - Design and architecture
- **Related Workflow Phase Issues**:
  - [Issue #836: Implementation] Generate Report - Implementation
  - [Issue #837: Packaging] Generate Report - Integration and Packaging
  - [Issue #838: Cleanup] Generate Report - Cleanup and Finalization

## Implementation Notes

**Status**: Ready to start (depends on Issue #834 complete)

**Dependencies**:

- Issue #834 (Plan) must be complete - defines specifications and requirements
- Can proceed in parallel with Issue #836 (Implementation) following TDD workflow
- Coordinates with Issue #836 for test-driven development cycle

**Test Coverage Goals**:

1. **HTML Report Generation** (20-25% of tests)
   - Basic HTML structure generation
   - File listing and navigation
   - Source code display with line numbers
   - Coverage highlighting (covered vs uncovered lines)
   - Summary statistics display
   - Links and navigation elements
   - CSS styling integration

2. **Text Summary Generation** (15-20% of tests)
   - Header formatting
   - File statistics table generation
   - Overall coverage summary
   - Column alignment and spacing
   - Color codes in text output (if terminal-based)
   - Sorting by coverage percentage

3. **Statistics Calculation** (20-25% of tests)
   - Per-file coverage percentage calculation
   - Overall project coverage percentage
   - Line counting (covered/uncovered/total)
   - File grouping and categorization
   - Edge cases (empty files, single-line files, large files)

4. **Formatting and Color Coding** (15-20% of tests)
   - Color mapping for HTML (green=covered, red=uncovered)
   - Proper HTML escaping for source code
   - Line number alignment and padding
   - File path formatting
   - Unicode/special character handling

5. **Integration Tests** (15-20% of tests)
   - End-to-end report generation workflow
   - Multiple file handling
   - Report output file creation
   - File handling and I/O operations
   - Integration with coverage data sources

**Key Test Files**:

1. `test_html_report.mojo` - HTML report structure, formatting, content
2. `test_text_summary.mojo` - Text summary structure, statistics, formatting
3. `test_statistics.mojo` - Coverage calculations, aggregations, percentages
4. `test_formatting.mojo` - Color coding, HTML escaping, alignment
5. `test_integration.mojo` - End-to-end report generation workflows

**TDD Approach**:

- Write tests BEFORE implementation in Issue #836
- Tests should initially fail (red phase)
- Implementation in Issue #836 makes tests pass (green phase)
- Refactor implementation while keeping tests green
- Iterate on test refinement as needed

**Test Data Strategy**:

Create comprehensive test fixtures covering:

1. **Coverage Data Samples**:
   - 100% covered file (all lines executed)
   - 0% covered file (no lines executed)
   - Partial coverage files (50%, 75%, 90% coverage)
   - Multiple files with varying coverage levels
   - Large files (1000+ lines)
   - Small files (1-5 lines)
   - Files with comments and docstrings

2. **Edge Cases**:
   - Empty source files
   - Files with only comments
   - Files with syntax that needs HTML escaping (`<`, `>`, `&`, quotes)
   - Very long lines (>200 characters)
   - Unicode characters in code
   - Mixed line endings (CRLF vs LF)

3. **Report Scenarios**:
   - Single file report
   - Multi-file report (5+ files)
   - Nested directory structure
   - Files with identical names in different directories

**Test Infrastructure**:

- Create fixture factory for generating sample coverage data
- Utility functions for comparing HTML/text output
- Mock file system operations where needed
- Validation utilities for report structure

**Next Steps**:

1. Review coverage report specifications from Issue #834
2. Create test directory structure (`tests/coverage/`)
3. Design test fixtures and sample data
4. Implement test cases following Mojo testing conventions
5. Coordinate with Issue #836 for TDD cycle
6. Ensure all tests pass before moving to implementation

## Test File Templates

### test_html_report.mojo

```mojo
fn test_html_report_structure() raises:
    """Test that generated HTML has required structure"""
    pass

fn test_html_report_file_listing() raises:
    """Test that all files appear in report"""
    pass

fn test_html_code_highlighting() raises:
    """Test that covered/uncovered lines are highlighted correctly"""
    pass

fn test_html_line_numbers() raises:
    """Test that line numbers are included and formatted correctly"""
    pass

fn test_html_summary_section() raises:
    """Test that summary statistics are included in report"""
    pass
```

### test_text_summary.mojo

```mojo
fn test_text_summary_header() raises:
    """Test text summary header format"""
    pass

fn test_text_summary_file_stats() raises:
    """Test file statistics table generation"""
    pass

fn test_text_summary_sorting() raises:
    """Test files are sorted by coverage percentage"""
    pass

fn test_text_summary_alignment() raises:
    """Test column alignment and spacing"""
    pass

fn test_text_summary_totals() raises:
    """Test overall coverage totals in summary"""
    pass
```

### test_statistics.mojo

```mojo
fn test_per_file_coverage_calculation() raises:
    """Test calculation of per-file coverage percentage"""
    pass

fn test_overall_coverage_calculation() raises:
    """Test calculation of overall project coverage"""
    pass

fn test_line_counting() raises:
    """Test accurate counting of covered/uncovered lines"""
    pass

fn test_statistics_with_edge_cases() raises:
    """Test statistics calculation with empty files and edge cases"""
    pass
```

### test_formatting.mojo

```mojo
fn test_color_mapping() raises:
    """Test color codes are applied correctly (green/red)"""
    pass

fn test_html_escaping() raises:
    """Test that code is properly HTML escaped"""
    pass

fn test_line_number_padding() raises:
    """Test line numbers are padded correctly"""
    pass

fn test_special_character_handling() raises:
    """Test handling of special and unicode characters"""
    pass
```

### test_integration.mojo

```mojo
fn test_full_report_generation() raises:
    """Test complete report generation workflow"""
    pass

fn test_multiple_file_report() raises:
    """Test report with multiple files"""
    pass

fn test_report_file_output() raises:
    """Test that report files are created correctly"""
    pass

fn test_coverage_data_parsing() raises:
    """Test parsing of coverage data from various sources"""
    pass
```

## Test Fixtures Structure

```text
tests/
├── coverage/
│   ├── conftest.mojo          # Shared fixtures and utilities
│   ├── test_html_report.mojo
│   ├── test_text_summary.mojo
│   ├── test_statistics.mojo
│   ├── test_formatting.mojo
│   ├── test_integration.mojo
│   └── fixtures/
│       ├── coverage_data/     # Sample coverage data files
│       │   ├── basic.json
│       │   ├── high_coverage.json
│       │   ├── low_coverage.json
│       │   └── edge_cases.json
│       └── sample_code/       # Sample source files
│           ├── fully_covered.mojo
│           ├── partial_covered.mojo
│           ├── uncovered.mojo
│           └── edge_cases.mojo
```

## Testing Checklist

### Before Starting Implementation Tests

- [ ] Review Issue #834 specifications thoroughly
- [ ] Understand report format requirements (HTML and text)
- [ ] Review expected outputs from planning phase
- [ ] Plan test data and fixtures

### During Test Development

- [ ] Write test for each major functionality
- [ ] Include edge case tests
- [ ] Create comprehensive test fixtures
- [ ] Document test purpose and expected behavior
- [ ] Ensure tests are independent and repeatable

### Before Marking Complete

- [ ] All tests are written and documented
- [ ] Test fixtures are comprehensive and reusable
- [ ] Tests fail initially (red phase)
- [ ] Tests are ready for implementation phase
- [ ] No implementation code is written yet

## Related Issues

- Issue #834: [Plan] Generate Report - Design and Documentation
- Issue #836: [Implementation] Generate Report - Implementation
- Issue #837: [Packaging] Generate Report - Integration and Packaging
- Issue #838: [Cleanup] Generate Report - Cleanup and Finalization
