# Issue #834: [Plan] Generate Report - Design and Documentation

## Objective

Design and document a comprehensive coverage report generation system that produces clear, navigable reports showing which code is tested and which is not. This planning phase will define detailed specifications, design the architecture and approach, document API contracts and interfaces, and create comprehensive design documentation to enable developers to identify gaps in test coverage and prioritize testing efforts.

## Deliverables

- Coverage report generation architecture design
- Report format specifications (HTML, text, JSON)
- HTML report UI/UX design with navigation and highlighting
- Per-file coverage statistics design
- Color coding and visual design specifications
- Summary report template and data structures
- API contracts for report generators
- Configuration format for report customization
- Comprehensive design documentation and specifications

## Success Criteria

- [ ] Coverage report architecture is fully specified
- [ ] Report formats are well-defined (HTML, text, JSON)
- [ ] HTML report design includes navigation, line highlighting, and statistics
- [ ] Color coding schema is defined (green for covered, red for uncovered)
- [ ] Per-file coverage calculation methodology is documented
- [ ] Overall coverage percentage calculation is specified
- [ ] Report generator API contracts are clearly documented
- [ ] File sorting by coverage percentage strategy is defined
- [ ] Summary statistics and key metrics are identified
- [ ] Design documentation is comprehensive and actionable
- [ ] Integration approach with coverage data sources is specified

## References

- **Source Plan**: `/notes/plan/06-agentic-workflows/03-testing-improvements/03-generate-report/plan.md`
- **Related Issues**:
  - #831 [Plan] Run Coverage Analysis - Design and Documentation
  - #832 [Test] Run Coverage Analysis - Test Suite
  - #833 [Implementation] Run Coverage Analysis - Implementation
  - #835 [Test] Generate Report - Test Suite
  - #836 [Implementation] Generate Report - Implementation
  - #837 [Packaging] Generate Report - Integration and Packaging
  - #838 [Cleanup] Generate Report - Cleanup and Finalization
- **Testing Improvements Section**: Issue tracking for comprehensive test infrastructure enhancements
- **Coverage Analysis Prerequisites**: Review Issue #831 for data source specifications

## Implementation Notes

*To be filled during implementation*

## Design Decisions

### 1. Coverage Report Architecture

The coverage report generation system follows a modular architecture with clear separation of concerns:

```text
Coverage Data (from Issue #831)
        ↓
   Data Parser
        ↓
   Report Generator (Multiple Formats)
     ├── HTML Generator
     ├── Text Generator
     └── JSON Generator
        ↓
   Report Output Files
```text

#### Component Responsibilities

1. **Data Parser**
   - Consumes coverage data from analysis phase (Issue #831)
   - Normalizes coverage metrics across formats
   - Validates data integrity
   - Handles missing or incomplete data gracefully

1. **Report Generators**
   - Format-specific report creation
   - Apply styling and formatting rules
   - Calculate derived metrics
   - Organize data for readability

1. **Output Handler**
   - Write reports to file system
   - Manage report file naming conventions
   - Create index/navigation files
   - Handle file permissions and organization

### 2. Report Formats Specification

#### 2.1 HTML Report

**Purpose**: Interactive, visually rich report for developers to explore coverage data.

### Features

- Syntax-highlighted source code with line numbers
- Color-coded coverage indicators
- Clickable navigation between files
- Expandable/collapsible sections
- Responsive design for multiple screen sizes
- Search functionality for finding specific files or functions
- Coverage statistics sidebar
- Tooltips showing coverage details on hover

### Structure

```text
html_report/
├── index.html              # Main entry point with overall statistics
├── css/
│   └── coverage.css        # Styling and color scheme
├── js/
│   └── coverage.js         # Navigation and interactivity
└── files/
    ├── file1.html          # Per-file report (syntax highlighted)
    ├── file2.html
    └── ...
```text

### Color Scheme

- **Green (#4CAF50)**: Lines covered by tests
- **Red (#F44336)**: Lines not covered by tests
- **Yellow (#FFC107)**: Partially covered lines (for branches)
- **Gray (#9E9E9E)**: Non-executable lines (comments, blank lines)

### Key Metrics Displayed

- Overall coverage percentage
- Coverage by file
- Coverage by function/method
- Top uncovered areas (sorted by coverage percentage)
- Top well-covered areas

#### 2.2 Text Summary Report

**Purpose**: Quick command-line summary for CI/CD pipelines and scripting.

### Format

```text
================================================================================
COVERAGE REPORT SUMMARY
================================================================================

Project: ml-odyssey
Generated: 2025-11-16 12:34:56 UTC
Coverage Tool: pytest-cov

OVERALL STATISTICS
------------------
Total Lines:        5,234
Covered Lines:      4,156
Uncovered Lines:    1,078
Coverage:           79.4%

TOP FILES BY COVERAGE
---------------------
✓ src/core/tensor.mojo            95.2%  (142/150 lines)
✓ src/core/array.mojo             92.1%  (187/203 lines)
✗ src/models/lenet5.mojo          42.3%  (95/225 lines)
✗ src/training/optimizer.mojo     38.9%  (88/226 lines)

UNCOVERED FILES
---------------
src/inference/onnx.mojo           0%     (0/156 lines)
src/export/tensorflow.mojo        0%     (0/198 lines)

COVERAGE DETAILS BY DIRECTORY
-----------------------------
src/core/                         91.3%  (624/683 lines)
src/models/                       65.2%  (445/682 lines)
src/training/                     72.1%  (398/552 lines)
src/inference/                    12.3%  (45/365 lines)
src/export/                       8.5%   (23/270 lines)

================================================================================
```text

### Report Components

1. Header with generation timestamp and tool info
1. Overall statistics summary
1. Top files by coverage (both high and low)
1. Directory-level coverage breakdown
1. Line count totals
1. Coverage percentage trends (if historical data available)

#### 2.3 JSON Format Report

**Purpose**: Machine-readable format for programmatic processing and integration.

### Schema

```json
{
  "metadata": {
    "generated_at": "2025-11-16T12:34:56Z",
    "tool": "pytest-cov",
    "project": "ml-odyssey",
    "version": "0.1.0"
  },
  "summary": {
    "total_lines": 5234,
    "covered_lines": 4156,
    "uncovered_lines": 1078,
    "coverage_percent": 79.4,
    "num_files": 45
  },
  "files": [
    {
      "path": "src/core/tensor.mojo",
      "coverage_percent": 95.2,
      "covered_lines": 142,
      "total_lines": 150,
      "uncovered_line_numbers": [23, 45, 78],
      "partially_covered_lines": []
    },
    {
      "path": "src/core/array.mojo",
      "coverage_percent": 92.1,
      "covered_lines": 187,
      "total_lines": 203,
      "uncovered_line_numbers": [15, 67, 89, 120, 145, 156, 189, 192, 198, 201, 203],
      "partially_covered_lines": []
    }
  ],
  "directories": [
    {
      "path": "src/core",
      "coverage_percent": 91.3,
      "covered_lines": 624,
      "total_lines": 683
    }
  ],
  "functions": [
    {
      "file": "src/core/tensor.mojo",
      "name": "Tensor.__init__",
      "coverage_percent": 100.0,
      "lines": [12, 13, 14]
    }
  ]
}
```text

### 3. HTML Report User Interface Design

#### 3.1 Main Index Page

### Layout

- Header with project title and metadata
- Summary statistics panel (overall coverage with gauge/progress bar)
- Search bar for quick file filtering
- Navigation tabs (Files, Directories, Uncovered Areas)
- File list with coverage indicators

### Summary Statistics Display

```text
┌─────────────────────────────────────────┐
│  Coverage: 79.4%                        │
│  ████████░ (4,156 / 5,234 lines)       │
├─────────────────────────────────────────┤
│ Files:     45      Covered:  42         │
│ Avg File:  117 lines        75.2%       │
└─────────────────────────────────────────┘
```text

#### 3.2 Per-File Report Page

### Features

- Source code with syntax highlighting
- Line numbers and coverage indicators
- Function/class navigation sidebar
- Inline coverage statistics
- Copy-to-clipboard functionality for code snippets
- Line-level tooltips showing coverage status

### Coverage Indicators

```text
  1 | fn main():                           # Fully covered
  2 |     let x = 5                        # Fully covered
  3 |     if x > 0:                        # Fully covered
  4 |         print("positive")            # NOT covered (red background)
  5 |     else:                            # Fully covered
  6 |         print("negative")            # Fully covered
  7 |
  8 | fn helper():                         # NOT covered (red background)
  9 |     return 42
```text

### Color Coding

- Green line numbers/backgrounds: Line executed in tests
- Red line numbers/backgrounds: Line NOT executed in tests
- Yellow backgrounds: Partially executed (branch coverage)
- Gray: Non-executable lines (comments, blank lines)

#### 3.3 Navigation Features

### Breadcrumb Navigation

- Show current file path in breadcrumb
- Enable quick navigation to parent directories
- Link to main index

### Sidebar Navigation

- File tree showing directory structure
- Filter by coverage threshold
- Sort options (by name, by coverage %)
- Expandable/collapsible file groups

### Search Features

- Quick file search with autocomplete
- Filter by coverage percentage
- Filter by file type/extension
- Show only uncovered files

### 4. Coverage Calculation Methodology

#### 4.1 Line Coverage

**Definition**: Percentage of executable lines that were executed during testing.

### Formula

```text
Line Coverage % = (Covered Lines / Total Executable Lines) × 100
```text

### Calculation Rules

- Count only executable lines (exclude comments, blank lines, decorators)
- Include all lines with code statements
- Track execution at bytecode instruction level
- Handle conditional branches (counted as covered if any branch executed)

#### 4.2 File Coverage

**Definition**: Coverage percentage for a specific source file.

### Calculation

```text
File Coverage % = (Covered Lines in File / Total Executable Lines in File) × 100
```text

### Aggregation

- Weight by file size to identify problem areas
- Track both absolute coverage % and line counts
- Support sorting by either metric

#### 4.3 Overall Coverage

**Definition**: Project-wide coverage percentage.

### Calculation

```text
Overall Coverage % = (Total Covered Lines / Total Executable Lines) × 100
```text

### Scope

- Include all source code files (exclude test files from coverage calculation)
- Exclude generated code (protobuf, etc.)
- Include vendored dependencies based on configuration

#### 4.4 Directory-Level Coverage

**Definition**: Coverage percentage for all files within a directory.

### Calculation

```text
Directory Coverage % = (Sum of Covered Lines in Directory / Sum of Total Lines in Directory) × 100
```text

### Recursive Calculation

- Calculate for each directory level
- Show summary statistics per directory
- Enable drill-down from directory to individual files

### 5. File Sorting and Prioritization

#### 5.1 Default Sort Orders

### By Coverage Percentage (Ascending)

- Purpose: Identify highest-priority areas for additional testing
- Shows files most needing test coverage first
- Example: `[0%, 5%, 12%, 42%, 78%, 95%]`

### By File Size

- Purpose: Identify largest untested code areas
- Sort by absolute line count of uncovered code
- Example: `[1078 uncovered, 556 uncovered, 234 uncovered, ...]`

### Alphabetically

- Purpose: Quick navigation by file name
- Secondary sort when coverage is equal

#### 5.2 Problem Area Highlighting

### Coverage Threshold Rules

- **Red**: Coverage < 50% (critical - needs testing)
- **Yellow**: Coverage 50-75% (warning - improve coverage)
- **Green**: Coverage > 75% (good - maintain level)

### Summary Metrics

- Top 10 files needing coverage
- Total uncovered lines across project
- Files with zero coverage (priority list)
- Coverage trend (if historical data available)

### 6. Report Configuration

#### 6.1 Configuration File Format

**Location**: `pyproject.toml` or dedicated `coverage_config.toml`

### Schema

```toml
[tool.coverage.report]
# Output directory for generated reports
output_dir = "htmlcov"

# Report formats to generate
formats = ["html", "text", "json"]

# HTML report options
[tool.coverage.report.html]
title = "ML Odyssey - Test Coverage Report"
show_contexts = true
skip_covered = false
sort_by = "coverage"  # "coverage", "name", "lines"

# Include/exclude patterns
[tool.coverage.report.filters]
include = [
    "src/",
    "lib/",
]
exclude = [
    "tests/",
    "scripts/",
    "build/",
    "*_pb2.py",  # Generated protobuf files
]

# Thresholds for pass/fail
[tool.coverage.report.thresholds]
minimum = 70  # Fail if overall coverage below this %
per_file_minimum = 50  # Warn if any file below this %

# Text report options
[tool.coverage.report.text]
skip_empty = true
skip_covered = false
precision = 2  # Decimal places for percentages
```text

#### 6.2 Command-Line Options

### Report Generation Command

```bash
# Generate all configured reports
coverage-report generate

# Generate specific format only
coverage-report generate --format html
coverage-report generate --format text
coverage-report generate --format json

# Custom output directory
coverage-report generate --output /tmp/reports

# Set minimum coverage threshold
coverage-report generate --min-coverage 80

# Sort by coverage percentage
coverage-report generate --sort-by coverage

# Exclude patterns
coverage-report generate --exclude "tests/*,build/*"
```text

### 7. Integration with Coverage Analysis (Issue #831)

#### 7.1 Data Flow

```text
Coverage Analysis (Issue #831)
    ↓
Coverage Data Files
    (.coverage, coverage.xml, or JSON)
    ↓
Report Generator (Issue #834)
    ↓
HTML, Text, JSON Reports
    ↓
CI/CD Pipeline / Developer Tools
```text

#### 7.2 Data Format Compatibility

### Supported Input Formats

- `.coverage` binary format (pytest-cov default)
- `coverage.xml` (XML format for interoperability)
- `coverage.json` (JSON for programmatic access)
- Custom JSON schema (Issue #831 specification)

### Format Detection

- Auto-detect input format based on file extension
- Support explicit format specification via CLI
- Validate data before processing
- Provide clear error messages for incompatible formats

#### 7.3 Error Handling

### Data Validation

- Check coverage data file exists and is readable
- Validate coverage data integrity
- Handle missing source files gracefully
- Report warnings for incomplete coverage data

### Fallback Handling

- Generate reports even if some files missing coverage data
- Mark missing files with "No data" indicator
- Continue processing remaining files on errors
- Log warnings for investigation

### 8. Performance Considerations

#### 8.1 Report Generation Performance

### Targets

- HTML report generation: < 5 seconds for typical project
- Text report generation: < 1 second
- JSON report generation: < 2 seconds

### Optimization Strategies

- Stream large HTML files to avoid memory overhead
- Cache parsed source code
- Parallel file processing (if applicable)
- Lazy loading in HTML report (load files on demand)

#### 8.2 HTML Report Size

### Target Sizes

- HTML report for typical project (50 files): < 10MB
- Index page: < 500KB
- Per-file report pages: < 200KB average

### Optimization Techniques

- Minify CSS and JavaScript
- Use relative paths for file references
- Compress common assets
- Remove redundant data from HTML files

### 9. API Contracts

#### 9.1 Report Generator Interface

### Base Generator Class

```python
class CoverageReportGenerator:
    """Base class for coverage report generators."""

    def __init__(self, coverage_data: CoverageData, config: ReportConfig):
        """Initialize generator with coverage data and configuration."""
        pass

    def generate(self, output_dir: Path) -> ReportOutput:
        """Generate coverage report.

        Returns:
            ReportOutput with paths to generated files and summary stats
        """
        pass

    def validate_data(self) -> bool:
        """Validate coverage data before generation."""
        pass
```text

#### 9.2 HTML Report Generator

```python
class HTMLReportGenerator(CoverageReportGenerator):
    """Generate interactive HTML coverage reports."""

    def generate_index(self) -> Path:
        """Generate main index.html with statistics."""
        pass

    def generate_file_report(self, file_path: str) -> Path:
        """Generate per-file HTML report."""
        pass

    def generate_assets(self) -> List[Path]:
        """Generate CSS, JavaScript, and other assets."""
        pass
```text

#### 9.3 Text Report Generator

```python
class TextReportGenerator(CoverageReportGenerator):
    """Generate text summary reports."""

    def generate_summary(self) -> str:
        """Generate text summary of coverage."""
        pass

    def generate_detailed(self) -> str:
        """Generate detailed per-file coverage report."""
        pass
```text

#### 9.4 JSON Report Generator

```python
class JSONReportGenerator(CoverageReportGenerator):
    """Generate machine-readable JSON reports."""

    def generate(self, output_dir: Path) -> Path:
        """Generate JSON coverage report."""
        pass

    def to_dict(self) -> Dict:
        """Convert coverage data to dictionary."""
        pass
```text

### 10. Validation and Quality Assurance

#### 10.1 Report Validation

### Validation Checks

- All uncovered line numbers are within file bounds
- Coverage percentages are between 0-100%
- Total lines match sum of covered + uncovered
- No duplicate entries in file listings
- All referenced source files are readable

#### 10.2 Output Verification

### Verification Steps

- HTML reports are valid, well-formed HTML
- CSS and JavaScript assets are properly linked
- All file paths are correctly computed
- Statistics calculations are accurate
- Navigation links are functional

#### 10.3 Consistency Checks

### Cross-Format Consistency

- Text and HTML reports show same statistics
- JSON data matches HTML/text reports
- Directory totals match sum of file coverage
- Overall coverage matches aggregation

### 11. Documentation Requirements

#### 11.1 User Documentation

### Content

- How to generate coverage reports
- Interpreting report statistics
- Understanding color coding
- Using HTML report navigation
- Filtering and searching reports
- Integration with CI/CD pipelines

#### 11.2 API Documentation

### Content

- Report generator class interfaces
- Configuration options and schema
- Input data format specifications
- Output format specifications
- Error handling and validation
- Performance characteristics

#### 11.3 Developer Guide

### Content

- Architecture and design decisions
- Extending report generators
- Adding new report formats
- Customizing styling and templates
- Performance optimization guide

## Next Steps

After this planning phase is complete:

1. **Issue #835**: Create test suite to validate coverage report generation
1. **Issue #836**: Implement the coverage report generation system
1. **Issue #837**: Integrate with CI/CD pipelines and packaging
1. **Issue #838**: Cleanup, finalization, and performance optimization
