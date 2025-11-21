# Issue #839: [Plan] Check Thresholds - Design and Documentation

## Objective

Design and document a comprehensive threshold validation system that checks test coverage percentages against configured minimum thresholds. This system ensures the ML Odyssey repository maintains adequate test coverage and prevents merging code with insufficient tests through automated CI/CD integration.

## Deliverables

- Threshold validation architecture and design specification
- Configuration schema and format definition
- API reference for threshold checking module
- Implementation guidelines and code structure
- Integration patterns with CI/CD systems
- Performance characteristics and optimization strategies
- Comprehensive design documentation in `/notes/review/`
- Issue-specific planning notes and decisions

## Success Criteria

- Comprehensive design specification document created
- Threshold configuration schema fully defined (per-file and overall targets)
- API contracts and interfaces documented
- CI/CD integration patterns documented
- Grace period mechanisms for new files specified
- Exit code requirements for CI/CD blocking defined
- Performance requirements and trade-offs analyzed
- All design decisions documented with rationale
- References to implementation phase issues established

## Inputs

- Coverage data formats (percentage values, per-file and overall metrics)
- Threshold configuration requirements (location, format, supported options)
- CI/CD integration requirements (exit codes, reporting)
- Project configuration structure from Issue #74 (Configs)
- Coverage tool outputs and formats
- Grace period requirements for new files
- Reporting format requirements

## Expected Outputs

- Threshold validation results (pass/fail status)
- Clear identification of files failing threshold requirements
- Actionable recommendations for coverage improvement
- CI/CD exit codes (0 for pass, non-zero for fail)
- Detailed violation reports with file-specific failures
- Measurement of coverage gaps vs. configured thresholds
- Performance metrics for threshold checking operations

## References

- [Issue #74: Configuration System](../../../../../../../notes/issues/74/README.md) - Configuration management foundation
- [Config Architecture](../../../../../../../notes/review/configs-architecture.md) - Comprehensive configuration system design
- [CI/CD Integration Patterns](../../../../../../../notes/review/orchestration-patterns.md) - Workflow integration guidelines
- [Skills Design](../../../../../../../notes/review/skills-design.md) - Agent skills for coverage analysis

## High-Level Design

### Architecture Overview

The threshold validation system consists of four primary components:

#### 1. Configuration Layer

- **Purpose**: Load and manage coverage threshold configuration
- **Sources**: Project configuration file (using Issue #74 infrastructure)
- **Format**: YAML/TOML with support for both overall and per-file thresholds
- **Scope**: Per-repository, per-section, and per-module configuration

#### 2. Validation Engine

- **Purpose**: Compare coverage data against configured thresholds
- **Inputs**: Coverage metrics (percentages, per-file data) and threshold configuration
- **Processing**: File-by-file and overall comparisons with grace period handling
- **Outputs**: Detailed validation results with violation detection

#### 3. Reporting System

- **Purpose**: Generate clear violation reports and improvement recommendations
- **Format**: Human-readable console output and machine-parseable structured format
- **Details**: File-level violations, coverage gaps, threshold requirements, recommendations
- **Integration**: Support for CI/CD systems and developer workflows

#### 4. CI/CD Integration

- **Purpose**: Enable automated blocking of insufficient coverage
- **Exit Codes**: 0 for pass, non-zero for fail (enables build failure)
- **Workflow Integration**: Runs after test execution, before merge gates
- **Configuration**: Determines whether violations block or warn

### Configuration Schema

#### Overall Threshold

```yaml
coverage:
  thresholds:
    overall:
      lines: 80.0          # Minimum line coverage percentage
      branches: 75.0       # Minimum branch coverage percentage
      functions: 85.0      # Minimum function coverage percentage
      statements: 80.0     # Minimum statement coverage percentage
```text

#### Per-File Thresholds

```yaml
coverage:
  thresholds:
    per_file:
      minimum: 70.0        # Minimum per-file coverage
      exclude_patterns:    # Patterns to exclude from per-file checks
        - "**/test_*.mojo"
        - "**/conftest.mojo"
        - "**/migrations/*"
```text

#### Grace Period Configuration

```yaml
coverage:
  thresholds:
    grace_period:
      enabled: true
      age_days: 7          # New files get grace period
      initial_minimum: 50.0  # Initial minimum for new files
```text

#### Blocking Configuration

```yaml
coverage:
  thresholds:
    blocking:
      enabled: true        # If false, report violations but don't fail CI
      strict_mode: true    # If true, fail on any violation
      allow_decrease: 0.5  # Allow up to 0.5% coverage decrease without review
```text

### API Design

#### Core Functions

**validate_coverage(coverage_data, threshold_config)**

- **Purpose**: Main validation entry point
- **Inputs**: Coverage metrics object, threshold configuration object
- **Returns**: ValidationResult object with detailed findings
- **Raises**: ConfigurationError for invalid configuration

**check_overall_coverage(coverage_data, thresholds)**

- **Purpose**: Validate overall project coverage metrics
- **Inputs**: Project-level coverage metrics, overall thresholds
- **Returns**: OverallResult with pass/fail status and gaps

**check_per_file_coverage(coverage_by_file, thresholds, exclude_patterns)**

- **Purpose**: Validate per-file coverage metrics
- **Inputs**: Per-file coverage map, per-file thresholds, exclusion patterns
- **Returns**: List of FileViolation objects

**apply_grace_period(file_path, grace_config)**

- **Purpose**: Determine if file is within grace period
- **Inputs**: File path and grace period configuration
- **Returns**: Boolean indicating grace period status

**generate_violation_report(validation_result)**

- **Purpose**: Create human-readable violation report
- **Inputs**: ValidationResult object
- **Returns**: Formatted report string with recommendations

### Data Structures

#### ValidationResult

```text
{
  status: 'pass' | 'fail',
  overall_violations: OverallViolation[],
  per_file_violations: FileViolation[],
  coverage_summary: CoverageSummary,
  recommendations: string[],
  exit_code: int
}
```text

#### OverallViolation

```text
{
  metric_type: 'lines' | 'branches' | 'functions' | 'statements',
  threshold: float,
  actual: float,
  gap: float,
  severity: 'critical' | 'major' | 'minor'
}
```text

#### FileViolation

```text
{
  file_path: string,
  metric_type: 'lines' | 'branches' | 'functions',
  threshold: float,
  actual: float,
  gap: float,
  in_grace_period: bool,
  affected_lines: int[]
}
```text

#### CoverageSummary

```text
{
  total_files_checked: int,
  files_with_violations: int,
  total_violations: int,
  coverage_metrics: {
    lines: float,
    branches: float,
    functions: float,
    statements: float
  }
}
```text

### Implementation Strategy

#### Phase 1: Core Validation Engine

1. Create ValidationEngine class
   - Load threshold configuration
   - Parse coverage data (multiple formats)
   - Compare metrics against thresholds
   - Detect violations

1. Implement metric comparison logic
   - File-by-file comparisons
   - Overall project metrics
   - Metric type handling (lines, branches, functions, statements)

1. Grace period mechanism
   - File age detection
   - Temporary threshold reduction for new files
   - Configurable grace period duration

#### Phase 2: Configuration Integration

1. Extend Issue #74 configuration system
   - Add coverage threshold section to project config
   - Support inheritance (default → paper → experiment)
   - Validate configuration schema

1. Configuration loading
   - Load from YAML/TOML files
   - Merge with defaults
   - Validate required fields

#### Phase 3: Reporting System

1. Violation report generation
   - Summary statistics
   - Per-file violations list
   - Coverage gap calculations
   - Improvement recommendations

1. Output formatting
   - Console output (human-readable)
   - JSON output (CI/CD integration)
   - HTML reports (detailed analysis)

#### Phase 4: CI/CD Integration

1. Exit code handling
   - Exit 0 on pass
   - Exit 1 on critical violations
   - Exit 2 on configuration errors

1. Workflow integration
   - GitHub Actions step configuration
   - Environment variable support
   - Parallel execution with other checks

### Configuration Location and Inheritance

Coverage thresholds follow the project's hierarchical configuration system (from Issue #74):

1. **Default Config** (`configs/default.yaml`)
   - Base thresholds for entire project
   - Applied to all papers and experiments

1. **Paper-Specific Config** (`configs/papers/<paper-name>/default.yaml`)
   - Override thresholds for specific paper implementations
   - Inherits defaults if not specified

1. **Experiment Config** (`configs/papers/<paper>/experiments/<exp-name>.yaml`)
   - Override thresholds for specific experiments
   - Inherits paper-level thresholds

### Grace Period Mechanism

New files (created within configurable period, e.g., 7 days) receive:

1. **Temporary Reduced Threshold**: Initial minimum (e.g., 50%) while ramping up tests
1. **Age-Based Exemption**: Files younger than configured age automatically pass per-file checks
1. **Milestone-Based Progression**: Increase thresholds as file matures
1. **Opt-In Early Enforcement**: Option to enforce full thresholds immediately for critical files

### CI/CD Integration Points

#### Integration Points

1. **After Test Execution**
   - Collect coverage data from pytest/coverage tools
   - Trigger threshold validation

1. **Before Merge Gates**
   - Block merge if violations detected (configurable)
   - Provide detailed feedback to developers

1. **Status Reporting**
   - Add checks to pull requests
   - Display coverage trends
   - Show violations preventing merge

#### Failure Handling

- **Strict Mode**: Any violation blocks CI (default)
- **Tolerant Mode**: Only critical violations block
- **Warning Mode**: Report violations but allow merge
- **Allow Decrease**: Permit small coverage decrease (e.g., 0.5%)

### Performance Considerations

1. **Validation Speed**
   - In-memory comparison of metrics (< 1ms typically)
   - Per-file validation scales linearly with file count
   - Grace period checks use cached file metadata

1. **Memory Usage**
   - Configuration loading: Minimal overhead
   - Coverage data: Depends on file count and metrics granularity
   - Result objects: Proportional to violation count

1. **Caching Strategy**
   - Cache configuration after initial load
   - Cache file metadata for grace period calculations
   - Reuse coverage data across multiple checks

### Error Handling and Edge Cases

#### Configuration Errors

- Missing required threshold values
- Invalid percentage values (not 0-100)
- Malformed configuration files
- Conflicting grace period settings

#### Coverage Data Issues

- Missing coverage metrics for files
- Inconsistent metric formats
- Coverage decreases from previous builds
- Partial coverage data

#### Grace Period Edge Cases

- File age calculation across time zones
- Rename/move of files
- Merge of branches with older files
- Coverage for deleted files in grace period

### Testing Strategy

1. **Unit Tests**
   - Threshold comparison logic
   - Grace period calculations
   - Configuration loading and validation
   - Report generation

1. **Integration Tests**
   - End-to-end validation workflow
   - Configuration inheritance from Issue #74
   - CI/CD exit code behavior
   - Grace period with real file system

1. **Performance Tests**
   - Validation speed with large project (1000+ files)
   - Configuration loading time
   - Memory usage during validation

1. **Regression Tests**
   - Coverage decrease detection
   - Threshold boundary conditions
   - Grace period transitions

## Implementation Phases

### Phase 1: Plan (Current - Issue #839)

- Complete design specification ✅
- Document architecture and API
- Define configuration schema
- Establish CI/CD integration patterns

### Phase 2: Test (Issue TBD)

- Write test specifications
- Create test fixtures with sample data
- Define test coverage targets (ideally 95%+)

### Phase 3: Implementation (Issue TBD)

- Implement validation engine
- Integrate with config system (Issue #74)
- Create reporting system
- Add CI/CD integration

### Phase 4: Package (Issue TBD)

- Package threshold checker module
- Create Python/Mojo wrapper
- Document installation and usage
- Add to project CI workflows

### Phase 5: Cleanup (Issue TBD)

- Refactor code based on implementation learnings
- Optimize performance
- Enhance error messages
- Complete documentation

## Design Decisions

### 1. Configuration Inheritance Over Duplication

**Decision**: Use hierarchical configuration system from Issue #74 instead of separate threshold config files.

### Rationale

- Eliminates duplication of threshold definitions
- Maintains single source of truth
- Supports paper-level and experiment-level overrides
- Aligns with project configuration philosophy

### 2. Grace Period Over Exemptions

**Decision**: Use temporary reduced thresholds for new files instead of complete exemptions.

### Rationale

- Ensures new code is tested from the start (TDD principle)
- Gradually enforces quality standards
- Prevents accumulation of untested code
- More practical than absolute exemptions

### 3. Configurable Blocking vs. Warnings

**Decision**: Support both strict (blocks merge) and tolerant (warns only) modes.

### Rationale

- Different projects have different risk tolerances
- Allows gradual enforcement for legacy code
- Supports critical vs. non-critical modules
- Enables iterative quality improvement

### 4. Per-File and Overall Thresholds

**Decision**: Support both per-file and overall project thresholds with separate configurations.

### Rationale

- Per-file ensures all code is tested
- Overall prevents coverage collapse in critical areas
- Different files may have different requirements
- Provides flexibility for large projects

### 5. Metric Flexibility

**Decision**: Support multiple coverage metrics (lines, branches, functions, statements) with independent thresholds.

### Rationale

- Different metrics catch different testing gaps
- Branch coverage is most comprehensive
- Line coverage is most common baseline
- Function coverage ensures all public APIs are tested

## Dependencies and Prerequisites

- **Issue #74**: Configuration system must be complete
  - Threshold config must be loadable through config system
  - Configuration inheritance must work correctly
  - Schema validation must be available

- **Coverage Tools**: Assumes pytest-cov or similar coverage tool installed
  - Must provide per-file coverage data
  - Must support JSON output for machine parsing
  - Must be integrated into CI workflow

- **CI/CD System**: GitHub Actions with Python/Mojo runtime
  - Must support exit code handling
  - Must provide coverage data to threshold checker
  - Must support status checks for merge gates

## Success Metrics

1. **Comprehensiveness**: All major components designed and specified
1. **Clarity**: Design is clear enough for engineers to implement
1. **Flexibility**: Supports various project configurations and requirements
1. **Alignment**: Integrates seamlessly with existing configuration system
1. **Documentation**: All design decisions documented with rationale
1. **Feasibility**: Design can be implemented in 1-2 weeks

## Implementation Notes

*To be filled in during implementation phases*

### Key Learnings

(Add notes here during implementation)

### Challenges Identified

(Document any challenges during implementation)

### Refinements Made

(Record design refinements discovered during implementation)

## Workflow

**Planning Phase**: Current issue (Issue #839)

### Requires

- Issue #74 (Configs) must be complete

### Can run in parallel with

- Other planning issues
- Unrelated implementation work

### Blocks

- Test phase (Issue TBD)
- Implementation phase (Issue TBD)
- Packaging phase (Issue TBD)

### Recommended sequence

1. Complete Issue #74 (Configs) ✅
1. Complete Issue #839 (Plan - Check Thresholds) ← Current
1. Create Issue TBD (Test - Check Thresholds)
1. Create Issue TBD (Impl - Check Thresholds)
1. Create Issue TBD (Package - Check Thresholds)
1. Create Issue TBD (Cleanup - Check Thresholds)

**Estimated Duration**: 1-2 days for comprehensive design documentation

## Related Issues

- [Issue #74: Configuration System](../../../../../../../notes/issues/74/README.md) - Foundation for threshold configuration
- Future testing phase issue (TBD)
- Future implementation phase issue (TBD)
- Future packaging phase issue (TBD)
- Future cleanup phase issue (TBD)

## Additional Context

### Motivation

Test coverage is a critical quality metric for the ML Odyssey project. As the codebase grows with implementations of multiple papers (LeNet, ResNet, Transformers, etc.), maintaining consistent test coverage becomes increasingly important. An automated threshold validation system:

1. **Prevents Coverage Decay**: Catches merges that reduce test coverage
1. **Enforces Quality Standards**: Requires minimum coverage before merge
1. **Provides Feedback**: Shows developers what coverage is missing
1. **Enables Gradual Improvement**: Grace periods allow ramping up coverage for existing code
1. **Supports CI/CD**: Integrates with GitHub Actions for automated enforcement

### Coverage Tools Integration

The system is designed to work with standard Python coverage tools:

- **pytest-cov**: Primary coverage tool for Mojo and Python tests
- **Coverage.py**: Underlying coverage engine
- **GitHub Actions**: Integration point for CI/CD

Coverage data is expected in the format provided by `coverage json` command:

```json
{
  "meta": {...},
  "files": {
    "src/module.mojo": {
      "summary": {
        "num_statements": 100,
        "num_missing": 10,
        "percent_covered": 90.0,
        "percent_covered_display": "90",
        "missing_lines": [...]
      },
      "executed_lines": [...],
      "missing_lines": [...]
    }
  }
}
```text

### Future Enhancements

Potential future extensions (out of scope for initial implementation):

1. **Coverage Trends**: Track coverage over time, visualize trends in dashboards
1. **Blame Attribution**: Link uncovered code to responsible developers/PRs
1. **Complexity-Weighted Coverage**: Higher thresholds for complex code
1. **Performance Profiling**: Integrate with performance metrics
1. **Mutation Testing**: Use mutation scores for more comprehensive coverage validation
1. **Coverage Badges**: Generate dynamic coverage badges for README
1. **Incremental Coverage**: Only check coverage for changed files

## Appendix: Coverage Metrics Reference

### Metric Definitions

1. **Line Coverage**: Percentage of executable lines executed during tests
   - Simplest metric, most commonly used
   - Baseline: typically 80-90%

1. **Branch Coverage**: Percentage of conditional branches executed
   - More comprehensive than line coverage
   - Accounts for if/else, loops, etc.
   - Typically 5-10% lower than line coverage

1. **Function Coverage**: Percentage of functions/methods called during tests
   - Ensures all public APIs are exercised
   - Can be 100% even if not all code paths covered
   - Good indicator of API completeness

1. **Statement Coverage**: Similar to line coverage but more precise
   - Counts logical statements vs. physical lines
   - More accurate for multi-statement lines
   - Usually very similar to line coverage

### Typical Coverage Targets

| Project Type | Target Line | Target Branch | Notes |
|---|---|---|---|
| Critical systems | 95%+ | 90%+ | High reliability requirements |
| Standard library | 85-90% | 80-85% | Balance coverage and productivity |
| Application code | 80-85% | 75-80% | Practical coverage level |
| Experimental code | 70-75% | 65-70% | Lower coverage while exploring |

