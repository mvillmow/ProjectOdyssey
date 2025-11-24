# Issue #814: [Plan] Validate Structure - Design and Documentation

## Objective

Design and document a comprehensive specification for a structure validation system that ensures paper directory organization and files meet repository requirements. This planning phase will establish the validation architecture, rules, error reporting format, and integration points with the paper implementation workflow.

## Deliverables

### Design Documentation

- **Validation Architecture Document** - Overall design approach, component responsibilities, and data flow
- **Structure Requirements Specification** - Complete definition of required directories and files for papers
- **Validation Rules Document** - Specific rules for each validation check
- **Error Reporting Format** - Standardized validation output and suggestions
- **Integration Points Document** - How validation integrates with CI/CD and paper workflow

### Technical Specifications

- **Validation Rules Set** - Comprehensive list of all checks (directory, file, naming, structure)
- **File Naming Conventions** - Standards for test files, source files, documentation files
- **Directory Structure Template** - Canonical directory layout for new papers
- **Report Schema** - Format for validation results (pass/fail, missing items, suggestions)

### API Specifications

- **Validation Module Interface** - Public functions and their contracts
- **Error Codes and Messages** - Standardized error messages and recovery suggestions
- **Configuration Schema** - Validation rules can be configured per paper type

### Documentation

- **Quick Start Guide** - How to run validation checks
- **Validation Rules Reference** - Detailed explanation of each validation rule
- **Example Validation Reports** - Sample output for both passing and failing validations
- **Integration Guide** - How to integrate validation in CI/CD pipelines

## Success Criteria

### Planning Complete

- [x] Validation system architecture designed
- [x] All required directories documented (src/, tests/, docs/)
- [x] File naming conventions specified
- [x] Validation rules formally specified (15+ specific checks)
- [x] Error codes and recovery suggestions defined
- [x] Integration points with CI/CD clarified
- [x] Validation report format specified
- [x] Downstream work identified and documented

### Documentation Complete

- [x] Comprehensive design document created
- [x] Validation rules reference guide written
- [x] API contracts documented
- [x] Example validation reports provided
- [x] Integration guide for CI/CD created
- [x] File naming conventions clearly specified

## References

- [Paper Directory Structure Design](../../../../../../../notes/review/paper-directory-structure.md)
- [CI/CD Integration Strategy](../../../../../../../notes/review/ci-cd-strategy.md)
- [Validation Architecture Review](../../../../../../../notes/review/validation-architecture.md)
- [Downstream Specifications (Validation Rules)](../../../../../../../notes/issues/814/validation-rules.md)
- [Example Validation Reports](../../../../../../../notes/issues/814/example-reports.md)

## Implementation Notes

### Validation System Architecture

The validation system is designed as a **modular, extensible validation framework** with the following components:

#### 1. Core Validation Engine

**Responsibility**: Orchestrate validation checks and aggregate results

### Interface

```mojo
fn validate_paper_structure(
    paper_path: String,
    config: ValidationConfig
) -> ValidationReport
```text

**Output**: `ValidationReport` containing:

- Overall pass/fail status
- List of validation checks with results
- List of missing items with locations
- Recovery suggestions for each failure

#### 2. Validation Check Modules

Organized by category with clear responsibility separation:

**Directory Checks** (`validate_directories()`):

- Check for required directories (src/, tests/, docs/)
- Check for proper nesting and organization
- Identify unexpected directories at root level

**File Checks** (`validate_files()`):

- Verify required files exist (README.md, etc.)
- Check file accessibility and readability
- Validate file completeness (non-empty core files)

**Naming Convention Checks** (`validate_naming()`):

- Verify test file naming patterns (`test_*.mojo`, `*_test.mojo`)
- Check source file naming conventions
- Validate documentation file naming

**Structure Integrity Checks** (`validate_structure_integrity()`):

- Verify proper file organization within directories
- Check for orphaned or misplaced files
- Validate cross-directory references

**Metadata Checks** (`validate_metadata()`):

- Verify paper name consistency (directory name vs metadata)
- Check for required metadata in README.md
- Validate author and date information

#### 3. Validation Rules Registry

**Design**: Pluggable rule system allowing configuration-driven validation

### Components

- Rule definitions (name, description, severity, check function)
- Rule groupings (directory, file, naming, metadata, structure)
- Severity levels (ERROR, WARNING, INFO)
- Customization per paper type

### Example Rule Definition

```yaml
- rule_id: REQUIRED_DIR_SRC
  name: "Required source directory exists"
  category: directory
  severity: ERROR
  description: "Paper must have src/ directory for implementation"
  fix_suggestion: "Create src/ directory at paper root"
```text

#### 4. Report Generator

**Responsibility**: Format validation results for various audiences

### Report Styles

- **Summary Report** - Quick pass/fail with critical items only
- **Detailed Report** - Full results with all details and suggestions
- **JSON Report** - Machine-readable format for CI/CD integration
- **Fix Checklist** - Actionable steps to resolve failures

### Required Directory Structure Specification

#### Standard Paper Structure (Full Requirements)

```text
papers/<paper-name>/
├── README.md                    # Paper overview and setup instructions
├── src/                         # Implementation source code
│   ├── model.mojo              # Model implementation (required)
│   ├── layers/                 # Layer implementations (if needed)
│   │   └── *.mojo
│   ├── loss.mojo               # Loss functions (if custom)
│   ├── optimizer.mojo          # Custom optimizers (if needed)
│   └── utils.mojo              # Helper functions
├── tests/                       # Test suite
│   ├── conftest.mojo           # Test configuration
│   ├── test_model.mojo         # Model tests (required)
│   ├── test_layers.mojo        # Layer tests (optional)
│   ├── test_loss.mojo          # Loss function tests (optional)
│   ├── test_training.mojo      # Training loop tests (optional)
│   └── fixtures/               # Test data and fixtures
│       ├── sample_data.mojo
│       └── expected_outputs.mojo
├── docs/                       # Documentation
│   ├── architecture.md         # Model architecture documentation
│   ├── training.md             # Training procedure documentation
│   ├── results.md              # Experimental results
│   ├── images/                 # Figures and diagrams
│   └── references.md           # Paper references
├── examples/                   # Example usage scripts (optional)
│   ├── train.mojo             # Training example
│   └── inference.mojo         # Inference example
├── configs/                    # Configuration files (if using config system)
│   ├── model.yaml
│   ├── training.yaml
│   └── data.yaml
└── .gitignore                 # Git ignore rules for paper-specific artifacts
```text

#### Minimal Paper Structure (Minimum Valid)

```text
papers/<paper-name>/
├── README.md                    # Must exist, must not be empty
├── src/
│   └── model.mojo              # Must exist, must not be empty
└── tests/
    └── test_model.mojo         # Must exist, must not be empty
```text

### Validation Rules Specification

#### Rule Categories and Checks

**Category: Required Directories** (Severity: ERROR)

1. **REQUIRED_DIR_SRC**
   - Check: `src/` directory exists
   - Message: "Paper must have `src/` directory containing implementation"
   - Fix: Create `src/` directory at paper root

1. **REQUIRED_DIR_TESTS**
   - Check: `tests/` directory exists
   - Message: "Paper must have `tests/` directory containing test suite"
   - Fix: Create `tests/` directory at paper root

1. **REQUIRED_DIR_DOCS**
   - Check: `docs/` directory exists
   - Message: "Paper should have `docs/` directory for documentation"
   - Fix: Create `docs/` directory at paper root

**Category: Required Files** (Severity: ERROR/WARNING)

1. **REQUIRED_FILE_README**
   - Check: `README.md` exists at paper root
   - Message: "Paper must have README.md at root level"
   - Fix: Create README.md with paper overview and setup instructions

1. **REQUIRED_FILE_MODEL**
   - Check: `src/model.mojo` exists and is not empty
   - Message: "Paper must have model implementation in src/model.mojo"
   - Fix: Create model.mojo with model class implementation

1. **REQUIRED_FILE_TEST_MODEL**
   - Check: `tests/test_model.mojo` exists and is not empty
   - Message: "Paper must have test_model.mojo with model tests"
   - Fix: Create test_model.mojo with test functions

**Category: File Naming Conventions** (Severity: WARNING)

1. **TEST_FILE_PREFIX_NAMING**
   - Check: All test files in `tests/` start with `test_` or end with `_test.mojo`
   - Message: "Test file '{filename}' doesn't follow naming convention (should be test_*.mojo or *_test.mojo)"
   - Fix: Rename file to match convention

1. **SOURCE_FILE_NAMING**
   - Check: Source files in `src/` follow snake_case naming
   - Message: "Source file '{filename}' should use snake_case naming (e.g., model_utils.mojo not modelUtils.mojo)"
   - Fix: Rename file to snake_case

1. **DOC_FILE_NAMING**
   - Check: Documentation files in `docs/` follow snake_case naming
   - Message: "Documentation file '{filename}' should use snake_case naming"
   - Fix: Rename file to snake_case

**Category: Directory Organization** (Severity: WARNING)

1. **NO_MOJO_IN_ROOT**
    - Check: No `.mojo` files exist at paper root level
    - Message: "Mojo files should be in src/ directory, not at paper root: {filenames}"
    - Fix: Move files to appropriate subdirectories (src/, tests/, etc.)

1. **TEST_FILES_IN_TEST_DIR**
    - Check: All test files are in `tests/` directory
    - Message: "Test files found outside tests/ directory: {paths}"
    - Fix: Move test files to tests/ directory

1. **README_IN_SUBDIRS**
    - Check: README.md files exist in src/, tests/, docs/ if those dirs have significant content
    - Message: "Directory {dirname}/ should have README.md documenting its contents"
    - Fix: Create README.md in subdirectory

**Category: File Content Validation** (Severity: ERROR)

1. **README_NOT_EMPTY**
    - Check: README.md is not empty
    - Message: "README.md exists but is empty or contains only whitespace"
    - Fix: Add content describing paper, setup instructions, and usage

1. **SOURCE_NOT_EMPTY**
    - Check: Core source files (model.mojo) are not empty
    - Message: "Source file {filename} is empty or contains only whitespace"
    - Fix: Implement the module or remove the empty file

1. **TESTS_NOT_EMPTY**
    - Check: Test files are not empty
    - Message: "Test file {filename} is empty or contains only whitespace"
    - Fix: Add test cases or remove the empty test file

**Category: Metadata Validation** (Severity: WARNING)

1. **README_HAS_TITLE**
    - Check: README.md starts with a heading (# Title)
    - Message: "README.md should start with paper title (# Paper Name)"
    - Fix: Add heading at beginning of README.md

1. **README_HAS_SETUP_SECTION**
    - Check: README.md contains setup/installation instructions
    - Message: "README.md should have Setup or Installation section"
    - Fix: Add section documenting how to set up and run the paper

1. **README_HAS_USAGE_SECTION**
    - Check: README.md contains usage examples or instructions
    - Message: "README.md should have Usage or Examples section"
    - Fix: Add section showing how to use the implementation

**Category: Structure Integrity** (Severity: INFO/WARNING)

1. **CONSISTENT_MODULE_NAMES**
    - Check: Test files match their corresponding source modules
    - Message: "Found test file test_foo.mojo but no src/foo.mojo"
    - Fix: Create matching source module or rename test file

1. **NO_CIRCULAR_IMPORTS**
    - Check: Source files don't create circular import dependencies
    - Message: "Circular import detected: {cycle}"
    - Fix: Restructure imports to remove cycles

### Validation Rules Configuration

Validation rules can be configured via a `validation.yaml` file in the paper directory:

```yaml
validation:
  # Which rule categories to enforce
  categories:
    - required_directories
    - required_files
    - naming_conventions
    - file_content

  # Rules to skip
  skip_rules:
    - REQUIRED_DIR_DOCS    # Optional for simple papers
    - README_HAS_USAGE_SECTION

  # Custom severity overrides
  severity_overrides:
    TEST_FILE_PREFIX_NAMING: WARNING  # Default is WARNING, keep as is

  # Minimum file size thresholds (in bytes)
  min_file_sizes:
    readme: 100
    source: 50
    test: 100
```text

### Error Reporting Format

#### Validation Report Structure

```text
ValidationReport {
  overall_status: PASS | FAIL | WARNING
  paper_path: String
  validation_timestamp: DateTime
  summary: {
    total_checks: Int
    passed: Int
    failed: Int
    warnings: Int
  }
  checks: [ValidationCheck]
  missing_items: [MissingItem]
  suggestions: [Suggestion]
  metadata: {
    validation_version: String
    config_file_used: String
  }
}
```text

#### Validation Check Result

```text
ValidationCheck {
  rule_id: String
  rule_name: String
  status: PASS | FAIL | WARNING
  severity: ERROR | WARNING | INFO
  description: String
  affected_items: [String]
}
```text

#### Missing Item Description

```text
MissingItem {
  type: DIRECTORY | FILE | METADATA
  name: String
  path: String
  required: bool
  suggestion: String
}
```text

#### Human-Readable Report Example

```text
Paper Structure Validation Report
===================================
Paper: papers/lenet5
Status: PASS ✓

Summary:
- Total checks: 20
- Passed: 20
- Failed: 0
- Warnings: 0

Structure Validation:
✓ Required directory 'src/' exists
✓ Required directory 'tests/' exists
✓ Required directory 'docs/' exists
✓ File 'README.md' exists and is not empty
✓ File 'src/model.mojo' exists and is not empty
✓ File 'tests/test_model.mojo' exists and is not empty
✓ All test files follow naming convention (test_*.mojo)
✓ No Mojo files at paper root level
✓ All Mojo files properly organized
✓ README.md has proper title (# ...)
✓ README.md has setup/installation section
✓ README.md has usage/examples section

Recommendations:
- Consider adding docs/architecture.md documenting the model design
- Consider adding docs/training.md documenting training procedure

Validation completed at: 2025-11-16T10:30:00Z
```text

### Integration Points

#### CI/CD Pipeline Integration

The validation system integrates with CI/CD at multiple points:

1. **Pre-commit Hook**
   - Run validation before allowing commits
   - Check only modified papers
   - Fail on ERROR severity, warn on WARNING

1. **Pull Request Checks**
   - Run on all papers modified in PR
   - Generate detailed HTML report as PR comment
   - Block merge if validation fails

1. **Paper Creation Workflow**
   - Run validation when new paper directory is created
   - Generate checklist for paper creator
   - Provide setup script to initialize structure

1. **Continuous Integration**
   - Run full validation on main branch after each merge
   - Generate validation report for all papers
   - Track validation metrics over time

#### Paper Implementation Workflow

The validation system guides the paper implementation process:

1. **Paper Initialization**
   - Create minimal directory structure
   - Generate README.md template
   - Provide validation checklist

1. **Development Phase**
   - Run validation after each significant change
   - Guide developers toward completing requirements
   - Provide helpful suggestions for missing pieces

1. **Submission Phase**
   - Final validation before paper is merged
   - Ensure all requirements met
   - Block merge if validation fails

1. **Maintenance Phase**
   - Periodic validation to catch regressions
   - Alert maintainers if structure degrades
   - Track paper health over time

### File Naming Conventions

#### Source Files (`src/` directory)

**Pattern**: `snake_case.mojo`

### Examples

- `model.mojo` - Main model implementation
- `data_loader.mojo` - Data loading utilities
- `training_loop.mojo` - Training logic
- `metrics.mojo` - Evaluation metrics
- `utils.mojo` - General utilities

### Anti-patterns to avoid

- `ModelClass.mojo` - Don't use PascalCase
- `model-file.mojo` - Don't use hyphens
- `model utils.mojo` - Don't use spaces
- `MODEL.mojo` - Don't use all caps

#### Test Files (`tests/` directory)

**Patterns** (either acceptable):

1. `test_*.mojo` (preferred)
   - `test_model.mojo`
   - `test_data_loader.mojo`
   - `test_training.mojo`

1. `*_test.mojo` (also acceptable)
   - `model_test.mojo`
   - `data_loader_test.mojo`
   - `training_test.mojo`

### Special files

- `conftest.mojo` - Pytest configuration and shared fixtures
- `fixtures/` - Directory for test data

### Anti-patterns

- `tests.mojo` - Should specify what's being tested
- `test.mojo` - Should be test_something.mojo
- `TestModel.mojo` - Should be test_model.mojo

#### Documentation Files (`docs/` directory)

**Pattern**: `snake_case.md` or descriptive names

### Standard files

- `architecture.md` - Model architecture and design
- `training.md` - Training procedure and hyperparameters
- `results.md` - Experimental results and analysis
- `references.md` - Paper references and sources

### Examples

- `layer_definitions.md`
- `custom_loss_functions.md`
- `data_preprocessing.md`

#### Configuration Files (`configs/` directory, if present)

**Pattern**: `snake_case.yaml` or `.toml`

### Standard files

- `model.yaml` - Model configuration
- `training.yaml` - Training configuration
- `data.yaml` - Data configuration

### Severity Levels

The validation system uses three severity levels:

- **ERROR**: Blocking issue that prevents the paper from being valid
  - Missing required directories or files
  - Empty required files
  - Critical naming violations

- **WARNING**: Issue that should be addressed but doesn't block validation
  - Suboptimal naming conventions
  - Missing optional documentation sections
  - Inconsistencies in structure

- **INFO**: Recommendations for improvement
  - Suggestions for additional documentation
  - Opportunities to add optional components
  - Best practice recommendations

### Downstream Work

This planning phase enables the following downstream work:

#### Issue #815 (Test) - Validation Testing

- Unit tests for each validation rule
- Integration tests for validation engine
- Test fixtures with various paper structures
- Edge case testing

#### Issue #816 (Implementation) - Validation System Implementation

- Core validation engine implementation
- Validation rule modules
- Report generator
- Error handling and messaging

#### Issue #817 (Integration) - CI/CD Integration

- Pre-commit hook integration
- GitHub Actions workflow integration
- Paper creation workflow integration
- Report generation and storage

#### Issue #818 (Cleanup) - Documentation and Polish

- Comprehensive validation guide
- Troubleshooting documentation
- Integration examples
- Performance optimization

## Status

✅ **COMPLETE** - Design and documentation phase completed. All specifications documented:

- Validation system architecture designed (5 core components identified)
- Required directory structure specified (full and minimal variants)
- 20+ validation rules formally documented with examples
- Error reporting format and samples defined
- Integration points with CI/CD documented
- File naming conventions fully specified
- Downstream work identified for Issues #815-818

Ready for Issue #815 (Testing) and Issue #816 (Implementation).
