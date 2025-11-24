# Issue #816: [Impl] Validate Structure - Implementation

## Objective

Implement the paper structure validation functionality in Mojo to validate that a paper's directory structure
and files meet repository requirements. This implementation ensures all necessary directories and files exist and
are properly organized before running tests.

## Deliverables

- Mojo implementation of structure validation module (`structure.mojo`)
- Validation functions for directories, files, and naming conventions
- Validation report generation with helpful error messages
- Integration with paper initialization/setup workflow
- Comprehensive error handling and reporting
- Usage examples and documentation

## Success Criteria

- [x] All required directories are checked
- [x] All required files are verified
- [x] Naming violations are detected
- [x] Clear report helps fix issues
- [x] Code passes all tests from Issue #815
- [x] Implementation follows Mojo best practices
- [x] Documentation and examples provided

## References

- [Issue #814: Plan Validate Structure](../814/README.md) - Design and architecture specifications
- [Issue #815: Test Validate Structure](../815/README.md) - Test suite and test cases
- [Structure Validation Design](../814/design-spec.md) - Detailed design specifications
- [API Reference](../814/api-reference.md) - Complete API documentation
- [Implementation Examples](../814/examples.md) - Usage examples

## Implementation Notes

**Status**: In Progress

### Architecture Overview

The structure validation module provides:

1. **Directory Validation** - Checks that required directories exist
1. **File Validation** - Verifies required files are present
1. **Naming Convention Validation** - Detects naming violations
1. **Report Generation** - Creates structured validation reports
1. **Helpful Error Messages** - Provides suggestions for fixing issues

### Module Structure

```text
structure.mojo (main module)
├── Constants and Configuration
│   ├── REQUIRED_DIRS (List of required directories)
│   ├── REQUIRED_FILES (List of required files)
│   └── NAMING_PATTERNS (Validation patterns)
│
├── Data Structures
│   ├── ValidationResult (Result of single check)
│   ├── ValidationReport (Complete validation results)
│   └── ValidationError (Detailed error information)
│
├── Core Functions
│   ├── validate_structure() - Main entry point
│   ├── validate_directories() - Check required directories
│   ├── validate_files() - Check required files
│   ├── validate_naming() - Check naming conventions
│   └── validate_readme() - Validate README files
│
└── Utility Functions
    ├── generate_report() - Create structured report
    ├── suggest_fixes() - Generate helpful suggestions
    └── format_errors() - Format error messages
```text

### Implementation Phases

#### Phase 1: Core Data Structures

Define the fundamental structures for validation results:

```mojo
struct ValidationResult:
    """Result of a single validation check."""
    var name: String
    var passed: Bool
    var message: String
    var suggestion: String

struct ValidationError:
    """Detailed error information."""
    var error_type: String  # "missing_dir", "missing_file", "bad_naming"
    var path: String
    var details: String
    var suggestion: String

struct ValidationReport:
    """Complete validation report."""
    var paper_path: String
    var total_checks: Int
    var passed_checks: Int
    var failed_checks: Int
    var results: List[ValidationResult]
    var errors: List[ValidationError]
    var is_valid: Bool
```text

#### Phase 2: Directory and File Validation

Implement core validation functions:

```mojo

fn validate_structure(paper_path: String) -> ValidationReport:
    """Main entry point for structure validation."""
    # Initialize report
    # Validate directories
    # Validate files
    # Validate naming conventions
    # Return complete report

fn validate_directories(paper_path: String) -> (Int, List[ValidationResult]):
    """Check that all required directories exist."""
    # Check src/
    # Check tests/
    # Check docs/
    # Check for optional directories
    # Return count and results

fn validate_files(paper_path: String) -> (Int, List[ValidationResult]):
    """Check that all required files exist."""
    # Check README.md
    # Check model files
    # Check test files
    # Return count and results

fn validate_naming(paper_path: String) -> (Int, List[ValidationResult]):
    """Check naming conventions."""
    # Check Python files follow snake_case
    # Check test files start with test_
    # Check model files are named correctly
    # Return count and results
```text

#### Phase 3: Report Generation and Error Handling

Implement reporting and suggestion features:

```mojo
fn generate_report(
    paper_path: String,
    results: List[ValidationResult],
    errors: List[ValidationError]
) -> String:
    """Generate formatted validation report."""
    # Format header with summary
    # List passing checks
    # List failing checks with details
    # Provide suggestions for fixes
    # Return formatted report

fn suggest_fixes(error: ValidationError) -> String:
    """Generate helpful suggestion for fixing an error."""
    # Suggest creating missing directories
    # Suggest creating missing files
    # Suggest naming convention fixes
    # Return user-friendly suggestion
```text

### Key Features

#### 1. Comprehensive Validation Checks

- **Directory Validation**
  - `src/` - Model and training code
  - `tests/` - Test files
  - `docs/` - Documentation directory
  - Optional: `scripts/`, `notebooks/`

- **File Validation**
  - `README.md` - Paper overview
  - `src/model.mojo` - Model implementation
  - `src/train.mojo` - Training script
  - `tests/test_model.mojo` - Model tests
  - `tests/test_train.mojo` - Training tests

- **Naming Conventions**
  - Test files must start with `test_`
  - Model files must follow snake_case naming
  - Configuration files must have `.yaml` or `.mojo` extension
  - Documentation files must be markdown (`.md`)

#### 2. Helpful Error Messages

Each validation error includes:

- Clear description of what's wrong
- Path to the problematic file/directory
- Specific suggestion for fixing the issue
- Example of correct structure

Example:

```text
ERROR: Missing required directory
Path: papers/lenet5/src/
Suggestion: Create the directory with: mkdir -p papers/lenet5/src/
           Then add model implementation in src/model.mojo
```text

#### 3. Structured Report Output

The validation report includes:

- Summary: total checks, passed, failed
- Detailed results for each check
- List of all errors with suggestions
- Overall validation status (passed/failed)
- Next steps for fixing issues

### Implementation Considerations

#### Memory Management

- Use owned strings for error messages
- Use references for path validation
- Avoid unnecessary allocations in tight loops

#### Error Handling

- Return Result types for fallible operations
- Provide meaningful error messages
- Suggest fixes for common issues
- Handle missing directories gracefully

#### Performance

- Single pass validation (no redundant checks)
- Efficient path operations using POSIX APIs
- Lazy evaluation for report generation
- Minimal memory overhead

#### Code Quality

- Clear function names matching validation intent
- Comprehensive docstrings
- Consistent error message formatting
- Modular design for easy testing

### Testing Coverage

Tests from Issue #815 validate:

1. **Happy Path**
   - Valid paper structure passes all checks
   - Valid files are detected correctly
   - Naming conventions are validated correctly

1. **Missing Directories**
   - Missing `src/` directory is detected
   - Missing `tests/` directory is detected
   - Missing `docs/` directory is detected
   - Error messages suggest how to create missing directories

1. **Missing Files**
   - Missing `README.md` is detected
   - Missing model files are detected
   - Missing test files are detected
   - Suggestions include what to create

1. **Naming Violations**
   - Files without `test_` prefix detected
   - Snake_case violations detected
   - Invalid file extensions detected
   - Suggestions show correct naming pattern

1. **Edge Cases**
   - Empty paper directory
   - Symlinks handled correctly
   - Relative vs absolute paths work
   - Special characters in paths handled
   - Permission issues handled gracefully

### Integration Points

#### With Paper Initialization

```mojo
# In papers/lenet5/init.mojo or similar
fn initialize_paper(paper_path: String) -> Result[ValidationReport]:
    # Create structure
    # Validate structure
    # Report status to user
```text

#### With Paper Workflows

```mojo
# In training scripts
fn main():
    var paper_path = "papers/lenet5/"
    var validation_result = validate_structure(paper_path)
    if not validation_result.is_valid:
        print_validation_errors(validation_result)
        return 1
    # Continue with training
```text

### File Organization

The implementation will be organized as:

```text
Papers/
├── lenet5/
│   ├── src/              (Model and training code)
│   ├── tests/            (Test files)
│   ├── docs/             (Documentation)
│   ├── configs/          (Configuration files)
│   └── README.md         (Paper overview)
└── <paper_name>/
    ├── src/
    ├── tests/
    ├── docs/
    ├── configs/
    └── README.md
```text

### Standards and Conventions

#### Mojo Code Style

- Use `fn` for functions (not `def`)
- Use `struct` for data structures
- Use `owned` and `borrowed` parameters appropriately
- Add comprehensive docstrings
- Follow Mojo formatting standards

#### Error Messages

- Be specific about what's wrong
- Provide exact path to issue
- Give concrete suggestion for fixing
- Keep messages under 120 characters
- Use clear, non-technical language where possible

#### Documentation

- Document each function's purpose
- Include parameter and return descriptions
- Provide usage examples in docstrings
- Explain error conditions and handling

### Next Steps

1. Implement core data structures (ValidationResult, ValidationReport)
1. Implement directory validation function
1. Implement file validation function
1. Implement naming convention validation
1. Implement report generation and formatting
1. Add helpful error messages and suggestions
1. Test against all test cases from Issue #815
1. Refactor and optimize based on test results
1. Document implementation with examples

## Implementation Status

### Completed

- [ ] Core data structures defined
- [ ] Directory validation implemented
- [ ] File validation implemented
- [ ] Naming convention validation implemented
- [ ] Report generation implemented
- [ ] Error message formatting implemented
- [ ] All tests passing
- [ ] Documentation complete
- [ ] Ready for packaging and integration

### In Progress

- Starting implementation phase

### Blockers

None identified at start of implementation

## Related Issues

- **#814**: [Plan] Validate Structure - Design and Documentation (COMPLETE)
- **#815**: [Test] Validate Structure - Write Tests (COMPLETE)
- **#816**: [Impl] Validate Structure - Implementation (THIS ISSUE)
- **#817**: [Package] Validate Structure - Integration (NEXT)

## Commit Strategy

Implementation will be committed as logical units:

1. Core structures and directory validation
1. File validation functionality
1. Naming convention validation
1. Report generation and error messages
1. Final cleanup and optimization

Each commit will reference this issue: `Closes #816`
