# Issue #754: [Plan] Validate Output - Design and Documentation

## Objective

Design and document the validation logic to verify that generated paper structures are complete and correct. This ensures all required directories and files exist, are properly formatted, and follow repository conventions.

## Deliverables

- **Validation report** with pass/fail status
- **List of missing or invalid items** with specific details
- **Suggestions for fixing issues** with actionable guidance
- **Comprehensive design documentation** in this README

## Success Criteria

- [ ] Validation detects missing directories
- [ ] Validation detects missing files
- [ ] Validation checks file format and content
- [ ] Clear report shows any issues found
- [ ] Design documentation is complete and comprehensive
- [ ] API contracts and interfaces are clearly defined

## Design Decisions

### Architecture

**Validation Strategy**: Implement a multi-level validation approach that checks:

1. **Structure validation** - Directory and file existence
1. **Content validation** - File format and syntax correctness
1. **Convention validation** - Repository standards compliance

**Rationale**: Separating validation into distinct levels allows for:

- Clear error categorization
- Progressive validation (fail fast on structure, then check content)
- Easier maintenance and extension
- Better error reporting

### API Design

### Core Interface

```mojo
struct ValidationReport:
    """Report containing validation results."""
    var status: ValidationStatus  # PASS, FAIL, WARNING
    var missing_directories: List[String]
    var missing_files: List[String]
    var invalid_files: List[FileValidationError]
    var suggestions: List[String]

fn validate_paper_structure(
    paper_path: Path,
    validation_rules: ValidationRules
) raises -> ValidationReport:
    """
    Validate a generated paper structure.

    Args:
        paper_path: Path to the generated paper directory
        validation_rules: Rules and checklist for validation

    Returns:
        ValidationReport with all validation results

    Raises:
        IOError: If paper_path doesn't exist or isn't accessible
    """
    pass
```text

**Rationale**: This API design provides:

- Clear input/output contracts
- Structured error reporting
- Extensible validation rules
- Type-safe error handling with Mojo's `raises`

### Validation Rules

### Required Directories

- `/papers/<paper-name>/` - Root paper directory
- `/papers/<paper-name>/src/` - Source code
- `/papers/<paper-name>/tests/` - Test files
- `/papers/<paper-name>/docs/` - Documentation

### Required Files

- `/papers/<paper-name>/README.md` - Paper overview
- `/papers/<paper-name>/plan.md` - Implementation plan
- `/papers/<paper-name>/pyproject.toml` - Project configuration

### Content Validation

- YAML files: Valid YAML syntax
- Markdown files: Basic structure (headers, no broken links)
- TOML files: Valid TOML syntax
- Mojo files: Basic syntax check (if present)

### Convention Validation

- File names follow kebab-case
- README.md contains required sections
- plan.md follows Template 1 format (9 sections)

**Rationale**: These rules ensure:

- Consistency across all generated papers
- Immediate detection of generation failures
- Compliance with repository standards
- Actionable feedback for corrections

### Error Reporting

### Error Message Format

```text
VALIDATION FAILED

Missing Directories (2):
  - /papers/lenet-5/tests/
  - /papers/lenet-5/docs/

Missing Files (1):
  - /papers/lenet-5/README.md

Invalid Files (1):
  - /papers/lenet-5/pyproject.toml
    Error: Invalid TOML syntax at line 5
    Suggestion: Check for missing quotes or brackets

Suggestions:
  - Run: mkdir -p /papers/lenet-5/tests /papers/lenet-5/docs
  - Create README.md using template: scripts/templates/paper_README.md
  - Fix TOML syntax: remove trailing comma at line 5
```text

**Rationale**: Clear, actionable error messages that:

- Group errors by category
- Show exact locations
- Provide specific fix suggestions
- Include runnable commands when possible

### Alternatives Considered

### Alternative 1: Simple boolean validation

- Just return True/False
- ❌ Rejected: Doesn't provide enough information for debugging
- ❌ No actionable guidance for users

**Alternative 2: Schema-based validation (JSON Schema, Pydantic)**

- Use formal schema definitions
- ❌ Rejected: Over-engineered for current needs
- ❌ Adds unnecessary dependencies
- ✅ Could revisit if validation becomes complex

### Alternative 3: Linter-style validation

- Validate everything, report all issues with severity levels
- ⚠️ Considered but simplified: Current approach uses this concept but keeps it lightweight
- ✅ May expand later if needed

### Implementation Approach

### Phase 1: Structure Validation

1. Check directory existence using Path.exists()
1. Check file existence for required files
1. Build list of missing items

### Phase 2: Content Validation

1. Read each file
1. Parse based on file type (YAML, TOML, Markdown)
1. Validate syntax and structure
1. Record errors with line numbers

### Phase 3: Convention Validation

1. Check naming conventions
1. Validate README.md structure
1. Validate plan.md format (if present)
1. Check for common issues

### Phase 4: Report Generation

1. Aggregate all findings
1. Generate suggestions based on errors
1. Format report for readability
1. Return structured ValidationReport

**Rationale**: Phased approach allows:

- Early exit on critical failures
- Progressive validation
- Clear separation of concerns
- Easier testing of each phase

## References

- **Source Plan**: [/home/mvillmow/ml-odyssey-manual/notes/plan/03-tooling/01-paper-scaffolding/02-directory-generator/03-validate-output/plan.md](notes/plan/03-tooling/01-paper-scaffolding/02-directory-generator/03-validate-output/plan.md)
- **Parent Component**: Directory Generator
- **5-Phase Workflow**: [/home/mvillmow/ml-odyssey-manual/notes/review/README.md](notes/review/README.md)
- **Mojo Language Guidelines**: [/home/mvillmow/ml-odyssey-manual/.claude/agents/mojo-language-review-specialist.md](.claude/agents/mojo-language-review-specialist.md)

### Related Issues

- **#755**: [Test] Validate Output - Write Tests
- **#756**: [Impl] Validate Output - Implementation
- **#757**: [Package] Validate Output - Integration and Packaging
- **#758**: [Cleanup] Validate Output - Refactor and Finalize

## Implementation Notes

*(This section will be filled as implementation progresses)*

### Key Considerations

1. **Keep validation simple** - Focus on essential checks, avoid over-engineering
1. **Provide actionable errors** - Every error message should tell users exactly what's wrong and how to fix it
1. **Never modify files** - Validation is read-only, never auto-fix
1. **Performance** - Validation should be fast (< 1 second for typical paper structure)
1. **Extensibility** - Design should allow adding new validation rules easily

### Open Questions

- Should validation be configurable (allow skipping certain checks)?
- Should we validate paper metadata (author, date, etc.)?
- How to handle optional directories/files?

### Testing Strategy

Tests should cover:

- Valid paper structure (all checks pass)
- Missing directories
- Missing files
- Invalid file content (syntax errors)
- Edge cases (empty files, special characters in names)
- Performance (large directory structures)
