# Issue #756: [Impl] Validate Output - Implementation

## Objective

Implement validation logic to verify that generated paper structures are complete and correct. This component ensures all required directories and files exist, are properly formatted, and follow repository conventions.

## Deliverables

- Validation function/module to check directory structure
- File existence verification logic
- Content validation for file format and syntax
- Report generation with pass/fail status and actionable error messages
- Integration with directory generator component

## Success Criteria

- [ ] Validation detects missing directories
- [ ] Validation detects missing files
- [ ] Validation checks file format and content
- [ ] Clear report shows any issues found
- [ ] All tests from issue #755 pass
- [ ] Code follows Mojo best practices and coding standards
- [ ] Code is clean, documented, and maintainable

## Design Decisions

### Architecture

**Validation Strategy**: Implement a multi-layer validation approach:

1. **Structural Validation**: Verify directory hierarchy matches expected structure
2. **File Validation**: Check all required files exist at correct paths
3. **Content Validation**: Parse and validate file contents (syntax, format)
4. **Report Generation**: Aggregate findings into actionable report

**Rationale**: Layered validation allows early detection of structural issues before expensive content parsing, and provides clear separation of concerns for testing and maintenance.

### API Design

**Function Signature** (Mojo):

```mojo
fn validate_paper_structure(
    root_path: Path,
    validation_rules: ValidationConfig
) raises -> ValidationReport:
    """
    Validate a generated paper structure.

    Args:
        root_path: Path to the paper directory root
        validation_rules: Configuration specifying required structure

    Returns:
        ValidationReport containing pass/fail status and details

    Raises:
        IOError: If root_path doesn't exist or is inaccessible
    """
    pass
```

**ValidationReport Structure**:

```mojo
struct ValidationReport:
    var passed: Bool
    var errors: List[ValidationError]
    var warnings: List[ValidationWarning]
    var summary: String

    fn to_string(self) -> String:
        """Generate human-readable report."""
        pass
```

**Rationale**: Using Mojo's type system and error handling provides compile-time safety and clear error propagation. The `ValidationReport` struct encapsulates all findings in a structured format suitable for both programmatic and human consumption.

### Validation Rules

**Required Directories**:
- `/papers/<paper-name>/`
- `/papers/<paper-name>/src/`
- `/papers/<paper-name>/tests/`
- `/papers/<paper-name>/data/`
- `/papers/<paper-name>/docs/`

**Required Files**:
- `/papers/<paper-name>/README.md`
- `/papers/<paper-name>/src/__init__.mojo` (or `.🔥`)
- `/papers/<paper-name>/tests/test_basic.mojo`

**Content Validation**:
- README.md must contain: title, overview, structure sections
- Mojo files must be syntactically valid (parseable)
- No placeholder text like "TODO" or "FIXME" in generated content

**Rationale**: These rules align with the repository conventions established in the foundation section and ensure generated papers have a complete, usable structure.

### Error Messages

**Design Principle**: Follow the project guideline - "Provide actionable error messages that tell users exactly what's wrong and how to fix it."

**Example Error Messages**:
- ❌ Bad: "Missing directory"
- ✅ Good: "Missing required directory: /papers/lenet5/tests/ - Create this directory to store test files"

- ❌ Bad: "Invalid README"
- ✅ Good: "README.md is missing required section 'Overview' - Add an ## Overview heading with 2-3 sentences describing the paper"

**Rationale**: Actionable messages reduce user friction and make the tool more helpful, especially for newcomers.

### Non-Modification Constraint

**Design Constraint**: "Validation should never modify files."

**Implementation**: All validation functions are read-only. No write operations are performed during validation. Use Mojo's `borrowed` parameter convention to enforce read-only access:

```mojo
fn validate_file_content(borrowed content: String) -> ValidationResult:
    # Cannot modify borrowed reference
    pass
```

**Rationale**: Separation of concerns - validation identifies issues, but modification is a separate user-initiated action. This prevents unexpected changes and maintains user trust.

### Alternatives Considered

**Alternative 1: Validator Class vs Functional Approach**
- **Considered**: Creating a `Validator` class with stateful configuration
- **Chosen**: Pure functions with configuration passed as parameters
- **Rationale**: Simpler, more testable, aligns with Mojo's functional programming capabilities and KISS principle

**Alternative 2: Fail-Fast vs Collect-All**
- **Considered**: Stop validation at first error (fail-fast)
- **Chosen**: Collect all errors and report together
- **Rationale**: More useful to users - see all issues at once rather than iteratively fixing and re-running

**Alternative 3: Schema-Based vs Programmatic Validation**
- **Considered**: Define validation rules in YAML/JSON schema
- **Chosen**: Programmatic validation in Mojo code
- **Rationale**: Better type safety, easier to test, more flexible for custom validation logic (YAGNI - schema approach adds complexity without clear benefit at this stage)

## References

### Source Plan
- [03-tooling/01-paper-scaffolding/02-directory-generator/03-validate-output/plan.md](../../../plan/03-tooling/01-paper-scaffolding/02-directory-generator/03-validate-output/plan.md)

### Related Issues
- #754: [Plan] Validate Output - Design and Documentation (parent planning phase)
- #755: [Test] Validate Output - Write Tests (TDD companion)
- #757: [Package] Validate Output - Integration and Packaging
- #758: [Cleanup] Validate Output - Refactor and Finalize

### Parent Component
- Directory Generator: [notes/plan/03-tooling/01-paper-scaffolding/02-directory-generator/plan.md](../../../plan/03-tooling/01-paper-scaffolding/02-directory-generator/plan.md)

### Architectural Context
- Repository conventions: [CLAUDE.md](../../../CLAUDE.md)
- Mojo coding patterns: [.claude/agents/mojo-language-review-specialist.md](../../../.claude/agents/mojo-language-review-specialist.md)
- Development principles: [CLAUDE.md#key-development-principles](../../../CLAUDE.md#key-development-principles)

## Implementation Notes

*This section will be populated during implementation with:*
- *Challenges encountered and solutions*
- *Performance considerations*
- *Dependencies discovered*
- *Testing insights*
- *Lessons learned*
