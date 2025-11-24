# Issue #744: [Plan] Create Structure - Design and Documentation

## Objective

Define the comprehensive design and architecture for implementing the core directory creation logic that builds the complete folder hierarchy for a new paper implementation. This planning phase establishes the specifications, API contracts, and design decisions that will guide the test, implementation, and packaging phases.

## Deliverables

### Design Documentation

- Architecture specification for directory creation system
- API contract definitions and interface design
- Error handling strategy and edge case analysis
- Data flow and processing pipeline design

### Technical Specifications

- Directory structure specification format
- Paper name normalization rules
- Permission setting requirements
- Logging and reporting format

### Component Outputs (from implementation)

- Created directory hierarchy following repository conventions
- Directory creation log with detailed operation tracking
- Error handling for existing directories and permission issues

## Success Criteria

- [ ] Architecture design is complete and documented
- [ ] API contracts clearly define inputs, outputs, and behavior
- [ ] Error handling strategy covers all edge cases
- [ ] Design decisions are documented with rationale
- [ ] Specifications are clear enough for test and implementation phases
- [ ] All required directories will be created per specification
- [ ] Directory names will follow naming conventions
- [ ] Permissions will be set correctly
- [ ] Existing directories will be handled gracefully

## Design Decisions

### 1. Implementation Language: Python

**Decision**: Use Python for directory creation automation

### Rationale

- Subprocess output capture (Mojo v0.25.7 limitation - cannot capture stdout/stderr)
- Robust os.makedirs() with exist_ok parameter for idempotent operations
- Better exception handling for filesystem operations
- Aligns with ADR-001 language selection strategy for tooling/automation

### Alternatives Considered

- Mojo: Not suitable due to subprocess limitations and immature filesystem API
- Bash scripts: Less maintainable, harder to test, poor error handling

### 2. Directory Creation Strategy: Idempotent Operations

**Decision**: Use `os.makedirs(path, exist_ok=True)` for all directory creation

### Rationale

- Safe to run multiple times without errors
- Handles existing directories gracefully
- Simplifies error handling logic
- Matches repository convention of idempotent operations

### Alternatives Considered

- Check-then-create: Race conditions possible
- Fail on existing: Not user-friendly, breaks idempotency

### 3. Paper Name Normalization

**Decision**: Convert paper names to lowercase with hyphens replacing spaces/special characters

### Rationale

- Consistent naming across filesystem
- Avoids shell escaping issues
- Follows established repository conventions
- Predictable and reversible transformation

### Normalization Rules

1. Convert to lowercase
1. Replace spaces with hyphens
1. Remove or replace special characters (keep only alphanumeric and hyphens)
1. Remove consecutive hyphens
1. Trim leading/trailing hyphens

**Example**: "LeNet-5: First CNN" → "lenet-5-first-cnn"

### 4. Directory Structure Specification Format

**Decision**: Use Python dictionary/JSON structure for directory hierarchy specification

### Rationale

- Easy to parse and validate
- Supports nested structures naturally
- Can be extended with metadata (permissions, templates, etc.)
- Human-readable and maintainable

### Example Structure

```python
{
    "name": "{paper_name}",
    "subdirs": [
        "src",
        "tests",
        "docs",
        "configs",
        "data"
    ]
}
```text

### 5. Logging Strategy

**Decision**: Structured logging with operation tracking and summary reporting

### Rationale

- Clear feedback for users
- Debugging support for failures
- Audit trail for created directories
- Supports both verbose and summary modes

### Log Levels

- INFO: Directory created successfully
- WARNING: Directory already exists (skipped)
- ERROR: Permission denied or other failures

### 6. Error Handling Approach

**Decision**: Continue-on-error with comprehensive reporting

### Rationale

- Create as much as possible even if some operations fail
- Provide complete error report at end
- User can fix issues and re-run (idempotent)
- Better UX than fail-fast approach

### Error Categories

1. Permission errors (report and continue)
1. Invalid path errors (fail early)
1. Disk space errors (fail early)
1. Existing directory (log and continue)

### 7. Permission Settings

**Decision**: Use default permissions (755 for directories) with option to customize

### Rationale

- Follows standard Unix conventions
- Secure by default (owner: rwx, group: rx, other: rx)
- Can be overridden if needed
- Consistent with repository practices

### 8. Validation Strategy

**Decision**: Post-creation validation of directory structure

### Rationale

- Confirms successful creation
- Detects partial failures
- Provides confidence in output
- Supports automated testing

### Validation Checks

1. All required directories exist
1. Directories are writable
1. Structure matches specification
1. No unexpected directories created

## Architecture

### Component Structure

```text
create_structure/
├── __init__.py           # Module exports
├── creator.py            # Main DirectoryCreator class
├── normalizer.py         # Paper name normalization
├── validator.py          # Structure validation
└── logging_config.py     # Logging setup
```text

### API Contract

#### DirectoryCreator Class

```python
class DirectoryCreator:
    """Creates directory structure for new paper implementations."""

    def __init__(self, base_path: str, structure_spec: Dict[str, Any]):
        """
        Initialize directory creator.

        Args:
            base_path: Target base directory path
            structure_spec: Directory structure specification
        """
        pass

    def create_structure(self, paper_name: str) -> CreationResult:
        """
        Create directory structure for a paper.

        Args:
            paper_name: Name of the paper (will be normalized)

        Returns:
            CreationResult with success/failure details

        Raises:
            ValueError: If base_path is invalid
            PermissionError: If base_path is not writable
        """
        pass

    def validate_structure(self, paper_path: str) -> ValidationResult:
        """
        Validate created directory structure.

        Args:
            paper_path: Path to paper directory

        Returns:
            ValidationResult with validation details
        """
        pass
```text

#### Helper Functions

```python
def normalize_paper_name(name: str) -> str:
    """
    Normalize paper name to valid directory name.

    Args:
        name: Original paper name

    Returns:
        Normalized directory name (lowercase, hyphens)
    """
    pass

def validate_base_path(path: str) -> bool:
    """
    Validate that base path exists and is writable.

    Args:
        path: Base directory path

    Returns:
        True if valid, False otherwise
    """
    pass
```text

#### Result Classes

```python
@dataclass
class CreationResult:
    """Result of directory creation operation."""
    success: bool
    created_dirs: List[str]
    skipped_dirs: List[str]
    errors: List[Tuple[str, str]]  # (path, error_message)
    summary: str

@dataclass
class ValidationResult:
    """Result of structure validation."""
    valid: bool
    missing_dirs: List[str]
    unexpected_dirs: List[str]
    permission_issues: List[str]
    summary: str
```text

### Data Flow

1. **Input**: Base path, paper name, structure specification
1. **Normalization**: Paper name → normalized directory name
1. **Validation**: Validate base path exists and is writable
1. **Creation**: Iterate through structure spec, create directories
1. **Logging**: Record each operation (created, skipped, error)
1. **Validation**: Verify created structure matches specification
1. **Output**: CreationResult with summary and details

### Error Handling Flow

1. **Invalid base path** → Raise ValueError immediately
1. **Permission denied on base** → Raise PermissionError immediately
1. **Permission denied on subdir** → Log error, continue, report in result
1. **Directory exists** → Log warning, continue
1. **Disk full** → Log error, fail operation
1. **Invalid directory name** → Log error, skip directory, continue

## References

### Source Plans

- [Create Structure Plan](notes/plan/03-tooling/01-paper-scaffolding/02-directory-generator/01-create-structure/plan.md)
- [Parent: Directory Generator Plan](notes/plan/03-tooling/01-paper-scaffolding/02-directory-generator/plan.md)

### Related Issues

- #745 - [Test] Create Structure - Write Tests (depends on this plan)
- #746 - [Implementation] Create Structure - Build Functionality (depends on this plan)
- #747 - [Packaging] Create Structure - Integration and Packaging (depends on this plan)
- #748 - [Cleanup] Create Structure - Refactor and Finalize (depends on parallel phases)

### Related Documentation

- [ADR-001: Language Selection for Tooling](../../review/adr/ADR-001-language-selection-tooling.md)
- [5-Phase Development Workflow](notes/review/README.md)
- [Repository Architecture](CLAUDE.md#repository-architecture)

### Team Resources

- [Agent Hierarchy](agents/hierarchy.md)
- [Delegation Rules](agents/delegation-rules.md)

## Implementation Notes

This section will be populated during the implementation phase with:

- Discoveries and insights from development
- Deviations from original design (with justification)
- Performance observations
- Edge cases encountered
- Lessons learned

---

**Status**: Planning Complete
**Phase**: Plan (1/5 in workflow: Plan → Test/Implementation/Packaging → Cleanup)
**Next Steps**: Test (#745), Implementation (#746), and Packaging (#747) phases can proceed in parallel
