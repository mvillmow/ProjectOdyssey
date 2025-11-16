# Issue #809: [Plan] Test Specific Paper - Design and Documentation

## Objective

Define the architecture and specifications for implementing logic to identify and focus testing on a specific paper
implementation. This component will enable the paper test script to parse paper identifiers, locate paper directories,
and set up the appropriate test context for validating a single paper implementation.

## Deliverables

- Comprehensive design documentation for paper identification logic
- API contracts and interface specifications
- Error handling strategy for invalid paper identifiers
- Paper metadata structure and loading mechanism
- Test context configuration specification

## Success Criteria

- [ ] Papers can be identified by name or path
- [ ] Invalid paper names are caught and reported with clear error messages
- [ ] Paper metadata is loaded correctly
- [ ] Test context is properly configured for single-paper testing
- [ ] Design supports flexible paper identification methods (full name, partial match, directory path)
- [ ] Performance considerations documented (metadata caching strategy)

## Design Decisions

### 1. Paper Identification Strategy

**Decision**: Support multiple identification methods with progressive fallback:

1. **Exact path match** - Direct path to paper directory (highest priority)
2. **Full name match** - Exact match against paper name in repository
3. **Partial name match** - Fuzzy/substring matching for convenience
4. **Interactive selection** - If multiple matches, prompt user to select

**Rationale**:

- Developers may not remember exact paper names during rapid iteration
- Direct path support enables integration with shell workflows (tab completion, scripting)
- Partial matching reduces friction during development
- Interactive fallback prevents ambiguity while maintaining usability

**Alternatives Considered**:

- **Exact name only**: Too rigid, poor developer experience
- **Path only**: Misses opportunity for convenient name-based lookup
- **Regex-based matching**: Overly complex for the use case

### 2. Paper Metadata Structure

**Decision**: Define minimal metadata structure for test context:

```python
@dataclass
class PaperMetadata:
    name: str                    # Full paper name (e.g., "lenet-5")
    path: Path                   # Absolute path to paper directory
    config_path: Path            # Path to paper configuration file
    test_directory: Path         # Path to paper-specific tests
    description: Optional[str]   # Short description from config
    tags: List[str]              # Classification tags (e.g., ["cnn", "vision"])
```

**Rationale**:

- Minimal fields cover immediate testing needs
- Extensible for future requirements (versioning, dependencies, etc.)
- Clear separation of path information and metadata
- Type-safe using Python dataclasses

**Alternatives Considered**:

- **Dictionary-based**: Less type safety, harder to maintain
- **Full configuration object**: Overly complex for test targeting needs

### 3. Error Handling Strategy

**Decision**: Implement graduated error messages with actionable guidance:

- **No match found**: List available papers with similarity ranking
- **Multiple matches**: Present interactive selection with paper descriptions
- **Invalid path**: Suggest correct directory structure
- **Missing metadata**: Provide template for paper configuration

**Rationale**:

- Clear error messages reduce debugging time
- Actionable guidance helps developers fix issues quickly
- Similar paper suggestions handle typos gracefully
- Interactive selection maintains flow without manual retry

**Alternatives Considered**:

- **Fail fast**: Poor developer experience, requires manual investigation
- **Auto-select first match**: Dangerous, could test wrong paper

### 4. Metadata Caching Strategy

**Decision**: Implement in-memory cache with lazy loading:

- Cache paper metadata on first access
- Invalidate on directory structure changes (use file modification times)
- Scope cache to script execution (no persistent state)

**Rationale**:

- Repeated test runs benefit from cached metadata
- File system overhead reduced for large repositories
- No stale data issues (cache per execution)
- Simple implementation without external dependencies

**Alternatives Considered**:

- **No caching**: Acceptable performance for small repos, scales poorly
- **Persistent cache**: Complexity of invalidation outweighs benefits
- **Full repository scan**: Too expensive for incremental testing

### 5. Directory Structure Assumptions

**Decision**: Papers are located in `papers/` directory with standard structure:

```text
papers/
├── lenet-5/
│   ├── config.yaml          # Paper metadata and configuration
│   ├── README.md            # Paper documentation
│   ├── src/                 # Implementation
│   └── tests/               # Paper-specific tests
└── alexnet/
    └── ...
```

**Rationale**:

- Consistent structure enables automated validation
- Config file provides single source of truth for metadata
- Separate test directory supports isolated paper testing
- Aligns with repository standards (see tooling/testing-tools plan)

**Alternatives Considered**:

- **Flexible structure**: Too complex to validate reliably
- **Flat structure**: Doesn't scale to many papers

## Technical Specifications

### API Interface

```python
def identify_paper(identifier: str, repository_root: Path) -> PaperMetadata:
    """
    Identify and load metadata for a specific paper.

    Args:
        identifier: Paper name, partial name, or path
        repository_root: Root directory of the repository

    Returns:
        PaperMetadata object with loaded configuration

    Raises:
        PaperNotFoundError: No matching paper found
        MultiplePapersFoundError: Ambiguous identifier (multiple matches)
        InvalidPaperError: Paper directory exists but invalid structure
    """
    pass

def list_available_papers(repository_root: Path) -> List[PaperMetadata]:
    """
    List all available papers in the repository.

    Args:
        repository_root: Root directory of the repository

    Returns:
        List of PaperMetadata objects for all valid papers
    """
    pass

def setup_test_context(paper: PaperMetadata) -> TestContext:
    """
    Configure test environment for a specific paper.

    Args:
        paper: PaperMetadata object for target paper

    Returns:
        TestContext configured for the paper
    """
    pass
```

### Error Classes

```python
class PaperError(Exception):
    """Base exception for paper-related errors."""
    pass

class PaperNotFoundError(PaperError):
    """Raised when paper identifier doesn't match any papers."""
    def __init__(self, identifier: str, suggestions: List[str]):
        self.identifier = identifier
        self.suggestions = suggestions
        super().__init__(f"Paper '{identifier}' not found")

class MultiplePapersFoundError(PaperError):
    """Raised when identifier matches multiple papers."""
    def __init__(self, identifier: str, matches: List[PaperMetadata]):
        self.identifier = identifier
        self.matches = matches
        super().__init__(f"Multiple papers match '{identifier}'")

class InvalidPaperError(PaperError):
    """Raised when paper directory has invalid structure."""
    def __init__(self, path: Path, missing_files: List[str]):
        self.path = path
        self.missing_files = missing_files
        super().__init__(f"Invalid paper at {path}")
```

### Configuration File Format

Papers use YAML configuration (`papers/<name>/config.yaml`):

```yaml
name: lenet-5
description: LeNet-5 Convolutional Neural Network (1998)
tags:
  - cnn
  - vision
  - classic
authors:
  - Yann LeCun
  - Léon Bottou
year: 1998
paper_url: http://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf
```

## References

- **Source Plan**: [notes/plan/03-tooling/02-testing-tools/02-paper-test-script/01-test-specific-paper/plan.md](../../../plan/03-tooling/02-testing-tools/02-paper-test-script/01-test-specific-paper/plan.md)
- **Parent Plan**: [notes/plan/03-tooling/02-testing-tools/02-paper-test-script/plan.md](../../../plan/03-tooling/02-testing-tools/02-paper-test-script/plan.md)
- **Related Issues**:
  - #810 - [Test] Test Specific Paper - Test Implementation
  - #811 - [Implementation] Test Specific Paper - Code Implementation
  - #812 - [Package] Test Specific Paper - Integration and Packaging
  - #813 - [Cleanup] Test Specific Paper - Refactor and Finalize
- **Repository Standards**: See testing tools section in parent plan

## Implementation Notes

*This section will be populated during implementation with findings, edge cases, and lessons learned.*

### Notes Added During Implementation

<!-- Add notes here as work progresses -->
