# Issue #774: [Plan] Output Formatting - Design and Documentation

## Objective

Design and document a comprehensive output formatting system for the paper scaffolding CLI tool. This system will
provide clear, well-formatted output including progress messages, success confirmations, error messages, and result
summaries to make the tool pleasant to use and help users understand what's happening.

## Deliverables

- **Output Format Specification**: Detailed specification for message templates, formatting rules, and output patterns
- **API Design Document**: Interface contracts for output formatting functions and classes
- **Progress Indicator Design**: Specification for progress indicators during long operations
- **Error Message Guidelines**: Standards for helpful, actionable error messages
- **Success Summary Format**: Template for final operation summaries with created file lists
- **Color Usage Guidelines**: Standards for appropriate use of terminal colors and emphasis
- **Architecture Documentation**: Component design and integration patterns

## Success Criteria

- [ ] Output format specification is complete and comprehensive
- [ ] API design covers all use cases (status, progress, errors, summaries)
- [ ] Progress indicators are designed for clarity and non-intrusive operation
- [ ] Error messages follow best practices for actionability and user guidance
- [ ] Success summary format clearly communicates operation results
- [ ] Color usage guidelines balance emphasis with readability
- [ ] Documentation is clear enough for implementation team to proceed
- [ ] Design is reviewed and approved by senior architects

## Design Decisions

### 1. Output Formatting Architecture

**Decision**: Use a centralized `OutputFormatter` class with specialized methods for different message types.

**Rationale**:

- Centralization ensures consistent formatting across the application
- Specialized methods (`success()`, `error()`, `progress()`, etc.) provide clear API
- Allows for easy testing and mocking of output in tests
- Supports both colored and non-colored output modes

**Alternatives Considered**:

- **Direct print statements**: Rejected - leads to inconsistent formatting and difficult testing
- **Logging-only approach**: Rejected - logging is for developers, CLI output is for users
- **Third-party CLI framework (Click, Rich)**: Deferred - start simple, can integrate later if needed

### 2. Color Scheme

**Decision**: Use ANSI color codes with automatic detection of terminal capabilities.

**Rationale**:

- Existing codebase already uses ANSI colors (see `scripts/create_issues.py`)
- Native terminal support, no dependencies required
- Can be disabled for non-terminal output (pipes, files, CI environments)
- Industry standard, widely supported

**Color Palette** (following existing patterns):

- **Success**: Green (`\033[92m`) - operation completions, confirmations
- **Error**: Red (`\033[91m`) - error messages, failures
- **Warning**: Yellow (`\033[93m`) - warnings, non-critical issues
- **Info**: Cyan (`\033[96m`) - general information, file paths
- **Header**: Magenta (`\033[95m`) - section headers, summaries
- **Bold**: (`\033[1m`) - emphasis for key information

**Alternatives Considered**:

- **No colors**: Rejected - reduces usability and clarity
- **16-color palette**: Rejected - overkill for CLI tool
- **Rich library**: Deferred - adds dependency, use native ANSI for MVP

### 3. Progress Indicators

**Decision**: Implement dual-mode progress indicators:

- **Interactive mode**: Use `tqdm` library for sophisticated progress bars (if available)
- **Fallback mode**: Simple text-based progress updates if `tqdm` not installed
- **Non-interactive mode**: Minimal progress output for scripts/CI

**Rationale**:

- `tqdm` already used in existing scripts (see `create_issues.py`)
- Graceful degradation for environments without `tqdm`
- Non-intrusive in CI/automated environments

**Alternatives Considered**:

- **Custom progress bars**: Rejected - reinventing the wheel, `tqdm` is mature
- **Spinner-only**: Rejected - less informative than progress bars
- **No progress indicators**: Rejected - poor UX for long operations

### 4. Message Structure

**Decision**: Standardize message format with consistent prefixes and structure:

```text
[PREFIX] Main message
  └─ Detail line 1
  └─ Detail line 2
```

**Prefixes**:

- `✓` - Success
- `✗` - Error
- `⚠` - Warning
- `ℹ` - Info
- `→` - Progress/Action

**Rationale**:

- Visual consistency aids scanning
- Hierarchical structure for detailed information
- Unicode symbols are widely supported in modern terminals
- Easy to parse visually

**Alternatives Considered**:

- **Text-only prefixes** (SUCCESS, ERROR): Rejected - verbose, less visual
- **Emoji-heavy**: Rejected - can be distracting, encoding issues
- **No prefixes**: Rejected - reduces scannability

### 5. Error Message Design

**Decision**: Three-part error message structure:

1. **What happened**: Clear description of the error
2. **Why it happened**: Context and root cause (when known)
3. **What to do**: Actionable next steps

**Example**:

```text
✗ Failed to create directory: /path/to/papers/invalid
  Cause: Permission denied
  Action: Check directory permissions or choose a different location
```

**Rationale**:

- Follows CLI UX best practices
- Reduces user frustration and support requests
- Actionable guidance improves user experience

**Alternatives Considered**:

- **Simple error strings**: Rejected - not helpful enough
- **Exception stack traces**: Rejected - too technical for end users
- **Error codes only**: Rejected - requires looking up documentation

### 6. Terminal Width Handling

**Decision**: Respect terminal width with automatic wrapping:

- Detect terminal width using `os.get_terminal_size()`
- Wrap long lines at word boundaries
- Default to 80 columns if width cannot be detected
- Allow override via environment variable (`COLUMNS`)

**Rationale**:

- Professional appearance across different terminal sizes
- Prevents text truncation or awkward wrapping
- Standard practice in CLI tools

**Alternatives Considered**:

- **Fixed 80 columns**: Rejected - wastes space on modern wide terminals
- **No wrapping**: Rejected - poor UX on narrow terminals
- **textwrap only**: Chosen - Python stdlib, no dependencies

### 7. Output Modes

**Decision**: Support three output modes:

1. **Interactive** (default): Full formatting, colors, progress bars
2. **Quiet**: Minimal output, errors only
3. **Verbose**: Detailed output with debug information

**Rationale**:

- Flexibility for different use cases (interactive vs. scripted)
- Standard pattern in Unix CLI tools
- Easy to implement with flag-based toggling

**Alternatives Considered**:

- **Single mode**: Rejected - not flexible enough
- **Five+ modes**: Rejected - unnecessary complexity

## Architecture Overview

### Component Structure

```text
output_formatting/
├── __init__.py              # Public API exports
├── formatter.py             # Main OutputFormatter class
├── progress.py              # Progress indicator implementations
├── messages.py              # Message template definitions
└── colors.py                # Color code constants and terminal detection
```

### Key Interfaces

#### OutputFormatter

```python
class OutputFormatter:
    """Centralized output formatting for CLI tool."""

    def __init__(self, mode: OutputMode = OutputMode.INTERACTIVE,
                 use_colors: Optional[bool] = None):
        """
        Initialize formatter.

        Args:
            mode: Output mode (interactive, quiet, verbose)
            use_colors: Enable/disable colors (auto-detect if None)
        """

    def success(self, message: str, details: Optional[List[str]] = None) -> None:
        """Print success message with optional details."""

    def error(self, message: str, cause: Optional[str] = None,
              action: Optional[str] = None) -> None:
        """Print error message with cause and suggested action."""

    def warning(self, message: str, details: Optional[List[str]] = None) -> None:
        """Print warning message with optional details."""

    def info(self, message: str, details: Optional[List[str]] = None) -> None:
        """Print info message with optional details."""

    def progress(self, iterable, desc: str, total: Optional[int] = None):
        """Create progress indicator for iterable operations."""

    def summary(self, title: str, items: Dict[str, Any]) -> None:
        """Print formatted summary with key-value pairs."""
```

#### ProgressIndicator

```python
class ProgressIndicator(Protocol):
    """Protocol for progress indicators."""

    def update(self, n: int = 1) -> None:
        """Update progress by n steps."""

    def close(self) -> None:
        """Clean up and finalize progress display."""
```

### Integration Points

1. **CLI Argument Parser**: Receives `--quiet`, `--verbose`, `--no-color` flags
2. **Error Handling**: Catches exceptions and formats with `formatter.error()`
3. **File Operations**: Reports progress during file creation/copying
4. **Validation**: Uses `formatter.warning()` for validation issues
5. **Completion**: Generates summary with `formatter.summary()`

## References

### Source Documentation

- **Source Plan**: [notes/plan/03-tooling/01-paper-scaffolding/03-cli-interface/03-output-formatting/plan.md](../../../plan/03-tooling/01-paper-scaffolding/03-cli-interface/03-output-formatting/plan.md)
- **Parent Plan**: [notes/plan/03-tooling/01-paper-scaffolding/03-cli-interface/plan.md](../../../plan/03-tooling/01-paper-scaffolding/03-cli-interface/plan.md)

### Related Issues

- **#775**: [Test] Output Formatting - Test implementation
- **#776**: [Implementation] Output Formatting - Build functionality
- **#777**: [Packaging] Output Formatting - Integration and packaging
- **#778**: [Cleanup] Output Formatting - Refactoring and finalization

### Related Components

- **Argument Parsing** (Issue #771): Provides flags for output mode configuration
- **User Prompts** (Issue #773): Uses output formatter for prompt display
- **Error Handling**: Integrates with error formatting system

### Existing Patterns

- **Color Usage**: See `scripts/create_issues.py` (lines 40-63) for existing ANSI color implementation
- **Progress Bars**: See `scripts/create_issues.py` for `tqdm` usage patterns
- **Message Formatting**: Review existing scripts for consistency

### Best Practices References

- [Bash Colors and Formatting](https://misc.flogisoft.com/bash/tip_colors_and_formatting)
- [Click Documentation - Colors](https://click.palletsprojects.com/en/8.1.x/utils/#ansi-colors)
- [Rich CLI Library](https://rich.readthedocs.io/) - for future enhancement ideas
- [TQDM Documentation](https://tqdm.github.io/) - progress bar library
- [CLI Guidelines](https://clig.dev/) - comprehensive CLI UX best practices

## Implementation Notes

This section will be populated during the implementation phase with:

- Challenges encountered during implementation
- Design refinements and adjustments
- Performance considerations
- Edge cases discovered
- Integration issues and resolutions

---

**Status**: Planning phase complete - ready for test/implementation phases

**Next Steps**:

1. Review this design with senior architects
2. Address any feedback or concerns
3. Proceed with test phase (Issue #775)
4. Begin implementation phase (Issue #776)
