# Issue #779: [Plan] CLI Interface - Design and Documentation

## Objective

Design and document a command-line interface for the paper scaffolding tool that provides an intuitive user experience. The CLI will handle argument parsing, prompt users for required information, and format output in a clear, helpful way. This planning phase establishes the architecture, API contracts, and design specifications for creating a user-friendly CLI that supports both interactive and non-interactive modes.

## Deliverables

1. **CLI Architecture Specification**
   - Component interaction diagram
   - Data flow between argument parser, prompts, and output formatter
   - Mode selection logic (interactive vs non-interactive)

1. **Argument Parsing Design**
   - Supported arguments and options specification
   - Validation rules and constraints
   - Help text and usage documentation format
   - Default values strategy

1. **User Prompts Design**
   - Interactive prompt workflow
   - Prompt text templates with examples
   - Input validation rules per field
   - Error handling and re-prompting logic

1. **Output Formatting Design**
   - Message templates for status, progress, success, and errors
   - Color usage guidelines
   - Progress indicator patterns
   - Summary format for created files

1. **API Contracts**
   - Interface definitions for each component
   - Input/output specifications
   - Error handling contracts

1. **Design Documentation**
   - `/notes/issues/779/README.md` (this file)
   - Design decisions and rationale
   - Usage examples and scenarios

## Success Criteria

- [ ] CLI architecture is clearly defined with component responsibilities
- [ ] All argument parsing rules are documented with examples
- [ ] Interactive prompt workflow is specified with validation rules
- [ ] Output formatting templates are designed for all message types
- [ ] API contracts are defined for component interfaces
- [ ] Design supports both interactive and non-interactive modes
- [ ] Help text format is comprehensive and user-friendly
- [ ] Error message patterns are helpful and actionable
- [ ] Progress indicators are designed for long-running operations
- [ ] Design documentation is complete and ready for implementation

## Design Decisions

### 1. Two-Mode Operation Strategy

**Decision**: Support both interactive mode (with prompts) and non-interactive mode (all arguments provided).

### Rationale

- **Interactive mode**: Best for first-time users and exploratory use cases
- **Non-interactive mode**: Essential for scripting, CI/CD integration, and automation
- **Flexibility**: Users can choose the mode that fits their workflow

### Implementation Approach

- Detect mode based on whether all required arguments are provided
- If any required argument is missing, enter interactive mode
- Allow `--no-interactive` flag to force error on missing arguments
- Interactive mode fills gaps in partially-provided arguments

### Alternatives Considered

- Interactive-only: Rejected - would prevent automation
- Non-interactive-only: Rejected - poor user experience for new users
- Explicit mode flag: Rejected - auto-detection is more intuitive

### 2. Argument Parsing Library Selection

**Decision**: Use Python's `argparse` standard library module.

### Rationale

- **Standard library**: No additional dependencies
- **Well-documented**: Extensive Python documentation and examples
- **Feature-complete**: Supports all required features (flags, options, help, validation)
- **Familiar**: Most Python developers know argparse

### API Design

```python
# Core arguments
--title, -t          Paper title (required if not interactive)
--author, -a         Paper author (required if not interactive)
--year, -y           Publication year (required if not interactive)
--output-dir, -o     Output directory (default: papers/<title-slug>)

# Optional metadata
--description, -d    Brief paper description
--arxiv-id          ArXiv identifier
--doi               DOI reference

# Behavior flags
--no-interactive    Fail on missing arguments instead of prompting
--verbose, -v       Show detailed progress
--dry-run           Show what would be created without creating
```text

### Alternatives Considered

- `click`: Rejected - adds dependency, more complex than needed
- `typer`: Rejected - adds dependency, requires Python 3.6+
- `getopt`: Rejected - lower-level, requires more boilerplate

### 3. Input Validation Strategy

**Decision**: Validate input at collection time (during parsing or prompting) with immediate feedback.

### Rationale

- **Early detection**: Catch errors before any file operations
- **Better UX**: Users get immediate feedback on invalid input
- **Fail fast**: Prevent partial operations that need rollback

### Validation Rules

- **Title**: 1-200 characters, no leading/trailing whitespace
- **Author**: 1-100 characters, no leading/trailing whitespace
- **Year**: 1900-2100, numeric
- **Output directory**: Valid path, does not already exist
- **ArXiv ID**: Format `YYMM.NNNNN` or `arch-ive/YYMMNNN`
- **DOI**: Format `10.NNNN/...`

### Error Handling

- Non-interactive mode: Print error and exit with code 1
- Interactive mode: Show error, re-prompt with guidance

### Alternatives Considered

- Post-collection validation: Rejected - poor UX, harder to recover
- Deferred validation: Rejected - could lead to partial operations

### 4. Output Formatting Approach

**Decision**: Use simple colored text output with clear sections and minimal decorations.

### Rationale

- **Clarity**: Focus on information, not decoration
- **Terminal compatibility**: Works on all standard terminals
- **Accessibility**: Color used for emphasis, not critical information
- **Maintainability**: Simple to implement and modify

### Color Usage

- **Green**: Success messages, checkmarks
- **Red**: Error messages, failures
- **Yellow**: Warnings, prompts
- **Blue**: Informational messages, headers
- **No color fallback**: All information available without color

### Message Templates

```text
# Progress
Creating paper structure for "Paper Title"...
  ✓ Created directory: papers/paper-title/
  ✓ Generated README.md
  ✓ Created implementation stubs
  ✓ Created test templates

# Success Summary
✓ Paper scaffolding complete!

Created files:
  - papers/paper-title/README.md
  - papers/paper-title/src/model.mojo
  - papers/paper-title/tests/test_model.mojo

Next steps:
  1. Review README.md
  2. Implement model in src/model.mojo
  3. Run tests with: mojo test papers/paper-title/tests/

# Error
✗ Error: Directory already exists: papers/paper-title/
  Use a different --output-dir or remove the existing directory.
```text

### Alternatives Considered

- Rich TUI library: Rejected - adds dependency, overkill for simple tool
- ASCII art banners: Rejected - distracting, wastes screen space
- JSON output mode: Deferred - can add later if needed for automation

### 5. Help Text Organization

**Decision**: Provide comprehensive help with examples, grouped by category.

### Format

```text
usage: scaffold-paper [OPTIONS]

Create a new paper implementation from templates.

Required arguments (interactive mode prompts if omitted):
  -t, --title TITLE      Paper title
  -a, --author AUTHOR    Paper author(s)
  -y, --year YEAR        Publication year

Optional arguments:
  -o, --output-dir DIR   Output directory (default: papers/<title-slug>)
  -d, --description DESC Brief description
  --arxiv-id ID          ArXiv identifier (e.g., 2301.12345)
  --doi DOI              DOI reference (e.g., 10.1234/example)

Behavior:
  --no-interactive       Fail on missing arguments instead of prompting
  -v, --verbose          Show detailed progress
  --dry-run              Show what would be created without creating
  -h, --help             Show this help message

Examples:
  # Interactive mode - prompts for required information
  $ scaffold-paper

  # Non-interactive mode - all arguments provided
  $ scaffold-paper -t "LeNet-5" -a "LeCun et al." -y 1998

  # Partial arguments - prompts for missing values
  $ scaffold-paper -t "ResNet" --arxiv-id 1512.03385

  # Dry run to preview without creating
  $ scaffold-paper -t "Test" -a "Author" -y 2024 --dry-run
```text

### Rationale

- **Examples**: Show common usage patterns
- **Grouped options**: Easier to scan and understand
- **Clear defaults**: Users know what happens if they omit options
- **Concise but complete**: All information needed without overwhelming

### 6. Component Architecture

**Decision**: Three separate components with clear responsibilities.

### Components

1. **Argument Parser** (`argument_parser.py`)
   - Parses command-line arguments using argparse
   - Returns dictionary of parsed values
   - Generates help text
   - Validates argument format (not business logic)

1. **User Prompter** (`user_prompter.py`)
   - Prompts for missing required values
   - Validates input according to business rules
   - Re-prompts on invalid input with guidance
   - Returns completed metadata dictionary

1. **Output Formatter** (`output_formatter.py`)
   - Formats progress messages
   - Displays success summaries
   - Formats error messages
   - Handles colored output with fallback

### Data Flow

```text
main.py
  ↓
[Parse Arguments] → args_dict (may have None values)
  ↓
[Prompt for Missing] → complete_metadata (all values filled)
  ↓
[Execute Scaffolding] → results
  ↓
[Format Output] → terminal display
```text

### Interface Contracts

```python
# Argument Parser
def parse_arguments(argv: List[str]) -> Dict[str, Any]:
    """Parse command-line arguments.

    Returns:
        Dictionary with keys: title, author, year, output_dir,
        description, arxiv_id, doi, no_interactive, verbose, dry_run
        Values may be None if not provided.
    """

# User Prompter
def prompt_for_metadata(partial_metadata: Dict[str, Any]) -> Dict[str, Any]:
    """Prompt user for any missing required metadata.

    Args:
        partial_metadata: Dictionary from argument parser

    Returns:
        Complete metadata with all required fields filled

    Raises:
        KeyboardInterrupt: If user cancels with Ctrl+C
    """

# Output Formatter
def format_progress(message: str, status: str = "info") -> None:
    """Print formatted progress message.

    Args:
        message: Message to display
        status: One of "info", "success", "warning", "error"
    """

def format_summary(created_files: List[str], paper_dir: str) -> None:
    """Print success summary with created files.

    Args:
        created_files: List of file paths that were created
        paper_dir: Root directory for the paper
    """
```text

### Rationale

- **Separation of concerns**: Each component has single responsibility
- **Testability**: Easy to unit test each component independently
- **Reusability**: Components can be used by other tools
- **Maintainability**: Changes to one component don't affect others

## References

### Source Plan

- [notes/plan/03-tooling/01-paper-scaffolding/03-cli-interface/plan.md](notes/plan/03-tooling/01-paper-scaffolding/03-cli-interface/plan.md)

### Parent Plan

- [notes/plan/03-tooling/01-paper-scaffolding/plan.md](notes/plan/03-tooling/01-paper-scaffolding/plan.md) - Paper Scaffolding Tool

### Child Components

This planning issue covers design for three implementation components:

1. **Argument Parsing** - [notes/plan/03-tooling/01-paper-scaffolding/03-cli-interface/01-argument-parsing/plan.md](notes/plan/03-tooling/01-paper-scaffolding/03-cli-interface/01-argument-parsing/plan.md)
1. **User Prompts** - [notes/plan/03-tooling/01-paper-scaffolding/03-cli-interface/02-user-prompts/plan.md](notes/plan/03-tooling/01-paper-scaffolding/03-cli-interface/02-user-prompts/plan.md)
1. **Output Formatting** - [notes/plan/03-tooling/01-paper-scaffolding/03-cli-interface/03-output-formatting/plan.md](notes/plan/03-tooling/01-paper-scaffolding/03-cli-interface/03-output-formatting/plan.md)

### Related Issues

- Issue #780: [Test] CLI Interface - Test Implementation
- Issue #781: [Implementation] CLI Interface - Build Functionality
- Issue #782: [Package] CLI Interface - Integration and Packaging
- Issue #783: [Cleanup] CLI Interface - Refactor and Finalize

### Related Documentation

- [CLAUDE.md](CLAUDE.md) - Project conventions and workflow
- [agents/README.md](agents/README.md) - Agent system documentation
- [agents/delegation-rules.md](agents/delegation-rules.md) - Coordination patterns

## Implementation Notes

*This section will be filled during the implementation phase with discoveries, decisions, and lessons learned.*
