# Issue #764: [Plan] Argument Parsing - Design and Documentation

## Objective

Design and document a command-line argument parsing system for the paper scaffolding tool that handles options and flags, validates input, and provides comprehensive help documentation. This planning phase establishes the architecture and specifications for argument parsing that will enable both interactive and non-interactive modes of operation.

## Deliverables

- **Parsed argument dictionary**: Structured data containing all command-line arguments with appropriate type conversion
- **Validation errors**: Clear error messages for invalid arguments, combinations, or missing required values
- **Help text and usage information**: Comprehensive documentation accessible via `--help` flag

## Success Criteria

- [ ] All supported arguments (title, author, output-dir, etc.) are correctly defined with proper types and defaults
- [ ] Invalid arguments are detected and reported with clear, actionable error messages
- [ ] Help text is comprehensive, well-formatted, and follows standard CLI conventions
- [ ] Default values are properly implemented and tested
- [ ] Short aliases (-t, -a, -o) work correctly alongside long-form options
- [ ] Argument validation catches common errors (invalid paths, missing required args, incompatible combinations)

## Design Decisions

### 1. Language Selection: Python with argparse

**Decision**: Implement argument parsing in Python using the standard library `argparse` module.

### Rationale

- **Follows ADR-001**: This is automation/tooling, not ML/AI implementation
- **Standard Library**: argparse is mature, well-tested, and part of Python stdlib (no external dependencies)
- **Rich Features**: Built-in support for help generation, type validation, default values, mutually exclusive groups
- **Consistency**: Aligns with existing tooling scripts (create_issues.py, regenerate_github_issues.py)
- **Documentation**: Extensive documentation and community resources available

### Alternatives Considered

- **Mojo**: Not suitable - this is a CLI tool requiring subprocess interaction and text processing (ADR-001 establishes Python for automation tasks)
- **click**: More features than needed; adds external dependency
- **sys.argv manual parsing**: Too low-level; reinvents the wheel

### 2. Argument Structure

**Decision**: Support both long-form and short-form arguments with sensible defaults.

### Core Arguments

```text
--title, -t         Paper title (required in non-interactive mode)
--author, -a        Paper author(s) (required in non-interactive mode)
--year, -y          Publication year (optional, default: current year)
--output-dir, -o    Output directory (optional, default: ./papers/<title-slug>)
--template          Template name (optional, default: "standard")
--interactive, -i   Interactive mode with prompts (default: True if args missing)
--help, -h          Show help message
--version, -v       Show version information
```text

### Rationale

- **Follows conventions**: Standard CLI patterns (--help, --version, short aliases)
- **Flexibility**: Supports both batch processing (all args) and interactive use (prompts for missing)
- **Discoverability**: Short aliases for frequently used options reduce typing

### 3. Validation Strategy

**Decision**: Implement three-tier validation: argparse built-in, custom validators, and business logic validation.

### Tier 1 - argparse built-in

- Type checking (str, int, Path)
- Required vs optional arguments
- Mutually exclusive groups (if needed)

### Tier 2 - Custom validators

- Path validation (parent directory exists, writable)
- Year validation (reasonable range, e.g., 1950-current year)
- Title/author format validation (non-empty, printable characters)

### Tier 3 - Business logic validation

- Template existence validation
- Output directory conflict detection (already exists with content)
- Cross-argument validation (e.g., template compatibility with year)

### Rationale

- **Fail Fast**: Catch errors before creating any files
- **Clear Messages**: Each tier provides specific, actionable error messages
- **Separation of Concerns**: Parsing logic separate from business logic

### 4. Help Text Design

**Decision**: Follow standard help text conventions with clear sections and examples.

### Structure

```text
usage: scaffold-paper [-h] [-v] [-t TITLE] [-a AUTHOR] [-y YEAR] [-o OUTPUT_DIR] [--template TEMPLATE] [-i]

Create a new research paper implementation scaffold.

optional arguments:
  -h, --help            show this help message and exit
  -v, --version         show program's version number and exit
  -t TITLE, --title TITLE
                        Paper title (e.g., "LeNet-5: Gradient-Based Learning")
  -a AUTHOR, --author AUTHOR
                        Paper author(s) (e.g., "LeCun et al.")
  -y YEAR, --year YEAR  Publication year (default: 2025)
  -o OUTPUT_DIR, --output-dir OUTPUT_DIR
                        Output directory (default: ./papers/<title-slug>)
  --template TEMPLATE   Template to use (default: standard)
  -i, --interactive     Force interactive mode with prompts

examples:
  # Interactive mode (prompts for inputs)
  $ scaffold-paper

  # Non-interactive mode (all args provided)
  $ scaffold-paper -t "LeNet-5" -a "LeCun et al." -y 1998

  # Custom output directory
  $ scaffold-paper -t "LeNet-5" -a "LeCun et al." -o ~/projects/lenet
```text

### Rationale

- **Discoverability**: Users can learn the tool through --help
- **Examples**: Real-world usage examples guide users
- **Standards**: Follows argparse conventions familiar to Python developers

### 5. Interactive vs Non-Interactive Mode

**Decision**: Automatically detect mode based on argument completeness; allow explicit override with --interactive.

### Logic

```python
if args.interactive or missing_required_args():
    # Launch interactive prompts
    collected_args = prompt_for_missing_args(args)
else:
    # Use provided arguments
    collected_args = args
```text

### Rationale

- **User-Friendly**: Default to interactive when args are missing (better UX)
- **Automation-Friendly**: Support fully non-interactive mode for scripts/CI
- **Explicit Control**: --interactive flag allows forcing interactive mode

### 6. Error Handling Strategy

**Decision**: Use argparse's error handling with custom formatting for better UX.

### Approach

- Let argparse handle basic errors (unknown args, type errors)
- Override `ArgumentParser.error()` to format messages consistently
- Provide "did you mean?" suggestions for typos
- Include relevant help section in error output

### Example Error Output

```text
error: argument --title/-t: required in non-interactive mode

Try 'scaffold-paper --help' for more information.
```text

### Rationale

- **Consistency**: All errors follow same format
- **Guidance**: Point users to help or correction
- **Standards**: Follows POSIX error message conventions

## Architecture Overview

```text
┌─────────────────────────────────────────────────────────────┐
│                     CLI Entry Point                         │
│                  (scripts/scaffold-paper)                   │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                  Argument Parser Module                     │
│              (tooling/scaffolding/arg_parser.py)            │
│                                                             │
│  ┌───────────────────────────────────────────────────┐    │
│  │  create_parser() -> ArgumentParser                │    │
│  │    - Define arguments (--title, --author, etc.)   │    │
│  │    - Set defaults and help text                   │    │
│  │    - Configure validation rules                   │    │
│  └───────────────────────────────────────────────────┘    │
│                         │                                   │
│                         ▼                                   │
│  ┌───────────────────────────────────────────────────┐    │
│  │  parse_arguments(args) -> Namespace               │    │
│  │    - Parse command-line args                      │    │
│  │    - Apply built-in validation                    │    │
│  │    - Return structured namespace                  │    │
│  └───────────────────────────────────────────────────┘    │
│                         │                                   │
│                         ▼                                   │
│  ┌───────────────────────────────────────────────────┐    │
│  │  validate_arguments(args) -> Result               │    │
│  │    - Custom validation (paths, years, etc.)       │    │
│  │    - Business logic validation                    │    │
│  │    - Return errors or validated args              │    │
│  └───────────────────────────────────────────────────┘    │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│              User Prompt Module (if interactive)            │
│           (tooling/scaffolding/user_prompts.py)             │
└─────────────────────────────────────────────────────────────┘
```text

## API Contract

### Public Interface

```python
from tooling.scaffolding.arg_parser import create_parser, parse_arguments, validate_arguments

# Create parser
parser = create_parser()

# Parse arguments
args = parse_arguments(parser, sys.argv[1:])

# Validate arguments
validation_result = validate_arguments(args)
if not validation_result.is_valid:
    print(f"Error: {validation_result.error_message}")
    sys.exit(1)

# Access parsed values
title = args.title
author = args.author
output_dir = args.output_dir
```text

### Return Types

```python
# argparse.Namespace with attributes
class ParsedArguments:
    title: str | None
    author: str | None
    year: int
    output_dir: Path | None
    template: str
    interactive: bool

# Validation result
class ValidationResult:
    is_valid: bool
    error_message: str | None
    validated_args: ParsedArguments | None
```text

## Integration Points

### Dependencies

- **Upstream**: None (entry point for CLI)
- **Downstream**: User Prompts module (interactive mode), Scaffold Generator module (uses parsed args)

### Data Flow

```text
Command Line → Argument Parser → Validation → [Interactive Prompts?] → Scaffold Generator
```text

## Testing Strategy

Tests will be defined in issue #765 (Test phase), but key test categories include:

- **Valid Arguments**: All supported argument combinations
- **Invalid Arguments**: Type errors, missing required, unknown args
- **Default Values**: Verify all defaults are applied correctly
- **Short Aliases**: Test -t, -a, -o, etc. work identically to long forms
- **Help Text**: Verify --help output is correct and complete
- **Error Messages**: Verify clear, actionable error messages
- **Interactive Detection**: Test mode detection logic
- **Edge Cases**: Empty strings, special characters, path traversal attempts

## Performance Considerations

- **Parse Time**: argparse is fast (<1ms for typical arguments)
- **Memory**: Negligible (single Namespace object)
- **Validation**: Custom validators should complete in <10ms total

## Security Considerations

- **Path Traversal**: Validate output_dir doesn't escape intended directory structure
- **Injection**: Sanitize inputs before passing to subprocess calls (in downstream modules)
- **Overwrite Protection**: Validate output directory doesn't contain existing work (in business logic)

## Future Enhancements

Potential future improvements (not in scope for this issue):

- Configuration file support (.scaffoldrc)
- Environment variable defaults (SCAFFOLD_AUTHOR, etc.)
- Shell completion scripts (bash, zsh)
- Argument validation plugins (custom validators for different paper types)
- Internationalization (i18n) for help text and error messages

## References

### Source Plan

- [notes/plan/03-tooling/01-paper-scaffolding/03-cli-interface/01-argument-parsing/plan.md](notes/plan/03-tooling/01-paper-scaffolding/03-cli-interface/01-argument-parsing/plan.md)

### Parent Plan

- [notes/plan/03-tooling/01-paper-scaffolding/03-cli-interface/plan.md](notes/plan/03-tooling/01-paper-scaffolding/03-cli-interface/plan.md)

### Related Issues

- Issue #765: [Test] Argument Parsing - Test Implementation
- Issue #766: [Implementation] Argument Parsing - Code Implementation
- Issue #767: [Packaging] Argument Parsing - Integration and Packaging
- Issue #768: [Cleanup] Argument Parsing - Refactor and Finalize

### Related Documentation

- [ADR-001: Language Selection for Tooling](../../review/adr/ADR-001-language-selection-tooling.md) - Justification for Python choice
- [Python argparse Documentation](https://docs.python.org/3/library/argparse.html) - Standard library reference
- [CLAUDE.md](CLAUDE.md) - Project conventions and language guidelines

### Existing Examples

- [scripts/create_single_component_issues.py](scripts/create_single_component_issues.py) - Example of Python argparse usage in project

## Implementation Notes

(This section will be populated during implementation with discoveries, challenges, and solutions)
