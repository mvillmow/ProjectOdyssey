# Issue #784: [Plan] Paper Scaffolding - Design and Documentation

## Objective

Create comprehensive design documentation and specifications for a CLI tool that generates complete directory
structure and boilerplate files for new paper implementations. This scaffolding system will use templates to create
consistent paper layouts with all required files, documentation, and test stubs, making it trivial to start a new
paper implementation while enforcing consistency across the repository.

## Deliverables

The following planning documentation and design specifications:

1. **Architecture Design**
   - Component architecture for scaffolding system
   - Integration with existing tooling infrastructure
   - Data flow and dependencies

1. **API Contracts and Interfaces**
   - Template system API
   - Directory generator interface
   - CLI interface specification

1. **Design Documentation**
   - Template system design (variable substitution, file generation)
   - Directory structure conventions and validation
   - User interaction patterns and CLI design

1. **Implementation Strategy**
   - Module breakdown and dependencies
   - Language selection (Mojo vs Python) with justification
   - Error handling and validation approach

## Success Criteria

- [ ] Complete architectural design documented
- [ ] API contracts defined for all three major components
- [ ] Design decisions documented with rationale
- [ ] Integration points with existing tooling identified
- [ ] Implementation approach specified for Test (#785), Implementation (#786), and Packaging (#787) phases
- [ ] Language selection justified per ADR-001 requirements
- [ ] Template design allows easy customization with paper metadata
- [ ] Generator design ensures valid directory structure creation
- [ ] CLI design provides intuitive interface for paper creation
- [ ] All generated papers will follow repository conventions

## Design Decisions

### 1. Three-Component Architecture

**Decision**: Split scaffolding tool into three independent, composable components:

1. **Template System** - File generation with variable substitution
1. **Directory Generator** - Structure creation and validation
1. **CLI Interface** - User interaction and orchestration

### Rationale

- **Modularity**: Each component can be developed, tested, and maintained independently
- **Reusability**: Template system and directory generator can be used programmatically without CLI
- **Testability**: Clear boundaries enable focused unit testing
- **Extensibility**: New template types or directory structures can be added without CLI changes

### Alternatives Considered

- **Monolithic Design**: Single script handling all functionality - Rejected due to complexity and testability concerns
- **Two-Component Split**: Combined template and directory generation - Rejected to maintain clear separation of
  concerns (template rendering vs file system operations)

### 2. Simple String Substitution for Templates

**Decision**: Use simple variable substitution (e.g., `{{paper_title}}`, `{{author}}`) instead of complex templating
engines like Jinja2.

### Rationale

- **Simplicity (KISS)**: Paper templates have straightforward variable needs
- **No External Dependencies**: Avoid adding templating engine dependencies
- **Easy to Create/Modify**: Contributors can understand and edit templates without learning templating syntax
- **Performance**: String substitution is faster than template parsing
- **Sufficient**: Project needs only basic variable replacement, not complex logic

### Alternatives Considered

- **Jinja2 Templates**: Full-featured templating - Rejected as over-engineering for current needs
- **Python f-strings**: Inline formatting - Rejected because templates need to be separate files
- **Mako Templates**: Another templating engine - Rejected for same reasons as Jinja2

### 3. Language Selection: Python for CLI Tool

**Decision**: Implement scaffolding tool in Python, not Mojo.

**Rationale** (per ADR-001):

- **Automation Category**: This is a development automation tool, not ML/AI implementation
- **Subprocess Requirements**: Tool needs to validate created structure, potentially run validation commands
- **File System Operations**: Heavy reliance on file creation, directory traversal, path manipulation
- **Python Strengths**: Excellent file I/O, mature libraries (pathlib, argparse), subprocess handling
- **Mojo Limitations**: Mojo v0.25.7 cannot capture subprocess output (documented limitation)

### Technical Justification

```python
# Example: Why Python is the right tool for this use case
# 1. Clean file system operations
from pathlib import Path
Path("papers/new-paper/src").mkdir(parents=True, exist_ok=True)

# 2. Simple template rendering
template = Path("templates/README.md").read_text()
rendered = template.replace("{{title}}", paper_title)

# 3. Subprocess validation with output capture
result = subprocess.run(["mojo", "format", "--check", "src/"],
                       capture_output=True, text=True)
if result.returncode != 0:
    print(f"Validation failed: {result.stderr}")
```text

**Note**: This is documented per ADR-001 Section 4.3 requirements for Python usage in automation.

### 4. Idempotent Directory Generation

**Decision**: Generator should be safe to run multiple times on the same target directory.

### Rationale

- **Safety**: Prevents accidental data loss if tool is run twice
- **Partial Recovery**: Can recover from interrupted executions
- **User-Friendly**: No catastrophic failures if user makes a mistake
- **Testing**: Simplifies test cleanup and validation

### Implementation Strategy

- Check if directories exist before creating
- Skip file generation if target file already exists
- Provide clear warnings about existing content
- Option to force overwrite with explicit flag

### 5. Interactive and Non-Interactive Modes

**Decision**: CLI supports both interactive prompts and command-line argument modes.

### Rationale

- **Interactive Mode**: Better UX for manual paper creation, guides users through process
- **Non-Interactive Mode**: Required for scripting and automation
- **Flexibility**: Users can choose workflow that fits their context

### Implementation

```bash
# Interactive mode (prompts for all info)
python scripts/scaffold_paper.py

# Non-interactive mode (all args provided)
python scripts/scaffold_paper.py --title "LeNet-5" --author "LeCun et al." --year 1998 --output papers/lenet5
```text

### 6. Repository Structure Conventions

**Decision**: Scaffolding follows established repository structure exactly.

### Structure

```text
papers/<paper-name>/
├── README.md              # Paper overview, architecture, references
├── src/
│   ├── __init__.mojo     # Module initialization
│   ├── model.mojo        # Model implementation
│   └── layers.mojo       # Custom layers (if needed)
├── tests/
│   ├── __init__.mojo     # Test module initialization
│   ├── test_model.mojo   # Model tests
│   └── test_layers.mojo  # Layer tests
└── docs/
    ├── architecture.md   # Detailed architecture documentation
    └── training.md       # Training approach and hyperparameters
```text

### Rationale

- **Consistency**: All papers have identical structure
- **Discoverability**: Developers know where to find components
- **Tooling**: Other tools can assume consistent structure
- **Documentation**: Clear conventions reduce cognitive load

### 7. Template Variables

**Decision**: Define standard variable set for all templates.

### Core Variables

- `{{paper_title}}` - Full paper title
- `{{paper_name}}` - Filesystem-safe name (lowercase, hyphens)
- `{{author}}` - Paper author(s)
- `{{year}}` - Publication year
- `{{date}}` - Current date (ISO 8601 format)
- `{{description}}` - Brief paper description

### Rationale

- **Standardization**: Consistent variable naming across templates
- **Completeness**: Covers all common paper metadata needs
- **Extensibility**: Can add custom variables without breaking existing templates

### 8. Validation and Error Handling

**Decision**: Validate inputs and outputs with clear error messages.

### Validation Points

1. **Input Validation**: Paper name is filesystem-safe, target directory is writable
1. **Template Validation**: All required variables are provided, templates exist
1. **Output Validation**: All expected files were created, structure matches specification

### Error Handling Strategy

- Fail early with clear error messages
- Provide actionable guidance for fixing issues
- Clean up partial state on failure (if safe)
- Log detailed information for debugging

### 9. TDD Approach

**Decision**: Follow test-driven development for all components.

### Test Coverage Requirements

- **Template System Tests**:
  - Variable substitution correctness
  - Missing variable handling
  - Template file loading
  - Edge cases (empty values, special characters)

- **Directory Generator Tests**:
  - Structure creation validation
  - Idempotency verification
  - Permission handling
  - Error recovery

- **CLI Tests**:
  - Argument parsing correctness
  - Interactive prompt flow
  - Output formatting
  - Error message clarity

## Architecture

### Component Diagram

```text
┌─────────────────────────────────────────────────┐
│              CLI Interface (#788)               │
│  - Argument parsing (argparse)                  │
│  - Interactive prompts                          │
│  - Output formatting and progress               │
└───────────────┬─────────────────────────────────┘
                │
                │ orchestrates
                │
    ┌───────────┴───────────┐
    │                       │
    ▼                       ▼
┌─────────────────┐  ┌──────────────────────┐
│ Template System │  │ Directory Generator  │
│  - Load templates│  │  - Create structure │
│  - Substitute vars│  │  - Generate files   │
│  - Render output│  │  - Validate output  │
└─────────────────┘  └──────────────────────┘
```text

### Data Flow

```text
User Input (CLI args or prompts)
    │
    ▼
Parse and Validate Arguments
    │
    ▼
Load Paper Metadata
    │
    ├──────────────────┬──────────────────┐
    ▼                  ▼                  ▼
Template Loading   Directory Creation   Validation
(Template System)  (Dir Generator)      (Dir Generator)
    │                  │                  │
    └──────────────────┴──────────────────┘
                       │
                       ▼
            Report Results to User
```text

### Module Breakdown

1. **scripts/scaffold_paper.py** (Main CLI script)
   - Entry point and argument parsing
   - User interaction and prompts
   - Orchestration of template and directory components
   - Progress reporting and error handling

1. **scripts/templates/** (Template files directory)
   - README.md.template
   - model.mojo.template
   - test_model.mojo.template
   - __init__.mojo.template
   - architecture.md.template

1. **scripts/lib/template_system.py** (Template rendering)
   - load_template(template_path: Path) -> str
   - substitute_variables(template: str, variables: dict) -> str
   - render_template(template_path: Path, variables: dict) -> str
   - validate_variables(required: set, provided: dict) -> bool

1. **scripts/lib/directory_generator.py** (Structure creation)
   - create_directory_structure(target: Path, paper_name: str) -> bool
   - generate_files(target: Path, templates: dict, variables: dict) -> list
   - validate_structure(target: Path) -> bool
   - cleanup_on_error(target: Path) -> None

## API Contracts

### Template System API

```python
class TemplateSystem:
    """Handles template loading and variable substitution."""

    def __init__(self, template_dir: Path):
        """Initialize with template directory path."""
        pass

    def load_template(self, template_name: str) -> str:
        """
        Load template file contents.

        Args:
            template_name: Name of template file (e.g., "README.md")

        Returns:
            Template contents as string

        Raises:
            FileNotFoundError: If template doesn't exist
        """
        pass

    def render(self, template_name: str, variables: dict) -> str:
        """
        Render template with variable substitution.

        Args:
            template_name: Name of template file
            variables: Dictionary of variable names to values

        Returns:
            Rendered template contents

        Raises:
            ValueError: If required variables are missing
        """
        pass

    def validate_variables(self, template_name: str, variables: dict) -> tuple[bool, list]:
        """
        Validate that all required variables are provided.

        Args:
            template_name: Name of template file
            variables: Dictionary of variable names to values

        Returns:
            Tuple of (is_valid: bool, missing_variables: list)
        """
        pass
```text

### Directory Generator API

```python
class DirectoryGenerator:
    """Handles directory structure creation and file generation."""

    def __init__(self, template_system: TemplateSystem):
        """Initialize with template system instance."""
        pass

    def create_structure(self, target_dir: Path, paper_name: str) -> bool:
        """
        Create paper directory structure.

        Args:
            target_dir: Parent directory for new paper
            paper_name: Filesystem-safe paper name

        Returns:
            True if successful, False otherwise

        Raises:
            PermissionError: If target directory is not writable
        """
        pass

    def generate_files(self, paper_dir: Path, variables: dict) -> list[Path]:
        """
        Generate all paper files from templates.

        Args:
            paper_dir: Paper root directory
            variables: Template variables

        Returns:
            List of generated file paths

        Raises:
            FileExistsError: If files already exist and overwrite=False
        """
        pass

    def validate_output(self, paper_dir: Path) -> tuple[bool, list]:
        """
        Validate generated paper structure.

        Args:
            paper_dir: Paper root directory

        Returns:
            Tuple of (is_valid: bool, issues: list)
        """
        pass
```text

### CLI Interface API

```python
class PaperScaffoldCLI:
    """Command-line interface for paper scaffolding tool."""

    def __init__(self):
        """Initialize CLI with argument parser."""
        pass

    def parse_arguments(self, args: list = None) -> argparse.Namespace:
        """
        Parse command-line arguments.

        Args:
            args: Argument list (defaults to sys.argv)

        Returns:
            Parsed arguments namespace
        """
        pass

    def prompt_for_metadata(self) -> dict:
        """
        Interactively prompt user for paper metadata.

        Returns:
            Dictionary of paper metadata
        """
        pass

    def run(self, args: list = None) -> int:
        """
        Execute scaffolding process.

        Args:
            args: Command-line arguments

        Returns:
            Exit code (0 for success, non-zero for failure)
        """
        pass
```text

## Integration Points

### With Existing Tooling

1. **Foundation Structure** (Section 01)
   - Uses established directory conventions
   - Follows repository structure patterns

1. **Testing Tools** (Section 03-02)
   - Generated tests integrate with test runners
   - Test file naming follows conventions

1. **Validation Tools** (Section 03-04)
   - Generated code passes validation checks
   - Structure validation uses same rules

1. **CI/CD Pipelines** (Section 05)
   - Generated papers can be tested in CI
   - Follows same quality standards

### With Development Workflow

1. **Pre-commit Hooks**
   - Generated Mojo code passes `mojo format`
   - Generated markdown passes `markdownlint-cli2`

1. **GitHub Issues**
   - Can generate issue templates for new papers
   - Links to planning workflow

1. **Documentation**
   - Generated docs follow markdown standards
   - README templates link to project docs

## Implementation Strategy

### Phase 1: Plan (#784) - Current Phase

- Create this comprehensive planning documentation
- Define all API contracts and interfaces
- Document design decisions with rationale
- Establish success criteria for subsequent phases

### Phase 2: Test (#785) - TDD Implementation

**Dependencies**: Requires Plan (#784) completion

### Approach

1. Write template system tests (variable substitution, file loading)
1. Write directory generator tests (structure creation, validation)
1. Write CLI tests (argument parsing, prompts, output)
1. All tests should fail initially (no implementation yet)
1. Document test coverage expectations (>80%)

### Deliverables

- `tests/test_template_system.py`
- `tests/test_directory_generator.py`
- `tests/test_scaffold_paper_cli.py`
- Test fixtures and sample templates

### Phase 3: Implementation (#786) - Make Tests Pass

**Dependencies**: Requires Test (#785) completion

### Approach

1. Implement template system to pass tests
1. Implement directory generator to pass tests
1. Implement CLI interface to pass tests
1. Iterate until all tests pass
1. Ensure >80% test coverage

### Deliverables

- `scripts/scaffold_paper.py` (main CLI)
- `scripts/lib/template_system.py`
- `scripts/lib/directory_generator.py`
- `scripts/templates/*.template` (template files)

### Phase 4: Packaging (#787) - Integration

**Dependencies**: Requires Implementation (#786) completion

### Approach

1. Create sample paper using tool (end-to-end validation)
1. Document CLI usage in README
1. Add scaffolding tool to project documentation
1. Verify integration with existing tooling
1. Create user guide with examples

### Deliverables

- Updated `scripts/README.md` with scaffolding documentation
- Example usage guide
- Integration verification report

### Phase 5: Cleanup (#788) - Finalization

**Dependencies**: Requires Packaging (#787) completion

### Approach

1. Refactor based on lessons learned
1. Improve error messages based on testing
1. Optimize performance if needed
1. Final documentation review
1. Code review and quality check

### Deliverables

- Refactored, production-ready code
- Comprehensive documentation
- Performance optimization notes

## References

- **Source Plan**: [/notes/plan/03-tooling/01-paper-scaffolding/plan.md](notes/plan/03-tooling/01-paper-scaffolding/plan.md)
- **Parent Plan**: [/notes/plan/03-tooling/plan.md](notes/plan/03-tooling/plan.md)
- **Child Plans**:
  - [Template System](notes/plan/03-tooling/01-paper-scaffolding/01-template-system/plan.md)
  - [Directory Generator](notes/plan/03-tooling/01-paper-scaffolding/02-directory-generator/plan.md)
  - [CLI Interface](notes/plan/03-tooling/01-paper-scaffolding/03-cli-interface/plan.md)
- **Language Selection**: [ADR-001: Language Selection for Tooling](../../review/adr/ADR-001-language-selection-tooling.md)
- **Related Issues**:
  - Issue #785: [Test] Paper Scaffolding
  - Issue #786: [Implementation] Paper Scaffolding
  - Issue #787: [Packaging] Paper Scaffolding
  - Issue #788: [Cleanup] Paper Scaffolding
- **Development Principles**: [CLAUDE.md](CLAUDE.md)
- **5-Phase Workflow**: [/notes/review/README.md](notes/review/README.md)

## Implementation Notes

This section will be populated during subsequent phases (Test, Implementation, Packaging, Cleanup) with:

- Lessons learned during implementation
- Deviations from original design (with justification)
- Performance observations
- User feedback and adjustments
- Edge cases discovered during testing

Currently empty as this is the planning phase.
