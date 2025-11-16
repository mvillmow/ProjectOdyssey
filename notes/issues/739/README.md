# Issue #739: [Plan] Template System - Design and Documentation

## Objective

Design and document a template system for generating paper files with customizable content. The system will use
simple variable substitution to insert paper-specific information (title, author, date, etc.) into boilerplate files,
enabling consistent and efficient creation of new paper implementations.

## Deliverables

- Template files for README, Mojo code, tests, and documentation
- Variable system for content customization with validation rules
- Template rendering engine using simple string substitution
- Comprehensive design documentation covering all three child components

## Success Criteria

- [ ] Templates cover all required paper files (README, implementation stubs, tests, documentation)
- [ ] Variables can be substituted in templates with clear naming conventions
- [ ] Rendering produces valid, formatted files with proper error handling
- [ ] All child plans are completed successfully (#740-#743)
- [ ] Design decisions documented with rationale

## Design Decisions

### 1. Simple String Substitution vs. Complex Templating Engine

**Decision**: Use simple string replacement (e.g., `{{PAPER_TITLE}}` → actual title) instead of a full templating
engine like Jinja2 or Mustache.

**Rationale**:

- Paper templates have straightforward substitution needs - primarily metadata fields
- Simple approach is easier to understand, maintain, and debug
- Reduces dependencies and complexity
- Sufficient for current requirements (YAGNI principle)
- Can upgrade later if more complex logic is needed

**Alternatives Considered**:

- Jinja2: Overkill for simple substitution; adds dependency
- Mustache: Similar to Jinja2, more than needed
- Python f-strings: Requires eval() which is a security risk
- Custom mini-language: Unnecessary complexity

### 2. Variable Naming Convention

**Decision**: Use uppercase with underscores for template placeholders (e.g., `PAPER_TITLE`, `AUTHOR_NAME`, `DATE`).

**Rationale**:

- Clear visual distinction from surrounding content
- Common convention in templating systems (environment variables, constants)
- Easy to identify in template files
- Reduces risk of accidental substitution

**Format**: `{{VARIABLE_NAME}}` with double curly braces for clear demarcation.

### 3. Template Storage Location

**Decision**: Store templates in a dedicated `templates/` directory structure within the tooling section.

**Rationale**:

- Centralized location for all template files
- Easy to locate and modify
- Supports template versioning and updates
- Allows for template categories (README, code, tests, docs)

**Structure**:

```text
templates/
├── readme/
│   └── README.md.template
├── code/
│   └── implementation.mojo.template
├── tests/
│   └── test_paper.mojo.template
└── docs/
    └── usage.md.template
```

### 4. Template Variable System Architecture

**Decision**: Three-tier variable system:

1. **Required variables** - Must be provided (PAPER_TITLE, AUTHOR_NAME)
2. **Optional variables with defaults** - Can be omitted (DATE defaults to current date)
3. **Computed variables** - Derived from other variables (PAPER_SLUG from PAPER_TITLE)

**Rationale**:

- Ensures essential information is always present
- Provides convenience with sensible defaults
- Reduces repetitive input
- Maintains consistency (computed values follow conventions)

### 5. Error Handling Strategy

**Decision**: Fail fast with clear error messages for missing required variables; warn but continue for optional
variables.

**Rationale**:

- Prevents incomplete file generation
- Clear feedback helps users correct issues quickly
- Warnings for optional variables maintain flexibility
- Aligns with principle of least astonishment (POLA)

**Error Message Format**:

```text
Error: Missing required variable 'PAPER_TITLE'
Please provide a value for PAPER_TITLE using --title option
```

### 6. Template Rendering Output

**Decision**: Render templates to in-memory strings first, validate, then write to disk atomically.

**Rationale**:

- Prevents partial file writes on error
- Allows validation before disk modification
- Supports dry-run mode for testing
- Cleaner error recovery

### 7. Implementation Language

**Decision**: Implement template rendering in Python (not Mojo) as part of tooling infrastructure.

**Rationale**:

- Tooling/automation context - Python is appropriate per ADR-001
- No performance-critical operations (one-time file generation)
- Better string manipulation libraries in Python
- Consistent with other paper scaffolding tools
- Can integrate with subprocess operations for file system tasks

**Reference**: [ADR-001](../../review/adr/ADR-001-language-selection-tooling.md) - Language selection strategy

## Architecture Overview

The template system consists of three main components:

### Component 1: Create Templates

**Purpose**: Design and create template files for all standard paper components.

**Key Files**:

- `templates/readme/README.md.template` - Paper documentation template
- `templates/code/implementation.mojo.template` - Mojo implementation stub
- `templates/tests/test_paper.mojo.template` - Test file template
- `templates/docs/usage.md.template` - Usage documentation template

**Design Principles**:

- Self-documenting with clear placeholder names
- Minimal but complete - cover essential structure
- Easy to read and modify
- Include inline comments for guidance

### Component 2: Template Variables

**Purpose**: Define variable schema, validation rules, and default values.

**Standard Variables**:

Required:

- `PAPER_TITLE` - Full paper title (e.g., "LeNet-5: Gradient-Based Learning")
- `AUTHOR_NAME` - Original paper author(s)
- `PAPER_YEAR` - Publication year

Optional with defaults:

- `DATE` - Creation date (defaults to current date)
- `IMPLEMENTER` - Who implemented it (defaults to system user)

Computed:

- `PAPER_SLUG` - URL-friendly version of title (e.g., "lenet-5")
- `PAPER_DIR` - Directory name derived from slug

**Validation Rules**:

- PAPER_TITLE: Non-empty, max 200 chars
- AUTHOR_NAME: Non-empty, max 100 chars
- PAPER_YEAR: Valid 4-digit year (1950-2025)
- DATE: Valid ISO 8601 format (YYYY-MM-DD)

### Component 3: Template Rendering

**Purpose**: Process templates and substitute variables to generate output files.

**Workflow**:

1. Load template file from disk
2. Parse for variable placeholders ({{VAR_NAME}})
3. Validate all required variables are provided
4. Apply default values for optional variables
5. Compute derived variables
6. Perform string substitution
7. Validate rendered output
8. Write to target file atomically

**Error Handling**:

- Missing required variables → Error and exit
- Invalid variable values → Error with validation message
- Missing optional variables → Warning and use default
- Template file not found → Error with helpful message
- Write permission issues → Error with file path

## Integration with Paper Scaffolding

The template system is the first component of the larger Paper Scaffolding tool:

```text
Paper Scaffolding CLI
├── Template System (this component)
│   ├── Template files
│   ├── Variable system
│   └── Rendering engine
├── Directory Generator
│   └── Creates paper directory structure
└── CLI Interface
    └── User-facing command-line tool
```

**Flow**:

1. User runs CLI: `paper-scaffold --title "LeNet-5" --author "LeCun et al."`
2. CLI validates input and prepares variables
3. Template System renders all template files
4. Directory Generator creates target structure
5. Rendered files written to appropriate locations
6. User gets complete, ready-to-use paper implementation

## References

- **Source Plan**: [notes/plan/03-tooling/01-paper-scaffolding/01-template-system/plan.md](../../plan/03-tooling/01-paper-scaffolding/01-template-system/plan.md)
- **Parent Component**: [notes/plan/03-tooling/01-paper-scaffolding/plan.md](../../plan/03-tooling/01-paper-scaffolding/plan.md)
- **Child Plans**:
  - Create Templates: [notes/plan/03-tooling/01-paper-scaffolding/01-template-system/01-create-templates/plan.md](../../plan/03-tooling/01-paper-scaffolding/01-template-system/01-create-templates/plan.md)
  - Template Variables: [notes/plan/03-tooling/01-paper-scaffolding/01-template-system/02-template-variables/plan.md](../../plan/03-tooling/01-paper-scaffolding/01-template-system/02-template-variables/plan.md)
  - Template Rendering: [notes/plan/03-tooling/01-paper-scaffolding/01-template-system/03-template-rendering/plan.md](../../plan/03-tooling/01-paper-scaffolding/01-template-system/03-template-rendering/plan.md)
- **Related Issues**:
  - #740 - [Test] Template System
  - #741 - [Implementation] Template System
  - #742 - [Package] Template System
  - #743 - [Cleanup] Template System
- **Architecture Decisions**:
  - [ADR-001: Language Selection for Tooling](../../review/adr/ADR-001-language-selection-tooling.md)
- **Development Principles**: [CLAUDE.md](../../../CLAUDE.md#key-development-principles)

## Implementation Notes

*This section will be populated during implementation phases (#740-#743) with:*

- Actual template file locations and formats
- Variable validation implementation details
- Rendering engine API and usage examples
- Edge cases discovered during testing
- Performance considerations if any
- Integration points with directory generator and CLI

---

**Planning Phase Completed**: 2025-11-15
**Next Steps**: Proceed to Test (#740), Implementation (#741), and Packaging (#742) phases in parallel
