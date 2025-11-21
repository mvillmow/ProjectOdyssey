# Issue #769: [Plan] User Prompts - Design and Documentation

## Objective

Design and document an interactive prompt system for collecting paper information from users when command-line arguments are not provided. The system will guide users through entering required metadata (title, author, description) with validation, defaults, and helpful error messages.

## Deliverables

- **Architecture Design**: Component structure and interaction patterns
- **API Specifications**: Function signatures, input/output contracts, validation rules
- **User Experience Design**: Prompt flow, message templates, error handling
- **Validation Rules**: Field requirements, format constraints, error messages
- **Integration Contracts**: Interface with argument parsing and paper scaffolding components

## Success Criteria

- [ ] Prompts are clear and informative with examples
- [ ] Input is validated in real-time with helpful feedback
- [ ] Default values are offered where appropriate
- [ ] Error messages help users fix mistakes
- [ ] Design supports both required and optional fields
- [ ] Conversational and user-friendly experience
- [ ] Architecture allows users to go back and change answers
- [ ] Non-interactive mode supported (all args provided)

## Design Decisions

### 1. Implementation Language

**Decision**: Use Python for the interactive prompt system.

### Rationale

- This is a CLI automation/tooling component (not ML/AI implementation)
- Requires robust user input handling and validation
- Benefits from Python's rich ecosystem for CLI tools (e.g., `prompt_toolkit`, `click`, or built-in `input()`)
- Follows ADR-001 guidance: "Python for automation tasks" when Mojo limitations apply
- No performance-critical operations - user interaction is I/O bound

### Justification per ADR-001

- Category: Automation/tooling (Section 03-tooling)
- Python is the right tool for interactive CLI applications
- Mojo lacks mature libraries for advanced prompt features (color, history, autocomplete)

### 2. Architecture Pattern

**Decision**: Implement a prompt orchestrator with field-specific validators.

### Components

```text
PromptOrchestrator
├── FieldDefinition (metadata: name, description, required, default, validator)
├── InputCollector (display prompts, collect input, handle retries)
├── Validator (validate input, generate error messages)
└── MetadataBuilder (construct validated paper metadata)
```text

### Rationale

- Separation of concerns: prompt logic, validation, metadata construction
- Extensible: easy to add new fields or validation rules
- Testable: each component can be unit tested independently
- Reusable: validators can be shared across fields

### 3. Prompt Flow

**Decision**: Sequential field-by-field prompts with immediate validation.

### Flow

1. Display field description and examples
1. Show default value (if available)
1. Collect user input
1. Validate immediately
1. If invalid: show error, re-prompt
1. If valid: move to next field
1. After all fields: display summary, confirm before proceeding

### Alternatives Considered

- **Form-style (all fields at once)**: Rejected - harder to validate incrementally, overwhelming for users
- **Wizard with back/forward navigation**: Deferred to future enhancement - adds complexity, not required for MVP

### Rationale

- Simple and intuitive
- Immediate feedback reduces frustration
- Easy to implement and test
- Matches user expectations for CLI tools

### 4. Field Definitions

### Required Fields

- **title**: Paper title (string, 1-200 chars, alphanumeric + punctuation)
- **author**: Author name(s) (string, 1-100 chars, letters + spaces + commas)

### Optional Fields

- **description**: Brief description (string, 0-500 chars, any printable characters)
- **year**: Publication year (integer, 1900-current year)

### Validation Rules

| Field | Required | Format | Example | Error Message |
|-------|----------|--------|---------|---------------|
| title | Yes | 1-200 chars, non-empty | "LeNet-5: Gradient-Based Learning" | "Title must be 1-200 characters" |
| author | Yes | 1-100 chars, letters/spaces/commas | "LeCun, Yann; Bottou, Leon" | "Author must be 1-100 characters" |
| description | No | 0-500 chars | "Convolutional neural network for digit recognition" | "Description too long (max 500 chars)" |
| year | No | Integer, 1900-{current_year} | "1998" | "Year must be between 1900 and {current_year}" |

### 5. Error Handling

**Decision**: Three-tier error handling strategy.

### Tiers

1. **Input Validation**: Catch format errors immediately (invalid chars, length violations)
1. **Semantic Validation**: Catch logical errors (year in future, duplicate paper names)
1. **System Errors**: Catch I/O errors (stdin closed, terminal not available)

### Error Message Format

```text
❌ Error: {clear description of what's wrong}
💡 Tip: {helpful suggestion to fix it}
📝 Example: {valid example}
```text

### Recovery Strategy

- Format errors: re-prompt with error message
- System errors: fall back to non-interactive mode or fail gracefully
- Max retries: 3 attempts per field, then abort with helpful message

### 6. Default Value Strategy

**Decision**: Provide smart defaults where possible, but require explicit confirmation.

**Default Sources** (priority order):

1. Environment variables (e.g., `ML_ODYSSEY_AUTHOR`)
1. Git config (e.g., `user.name` for author)
1. Previous session data (cached from last run)
1. Sensible fallbacks (e.g., current year)

### Display Format

```text
Enter paper title [default: {value}]:
```text

### Rationale

- Reduces typing for common values
- Speeds up workflow for power users
- Clear what the default is (transparency)

### 7. Integration Points

### Input from Argument Parsing

- Receive dictionary of parsed command-line arguments
- Only prompt for missing required fields
- Merge prompted values with CLI arguments

### Output to Paper Scaffolding

- Return validated `PaperMetadata` dictionary
- Keys: `title`, `author`, `description`, `year`
- All values are validated and sanitized

### Interface

```python
def collect_missing_metadata(
    parsed_args: Dict[str, Any],
    required_fields: List[str],
    optional_fields: List[str]
) -> Dict[str, Any]:
    """
    Collect missing metadata via interactive prompts.

    Args:
        parsed_args: Command-line arguments already provided
        required_fields: Fields that must be collected
        optional_fields: Fields that can be skipped

    Returns:
        Complete metadata dictionary with all values

    Raises:
        ValidationError: If user input is invalid after max retries
        SystemError: If terminal is not available
    """
```text

## References

- **Source Plan**: [/notes/plan/03-tooling/01-paper-scaffolding/03-cli-interface/02-user-prompts/plan.md](notes/plan/03-tooling/01-paper-scaffolding/03-cli-interface/02-user-prompts/plan.md)
- **Parent Component**: [/notes/plan/03-tooling/01-paper-scaffolding/03-cli-interface/plan.md](notes/plan/03-tooling/01-paper-scaffolding/03-cli-interface/plan.md)
- **Language Guidance**: [ADR-001: Language Selection for Tooling](../../review/adr/ADR-001-language-selection-tooling.md)

### Related Issues

- Issue #770: [Test] User Prompts - Write Tests
- Issue #771: [Implementation] User Prompts - Build Functionality
- Issue #772: [Packaging] User Prompts - Integration
- Issue #773: [Cleanup] User Prompts - Refactoring

### Dependencies

- Argument parsing component must be completed first (#766)
- Template system provides paper structure (#754-757)

## Implementation Notes

This section will be populated during the implementation phase with:

- Challenges encountered during development
- Design adjustments made during implementation
- Performance observations
- User feedback and usability findings
- Technical debt or future improvements identified
