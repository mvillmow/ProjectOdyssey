# Issue #508: [Plan] Template Variables - Design and Documentation

## Objective

Define and implement the variable system for template customization. Variables allow templates to be populated with
paper-specific information like title, author, date, and other metadata.

## Deliverables

- Variable definition schema
- Variable validation rules
- Default variable values
- Variable documentation

## Success Criteria

- [ ] All required variables are defined
- [ ] Variables have clear, consistent naming
- [ ] Validation catches invalid values
- [ ] Documentation explains variable usage

## Design Decisions

### Variable Naming Convention

**Decision**: Use uppercase with underscores for template placeholders (e.g., `PAPER_TITLE`, `AUTHOR_NAME`)

### Rationale

- Clear visual distinction from surrounding text in templates
- Follows common convention for template variables (Jinja2, Mustache, etc.)
- Easy to identify and search for in template files
- Reduces chance of accidental substring replacement

### Variable Categories

Based on the parent plan context, variables fall into these categories:

1. **Required Paper Metadata**
   - `PAPER_TITLE` - Full title of the research paper
   - `PAPER_AUTHOR` - Primary author(s) of the paper
   - `PAPER_YEAR` - Publication year
   - `PAPER_VENUE` - Publication venue (journal/conference)

1. **Optional Paper Information**
   - `PAPER_DOI` - Digital Object Identifier
   - `PAPER_ARXIV_ID` - arXiv identifier
   - `PAPER_URL` - Link to paper PDF or abstract

1. **Implementation Metadata**
   - `IMPLEMENTATION_AUTHOR` - Name of person implementing the reproduction
   - `IMPLEMENTATION_DATE` - Date of implementation creation
   - `IMPLEMENTATION_STATUS` - Status (e.g., "In Progress", "Complete")

1. **Project Structure**
   - `PROJECT_NAME` - Name of the implementation project
   - `MODULE_NAME` - Python/Mojo module name (lowercase, underscores)
   - `CLASS_NAME` - Main model class name (PascalCase)

### Validation Strategy

**Decision**: Use type-based validation with format checks

### Validation Rules

1. **Required vs Optional**: Distinguish between mandatory and optional variables
1. **Format Validation**:
   - Dates: ISO 8601 format (YYYY-MM-DD)
   - Years: 4-digit integer (1900-2100)
   - DOI: Standard DOI format (10.xxxx/yyyy)
   - Module names: Valid Python/Mojo identifier (lowercase, underscores only)
   - Class names: Valid PascalCase identifier

1. **Length Constraints**:
   - Titles: 1-200 characters
   - Authors: 1-500 characters (to accommodate multiple authors)
   - URLs: Valid URL format

### Default Values

**Decision**: Provide sensible defaults for optional variables

### Default Strategy

1. **Date/Time Defaults**:
   - `IMPLEMENTATION_DATE`: Current date in ISO format
   - `PAPER_YEAR`: None (must be provided)

1. **Status Defaults**:
   - `IMPLEMENTATION_STATUS`: "In Progress"

1. **Empty/None Defaults**:
   - Optional metadata (DOI, arXiv ID, URL): Empty string or None
   - Allow template to handle missing values gracefully

### Template Substitution Approach

**Decision**: Use simple string substitution, not complex templating engine

**Rationale** (from parent plan):

- Keeps implementation simple and maintainable
- Sufficient for the use case (paper scaffolding)
- Easy to debug and understand
- No additional dependencies required

### Implementation Approach

- Python's `str.replace()` or `str.format()` for basic substitution
- Consider `string.Template` for safety (prevents code injection)
- Process variables in deterministic order to avoid conflicts

### Variable Schema Format

**Decision**: Use structured schema for variable definitions

**Schema Structure** (YAML or Python dict):

```yaml
variables:
  PAPER_TITLE:
    type: string
    required: true
    description: "Full title of the research paper"
    validation:
      min_length: 1
      max_length: 200
    example: "Gradient-Based Learning Applied to Document Recognition"

  PAPER_AUTHOR:
    type: string
    required: true
    description: "Primary author(s) of the paper"
    validation:
      min_length: 1
      max_length: 500
    example: "LeCun, Y., Bottou, L., Bengio, Y., & Haffner, P."

  IMPLEMENTATION_DATE:
    type: date
    required: false
    default: "today"
    description: "Date of implementation creation"
    validation:
      format: "YYYY-MM-DD"
    example: "2025-11-15"

  MODULE_NAME:
    type: identifier
    required: true
    description: "Python/Mojo module name"
    validation:
      pattern: "^[a-z][a-z0-9_]*$"
    example: "lenet5"
```text

### Error Handling

**Decision**: Fail fast with clear error messages

### Error Scenarios

1. **Missing Required Variable**: Raise error listing missing variable name
1. **Invalid Format**: Show expected format and received value
1. **Validation Failure**: Explain which constraint was violated
1. **Unknown Variable**: Warn about undefined variables in templates (potential typo)

### Documentation Requirements

**Decision**: Document each variable with examples and constraints

### Documentation Sections

1. **Variable Reference Table**: All variables with type, required/optional, description
1. **Usage Examples**: Show common template usage patterns
1. **Validation Rules**: Explain constraints and valid formats
1. **Default Values**: List all defaults and how they are computed
1. **Error Messages**: Common errors and how to fix them

## References

### Source Plan

- [notes/plan/03-tooling/01-paper-scaffolding/01-template-system/02-template-variables/plan.md](../../../plan/03-tooling/01-paper-scaffolding/01-template-system/02-template-variables/plan.md)

### Parent Plan

- [Template System](../../../plan/03-tooling/01-paper-scaffolding/01-template-system/plan.md)

### Related Issues

- Issue #509: [Test] Template Variables - Test suite
- Issue #510: [Impl] Template Variables - Implementation
- Issue #511: [Package] Template Variables - Integration
- Issue #512: [Cleanup] Template Variables - Finalization

### Related Components

- Issue #503: [Plan] Create Templates - Defines template files that will use these variables
- Issue #513: [Plan] Template Rendering - Will consume variable definitions for rendering

## Implementation Notes

(To be filled during implementation phase)
