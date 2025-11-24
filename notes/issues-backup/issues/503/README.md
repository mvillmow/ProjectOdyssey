# Issue #503: [Plan] Create Templates - Design and Documentation

## Objective

Design and create template files for all standard paper components including README, implementation stubs, test files, and documentation. Templates serve as the foundation for generating new paper implementations using simple variable substitution.

## Deliverables

- README.md template with paper metadata placeholders
- Mojo implementation file templates for common patterns
- Test file templates with example test structures
- Documentation templates for usage guides

## Success Criteria

- [ ] Templates exist for all required paper files (README, Mojo code, tests, docs)
- [ ] Templates include appropriate placeholders using consistent naming (e.g., {{PAPER_TITLE}}, {{AUTHOR_NAME}})
- [ ] Template format is consistent and readable across all file types
- [ ] Templates can be easily customized for different papers
- [ ] Templates are self-documenting with clear placeholder names
- [ ] Templates follow repository conventions for structure and formatting

## Design Decisions

### Template Structure

**Decision**: Use simple string substitution with double-brace placeholders (e.g., `{{VARIABLE_NAME}}`)

### Rationale

- Avoids dependency on complex templating engines (Jinja2, Mustache, etc.)
- Easy to create and modify templates without learning new syntax
- Clear visual distinction between template variables and code
- Aligns with parent plan guidance: "Use simple string substitution for templates"

### Template Files to Create

Based on repository structure and paper requirements, we need templates for:

1. **README.md Template**
   - Paper title, author, publication year
   - Abstract/summary section
   - Implementation status
   - References to original paper
   - Usage instructions

1. **Mojo Implementation Template**
   - Module docstring with paper reference
   - Common imports (stdlib, tensor operations)
   - Placeholder struct/class definitions
   - Main model implementation stub
   - Forward pass skeleton

1. **Test File Template**
   - Test module structure
   - Example test cases for model initialization
   - Example test cases for forward pass
   - Placeholder for paper-specific tests
   - Import statements for testing framework

1. **Documentation Template**
   - Architecture overview section
   - API reference placeholder
   - Usage examples section
   - Performance notes section

### Placeholder Naming Convention

**Decision**: Use UPPERCASE_SNAKE_CASE for all template variables

### Standard Variables

- `{{PAPER_TITLE}}` - Full paper title
- `{{AUTHOR_NAME}}` - Primary author name(s)
- `{{PUBLICATION_YEAR}}` - Year paper was published
- `{{PAPER_URL}}` - Link to original paper (arXiv, journal, etc.)
- `{{IMPLEMENTATION_DATE}}` - Date template was instantiated
- `{{MODEL_NAME}}` - Primary model/architecture name
- `{{DESCRIPTION}}` - Brief description of the paper

### Rationale

- Consistent with common templating conventions
- Easy to identify in template files
- Reduces risk of accidental substitution of real code
- Self-documenting variable names

### Template Organization

**Decision**: Store templates in `templates/paper/` directory structure

### Structure

```text
templates/
└── paper/
    ├── README.md.template
    ├── model.mojo.template
    ├── test_model.mojo.template
    └── docs.md.template
```text

### Rationale

- Centralized location for all paper templates
- `.template` extension clearly identifies template files
- Hierarchical organization allows for future template categories
- Easy to locate and modify templates

### Template Content Strategy

**Decision**: Start with minimal templates covering essential structure

### Minimal Template Approach

- Include only required sections and boilerplate
- Avoid over-prescriptive content that limits flexibility
- Provide clear comments/guidance within templates
- Allow for easy extension and customization

### Rationale

- Aligns with plan guidance: "Start with minimal templates"
- Easier to maintain and evolve
- Reduces friction when customizing for specific papers
- Follows YAGNI principle (You Aren't Gonna Need It)

### Template Validation

**Decision**: Templates must pass linting and formatting checks when placeholders are replaced

### Validation Requirements

- Mojo templates must be valid Mojo syntax (after substitution)
- Markdown templates must pass markdownlint checks
- All templates must follow repository coding standards
- Templates must include proper docstrings and comments

### Rationale

- Ensures generated code meets quality standards
- Reduces manual cleanup after template instantiation
- Provides good starting point for implementation
- Aligns with pre-commit hooks and CI checks

## References

### Source Plan

- [Plan File](../../plan/03-tooling/01-paper-scaffolding/01-template-system/01-create-templates/plan.md)
- [Parent Plan - Template System](../../plan/03-tooling/01-paper-scaffolding/01-template-system/plan.md)
- [Grandparent Plan - Paper Scaffolding](../../plan/03-tooling/01-paper-scaffolding/plan.md)

### Related Issues

- Issue #503: [Plan] Create Templates (this issue)
- Issue #504: [Test] Create Templates
- Issue #505: [Impl] Create Templates
- Issue #506: [Package] Create Templates
- Issue #507: [Cleanup] Create Templates

### Supporting Documentation

- [Repository Structure](../../plan/01-foundation/01-directory-structure/plan.md)
- [Template System Overview](../../plan/03-tooling/01-paper-scaffolding/01-template-system/plan.md)
- [Mojo Coding Standards](../../../CLAUDE.md#language-preference)
- [Markdown Standards](../../../CLAUDE.md#markdown-standards)

### Sibling Components

- [02-template-variables](../../plan/03-tooling/01-paper-scaffolding/01-template-system/02-template-variables/plan.md) - Variable system for customization
- [03-template-rendering](../../plan/03-tooling/01-paper-scaffolding/01-template-system/03-template-rendering/plan.md) - Template rendering engine

## Implementation Notes

*This section will be populated during implementation with findings, decisions, and lessons learned.*

### Design Considerations

Key design considerations to address during implementation:

1. **Template Location**: Verify templates directory exists or should be created
1. **File Extensions**: Confirm `.template` extension or use different convention
1. **Variable Syntax**: Ensure `{{VAR}}` syntax doesn't conflict with Mojo/Markdown syntax
1. **Example Content**: Determine level of example content to include in templates
1. **Comments**: Balance between helpful guidance and template clutter

### Open Questions

Questions to resolve during implementation:

1. Should templates include example imports for common libraries?
1. Should test templates include fixtures or just basic test structure?
1. Should README template include badges (CI, license, etc.)?
1. Should templates include placeholder license headers?
1. Should templates reference specific Mojo stdlib versions?

### Dependencies

External dependencies to consider:

- Repository must have `templates/` directory created
- Templates must align with current Mojo language version
- Templates must be compatible with pre-commit hooks
- Templates must follow markdown linting rules

### Success Metrics

Quantifiable metrics for template quality:

- All 4 template files created (README, model, test, docs)
- Minimum 5 standard variables defined and used consistently
- Templates pass validation when placeholders replaced with sample values
- Templates generate files that pass pre-commit hooks
- Templates can be instantiated in under 60 seconds (manual substitution)

---

**Planning Phase Status**: In Progress

### Next Steps

1. Complete design decisions above
1. Hand off to Test phase (Issue #504)
1. Hand off to Implementation phase (Issue #505)
1. Coordinate with Packaging phase (Issue #506)
