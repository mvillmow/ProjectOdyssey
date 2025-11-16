# Issue #736: [Impl] Template Rendering - Implementation

## Objective

Implement the rendering engine that processes templates and substitutes variables to generate final output files. The renderer takes template files and variable values, performs substitution, and produces ready-to-use paper files.

## Deliverables

- Rendered files with substituted content
- Error messages for missing/invalid variables
- Rendering status and logs

## Success Criteria

- [ ] Renderer correctly substitutes all variables
- [ ] Missing variables are detected and reported
- [ ] Rendered files are properly formatted
- [ ] Error handling provides clear messages

## Design Decisions

### 1. Variable Substitution Strategy

**Decision**: Use simple string replacement for variable substitution instead of a complex templating engine.

**Rationale**:
- The use case is straightforward: replace placeholders with values
- Reduces external dependencies and complexity
- Easier to maintain and debug
- Sufficient for paper scaffolding needs (title, author, date, etc.)

**Alternatives Considered**:
- **Complex templating engine (e.g., Jinja2-like)**: Rejected due to unnecessary complexity for simple variable substitution
- **Custom parser with expression support**: Rejected as YAGNI (You Ain't Gonna Need It) - no requirement for expressions or logic
- **Format strings**: Rejected due to potential security issues with arbitrary code execution

**Implementation Approach**: Use find-and-replace pattern matching for placeholders (e.g., `{{variable_name}}`)

### 2. Error Handling for Missing Variables

**Decision**: Detect and report missing variables with clear error messages.

**Rationale**:
- Prevents silent failures where templates have unreplaced placeholders
- Provides actionable feedback to users
- Supports debugging template issues
- Maintains data integrity in generated files

**Implementation Approach**:
- Parse template for all variable placeholders
- Check each placeholder against provided variable values
- Collect all missing variables before failing (don't fail on first missing variable)
- Report comprehensive list of missing variables in error message

### 3. File I/O Strategy

**Decision**: Load entire template files into memory, process, and write output files.

**Rationale**:
- Template files are small (paper scaffolding files)
- Simplifies processing logic
- Allows for validation before writing
- Atomic write operations (all-or-nothing)

**Alternatives Considered**:
- **Streaming processing**: Rejected due to unnecessary complexity for small files
- **In-place modification**: Rejected as templates should remain unchanged

### 4. Placeholder Format

**Decision**: Use double-brace format for placeholders: `{{variable_name}}`

**Rationale**:
- Familiar syntax from popular templating engines
- Visually distinct from normal text
- Unlikely to appear in normal documentation
- Easy to parse with simple string operations

**Alternatives Considered**:
- `${variable_name}`: Rejected as it may conflict with shell variable syntax
- `%variable_name%`: Rejected as less common and harder to visually distinguish
- `<variable_name>`: Rejected as it may conflict with HTML/XML tags

## References

### Source Documentation

- **Plan File**: `/home/mvillmow/ml-odyssey-manual/notes/plan/03-tooling/01-paper-scaffolding/01-template-system/03-template-rendering/plan.md`
- **Parent Component**: Template System (#739)

### Related Issues

- **#735**: [Test] Template Rendering - Write Tests (parallel phase)
- **#737**: [Package] Template Rendering - Integration and Packaging (parallel phase)
- **#738**: [Cleanup] Template Rendering - Refactor and Finalize (sequential phase)
- **#739**: [Plan] Template System - Design and Documentation (parent component)

### Shared Documentation

- **Agent Hierarchy**: `/home/mvillmow/ml-odyssey-manual/agents/hierarchy.md`
- **Delegation Rules**: `/home/mvillmow/ml-odyssey-manual/agents/delegation-rules.md`
- **Mojo Language Review**: `/home/mvillmow/ml-odyssey-manual/.claude/agents/mojo-language-review-specialist.md`

## Implementation Notes

### Key Requirements

1. **Load template files from disk**
   - Read template file content
   - Handle file I/O errors gracefully
   - Support various file encodings (default to UTF-8)

2. **Parse template content for variable placeholders**
   - Identify all `{{variable_name}}` patterns
   - Extract variable names
   - Maintain placeholder positions for substitution

3. **Substitute variables with provided values**
   - Replace each placeholder with corresponding value
   - Handle missing variables appropriately
   - Preserve file formatting and structure

4. **Write rendered content to output files**
   - Create output directory if needed
   - Write substituted content atomically
   - Set appropriate file permissions
   - Log rendering status

### Edge Cases to Handle

- **Missing variables**: Detect and report all missing variables before failing
- **Malformed placeholders**: Handle incomplete or nested braces (e.g., `{variable}`, `{{{variable}}}`)
- **Empty values**: Allow empty string values for variables
- **Special characters**: Ensure values with special characters are properly substituted
- **File permissions**: Handle read/write permission errors
- **Large files**: While unlikely, ensure reasonable performance for larger templates
- **Duplicate placeholders**: Multiple occurrences of same variable should all be replaced

### Language Selection

**Language**: Mojo (required for all ML/AI implementations and core tooling)

**Justification**: Template rendering is core tooling functionality that benefits from:
- Type safety for variable validation
- Performance for file I/O operations
- Memory safety for string operations
- Consistency with project architecture

**See**: [ADR-001](/home/mvillmow/ml-odyssey-manual/notes/review/adr/ADR-001-language-selection-tooling.md) for language selection strategy.

### Implementation Steps

Following TDD principles:

1. **Read tests from #735** to understand expected behavior
2. **Implement template file loading**
   - Create function to read template files
   - Handle file I/O errors
3. **Implement placeholder parsing**
   - Create function to extract all variable names from template
   - Handle malformed placeholders
4. **Implement variable substitution**
   - Create function to replace placeholders with values
   - Validate all required variables are provided
5. **Implement output file writing**
   - Create function to write rendered content
   - Handle directory creation if needed
6. **Add error handling and logging**
   - Clear error messages for missing variables
   - Status logging for successful renders

### Testing Integration

This implementation should pass all tests defined in issue #735. Key test scenarios:

- Basic variable substitution
- Multiple variables in single template
- Missing variable detection
- Empty variable values
- Special characters in values
- File I/O error handling

## Notes

- Keep implementation simple following KISS principle
- Follow Mojo best practices (prefer `fn` over `def`, use ownership model)
- Ensure code is well-documented with clear docstrings
- Log important operations for debugging
- Use type hints for all function parameters and returns
