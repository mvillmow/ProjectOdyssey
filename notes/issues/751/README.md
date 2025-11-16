# Issue #751: [Impl] Generate Files - Implementation

## Objective

Implement file generation logic that uses templates to create all required files for a new paper implementation. This component is part of the Directory Generator subsystem within the Paper Scaffolding Tool, responsible for rendering templates with paper-specific metadata and writing files to appropriate locations in the generated paper structure.

## Deliverables

- Generated README.md for new paper implementations
- Implementation stub files (.mojo) with proper structure
- Test file stubs following project conventions
- Documentation files (notes, configuration)
- File generation functions with proper error handling
- File permission management logic

## Success Criteria

- [ ] All required files are generated from templates
- [ ] File content is properly rendered with paper metadata
- [ ] Files are placed in correct locations within directory structure
- [ ] File permissions are set appropriately (executable for scripts, read-write for docs)
- [ ] Generated files use proper encoding (UTF-8) and line endings
- [ ] Files are created in consistent order (README first, then code, then tests)
- [ ] Existing files are not overwritten without warning
- [ ] Implementation passes all tests from issue #752
- [ ] Code follows Mojo best practices and coding standards
- [ ] All functions are documented with clear docstrings

## Design Decisions

### 1. File Generation Order

**Decision**: Generate files in a specific order - README first, then implementation files, then test files.

**Rationale**:
- README provides context for developers examining the generated structure
- Implementation files define the structure that tests will validate
- Tests are last since they depend on implementation structure
- Consistent ordering makes debugging easier and provides predictable behavior

**Alternatives Considered**:
- Alphabetical order: Less intuitive, doesn't reflect logical dependencies
- Random order: Unpredictable, harder to debug

### 2. Overwrite Protection

**Decision**: Do not overwrite existing files without explicit warning.

**Rationale**:
- Prevents accidental data loss
- Makes the tool safer for re-running on existing directories
- Provides clear feedback when files already exist
- Aligns with idempotent design principle from parent plan

**Implementation**: Check file existence before writing, return warnings for existing files.

### 3. File Permissions

**Decision**: Set appropriate permissions based on file type (scripts executable, docs read-write).

**Rationale**:
- Follows Unix conventions
- Scripts need execute permissions to run
- Documentation files should be read-write for editing
- Proper permissions prevent common usage errors

**Implementation**: Use platform-appropriate chmod after file creation.

### 4. Template Rendering

**Decision**: Accept pre-rendered template content rather than rendering templates directly.

**Rationale**:
- Separation of concerns: template rendering is handled by template system
- This component focuses solely on file I/O operations
- Easier to test (no template dependencies)
- More flexible (can accept content from any source)

**Alternatives Considered**:
- Direct template rendering: Creates tight coupling, harder to test
- Combined approach: Violates single responsibility principle

### 5. Encoding and Line Endings

**Decision**: Use UTF-8 encoding with platform-appropriate line endings.

**Rationale**:
- UTF-8 is standard for modern codebases, supports all characters
- Platform-appropriate line endings prevent Git issues
- Matches repository pre-commit hooks configuration
- Ensures generated files are immediately usable

**Implementation**: Explicit UTF-8 encoding, let Python handle line endings based on platform.

## Architecture Notes

### Component Context

This component is part of the 3-tier Paper Scaffolding Tool architecture:

1. **Template System** - Handles template loading and rendering
2. **Directory Generator** (parent component)
   - Create Structure (sibling - creates directories)
   - **Generate Files (this component)** - writes rendered files
   - Validate Output (sibling - verifies structure)
3. **CLI Interface** - User-facing command-line tool

### Integration Points

**Inputs from**:
- Template system provides rendered content
- Create Structure component provides target paths
- Paper metadata from CLI interface

**Outputs to**:
- Validate Output component checks generated files
- File system receives written files
- Error/warning messages to CLI

### Key Interfaces

```mojo
fn generate_file(path: Path, content: String, permissions: FilePermissions) -> Result[None, Error]:
    """Write a single file with specified content and permissions."""
    pass

fn generate_files(file_specs: List[FileSpec]) -> Result[GenerationReport, Error]:
    """Generate multiple files from specifications, returning summary report."""
    pass

fn check_overwrite(path: Path) -> Result[Bool, Error]:
    """Check if file exists and would be overwritten."""
    pass
```

## Implementation Strategy

### Core Functionality

1. **Path Resolution**
   - Resolve relative paths to absolute paths
   - Validate parent directories exist (created by Create Structure)
   - Handle platform-specific path separators

2. **File Writing**
   - Write content with UTF-8 encoding
   - Set permissions after writing
   - Atomic writes where possible (write to temp, then rename)

3. **Error Handling**
   - Return detailed errors for I/O failures
   - Collect warnings for existing files
   - Provide context in error messages (which file, what operation)

4. **Validation**
   - Verify written files are readable
   - Check file sizes match expectations
   - Validate permissions were set correctly

### Testing Strategy (TDD)

Tests are defined in issue #752. Implementation should be driven by:
- Unit tests for individual file operations
- Integration tests for multi-file generation
- Error case tests (permissions, disk full, etc.)
- Edge case tests (empty content, special characters)

## References

### Source Documentation

- **Source Plan**: `/notes/plan/03-tooling/01-paper-scaffolding/02-directory-generator/02-generate-files/plan.md`
- **Parent Plan**: `/notes/plan/03-tooling/01-paper-scaffolding/02-directory-generator/plan.md`

### Related Issues

- **#752** - [Test] Generate Files - Test suite for this implementation
- **#753** - [Impl] Generate Files (duplicate, should be different component)
- **#754** - [Package] Generate Files - Integration and packaging
- **755** - [Cleanup] Generate Files - Refactoring and finalization

### Related Components

- **#748** - Create Structure (sibling component)
- **#757** - Validate Output (sibling component)
- Directory Generator (parent component)
- Template System (dependency)

### Documentation

- **Mojo Language Guidelines**: `/agents/mojo-language-review-specialist.md`
- **Implementation Specialist Role**: `/.claude/agents/implementation-specialist.md`
- **TDD Workflow**: `/notes/review/README.md` (5-phase development)

## Implementation Notes

*This section will be filled during implementation with:*
- *Discoveries and insights from coding*
- *Challenges encountered and solutions*
- *Performance considerations*
- *Technical decisions made during implementation*
- *Integration issues and resolutions*
