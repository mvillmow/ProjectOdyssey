# Issue #749: [Plan] Generate Files - Design and Documentation

## Objective

Design the file generation logic that uses templates to create all required files for a new paper implementation,
including rendering templates with paper-specific metadata and writing files to appropriate locations.

## Deliverables

- **Comprehensive Design Document**: Architecture specification for file generation system
- **API Contracts**: Interface definitions for file generation functions
- **File Generation Strategy**: Order, validation, and error handling approach
- **Integration Specification**: How file generation integrates with template rendering and directory creation

## Success Criteria

- [ ] All required files are identified and documented
- [ ] File content rendering process is designed
- [ ] File placement strategy is defined
- [ ] File permissions approach is specified
- [ ] Error handling for overwrites and conflicts is designed
- [ ] Integration with template system is documented

## Design Decisions

### Architecture Overview

The file generation system is the final step in the paper scaffolding workflow:

1. **Template System** → Renders templates with paper-specific variables
1. **Directory Generator** → Creates folder structure
1. **File Generator** (this component) → Writes rendered content to files

### Key Architectural Decisions

#### 1. File Generation Order

**Decision**: Generate files in a specific, consistent order: README → Implementation → Tests → Documentation

### Rationale

- README first provides immediate context if generation fails mid-process
- Implementation stubs before tests follows TDD workflow expectations
- Documentation last as it may reference other files
- Consistent ordering aids debugging and validation

### Alternatives Considered

- Alphabetical order: Rejected - no semantic meaning
- Parallel generation: Rejected - adds complexity without clear benefit for this use case

#### 2. Overwrite Protection

**Decision**: Never overwrite existing files without explicit warning/confirmation

### Rationale

- Prevents accidental data loss
- Aligns with principle of least surprise (POLA)
- Supports incremental/partial regeneration workflows

### Implementation

- Check file existence before writing
- Return clear error messages indicating which files already exist
- Provide option for force-overwrite flag (for automation scenarios)

#### 3. File Permissions

**Decision**: Set permissions to standard repository defaults (0644 for files, 0755 for directories)

### Rationale

- Consistent with Unix conventions
- Readable by all, writable by owner
- No execute permissions needed for source files

### Platform Considerations

- Use Python's `os.chmod()` with explicit mode values
- Handle Windows gracefully (permissions may be no-op)

#### 4. Encoding and Line Endings

**Decision**: UTF-8 encoding, platform-native line endings

### Rationale

- UTF-8 is universal standard for source code
- Git handles line ending normalization via `.gitattributes`
- Platform-native avoids surprising developers on different OSes

### Implementation

- Python default text mode handles line endings
- Explicit `encoding='utf-8'` in all file operations

#### 5. Validation Strategy

**Decision**: Validate rendered content before writing files

### Rationale

- Catch template errors early (before filesystem changes)
- Ensures atomic-like behavior (all files valid or none written)
- Reduces cleanup complexity on errors

### Validation Checks

- Content is non-empty
- Content is valid UTF-8
- No null bytes or control characters (except newlines/tabs)
- File paths are valid and within target directory

### API Design

#### Primary Interface

```python
def generate_files(
    rendered_templates: Dict[str, str],
    target_directory: Path,
    overwrite: bool = False
) -> GenerationResult:
    """
    Generate files from rendered templates.

    Args:
        rendered_templates: Mapping of relative file paths to rendered content
        target_directory: Root directory for file generation
        overwrite: If True, overwrite existing files; if False, raise error

    Returns:
        GenerationResult with created files, skipped files, and any errors

    Raises:
        FileExistsError: If files exist and overwrite=False
        ValidationError: If rendered content fails validation
        PermissionError: If insufficient permissions to write files
    """
```text

#### Result Type

```python
@dataclass
class GenerationResult:
    created_files: List[Path]
    skipped_files: List[Path]  # Already existed
    errors: List[str]
    success: bool
```text

#### File Type Mapping

| Template Type | Target Location | Filename Pattern |
|--------------|----------------|------------------|
| README | `{paper_dir}/` | `README.md` |
| Implementation | `{paper_dir}/src/` | `{module_name}.mojo` |
| Tests | `{paper_dir}/tests/` | `test_{module_name}.mojo` |
| Documentation | `{paper_dir}/docs/` | `{doc_name}.md` |

### Error Handling

#### Error Categories

1. **Pre-generation Errors** (fail before any files written):
   - Invalid target directory
   - Invalid file paths (outside target directory)
   - Content validation failures

1. **Generation Errors** (may leave partial state):
   - Permission errors during file writing
   - Disk space errors
   - I/O errors

1. **Post-generation Errors** (files created, but verification failed):
   - Permission setting failures
   - Validation of written files

#### Recovery Strategy

- Pre-generation errors: Return immediately, no cleanup needed
- Generation errors: Attempt to clean up any created files
- Post-generation errors: Log warnings but don't delete files

### Integration Points

#### Inputs from Template Rendering System

The template rendering component (issue #730) provides:

- `Dict[str, str]` mapping file paths to rendered content
- All variable substitution already completed
- Content ready to write (no further processing needed)

#### Inputs from Directory Structure Creation

The directory creation component (issue #744) provides:

- Root directory path for the paper
- Guarantee that target directories exist
- Valid directory permissions set

#### Outputs to Validation System

The output validation component (issue #754) expects:

- List of all created files
- Ability to read and validate each file
- Clear error messages if generation failed

### Performance Considerations

- **Small file count**: Typically 5-15 files per paper (no optimization needed)
- **Sequential I/O**: Files small enough that async I/O provides no benefit
- **Synchronous API**: Simpler error handling, adequate performance

### Testing Strategy

Refer to issue #750 for comprehensive test plan. Key test scenarios:

1. Successful generation of all file types
1. Overwrite protection (error when files exist)
1. Forced overwrite with flag
1. Partial failure handling
1. Permission errors
1. Invalid file paths
1. Content validation failures

## References

### Source Plans

- **Component Plan**: [/home/mvillmow/ml-odyssey-manual/notes/plan/03-tooling/01-paper-scaffolding/02-directory-generator/02-generate-files/plan.md](notes/plan/03-tooling/01-paper-scaffolding/02-directory-generator/02-generate-files/plan.md)
- **Parent Plan**: [/home/mvillmow/ml-odyssey-manual/notes/plan/03-tooling/01-paper-scaffolding/02-directory-generator/plan.md](notes/plan/03-tooling/01-paper-scaffolding/02-directory-generator/plan.md)

### Related Issues

- **#750**: [Test] Generate Files - Test implementation and validation
- **#751**: [Implementation] Generate Files - Build the file generation system
- **#752**: [Package] Generate Files - Integration and packaging
- **#753**: [Cleanup] Generate Files - Refactor and finalize

### Upstream Dependencies

- **#730**: [Implementation] Template Rendering - Provides rendered content
- **#744**: [Implementation] Create Structure - Provides target directories

### Downstream Dependencies

- **#754**: [Test] Validate Output - Validates generated files

### Architecture Documentation

- [Agent Hierarchy](agents/hierarchy.md)
- [5-Phase Workflow](notes/review/README.md)
- [Paper Scaffolding Overview](notes/plan/03-tooling/01-paper-scaffolding/plan.md)

## Implementation Notes

*This section will be populated during the implementation phase (issue #751) with:*

- Implementation discoveries
- Design adjustments
- Edge cases encountered
- Performance observations
- Integration challenges

---

**Status**: Planning Complete
**Next Phase**: Testing (issue #750) and Implementation (issue #751) can proceed in parallel
