# Issue #759: [Plan] Directory Generator - Design and Documentation

## Objective

Design and document the directory generation system that creates the complete folder structure for new paper
implementations. This planning phase defines the architecture, API contracts, and design decisions for a robust,
idempotent directory generator that ensures all required directories are created in the proper hierarchy and validates
the output structure.

## Deliverables

- Complete directory hierarchy creation specification
- Generated files from templates specification
- Validation report format and rules
- Creation summary output specification
- Architecture and design documentation
- API contracts and interfaces

## Success Criteria

- [ ] All required directories specification is defined
- [ ] File placement rules are documented
- [ ] Repository conventions compliance is specified
- [ ] Validation rules and success criteria are defined
- [ ] All child plans (#765, #771, #777) are integrated into design
- [ ] Architecture supports idempotent operations
- [ ] Clear feedback mechanism is specified

## Design Decisions

### 1. Architecture Overview

The Directory Generator follows a three-stage pipeline architecture:

1. **Create Structure** - Build directory hierarchy
2. **Generate Files** - Populate directories with template-based files
3. **Validate Output** - Verify completeness and correctness

This separation of concerns allows each stage to be tested independently and enables partial recovery from failures.

### 2. Idempotency Strategy

**Decision**: Generator must be safe to run multiple times on the same target.

**Rationale**:

- Users may need to re-run after interruption
- Partial failures should be recoverable
- Updates to templates should be re-applicable

**Implementation**:

- Check for existing directories before creation
- Never overwrite existing files without explicit user consent
- Provide clear status messages about what already exists vs. what was created
- Use atomic operations where possible

### 3. Directory Structure Specification

**Standard Paper Layout**:

```text
papers/<paper-name>/
├── src/                    # Implementation code (.mojo files)
├── tests/                  # Test files
├── docs/                   # Documentation
├── notebooks/              # Jupyter notebooks (if applicable)
├── data/                   # Sample data or datasets
└── README.md               # Paper-specific documentation
```

**Naming Conventions**:

- Paper names are normalized to lowercase with hyphens (e.g., "LeNet-5" → "lenet-5")
- Source files use snake_case (e.g., "conv_layer.mojo")
- Test files mirror source with "test_" prefix (e.g., "test_conv_layer.mojo")

### 4. Template Integration

**Decision**: Directory generator consumes output from Template System (#753).

**Interface Contract**:

- Template system provides rendered file content
- Directory generator handles file I/O and path resolution
- Clear separation: templates handle "what", generator handles "where"

**Rationale**:

- Single responsibility - templates focus on content, generator on structure
- Allows independent evolution of each system
- Simplifies testing (can mock template output)

### 5. Validation Approach

**Decision**: Validation runs as separate stage after generation, never modifies files.

**Validation Checks**:

1. **Completeness** - All required directories exist
2. **Structure** - Directories follow repository conventions
3. **Files** - All expected files are present
4. **Format** - Files are properly formatted (encoding, line endings)
5. **Syntax** - Generated code has no obvious syntax errors

**Output**: Structured validation report with clear pass/fail status and actionable error messages.

### 6. Error Handling Strategy

**Graceful Degradation**:

- Continue processing remaining items after non-fatal errors
- Collect all errors and report at end
- Distinguish between fatal (stop immediately) and recoverable errors

**Error Categories**:

- **Fatal**: Target directory unwritable, invalid paper metadata
- **Recoverable**: Individual file creation failure, validation warning
- **Informational**: Directory already exists, file skipped

### 7. Progress Reporting

**Decision**: Provide real-time feedback during generation.

**Output Format**:

```text
Creating directory structure...
  ✓ papers/lenet-5/
  ✓ papers/lenet-5/src/
  ✓ papers/lenet-5/tests/
  ⚠ papers/lenet-5/docs/ (already exists)

Generating files...
  ✓ papers/lenet-5/README.md
  ✓ papers/lenet-5/src/model.mojo
  ✓ papers/lenet-5/tests/test_model.mojo

Validating output...
  ✓ Directory structure complete
  ✓ All required files present
  ✓ File format valid

Summary: Created 5 directories, generated 8 files
```

### 8. API Design

**Primary Interface**:

```python
def generate_paper_structure(
    target_dir: Path,
    paper_name: str,
    paper_metadata: dict,
    templates: dict[str, str],
    dry_run: bool = False
) -> GenerationReport:
    """
    Generate complete directory structure for a paper implementation.

    Args:
        target_dir: Base directory for paper (e.g., papers/)
        paper_name: Name of paper (will be normalized)
        paper_metadata: Metadata for template rendering
        templates: Rendered template content {filename: content}
        dry_run: If True, report what would be done without creating anything

    Returns:
        GenerationReport with status, created items, errors
    """
```

**Supporting Types**:

```python
@dataclass
class GenerationReport:
    success: bool
    directories_created: list[Path]
    files_created: list[Path]
    skipped_items: list[tuple[Path, str]]  # (path, reason)
    errors: list[tuple[Path, str]]  # (path, error_message)
    validation_report: ValidationReport
```

### 9. Alternatives Considered

**Alternative 1: Single-Pass Generation**

- Combine directory creation and file generation in one pass
- **Rejected**: Harder to test, less flexible, couples structure to content

**Alternative 2: Configuration-Driven Structure**

- Use YAML/JSON to define directory structure
- **Rejected**: Over-engineering for current needs, harder to maintain

**Alternative 3: File Overwrite by Default**

- Always overwrite existing files
- **Rejected**: Dangerous, could lose user modifications

### 10. Dependencies and Integration

**Depends On**:

- Template System (#753) - Provides rendered file content
- Repository conventions (established in Foundation section)

**Depended On By**:

- CLI Interface (#783) - Consumes generator API
- Integration tests (#762) - Validates end-to-end workflow

**Integration Points**:

- Template System: Receives rendered content via dict[str, str]
- CLI: Invoked by CLI with user-provided parameters
- Validation: Uses repository conventions for correctness checks

## References

- **Source Plan**: [notes/plan/03-tooling/01-paper-scaffolding/02-directory-generator/plan.md](../../plan/03-tooling/01-paper-scaffolding/02-directory-generator/plan.md)
- **Parent Plan**: [notes/plan/03-tooling/01-paper-scaffolding/plan.md](../../plan/03-tooling/01-paper-scaffolding/plan.md)
- **Related Issues**:
  - #753 - [Plan] Template System (dependency)
  - #760 - [Test] Directory Generator
  - #761 - [Implementation] Directory Generator
  - #762 - [Packaging] Directory Generator
  - #763 - [Cleanup] Directory Generator
  - #765 - [Plan] Create Structure (child component)
  - #771 - [Plan] Generate Files (child component)
  - #777 - [Plan] Validate Output (child component)
  - #783 - [Plan] CLI Interface (consumer)
- **Documentation**:
  - [5-Phase Development Workflow](../../review/README.md)
  - [Repository Architecture](../../../CLAUDE.md#repository-architecture)
  - [Agent Hierarchy](../../../agents/hierarchy.md)

## Implementation Notes

This section will be populated during the implementation phase with:

- Architectural refinements discovered during implementation
- Edge cases and special handling requirements
- Performance considerations and optimizations
- Testing strategies and coverage gaps
- Integration challenges and solutions
