# Issue #746: [Impl] Create Structure - Implementation

## Objective

Implement the core directory creation logic that builds the complete folder hierarchy for a new paper implementation. This component creates all necessary subdirectories following the repository's established conventions, enabling automated paper scaffolding.

## Deliverables

- Created directory hierarchy matching the template structure
- Directory creation log showing operations performed
- Error handling for existing directories (graceful handling without overwriting)
- Normalized directory naming (lowercase, hyphens instead of spaces)

## Success Criteria

- [ ] All required directories are created (src, tests, docs, examples, data/{raw,processed,cache}, scripts, configs, notebooks)
- [ ] Directory names follow naming conventions (lowercase with hyphens)
- [ ] Permissions are set correctly (readable/writable by owner, readable by group)
- [ ] Existing directories are handled gracefully (no errors when re-running)
- [ ] All tests pass (from issue #745)
- [ ] Code follows Mojo best practices (prefer `fn` over `def`, use `owned`/`borrowed` for parameters)
- [ ] Implementation is clean, documented, and maintainable

## Design Decisions

### Architecture

**Directory Creation Strategy**: Implement a single-pass creation algorithm that:

1. **Validates inputs first** - Check target base directory exists and is writable before any operations
2. **Normalizes paper name** - Convert to lowercase, replace spaces/underscores with hyphens for consistency
3. **Creates parent directory** - Create main paper directory using normalized name
4. **Creates subdirectories** - Create all standard subdirectories in one pass
5. **Sets permissions** - Apply standard permissions (0755 for directories)
6. **Logs operations** - Return detailed log of what was created

**Rationale**: This approach follows the Fail-Fast principle - validate everything before making changes. It's simple (KISS), doesn't add unnecessary features (YAGNI), and follows the Single Responsibility Principle (each step has one job).

### Directory Structure

Based on `papers/_template/`, the standard structure is:

```text
<paper-name>/
├── src/              # Source code
├── tests/            # Test files
├── examples/         # Usage examples
├── data/             # Data files
│   ├── raw/          # Raw data
│   ├── processed/    # Processed data
│   └── cache/        # Cached results
├── scripts/          # Utility scripts
├── configs/          # Configuration files
└── notebooks/        # Jupyter notebooks
```

**Rationale**: This structure separates concerns clearly (Separation of Concerns principle) and follows established repository conventions for consistency.

### Implementation Language

**Decision**: Use Mojo for implementation.

**Rationale**:

- Mojo is the preferred language for ML/AI implementations per ADR-001
- Directory creation is a core ML tooling operation, not automation
- Mojo provides sufficient filesystem APIs for this task
- Type safety and performance benefits outweigh convenience of Python

**Alternative Considered**: Python with `os.makedirs()` was considered but rejected to maintain consistency with project language preferences.

### Error Handling

**Decision**: Gracefully handle existing directories without errors.

**Rationale**: The tool should be idempotent (running it multiple times is safe). If a directory already exists, log it but don't fail. This follows the Principle of Least Astonishment - users expect tools to be safe to re-run.

### Naming Convention

**Decision**: Normalize paper names to lowercase with hyphens.

**Examples**:

- "LeNet-5" → "lenet-5"
- "Attention Is All You Need" → "attention-is-all-you-need"
- "BERT_Model" → "bert-model"

**Rationale**: Consistent naming prevents issues with case-sensitive filesystems, spaces in paths, and URL generation. Hyphens are more readable than underscores for multi-word names.

## API Contract

### Function Signature

```mojo
fn create_paper_structure(
    base_path: String,
    paper_name: String
) raises -> DirectoryCreationResult:
    """Create complete directory structure for a new paper implementation.

    Args:
        base_path: Path to parent directory (must exist and be writable)
        paper_name: Name of the paper (will be normalized)

    Returns:
        DirectoryCreationResult containing:
        - created_directories: List of created directory paths
        - skipped_directories: List of already-existing directories
        - normalized_name: The normalized paper name used

    Raises:
        ValueError: If base_path doesn't exist or isn't writable
        PermissionError: If unable to create directories due to permissions
    """
```

### Expected Behavior

**Input validation**:

- `base_path` must exist
- `base_path` must be writable
- `paper_name` must be non-empty

**Directory creation**:

- Create `<base_path>/<normalized_name>/`
- Create all standard subdirectories
- Skip directories that already exist (log them)
- Set permissions to 0755 (rwxr-xr-x)

**Return value**:

- List of successfully created directories
- List of skipped (existing) directories
- Normalized paper name used

## References

### Source Plan

- [Plan: Create Structure](/home/mvillmow/ml-odyssey-manual/notes/plan/03-tooling/01-paper-scaffolding/02-directory-generator/01-create-structure/plan.md)
- [Parent Plan: Directory Generator](/home/mvillmow/ml-odyssey-manual/notes/plan/03-tooling/01-paper-scaffolding/02-directory-generator/plan.md)

### Related Issues

- Issue #744: [Plan] Create Structure - Design and Documentation (planning phase)
- Issue #745: [Test] Create Structure - Write Tests (test phase)
- Issue #747: [Package] Create Structure - Integration and Packaging (packaging phase)
- Issue #748: [Cleanup] Create Structure - Refactor and Finalize (cleanup phase)

### Template Reference

- Template structure: `/home/mvillmow/ml-odyssey-manual/papers/_template/`

### Documentation

- [Mojo Language Review Specialist](/home/mvillmow/ml-odyssey-manual/.claude/agents/mojo-language-review-specialist.md) - Mojo coding patterns
- [ADR-001: Language Selection for Tooling](/home/mvillmow/ml-odyssey-manual/notes/review/adr/ADR-001-language-selection-tooling.md) - Language choice rationale
- [Repository Structure](/home/mvillmow/ml-odyssey-manual/README.md#repository-architecture) - Overall project organization

## Implementation Notes

(This section will be populated during implementation with:)

- Actual filesystem APIs used
- Edge cases discovered
- Performance observations
- Deviations from plan (if any)
- Lessons learned
