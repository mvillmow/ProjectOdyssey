# Issue #789: [Plan] Discover Tests

## Objective

Design and document the test discovery component that automatically finds all test files in the repository by
traversing the directory structure and identifying Mojo and Python test files following established naming conventions.

## Deliverables

- Test discovery logic specification
- API contract definition for test discovery interface
- File pattern matching rules (test_*.mojo, test_*.py, *_test.mojo)
- Exclusion pattern specification for non-test directories
- Test metadata structure design (paper association, file type)
- Discovery statistics format specification
- Caching strategy for repeated runs
- Comprehensive design documentation

## Success Criteria

- [ ] Test discovery algorithm is fully specified
- [ ] File matching patterns are clearly defined
- [ ] Exclusion rules cover all non-test directories
- [ ] Test metadata structure supports paper association tracking
- [ ] Performance considerations are documented (fast traversal, caching)
- [ ] API interface is documented with clear input/output contracts
- [ ] Edge cases are identified and handling strategies documented
- [ ] Integration points with test runner are defined

## Design Decisions

### 1. File Naming Conventions

**Decision**: Support multiple test file naming patterns

### Rationale

- Flexibility for different testing styles and preferences
- Align with both Mojo and Python community conventions
- Support legacy code that may use different patterns

### Patterns Supported

- `test_*.mojo` - Prefix pattern (standard Python/pytest style)
- `*_test.mojo` - Suffix pattern (common in Go, C++)
- `test_*.py` - Python test files

### Alternatives Considered

- Single pattern only (`test_*.mojo`) - Rejected due to lack of flexibility
- Configuration-based patterns - Deferred to future enhancement

### 2. Directory Traversal Strategy

**Decision**: Use standard filesystem walking with explicit exclusions

### Rationale

- Simple, well-understood approach
- Easy to debug and maintain
- Predictable performance characteristics
- Standard library support in both Mojo and Python

### Exclusions

- Hidden directories (`.git`, `.pixi`, etc.)
- Build artifacts (`build/`, `dist/`, `__pycache__/`)
- Virtual environments (`venv/`, `.venv/`)
- Documentation directories (`docs/` unless tests present)
- Node modules (`node_modules/`)

### Alternatives Considered

- Glob-based discovery - Less flexible for complex exclusions
- Git-based discovery - Unnecessary dependency on git
- Configuration-based discovery - Over-engineered for current needs

### 3. Test Metadata Structure

**Decision**: Capture paper association, file type, and path information

### Metadata Fields

- `path`: Absolute path to test file
- `relative_path`: Path relative to repository root
- `file_type`: 'mojo' or 'python'
- `paper_name`: Extracted from directory structure (e.g., 'lenet5', 'alexnet')
- `test_category`: 'unit', 'integration', 'validation' (extracted from path)
- `file_size`: Size in bytes (for statistics)
- `last_modified`: Timestamp (for cache invalidation)

### Rationale

- Paper association enables filtering tests by implementation
- File type allows appropriate test runner selection
- Category information supports focused test execution
- Timestamps enable smart cache invalidation

### Alternatives Considered

- Minimal metadata (path only) - Insufficient for filtering and reporting
- Extended metadata (AST parsing for test names) - Over-engineered for discovery phase

### 4. Caching Strategy

**Decision**: Implement filesystem-based caching with timestamp validation

**Cache Location**: `.cache/test_discovery.json` in repository root

### Cache Structure

```json
{
  "last_scan": "2025-11-15T10:30:00Z",
  "repository_root": "/path/to/repo",
  "discovered_tests": [
    {
      "path": "/path/to/test_file.mojo",
      "relative_path": "papers/lenet5/tests/test_model.mojo",
      "file_type": "mojo",
      "paper_name": "lenet5",
      "test_category": "unit",
      "file_size": 1024,
      "last_modified": "2025-11-14T15:20:00Z"
    }
  ],
  "statistics": {
    "total_tests": 42,
    "mojo_tests": 35,
    "python_tests": 7,
    "papers_covered": 3
  }
}
```text

### Cache Invalidation

- Directory structure changes detected via last_modified timestamps
- Explicit cache clear option via CLI flag
- Automatic invalidation after 24 hours

### Rationale

- Speeds up repeated test runs
- Simple JSON format is human-readable and debuggable
- Timestamp validation ensures cache freshness
- Statistics pre-computation improves reporting speed

### Alternatives Considered

- No caching - Poor performance for large repositories
- In-memory caching only - Lost between runs
- Database caching (SQLite) - Over-engineered for simple key-value storage

### 5. Performance Optimization

**Decision**: Implement parallel directory traversal with early termination

### Techniques

- Parallel scanning of top-level paper directories
- Early termination on excluded directories (don't recurse)
- Lazy loading of metadata (only when requested)
- Pre-compiled regex patterns for file matching

### Expected Performance

- < 100ms for cached discovery
- < 500ms for fresh discovery on typical repository (10-20 papers, 100-200 test files)
- Linear scaling with number of test files

### Rationale

- Fast discovery enables frequent test runs
- Parallel scanning leverages multi-core processors
- Early termination reduces unnecessary filesystem operations

### Alternatives Considered

- Sequential scanning - Slower but simpler (deferred for initial implementation)
- Watchdog-based discovery - Complex, adds runtime dependency

### 6. API Design

### Function Signature

```mojo
fn discover_tests(
    root_dir: String,
    patterns: List[String] = ["test_*.mojo", "*_test.mojo", "test_*.py"],
    exclude_dirs: List[String] = [".git", "build", "__pycache__"],
    use_cache: Bool = True
) -> DiscoveryResult:
    """Discover all test files in the repository.

    Args:
        root_dir: Repository root directory path
        patterns: File name patterns to match (glob-style)
        exclude_dirs: Directory names to skip during traversal
        use_cache: Whether to use cached results if available

    Returns:
        DiscoveryResult containing:
            - tests: List[TestMetadata] - Discovered test files with metadata
            - statistics: DiscoveryStats - Summary statistics
            - cache_hit: Bool - Whether results came from cache
    """
    pass
```text

### Rationale

- Clear, self-documenting function signature
- Sensible defaults reduce boilerplate
- Return type encapsulates all discovery information
- Cache control gives users flexibility

### Alternatives Considered

- Class-based API - More complex for simple use case
- Multiple functions (discover, discover_cached) - Less discoverable
- Configuration object parameter - Over-engineered for 4 parameters

## References

- **Source Plan**: [/notes/plan/03-tooling/02-testing-tools/01-test-runner/01-discover-tests/plan.md](notes/plan/03-tooling/02-testing-tools/01-test-runner/01-discover-tests/plan.md)
- **Parent Plan**: [/notes/plan/03-tooling/02-testing-tools/01-test-runner/plan.md](notes/plan/03-tooling/02-testing-tools/01-test-runner/plan.md)
- **Testing Tools Plan**: [/notes/plan/03-tooling/02-testing-tools/plan.md](notes/plan/03-tooling/02-testing-tools/plan.md)
- **Related Issues**:
  - #790 - [Test] Discover Tests
  - #791 - [Implementation] Discover Tests
  - #792 - [Packaging] Discover Tests
  - #793 - [Cleanup] Discover Tests
- **Architecture Documentation**: [/agents/hierarchy.md](agents/hierarchy.md)
- **Language Guidelines**: [/notes/review/adr/ADR-001-language-selection-tooling.md](notes/review/adr/ADR-001-language-selection-tooling.md)

## Implementation Notes

(This section will be populated during implementation phases with findings, challenges, and decisions made during
development)
