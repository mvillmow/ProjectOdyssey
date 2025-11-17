# Foundation Tests

This directory contains tests for foundational repository components.

## Test Files

### test_papers_directory.py

Comprehensive test suite for `papers/` directory creation and management.

**Test Coverage**:

- **Unit Tests** (3 tests):
  - `test_create_papers_directory_success` - Verify successful directory creation
  - `test_papers_directory_permissions` - Verify read/write/execute permissions
  - `test_papers_directory_location` - Verify correct path and location

- **Edge Cases** (4 tests):
  - `test_create_papers_directory_already_exists` - Idempotent creation (no error when exists)
  - `test_create_papers_directory_parent_missing` - Parent directory creation with `parents=True`
  - `test_create_papers_directory_permission_denied` - Permission error handling
  - `test_create_papers_directory_without_exist_ok` - FileExistsError when `exist_ok=False`

- **Integration Tests** (3 tests):
  - `test_can_create_subdirectory_in_papers` - Verify subdirectory creation works
  - `test_papers_directory_can_contain_files` - Verify file creation works
  - `test_papers_directory_listing` - Verify directory listing works

- **Real-World Scenarios** (1 test):
  - `test_complete_workflow_papers_directory` - Complete workflow simulation

**Total Tests**: 11

**Test Principles**: Follows FIRST principles (Fast, Isolated, Repeatable, Self-validating, Timely)

### test_supporting_directories.py

Comprehensive test suite for supporting directories validation (benchmarks/, docs/, agents/, tools/, configs/).

**Test Coverage**:

- **Existence Tests** (5 tests):
  - `test_benchmarks_directory_exists` - Verify benchmarks/ exists
  - `test_docs_directory_exists` - Verify docs/ exists
  - `test_agents_directory_exists` - Verify agents/ exists
  - `test_tools_directory_exists` - Verify tools/ exists
  - `test_configs_directory_exists` - Verify configs/ exists

- **Location Tests** (3 tests):
  - `test_supporting_directories_at_root` - All directories at repository root
  - `test_directory_names_correct` - Directory names match specification
  - `test_directory_permissions` - Correct read/write/execute permissions

- **README Validation** (3 tests):
  - `test_each_directory_has_readme` - Every directory has README.md
  - `test_readme_not_empty` - READMEs have substantial content
  - `test_readme_has_title` - READMEs have markdown headers

- **Structure Tests** (5 tests):
  - `test_benchmarks_subdirectory_structure` - Validate benchmarks/ structure
  - `test_docs_subdirectory_structure` - Validate docs/ 4-tier structure
  - `test_agents_subdirectory_structure` - Validate agents/ structure
  - `test_tools_subdirectory_structure` - Validate tools/ structure
  - `test_configs_subdirectory_structure` - Validate configs/ structure

- **Integration Tests** (3 tests):
  - `test_all_supporting_directories_present` - All 5 directories exist
  - `test_directories_ready_for_content` - Can create subdirectories and files
  - `test_supporting_directories_relationship` - Flat structure at root

- **Real-World Scenarios** (1 test):
  - `test_complete_supporting_directories_workflow` - Complete workflow validation

**Total Tests**: 20

**Test Principles**: Follows FIRST principles, uses real filesystem (no mocks), validates critical paths

### conftest.py

Shared pytest fixtures for foundation tests:

- `repo_root` - Real repository root directory
- `benchmarks_dir` - Path to benchmarks/ directory
- `docs_dir` - Path to docs/ directory
- `agents_dir` - Path to agents/ directory
- `tools_dir` - Path to tools/ directory
- `configs_dir` - Path to configs/ directory
- `supporting_dirs` - Dictionary of all supporting directory paths

## Running Tests

### Run all foundation tests

```bash
pytest tests/foundation/
```

### Run specific test file

```bash
pytest tests/foundation/test_papers_directory.py
```

### Run with verbose output

```bash
pytest tests/foundation/test_papers_directory.py -v
```

### Run specific test

```bash
pytest tests/foundation/test_papers_directory.py::TestPapersDirectoryCreation::test_create_papers_directory_success
```

## Test Structure

Tests are organized into classes by category:

- `TestPapersDirectoryCreation` - Basic directory creation tests
- `TestPapersDirectoryEdgeCases` - Edge cases and error handling
- `TestPapersDirectoryIntegration` - Integration tests
- `TestPapersDirectoryRealWorld` - Real-world scenario tests

## Fixtures

- `mock_repo_root` - Provides temporary directory as repository root
- `papers_dir` - Provides path to papers directory for testing

## Coverage

Tests achieve comprehensive coverage of directory creation functionality:

- Happy path scenarios
- Edge cases (already exists, permissions, parent creation)
- Error handling (permission denied, file exists)
- Integration scenarios (subdirectories, files, listing)
- Real-world workflows

**Estimated Coverage**: >95% for directory creation logic

## Notes

- Tests use pytest's `tmp_path` fixture for isolation
- No side effects on actual filesystem
- Tests are platform-independent (Linux, macOS, Windows)
- All tests pass in <0.1 seconds (Fast principle)
