# Issue #436: [Package] Setup Testing

## Objective

Package the testing framework setup components for distribution and reuse, creating installable test infrastructure that can be used across the project and potentially in other projects.

## Deliverables

- Packaged test framework configuration
- Reusable test runner scripts
- CI integration templates
- Installation documentation
- Usage examples for other projects

## Success Criteria

- [ ] Test framework is packaged and installable
- [ ] Test runner can be invoked from any project location
- [ ] CI templates are reusable and well-documented
- [ ] Package includes clear usage instructions
- [ ] Other project components can use the test framework easily

## Current State Analysis

### Existing Package Structure

**Test Infrastructure Location**: `/tests/shared/conftest.mojo`
**Test Helpers**: `/tests/helpers/*.mojo`
**Test Suites**: Distributed across `/tests/` subdirectories

**Current Organization**:
```text
tests/
├── shared/
│   └── conftest.mojo      # Central test utilities (337 lines)
├── helpers/
│   ├── assertions.mojo     # ExTensor assertions (365 lines)
│   ├── fixtures.mojo       # Fixture placeholders (17 lines)
│   └── utils.mojo          # Utility placeholders (14 lines)
└── [component_tests]/      # Various test suites
```

### What Packaging Means

**For Testing Infrastructure**, packaging involves:

1. **Making Test Utilities Reusable**:
   - Ensure `conftest.mojo` can be imported from any test
   - Provide clear API for test utilities
   - Document usage patterns

2. **Creating Test Templates**:
   - Template test files for new components
   - CI configuration templates
   - Test structure examples

3. **Documentation for Reuse**:
   - How to write new tests using the framework
   - How to set up testing for new components
   - How to integrate with CI

**Note**: Unlike code packages (`.mojopkg`), testing infrastructure is typically:
- Used internally within the project
- Distributed as source files
- Documented for reuse patterns

## Packaging Approach

### 1. Test Utilities as Importable Module

**Current State**: `conftest.mojo` is standalone in `/tests/shared/`

**Packaging Goal**: Make utilities importable from any test file

**Approach**:
```mojo
// In any test file
from tests.shared.conftest import assert_almost_equal, TestFixtures, BenchmarkResult
from tests.helpers.assertions import assert_shape, assert_all_close
```

**Deliverable**: Verify import paths work correctly across test suite

### 2. Test Runner Distribution

**If Implemented in #435**: Package test runner as executable script

**Options**:
- Shell script: `scripts/run_tests.sh`
- Python script: `scripts/run_tests.py`
- Mojo executable: `scripts/run_tests.mojo` (if feasible)

**Packaging**:
```bash
# Make executable
chmod +x scripts/run_tests.sh

# Add to PATH or document usage
./scripts/run_tests.sh [options]
```

### 3. CI Templates

**Goal**: Provide reusable CI configuration for testing

**Template**: `.github/workflows/test-template.yml`
```yaml
name: Component Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Setup Environment
        run: # Setup steps
      - name: Run Tests
        run: ./scripts/run_tests.sh --component ${{ matrix.component }}
    strategy:
      matrix:
        component: [core, data, training, utils]
```

**Deliverable**: Template with documentation on customization

### 4. Documentation Package

**Contents**:
- `docs/testing/setup-guide.md` - How to set up testing for new components
- `docs/testing/writing-tests.md` - Test writing best practices
- `docs/testing/ci-integration.md` - CI setup instructions
- `docs/testing/api-reference.md` - Test utilities API reference

**Structure**:
```markdown
# Testing Setup Guide

## Quick Start

1. Create test directory: `tests/your_component/`
2. Import test utilities: `from tests.shared.conftest import ...`
3. Write test: `fn test_feature() raises: ...`
4. Run test: `mojo test tests/your_component/test_feature.mojo`

## Available Utilities

- Assertions: assert_true, assert_equal, assert_almost_equal
- Fixtures: TestFixtures.set_seed(), TestFixtures.deterministic_seed()
- Generators: create_test_vector(), create_test_matrix()
```

## Deliverables Summary

### 1. Importable Test Utilities
- [x] conftest.mojo already exists and is importable
- [ ] Document import patterns
- [ ] Verify imports work across test suite

### 2. Test Runner (if implemented)
- [ ] Package as executable script
- [ ] Add to project scripts/
- [ ] Document usage and options

### 3. CI Templates
- [ ] Create reusable workflow template
- [ ] Document customization points
- [ ] Provide examples for common scenarios

### 4. Documentation
- [ ] Setup guide for new components
- [ ] Test writing best practices
- [ ] API reference for test utilities
- [ ] CI integration instructions

## References

- **Source Plan**: [notes/plan/02-shared-library/04-testing/01-test-framework/01-setup-testing/plan.md](../../../plan/02-shared-library/04-testing/01-test-framework/01-setup-testing/plan.md)
- **Related Issues**:
  - Issue #433: [Plan] Setup Testing
  - Issue #434: [Test] Setup Testing
  - Issue #435: [Impl] Setup Testing
  - Issue #437: [Cleanup] Setup Testing
- **Existing Code**: `/tests/shared/conftest.mojo`, `/tests/helpers/`

## Implementation Notes

### Packaging Decisions

**Key Insight**: Testing infrastructure packaging is about **organization and documentation**, not creating distributable binaries.

**Focus Areas**:
1. **Organization**: Ensure test utilities are well-organized and easy to import
2. **Documentation**: Provide clear guides for using the test framework
3. **Templates**: Offer reusable patterns for common testing scenarios
4. **CI Integration**: Make it easy to add testing to new components

### Minimal Changes Principle

**Current State Assessment**:
- Test utilities already exist in `conftest.mojo`
- Tests are already organized by component
- Imports already work (tests are passing)

**Packaging Needs**:
- Document the existing organization
- Create templates for new tests
- Document CI integration patterns
- **Don't rebuild** - document and template what exists

### Next Steps

1. Document current import patterns
2. Create test file templates
3. Document CI integration approach
4. Create testing setup guide
5. Verify documentation with new test creation
