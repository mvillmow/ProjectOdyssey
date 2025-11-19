# Issues #770-813 Analysis & Implementation Plan

**Analysis Date**: 2025-11-19
**Repository**: mvillmow/ml-odyssey
**Branch**: claude/analyze-github-issues-015tGYGVL5wfw1697fDgCFtM
**Scope**: Tooling Section - Paper Scaffold and Test Runner

## Executive Summary

- **Total Issues**: 44 (from #770 to #813)
- **Complete (should close)**: 10 issues (planning phases all closed)
- **Partial (needs work)**: 8 issues (implementation exists but incomplete)
- **Missing (full implementation)**: 4 issues (user prompts functionality)
- **Duplicate (exists as skill)**: 22 issues (test runner implemented as mojo-test-runner skill)

### Quick Status

| Status | Count | Issues |
|--------|-------|--------|
| ✅ COMPLETE | 10 | #774-779, #784, #789, #794, #799, #804, #809 (all [Plan] phases) |
| ⚠️ PARTIAL | 8 | #780-783 (CLI Interface), #785-788 (Paper Scaffolding) |
| ❌ MISSING | 4 | #770-773 (User Prompts) |
| 🔄 DUPLICATE | 22 | #790-793, #795-798, #800-803, #805-808, #810-813 (Test Runner features) |

## Issue-by-Issue Status

| Issue | Title | Status | Repo State | Action |
|-------|-------|--------|------------|--------|
| #770 | [Test] User Prompts | Open | Missing | Implement interactive prompts |
| #771 | [Impl] User Prompts | Open | Missing | Implement interactive prompts |
| #772 | [Package] User Prompts | Open | Missing | Implement interactive prompts |
| #773 | [Cleanup] User Prompts | Open | Missing | Implement interactive prompts |
| #774 | [Plan] Output Formatting | **Closed** | Complete | Already done ✓ |
| #775 | [Test] Output Formatting | **Closed** | Complete | Already done ✓ |
| #776 | [Impl] Output Formatting | **Closed** | Complete | Already done ✓ |
| #777 | [Package] Output Formatting | **Closed** | Complete | Already done ✓ |
| #778 | [Cleanup] Output Formatting | **Closed** | Complete | Already done ✓ |
| #779 | [Plan] CLI Interface | **Closed** | Complete | Already done ✓ |
| #780 | [Test] CLI Interface | Open | Partial | Add CLI argument parsing tests |
| #781 | [Impl] CLI Interface | Open | Partial | CLI exists, needs integration |
| #782 | [Package] CLI Interface | Open | Partial | Verify packaging |
| #783 | [Cleanup] CLI Interface | Open | Partial | Cleanup after implementation |
| #784 | [Plan] Paper Scaffolding | **Closed** | Complete | Already done ✓ |
| #785 | [Test] Paper Scaffolding | Open | Partial | Tests exist, verify completeness |
| #786 | [Impl] Paper Scaffolding | Open | Partial | Implementation exists, verify |
| #787 | [Package] Paper Scaffolding | Open | Partial | Verify packaging |
| #788 | [Cleanup] Paper Scaffolding | Open | Partial | Cleanup after verification |
| #789 | [Plan] Discover Tests | **Closed** | Complete | Already done ✓ |
| #790 | [Test] Discover Tests | Open | Duplicate | Close - mojo-test-runner skill |
| #791 | [Impl] Discover Tests | Open | Duplicate | Close - mojo-test-runner skill |
| #792 | [Package] Discover Tests | Open | Duplicate | Close - mojo-test-runner skill |
| #793 | [Cleanup] Discover Tests | Open | Duplicate | Close - mojo-test-runner skill |
| #794 | [Plan] Run Tests | **Closed** | Complete | Already done ✓ |
| #795 | [Test] Run Tests | Open | Duplicate | Close - mojo-test-runner skill |
| #796 | [Impl] Run Tests | Open | Duplicate | Close - mojo-test-runner skill |
| #797 | [Package] Run Tests | Open | Duplicate | Close - mojo-test-runner skill |
| #798 | [Cleanup] Run Tests | Open | Duplicate | Close - mojo-test-runner skill |
| #799 | [Plan] Report Results | **Closed** | Complete | Already done ✓ |
| #800 | [Test] Report Results | Open | Duplicate | Close - mojo-test-runner skill |
| #801 | [Impl] Report Results | Open | Duplicate | Close - mojo-test-runner skill |
| #802 | [Package] Report Results | Open | Duplicate | Close - mojo-test-runner skill |
| #803 | [Cleanup] Report Results | Open | Duplicate | Close - mojo-test-runner skill |
| #804 | [Plan] Test Runner | **Closed** | Complete | Already done ✓ |
| #805 | [Test] Test Runner | Open | Duplicate | Close - mojo-test-runner skill |
| #806 | [Impl] Test Runner | Open | Duplicate | Close - mojo-test-runner skill |
| #807 | [Package] Test Runner | Open | Duplicate | Close - mojo-test-runner skill |
| #808 | [Cleanup] Test Runner | Open | Duplicate | Close - mojo-test-runner skill |
| #809 | [Plan] Test Specific Paper | **Closed** | Complete | Already done ✓ |
| #810 | [Test] Test Specific Paper | Open | Missing | Implement paper filtering |
| #811 | [Impl] Test Specific Paper | Open | Missing | Implement paper filtering |
| #812 | [Package] Test Specific Paper | Open | Missing | Implement paper filtering |
| #813 | [Cleanup] Test Specific Paper | Open | Missing | Implement paper filtering |

## Components Analysis

### Component 1: User Prompts (Issues #770-773)

**Issues**: #770-773 (Test, Impl, Package, Cleanup)

**Status**: ❌ MISSING

**Repository State**:
- `tools/paper-scaffold/scaffold.py` - CLI with argparse only
- `tools/paper-scaffold/scaffold_enhanced.py` - CLI with argparse only
- No interactive prompt functionality exists

**Gap**: The paper scaffold tools accept arguments via CLI flags but do NOT prompt users interactively when arguments are missing. The issues require:
- Interactive prompts for paper metadata (title, author, description)
- Real-time input validation
- Default values where appropriate
- Helpful error messages

**Current Behavior**:
```bash
# Works - all args provided
python scaffold_enhanced.py --paper "LeNet-5" --title "..." --authors "..."

# Fails - missing args (should prompt instead)
python scaffold_enhanced.py --paper "LeNet-5"
# Error: argument --title is required
```

**Expected Behavior**:
```bash
python scaffold_enhanced.py --paper "LeNet-5"
# Prompts:
# > Paper title: [user types here]
# > Authors: [user types here]
# > Year [2025]: [user types or accepts default]
```

### Component 2: Output Formatting (Issues #774-778)

**Issues**: #774-778 (Plan, Test, Impl, Package, Cleanup)

**Status**: ✅ COMPLETE (All closed)

**Repository State**:
- Implemented in `tools/paper-scaffold/scaffold_enhanced.py:36-83` (CreationResult.summary())
- Comprehensive output formatting with:
  - Progress messages during generation
  - Success/failure indicators (✓, ⚠, ✗)
  - Validation results
  - Error reporting

**Evidence**:
```python
# tools/paper-scaffold/scaffold_enhanced.py
def summary(self) -> str:
    """Generate human-readable summary."""
    lines = []
    lines.append("=" * 60)
    lines.append("GENERATION SUMMARY")
    # ... formatted output ...
```

**Action**: None needed - functionality is complete.

### Component 3: CLI Interface (Issues #779-783)

**Issues**: #779-783 (Plan, Test, Impl, Package, Cleanup)

**Status**: ⚠️ PARTIAL (Plan closed, others open)

**Repository State**:
- Planning complete (#779 closed)
- Implementation exists: `scaffold.py` and `scaffold_enhanced.py` both have argparse CLIs
- Tests missing: No tests for CLI argument parsing in `tests/tooling/test_paper_scaffold.py`

**Gap**: Tests for:
- Argument parsing validation
- Help text generation
- Error handling for invalid arguments
- Default value handling

**Current Tests** (from tests/tooling/test_paper_scaffold.py):
- ✓ Paper name normalization
- ✓ Directory creation
- ✓ Template rendering
- ✓ File overwrite protection
- ✗ CLI argument parsing (MISSING)
- ✗ Help text (MISSING)
- ✗ Error messages (MISSING)

### Component 4: Paper Scaffolding (Issues #784-788)

**Issues**: #784-788 (Plan, Test, Impl, Package, Cleanup)

**Status**: ⚠️ PARTIAL (Plan closed, others open)

**Repository State**:
- Planning complete (#784 closed)
- Implementation exists:
  - `tools/paper-scaffold/scaffold.py` (basic version)
  - `tools/paper-scaffold/scaffold_enhanced.py` (enhanced with validation)
  - `tools/paper-scaffold/validate.py` (structure validation)
- Tests exist: `tests/tooling/test_paper_scaffold.py` (150+ lines)
- Templates exist: `tools/paper-scaffold/templates/` directory

**Gap**: Verification needed:
- Are tests comprehensive enough?
- Is packaging/integration complete?
- Should issues be closed?

**Test Coverage** (from test_paper_scaffold.py):
- ✓ Paper name normalization (Issue #744)
- ✓ Directory creation idempotency (Issue #744)
- ✓ Template rendering (Issue #749)
- ✓ File overwrite protection (Issue #749)
- ? Comprehensive validation (Issue #754)
- ? All edge cases covered

### Component 5: Test Discovery (Issues #789-793)

**Issues**: #789-793 (Plan, Test, Impl, Package, Cleanup)

**Status**: 🔄 DUPLICATE (Plan closed, functionality exists as skill)

**Repository State**:
- Planning complete (#789 closed)
- Functionality exists in `.claude/skills/mojo-test-runner/`
- `mojo test` command has built-in test discovery

**Evidence from mojo-test-runner skill**:
```markdown
## Test Discovery

Mojo discovers tests by:
- Files matching `test_*.mojo` or `*_test.mojo`
- Functions starting with `test_`
- In specified directory or file
```

**Rationale for DUPLICATE**:
The issues were created before the mojo-test-runner skill was implemented. The skill now provides:
- Test discovery via `mojo test` built-in functionality
- Test execution via run_tests.sh script
- Integration with TDD workflow

**Recommendation**: Close issues #790-793 as duplicate/implemented by mojo-test-runner skill.

### Component 6: Test Execution (Issues #794-798)

**Issues**: #794-798 (Plan, Test, Impl, Package, Cleanup)

**Status**: 🔄 DUPLICATE (Plan closed, functionality exists as skill)

**Repository State**:
- Planning complete (#794 closed)
- Implementation exists in `.claude/skills/mojo-test-runner/scripts/run_tests.sh`
- Uses `mojo test` command for execution

**Evidence**:
```bash
# From .claude/skills/mojo-test-runner/scripts/run_tests.sh
# Provides:
# - Test isolation via mojo test
# - Output capture
# - Error handling
# - Execution statistics
```

**Recommendation**: Close issues #795-798 as duplicate/implemented by mojo-test-runner skill.

### Component 7: Test Reporting (Issues #799-803)

**Issues**: #799-803 (Plan, Test, Impl, Package, Cleanup)

**Status**: 🔄 DUPLICATE (Plan closed, functionality exists)

**Repository State**:
- Planning complete (#799 closed)
- `mojo test` provides built-in reporting
- mojo-test-runner skill documents reporting capabilities

**Evidence**:
```markdown
## Test Reporting

### Basic Report
mojo test tests/  # Shows pass/fail summary

### Verbose Report
mojo test -v tests/  # Shows detailed output for each test
```

**Recommendation**: Close issues #800-803 as duplicate/implemented by mojo test and skill.

### Component 8: Unified Test Runner (Issues #804-808)

**Issues**: #804-808 (Plan, Test, Impl, Package, Cleanup)

**Status**: 🔄 DUPLICATE (Plan closed, unified runner exists as skill)

**Repository State**:
- Planning complete (#804 closed)
- mojo-test-runner skill provides unified test runner combining:
  - Discovery (via `mojo test` built-in)
  - Execution (via run_tests.sh)
  - Reporting (via `mojo test` output)

**Evidence**:
```markdown
# .claude/skills/mojo-test-runner/SKILL.md
This skill runs Mojo tests using the `mojo test` command with various
filtering and reporting options.
```

**Recommendation**: Close issues #805-808 as duplicate/implemented by mojo-test-runner skill.

### Component 9: Test Specific Paper (Issues #809-813)

**Issues**: #809-813 (Plan, Test, Impl, Package, Cleanup)

**Status**: ❌ MISSING (Plan closed, implementation needed)

**Repository State**:
- Planning complete (#809 closed)
- No implementation exists for filtering tests by specific paper
- mojo-test-runner skill runs ALL tests or tests from a specific directory

**Gap**: Functionality to:
- Parse paper identifier from user input (name or path)
- Locate paper directory in repository
- Load paper metadata
- Filter tests to run only tests for that specific paper

**Current Limitation**:
```bash
# Can run all tests
mojo test tests/

# Can run tests in a directory
mojo test papers/lenet-5/tests/

# CANNOT filter by paper name
mojo test --paper lenet-5  # Does not exist
```

**Expected Behavior**:
```bash
# Filter by paper name
./run_tests.sh --paper lenet-5

# Filter by partial match
./run_tests.sh --paper lenet

# Filter by path
./run_tests.sh --paper papers/lenet-5/
```

## Implementation Plan

### Priority 1: CRITICAL - Close Duplicate Issues

**Component**: Test Runner (Discover, Run, Report, Unified)

**Issues to Close**: #790-793, #795-798, #800-803, #805-808 (22 issues total)

**Justification**:
- All functionality implemented in `.claude/skills/mojo-test-runner/`
- Planning phases (#789, #794, #799, #804) already closed
- Implementation phases were superseded by skill-based approach
- Keeping them open creates confusion and duplicate work

**Actions**:
1. Close #790-793 with comment: "Superseded by mojo-test-runner skill (Test Discovery)"
2. Close #795-798 with comment: "Superseded by mojo-test-runner skill (Test Execution)"
3. Close #800-803 with comment: "Superseded by mojo-test-runner skill (Test Reporting)"
4. Close #805-808 with comment: "Superseded by mojo-test-runner skill (Unified Test Runner)"

**Success Criteria**: 22 issues closed with clear explanation of implementation location.

---

### Priority 2: HIGH - Verify Paper Scaffolding Completion

**Component**: Paper Scaffolding

**Issues**: #785-788 (Test, Impl, Package, Cleanup)

**Current State**:
- Implementation exists: scaffold_enhanced.py, validate.py
- Tests exist: test_paper_scaffold.py
- Templates exist: templates/ directory

**Needed**:
1. **Verify Test Coverage**
   - Run tests: `pytest tests/tooling/test_paper_scaffold.py -v`
   - Check coverage: Do tests cover all success criteria from #785?
   - Add missing tests if needed

2. **Verify Implementation Completeness**
   - Check against success criteria from #786
   - Verify all deliverables from #784 are implemented
   - Test end-to-end: Create a paper, verify structure

3. **Verify Packaging**
   - Check if tool is properly integrated in repository
   - Verify documentation exists (README.md in tools/paper-scaffold/)
   - Check if tool is referenced in main docs

4. **Close Issues if Complete**
   - If all criteria met, close #785-788
   - If gaps found, document them and implement

**Files to Review**:
- `tools/paper-scaffold/scaffold_enhanced.py` (489 lines)
- `tools/paper-scaffold/validate.py` (9301 bytes)
- `tests/tooling/test_paper_scaffold.py` (150+ lines)
- `tools/paper-scaffold/README_ENHANCED.md` (7065 bytes)

**Success Criteria**:
- All tests passing
- Coverage > 80% for paper-scaffold code
- Documentation complete
- Issues closed or action items identified

---

### Priority 3: HIGH - Interactive User Prompts

**Component**: User Prompts (Missing Functionality)

**Issues**: #770-773 (Test, Impl, Package, Cleanup)

**Current State**: CLI arguments only, no interactive prompts

**Implementation Needed**:

1. **[Test] User Prompts (#770)**
   - Tests: Create `tests/tooling/test_user_prompts.py`
   - Test interactive input collection
   - Test validation logic
   - Test default value handling
   - Test error message display

2. **[Impl] User Prompts (#771)**
   - Files: Create `tools/paper-scaffold/prompts.py`
   - Implement:
     ```python
     class InteractivePrompter:
         def prompt_for_metadata(self) -> Dict[str, str]:
             """Prompt user for paper metadata interactively."""
             # Prompt for title (required)
             # Prompt for authors (required)
             # Prompt for year (optional, default: current year)
             # Prompt for URL (optional)
             # Prompt for description (optional)
             # Validate inputs
             # Return metadata dict
     ```
   - Integration: Modify scaffold_enhanced.py main() to use prompts when args missing

3. **[Package] User Prompts (#772)**
   - Verify: Interactive mode works end-to-end
   - Test: Both CLI and interactive modes work
   - Document: Update README with interactive mode examples

4. **[Cleanup] User Prompts (#773)**
   - Refactor: Clean up prompt code
   - Document: Add docstrings and comments
   - Validate: Final testing

**Dependencies**:
- Python input() for prompts
- Validation logic (reuse from existing code)
- Default value handling

**Test Cases**:
```python
def test_prompt_with_defaults():
    """Test prompts accept default values."""

def test_prompt_validation():
    """Test invalid input is rejected with helpful message."""

def test_prompt_required_fields():
    """Test required fields cannot be skipped."""
```

**Integration**:
```python
# In scaffold_enhanced.py main()
if not all([args.title, args.authors]):  # Some args missing
    from prompts import InteractivePrompter
    prompter = InteractivePrompter()
    metadata = prompter.prompt_for_metadata(
        existing=vars(args)  # Use provided args as defaults
    )
else:
    # Use CLI args as before
    metadata = {...}
```

**Success Criteria**:
- User can create paper with no CLI args (fully interactive)
- User can mix CLI args and prompts (partial interactive)
- Validation prevents invalid inputs
- Clear error messages guide users
- Tests verify all functionality

**Estimated Complexity**: Medium (3-5 hours)

---

### Priority 4: MEDIUM - CLI Interface Tests

**Component**: CLI Interface

**Issues**: #780-783 (Test, Impl, Package, Cleanup)

**Current State**: Implementation exists, tests missing

**Implementation Needed**:

1. **[Test] CLI Interface (#780)**
   - File: Add to `tests/tooling/test_paper_scaffold.py`
   - Test argparse configuration
   - Test help text generation
   - Test error handling for invalid args
   - Test default values

**Test Cases**:
```python
class TestCLIInterface:
    """Test CLI argument parsing (Issue #780)."""

    def test_help_text(self):
        """Test --help displays usage information."""
        result = subprocess.run(
            ["python", "scaffold_enhanced.py", "--help"],
            capture_output=True
        )
        assert "Paper name" in result.stdout.decode()

    def test_required_args(self):
        """Test required arguments are enforced."""
        result = subprocess.run(
            ["python", "scaffold_enhanced.py"],
            capture_output=True
        )
        assert result.returncode != 0

    def test_default_values(self):
        """Test default values are applied."""
        # Mock argparse to verify defaults
```

2. **[Impl] CLI Interface (#781)**
   - Status: Already implemented
   - Action: Verify completeness against #779 specs

3. **[Package] CLI Interface (#782)**
   - Verify: Tool can be installed/used
   - Document: Installation instructions

4. **[Cleanup] CLI Interface (#783)**
   - Final review and documentation

**Success Criteria**:
- All CLI tests passing
- Help text comprehensive
- Error messages helpful
- Issues closed

**Estimated Complexity**: Simple (1-2 hours)

---

### Priority 5: LOW - Test Specific Paper

**Component**: Paper-Specific Test Filtering

**Issues**: #810-813 (Test, Impl, Package, Cleanup)

**Current State**: Planning complete, implementation missing

**Implementation Needed**:

1. **[Test] Test Specific Paper (#810)**
   - Tests: Create `tests/tooling/test_paper_filter.py`
   - Test paper name parsing
   - Test paper directory resolution
   - Test metadata loading
   - Test test filtering logic

2. **[Impl] Test Specific Paper (#811)**
   - File: Modify `.claude/skills/mojo-test-runner/scripts/run_tests.sh`
   - Add `--paper <name>` option
   - Implement paper directory lookup
   - Filter tests to paper directory

**Implementation Approach**:
```bash
# In run_tests.sh
if [ "$PAPER_NAME" ]; then
    # Find paper directory
    PAPER_DIR=$(find papers/ -type d -name "*$PAPER_NAME*" | head -1)

    if [ -z "$PAPER_DIR" ]; then
        echo "Error: Paper '$PAPER_NAME' not found"
        exit 1
    fi

    # Run tests only for this paper
    mojo test "$PAPER_DIR/tests/"
else
    # Run all tests
    mojo test tests/
fi
```

3. **[Package] Test Specific Paper (#812)**
   - Integration testing
   - Documentation

4. **[Cleanup] Test Specific Paper (#813)**
   - Final refinement

**Success Criteria**:
- Can filter tests by paper name
- Can filter by partial match
- Can filter by directory path
- Clear error for non-existent papers

**Estimated Complexity**: Simple (2-3 hours)

---

## Recommended Closure List

### Close Immediately (22 issues)

**Duplicate/Superseded by Skills**:
- #790-793: Test Discovery → mojo-test-runner skill
- #795-798: Test Execution → mojo-test-runner skill
- #800-803: Test Reporting → mojo-test-runner skill
- #805-808: Unified Test Runner → mojo-test-runner skill

**Closure Comment Template**:
```markdown
Closing this issue as the functionality has been implemented in the
`.claude/skills/mojo-test-runner/` skill.

**Implementation Details**:
- Test Discovery: Built into `mojo test` command
- Test Execution: `scripts/run_tests.sh`
- Test Reporting: `mojo test` built-in output
- Documentation: `.claude/skills/mojo-test-runner/SKILL.md`

The skill-based approach is more maintainable and better integrated
with the project's automation system.

See: `.claude/skills/mojo-test-runner/` for complete implementation.
```

---

## Recommended Deferral List

None - all issues should either be implemented or closed as duplicate.

---

## Next Steps

### Immediate Actions (Priority 1)

1. **Close 22 duplicate issues** (#790-793, #795-798, #800-803, #805-808)
   - Use template closure comment
   - Reference mojo-test-runner skill
   - Mark as "wontfix" or "duplicate" label

### Short-term Actions (Priority 2-3)

2. **Verify Paper Scaffolding** (#785-788)
   - Run existing tests
   - Check coverage
   - Close if complete, or document gaps

3. **Implement User Prompts** (#770-773)
   - Create prompts.py module
   - Add tests
   - Integrate with scaffold_enhanced.py
   - Close issues when complete

### Medium-term Actions (Priority 4-5)

4. **Add CLI Tests** (#780-783)
   - Expand test_paper_scaffold.py
   - Verify CLI completeness
   - Close issues

5. **Implement Paper Filtering** (#810-813)
   - Enhance run_tests.sh
   - Add paper lookup logic
   - Test filtering
   - Close issues

---

## Repository State Summary

### What Exists

**Paper Scaffolding** (mostly complete):
- ✓ `tools/paper-scaffold/scaffold.py` - Basic scaffold
- ✓ `tools/paper-scaffold/scaffold_enhanced.py` - Enhanced with validation (489 lines)
- ✓ `tools/paper-scaffold/validate.py` - Structure validation (9301 bytes)
- ✓ `tools/paper-scaffold/templates/` - Template files
- ✓ `tests/tooling/test_paper_scaffold.py` - Tests (150+ lines)
- ⚠️ Output formatting implemented (CreationResult.summary())
- ✗ Interactive prompts missing

**Test Runner** (complete via skill):
- ✓ `.claude/skills/mojo-test-runner/` - Complete skill
- ✓ `.claude/skills/mojo-test-runner/scripts/run_tests.sh` - Test execution
- ✓ Test discovery via `mojo test` built-in
- ✓ Test reporting via `mojo test` output
- ✗ Paper-specific filtering missing

### What's Missing

1. **Interactive User Prompts** (4 issues)
   - Not implemented
   - Need: prompts.py module
   - Need: Tests for prompting
   - Need: Integration with scaffold_enhanced.py

2. **CLI Tests** (4 issues)
   - Implementation exists
   - Tests missing
   - Need: Expand test_paper_scaffold.py

3. **Paper-Specific Test Filtering** (4 issues)
   - Not implemented
   - Need: Enhance run_tests.sh
   - Need: Tests

### What Should Be Closed

- 22 issues superseded by mojo-test-runner skill
- Potentially 4-8 more after verification (paper scaffolding)

---

## Conclusion

Out of 44 issues:
- **10 already closed** (all planning phases)
- **22 should be closed** (duplicate/superseded by skill)
- **8 need verification** (paper scaffolding may be complete)
- **4 need implementation** (user prompts)

**Recommended Approach**:
1. Close 22 duplicate issues immediately (reduce noise)
2. Verify paper scaffolding completion (possibly close 4-8 more)
3. Implement user prompts (4 issues, medium complexity)
4. Add CLI tests (4 issues, simple)
5. Implement paper filtering (4 issues, simple)

This will clean up the issue tracker significantly and focus effort on the truly missing functionality.
