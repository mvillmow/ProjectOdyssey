# Agent Testing - Test Plan

## Overview

This document defines the comprehensive test plan for validating the 6-level agent hierarchy system in ml-odyssey.

## Test Objectives

1. **Configuration Validation**: Ensure all agent configurations are syntactically correct and complete
1. **Discovery & Loading**: Verify Claude Code can discover and load agents correctly
1. **Delegation Patterns**: Validate delegation chains work across all 6 levels
1. **Workflow Integration**: Test integration with 5-phase workflow and git worktrees
1. **Mojo Patterns**: Ensure Mojo-specific guidance is present in implementation agents

## Test Coverage Matrix

| Test Area | Script | Coverage | Priority |
|-----------|--------|----------|----------|
| Configuration Validation | `validate_configs.py` | All agent configs | Critical |
| Agent Discovery | `test_loading.py` | Agent file discovery, activation patterns | Critical |
| Delegation Chains | `test_delegation.py` | Level 0→1→2→3→4→5 delegation | High |
| Workflow Integration | `test_integration.py` | 5-phase workflow, worktrees, parallel execution | High |
| Mojo Patterns | `test_mojo_patterns.py` | Mojo-specific guidelines | Medium |

## Test Scripts

### 1. validate_configs.py

**Purpose**: Validate YAML frontmatter and configuration completeness

### Tests

- YAML frontmatter syntax is valid
- Required fields present: name, description, tools, model
- Tool specifications are valid Claude Code tools
- Descriptions are clear enough for auto-invocation
- File naming follows conventions
- Content structure includes expected sections

### Expected Results

- All agent configs pass validation
- No syntax errors
- Clear descriptions that would trigger auto-invocation
- Consistent file naming (matches frontmatter name)

### Success Criteria

- 100% of agent configs pass validation
- Zero critical errors
- Minimal warnings (descriptive issues only)

### 2. test_loading.py

**Purpose**: Test agent discovery and loading

### Tests

- Agent files discovered in `.claude/agents/`
- Configurations load without errors
- Activation patterns detected from descriptions
- Hierarchy coverage (all levels 0-5 represented)
- Tool usage analysis
- Model distribution

### Expected Results

- All `.md` files in `.claude/agents/` discovered
- Configurations parse successfully
- Each description contains activation keywords
- All 6 levels have at least one agent
- Tool usage is appropriate for agent roles
- Model selection is appropriate (sonnet for most, opus for complex)

### Success Criteria

- All agent files discovered
- Zero loading errors
- All levels 0-5 have agent coverage
- Activation keywords present in all descriptions

### 3. test_delegation.py

**Purpose**: Validate delegation patterns across hierarchy

### Tests

- Level 0 → Level 1 delegation defined
- Level 1 → Level 2 delegation defined
- Level 2 → Level 3 delegation defined
- Level 3 → Level 4 delegation defined
- Level 4 → Level 5 delegation defined
- Escalation paths defined (Level N → Level N-1)
- Escalation triggers documented
- Horizontal coordination patterns present

### Expected Results

- Clear delegation chains from top to bottom
- Each level (except 5) delegates to level below
- Each level (except 0) escalates to level above
- Escalation triggers defined for common scenarios
- Coordination patterns for same-level agents

### Success Criteria

- Complete delegation chains for all levels
- Escalation paths defined for levels 1-5
- Escalation triggers documented
- Horizontal coordination mentioned

### 4. test_integration.py

**Purpose**: Test workflow and worktree integration

### Tests

- All 5 phases covered by agents (Plan, Test, Implementation, Packaging, Cleanup)
- Level-phase alignment (right levels in right phases)
- Parallel execution support documented
- Git worktree compatibility mentioned
- Coordination scenarios defined

### Expected Results

- Plan phase: Levels 0-3 participate
- Test/Impl/Package phases: Levels 3-5 participate (parallel)
- Cleanup phase: All levels participate
- Parallel execution guidance for levels 3-5
- Worktree coordination patterns documented

### Success Criteria

- All 5 phases have agent coverage
- Level-phase alignment matches expected patterns
- Parallel execution guidance present
- Worktree coordination documented

### 5. test_mojo_patterns.py

**Purpose**: Validate Mojo-specific guidance

### Tests

- fn vs def guidance in implementation agents
- struct vs class guidance in implementation agents
- SIMD optimization guidance where appropriate
- Memory management (owned, borrowed) patterns
- Performance optimization context
- Type safety guidance

### Expected Results

- Implementation agents (levels 3-5) have comprehensive Mojo guidance
- fn vs def: When to use each
- struct vs class: When to use each
- SIMD: How to leverage vectorization
- Memory: owned, borrowed, inout patterns
- Performance: @parameter, compile-time optimization

### Success Criteria

- 80%+ of implementation agents have fn vs def guidance
- 80%+ of implementation agents have struct vs class guidance
- 50%+ of implementation agents have SIMD guidance
- 80%+ of implementation agents have memory management guidance

## Test Scenarios

### Scenario 1: New Agent Configuration

**Setup**: Create a new agent configuration file

### Steps

1. Run `validate_configs.py` on new file
1. Check for YAML syntax errors
1. Check for required fields
1. Check for clear description
1. Verify file naming

**Expected**: Configuration validates without errors

### Scenario 2: Complete Delegation Chain

**Setup**: Trace delegation from Level 0 to Level 5

### Steps

1. Run `test_delegation.py`
1. Identify Chief Architect (Level 0)
1. Trace delegation to Section Orchestrators (Level 1)
1. Trace delegation to Module Design Agents (Level 2)
1. Trace delegation to Component Specialists (Level 3)
1. Trace delegation to Implementation Engineers (Level 4)
1. Trace delegation to Junior Engineers (Level 5)

**Expected**: Complete chain with clear handoff at each level

### Scenario 3: 5-Phase Workflow Execution

**Setup**: Simulate full workflow execution

### Steps

1. Run `test_integration.py`
1. Identify Plan phase agents
1. Identify parallel phase agents (Test/Impl/Package)
1. Identify Cleanup phase agents
1. Check coordination patterns

### Expected

- Plan phase completes first
- Parallel phases can run simultaneously
- Cleanup phase runs last
- Clear coordination between parallel agents

### Scenario 4: Mojo Implementation Guidance

**Setup**: Check implementation agent for Mojo guidance

### Steps

1. Run `test_mojo_patterns.py`
1. Identify implementation engineers
1. Check for fn vs def guidance
1. Check for struct vs class guidance
1. Check for SIMD optimization guidance
1. Check for memory management patterns

**Expected**: Comprehensive Mojo guidance for implementation

### Scenario 5: Cross-Worktree Coordination

**Setup**: Simulate Test Engineer and Implementation Engineer coordination

### Steps

1. Check both agents have worktree guidance
1. Check both agents mention TDD workflow
1. Check coordination patterns documented
1. Verify parallel execution support

**Expected**: Clear guidance for working in separate worktrees with TDD coordination

## Test Data

### Mock Agent Configurations

For initial testing (before issue #64 completes), create mock agents:

1. **chief-architect.md** (Level 0)
1. **foundation-orchestrator.md** (Level 1)
1. **architecture-design.md** (Level 2)
1. **senior-implementation-specialist.md** (Level 3)
1. **implementation-engineer.md** (Level 4)
1. **junior-implementation-engineer.md** (Level 5)

Each mock should:

- Have valid YAML frontmatter
- Include delegation/escalation information
- Mention workflow phases
- Include Mojo patterns (for implementation agents)

## Success Criteria

### Critical Success Factors

1. **All configurations valid**: 100% pass `validate_configs.py`
1. **Complete hierarchy**: All levels 0-5 have agents
1. **Clear delegation**: Complete delegation chains defined
1. **Workflow coverage**: All 5 phases covered
1. **Mojo guidance**: 80%+ implementation agents have comprehensive guidance

### Quality Metrics

- **Configuration Quality**: Zero critical errors, less than 10 warnings
- **Coverage**: 100% of expected levels have agents
- **Delegation Completeness**: 100% of agents have delegation/escalation defined
- **Workflow Alignment**: 100% of agents participate in appropriate phases
- **Mojo Completeness**: 80%+ of implementation agents have core patterns

## Test Execution Plan

### Phase 1: Mock Agent Testing (Initial)

1. Create mock agent configurations
1. Run all validation scripts
1. Identify any script issues
1. Fix validation scripts
1. Document baseline results

### Phase 2: Real Agent Testing (After Issue #64)

1. Point tests at `.claude/agents/` from issue #64
1. Run all validation scripts
1. Document actual results
1. Identify gaps in agent configurations
1. Create issues for any missing patterns

### Phase 3: Continuous Testing

1. Run tests on each new agent added
1. Run full suite periodically
1. Update tests as patterns evolve
1. Maintain test results documentation

## Test Environment

### Requirements

- Python 3.7+
- Access to `.claude/agents/` directory
- Git worktree capability (for integration tests)

### Setup

```bash
# Run from repository root or worktree
cd worktrees/issue-63-test-agents

# Run individual tests
python3 tests/agents/validate_configs.py
python3 tests/agents/test_loading.py
python3 tests/agents/test_delegation.py
python3 tests/agents/test_integration.py
python3 tests/agents/test_mojo_patterns.py

# Run with specific directory
python3 tests/agents/validate_configs.py /path/to/.claude/agents
```text

## Test Results Documentation

Results will be documented in `/notes/issues/63/test-results.md`:

- Test execution date/time
- Tests run
- Pass/fail status
- Error count
- Warning count
- Specific issues found
- Recommendations for improvement

## References

- `/agents/agent-hierarchy.md` - Complete agent specifications
- `/agents/delegation-rules.md` - Delegation patterns
- `/notes/review/orchestration-patterns.md` - Detailed orchestration patterns
- `/notes/review/skills-design.md` - Skills integration
- Claude Code sub-agents documentation

## Next Steps

After test execution:

1. Document results in `test-results.md`
1. Create GitHub issues for any gaps found
1. Update agent configurations as needed
1. Re-run tests to verify fixes
1. Integrate into CI/CD (future)
