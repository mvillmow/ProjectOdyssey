# Git Worktree Strategy for Issues 62-67 (and 68-73)

## Overview

This document defines the git worktree strategy for implementing the multi-level agent hierarchy across issues
62-67 (agents/tools) and the new issues 68-73 (skills).

## Worktree Mapping

### Issues 62-66: Agents Directory

#### Issue 62: [Plan] Agents - Design and Documentation

- **Worktree**: `worktrees/issue-62-plan-agents/`
- **Branch**: `62-plan-agents`
- **Phase**: Plan (Sequential, must complete first)
- **Scope**: Design .claude/agents/ structure and agents/ documentation
- **Agents**: Architecture Design Agent, Documentation Specialist
- **Deliverables**:
  - Updated plan.md with 6-level hierarchy
  - Architecture decision records (tracked in notes/issues/)
  - Agent specifications for all levels

#### Issue 63: [Test] Agents - Write Tests

- **Worktree**: `worktrees/issue-63-test-agents/`
- **Branch**: `63-test-agents`
- **Phase**: Test (Parallel after Plan)
- **Scope**: Test that agent configurations are valid and loadable
- **Agents**: Test Design Specialist, Test Engineer
- **Deliverables**:
  - Validation tests for agent .md files
  - Tests for agent delegation logic
  - Integration tests for agent loading

#### Issue 64: [Impl] Agents - Implementation

- **Worktree**: `worktrees/issue-64-impl-agents/`
- **Branch**: `64-impl-agents`
- **Phase**: Implementation (Parallel after Plan)
- **Scope**: Create actual .claude/agents/ configs and agents/ docs
- **Agents**: Senior Implementation Specialist, Implementation Engineers
- **Deliverables**:
  - .claude/agents/ directory with all agent configs
  - agents/ directory with documentation and templates
  - Agent configuration files for all 6 levels

#### Issue 65: [Package] Agents - Integration and Packaging

- **Worktree**: `worktrees/issue-65-pkg-agents/`
- **Branch**: `65-pkg-agents`
- **Phase**: Packaging (Parallel after Plan)
- **Scope**: Integrate agents with repository workflow
- **Agents**: Integration Design Agent, Implementation Engineers
- **Deliverables**:
  - Integration documentation
  - Setup scripts for agent initialization
  - Team onboarding guide

#### Issue 66: [Cleanup] Agents - Refactor and Finalize

- **Worktree**: `worktrees/issue-66-cleanup-agents/`
- **Branch**: `66-cleanup-agents`
- **Phase**: Cleanup (Sequential after Test/Impl/Package)
- **Scope**: Final review, refactoring, and documentation updates
- **Agents**: All agents from previous phases
- **Deliverables**:
  - Refactored and polished code
  - Complete documentation
  - Quality assurance sign-off

---

### Issue 67: Tools Directory

#### Issue 67: [Plan] Tools - Design and Documentation

- **Worktree**: `worktrees/issue-67-plan-tools/`
- **Branch**: `67-plan-tools`
- **Phase**: Plan (Sequential)
- **Scope**: Design tools/ directory structure
- **Agents**: Architecture Design Agent
- **Deliverables**:
  - Updated plan.md clarifying tools/ purpose
  - Tool categories and organization (tracked docs)
  - Integration with existing scripts/

**Note**: Issues 68-72 for Tools (Test/Impl/Package/Cleanup) will be similar to Agents structure

---

### Issues 68-73: Skills Directory (NEW)

#### Issue 68: [Plan] Skills - Design and Documentation

- **Worktree**: `worktrees/issue-68-plan-skills/`
- **Branch**: `68-plan-skills`
- **Phase**: Plan (Sequential)
- **Scope**: Design .claude/skills/ structure
- **Agents**: Architecture Design Agent, Skills Designer
- **Deliverables**:
  - New plan.md for skills
  - Skills taxonomy (Tier 1/2/3) (tracked in notes/issues/)
  - Skills vs sub-agents decision matrix (tracked)

#### Issue 69: [Test] Skills - Write Tests

- **Worktree**: `worktrees/issue-69-test-skills/`
- **Branch**: `69-test-skills`
- **Phase**: Test (Parallel after Plan)
- **Scope**: Test skill loading and activation
- **Deliverables**: Skill validation tests

#### Issue 70: [Impl] Skills - Implementation

- **Worktree**: `worktrees/issue-70-impl-skills/`
- **Branch**: `70-impl-skills`
- **Phase**: Implementation (Parallel after Plan)
- **Scope**: Create .claude/skills/ with initial skills
- **Deliverables**: Working skill configurations

#### Issue 71: [Package] Skills - Integration and Packaging

- **Worktree**: `worktrees/issue-71-pkg-skills/`
- **Branch**: `71-pkg-skills`
- **Phase**: Packaging (Parallel after Plan)
- **Scope**: Integrate skills with agents
- **Deliverables**: Integration documentation

#### Issue 72: [Cleanup] Skills - Refactor and Finalize

- **Worktree**: `worktrees/issue-72-cleanup-skills/`
- **Branch**: `72-cleanup-skills`
- **Phase**: Cleanup (Sequential)
- **Scope**: Final polish
- **Deliverables**: Production-ready skills

---

## Workflow Phases

### Phase 1: Plan (Sequential)

**Active Worktrees**: 62, 67, 68 (Plan issues only)
**Dependencies**: None → Start immediately
**Completion**: All plan.md files created and reviewed

### Phase 2: Parallel Development

**Active Worktrees**: 63-65, 69-71 (Test/Impl/Package)
**Dependencies**: Requires respective Plan issues (62, 67, 68) complete
**Coordination**:

- Test and Impl coordinate for TDD
- Package integrates Test and Impl artifacts

### Phase 3: Cleanup (Sequential)

**Active Worktrees**: 66, 72 (Cleanup issues)
**Dependencies**: Requires Test/Impl/Package complete
**Integration**: Merge all parallel work, resolve issues

---

## Coordination Patterns

### Between Agents Issues (62-66)

- Issue 64 (Impl) creates actual .claude/agents/ files
- Issue 63 (Test) validates those files work
- Issue 65 (Package) documents how to use them
- All coordinate through plan.md specifications from Issue 62

### Between Agents and Skills (62-66 ↔ 68-72)

- Skills (68-72) can start Plan phase in parallel with Agents
- Skills Implementation (70) should coordinate with Agents Implementation (64)
- Some skills may be used BY agents, document these dependencies

### Between Issues and Tools (67)

- Tools (67) is simpler, primarily documentation
- Tools can proceed independently
- Tools may reference agents and skills in documentation

---

## Worktree Creation Sequence

```bash

# Phase 1: Create Plan worktrees

git worktree add worktrees/issue-62-plan-agents 62-plan-agents
git worktree add worktrees/issue-67-plan-tools 67-plan-tools
git worktree add worktrees/issue-68-plan-skills 68-plan-skills

# Phase 2: Create parallel worktrees (after Plan complete)
# Agents

git worktree add worktrees/issue-63-test-agents 63-test-agents
git worktree add worktrees/issue-64-impl-agents 64-impl-agents
git worktree add worktrees/issue-65-pkg-agents 65-pkg-agents

# Skills

git worktree add worktrees/issue-69-test-skills 69-test-skills
git worktree add worktrees/issue-70-impl-skills 70-impl-skills
git worktree add worktrees/issue-71-pkg-skills 71-pkg-skills

# Phase 3: Create cleanup worktrees (after parallel complete)

git worktree add worktrees/issue-66-cleanup-agents 66-cleanup-agents
git worktree add worktrees/issue-72-cleanup-skills 72-cleanup-skills
```

---

## PR Strategy

### Option 1: One PR per Phase (Recommended)

- **PR 1**: Plan issues (62, 67, 68) → Merge plan.md updates together
- **PR 2**: Agents Test/Impl/Package (63-65) → Merge agents work together
- **PR 3**: Skills Test/Impl/Package (69-71) → Merge skills work together
- **PR 4**: Tools Test/Impl/Package → Merge tools work
- **PR 5**: Cleanup (66, 72) → Final polish

**Advantages**: Logical grouping, easier review, clear milestones

### Option 2: One PR per Issue

- Each issue gets its own PR
- PRs can be reviewed in parallel
- More PRs to manage (12+ PRs total)

**Recommendation**: Use Option 1 for cleaner workflow

---

## Cleanup After Merge

```bash

# After each PR merges, remove worktrees

git worktree remove worktrees/issue-62-plan-agents
git worktree remove worktrees/issue-63-test-agents

# ... etc

```

---

## Notes

- All agents working on a PR work at the worktree level
- Each worktree is isolated, preventing conflicts
- Worktrees can be created/destroyed as needed
- Main branch remains clean during development
- Easy to switch context between issues

---

## References

- [Git Worktree Documentation](https://git-scm.com/docs/git-worktree)
- [Agent Hierarchy Design](./agent-hierarchy.md)
- [Orchestration Patterns](./orchestration-patterns.md)
