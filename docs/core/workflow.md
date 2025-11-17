# 5-Phase Development Workflow

## Overview

ML Odyssey uses a hierarchical 5-phase workflow to develop features, papers, and tooling. Each component progresses
sequentially through phases, with Test/Implementation/Package running in parallel after planning completes.

**Phases**: Plan → [Test | Implementation | Package] → Cleanup

This workflow ensures clear specifications before implementation, efficient parallel development, and systematic
cleanup of issues discovered during execution.

## Phase 1: Plan (Sequential)

**Goal**: Design specifications before any implementation begins

**Activities**:

- Analyze requirements and objectives
- Design component architecture and interfaces
- Create detailed specifications for all other phases
- Document decisions and rationale
- Identify dependencies and integration points

**Outputs**:

- `plan.md` file with 9-section format (task-relative in `notes/plan/`)
- Component specifications
- API contracts and interfaces
- Acceptance criteria

**Success Criteria**:

- All specifications clear and complete
- No ambiguity for implementation teams
- Dependencies clearly documented
- Tests can be written from specification alone

**Key Principle**: Never skip planning. Good specifications save time in all downstream phases.

## Phase 2-4: Parallel Development

After Plan completes, three teams work simultaneously in separate git worktrees:

### Phase 2: Test (TDD Approach)

**Goal**: Write comprehensive tests before implementation

**Activities**:

- Write unit tests for all specifications
- Design test fixtures and mocks
- Create integration test scenarios
- Document test coverage and edge cases

**Outputs**:

- Test files (`test_*.py`, `test_*.mojo`)
- Test fixtures and utilities
- Test documentation

**Coordinates With**: Implementation team (tests drive implementation via TDD)

### Phase 3: Implementation

**Goal**: Build functionality to pass tests

**Activities**:

- Implement functions and classes
- Write docstrings and inline comments
- Follow specifications exactly
- Integrate with passing tests

**Outputs**:

- Source code files (`.py`, `.mojo`)
- Module organization and structure
- Function implementations

**Coordinates With**: Test team (implements to pass tests), Documentation team (docstring content)

### Phase 4: Package

**Goal**: Create distributable artifacts

**Activities**:

- Build binary packages (`.mojopkg` for Mojo modules)
- Create distribution archives (`.tar.gz`, `.zip`)
- Configure package metadata and installation
- Test package installation in clean environment
- Add to existing packages
- Create CI/CD workflows

**Outputs**:

- Package files (`.mojopkg`, `.tar.gz`, etc.)
- Installation instructions
- CI/CD configuration
- Package metadata

**Note**: Packaging creates actual distributable artifacts, not just documentation.

## Phase 5: Cleanup (Sequential)

**Goal**: Fix issues discovered during parallel phases

**Activities**:

- Collect issues from all teams
- Prioritize cleanup work
- Refactor duplicated code
- Fix integration issues
- Consolidate documentation
- Optimize performance

**Outputs**:

- Refactored code
- Final documentation
- Performance improvements
- Bug fixes

**Success Criteria**:

- No outstanding technical debt
- Code meets quality standards
- All tests pass
- Documentation complete and accurate

## Workflow Diagram

```text
```text

┌─────────────┐
│  Phase 1    │
│   PLAN      │ Design specifications
│ (Sequential)│
└──────┬──────┘
       │
       ├─────────────────────────────────────────┐
       │                                         │
       ▼                                         ▼
┌──────────────┐                        ┌──────────────┐
│   Phase 2    │                        │   Phase 3    │
│    TEST      │◄──────TDD────────►    │  IMPLEMENT   │
│  (Parallel)  │                        │  (Parallel)  │
└──────────────┘                        └──────────────┘
       │                                         │
       └─────────────────────┬───────────────────┘
                             │
                             ▼
                      ┌──────────────┐
                      │   Phase 4    │
                      │   PACKAGE    │
                      │  (Parallel)  │
                      └──────┬───────┘
                             │
                             ▼
                      ┌──────────────┐
                      │   Phase 5    │
                      │   CLEANUP    │
                      │ (Sequential) │
                      └──────────────┘

```text

## Example: LeNet-5 Paper Implementation

### Phase 1: Plan

Architect designs:

- Model architecture (conv layers, pooling, dense layers)
- Data pipeline specifications
- Training loop design
- Evaluation metrics

Creates `plan.md` with component breakdown.

### Phase 2-4: Parallel Development

**Test Team**:

- Writes tests for Conv2D layer
- Creates test data fixtures
- Tests training loop

**Implementation Team**:

- Implements Conv2D layer (to pass tests)
- Implements pooling layers
- Implements training loop

**Package Team**:

- Creates `ml_odyssey.papers` package
- Bundles model weights
- Creates installation scripts

### Phase 5: Cleanup

- Consolidates duplicate utilities
- Optimizes tensor operations
- Finalizes documentation
- Verifies complete implementation

## Coordination Patterns

### TDD Coordination (Phase 2-3)

Tests and implementation coordinate through specification:

1. Test team implements test from specification
2. Implementation team writes code to pass tests
3. Both teams iterate together on edge cases
4. TDD ensures code quality and completeness

### Parallel Execution (Phase 2-4)

Each team works in separate git worktree:

```text
```text

worktrees/
├── issue-XX-plan/           → Plan phase (completed)
├── issue-YY-test/           → Test team works here
├── issue-ZZ-impl/           → Implementation team works here
└── issue-AA-package/        → Package team works here

```text

### Integration (Phase 4-5)

Package team integrates work from all teams:

1. Cherry-pick test commits into packaging worktree
2. Cherry-pick implementation commits
3. Verify tests pass with implementation
4. Create packages with all components
5. Test installation and usage

## Key Principles

1. **Specification First**: Plan phase provides specs for all other phases
2. **Parallel Execution**: Test/Impl/Package maximize productivity
3. **TDD Integration**: Tests drive implementation quality
4. **Clear Handoffs**: Each phase completes cleanly before next
5. **Minimal Delays**: Cleanup phase collects issues, not discovers them

## Getting Started

Each phase has its own GitHub issue with detailed instructions:

- **Plan Issue**: Contains template and specification requirements
- **Test Issue**: Contains test framework and fixture guidelines
- **Implementation Issue**: Contains coding standards and integration points
- **Package Issue**: Contains distribution requirements and CI/CD setup
- **Cleanup Issue**: Contains refactoring and finalization checklist

See `agents/hierarchy.md` for which specialists lead each phase.

## References

- **Orchestration Patterns** (`notes/review/orchestration-patterns.md`) - Detailed coordination rules
- **Agent Hierarchy** (`agents/hierarchy.md`) - Which agents lead each phase
- **CLAUDE.md** (repo root) - Complete phase definitions
- **Worktree Strategy** (`notes/review/worktree-strategy.md`) - Git workflow coordination
