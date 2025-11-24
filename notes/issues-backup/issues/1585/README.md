# Issue #1585: Create Agent System Usage Examples

## Overview

Create comprehensive examples documentation showing how to use the agent system for common development tasks.

## Problem

New contributors lack clear examples of:

- How to invoke agents for different tasks
- What agents to use when
- How agents coordinate and delegate
- Best practices for agent usage

## Proposed Content

Create `agents/docs/examples.md` with:

### Example Categories

1. **Planning a New Feature**
   - Using chief-architect
   - Breaking down work
   - Creating plan files

1. **Implementing a Component**
   - Using orchestrators
   - Delegating to specialists
   - Following 5-phase workflow

1. **Code Review**
   - Using code-review-orchestrator
   - Routing to specialists
   - Addressing feedback

1. **Testing**
   - Using test-specialist
   - TDD workflow
   - Running tests

1. **Documentation**
   - Using documentation-specialist
   - Writing ADRs
   - Updating guides

### Each Example Shows

- Initial prompt/request
- Agent selection rationale
- Expected output
- Follow-up steps
- Common pitfalls

## Benefits

- Faster onboarding
- Better agent usage
- Fewer mistakes
- Clearer expectations

## Status

**COMPLETED** ✅

Created comprehensive agent usage examples documentation at `agents/docs/examples.md`.

## Implementation Details

Created `/home/mvillmow/ml-odyssey/agents/docs/examples.md` with 7 comprehensive examples:

### Example 1: Planning a New Feature

- Scenario: Adding GELU activation to shared library
- Shows: Orchestrator delegation, issue creation, design spec generation
- Demonstrates: Proper planning workflow before implementation

### Example 2: Implementing a Component

- Scenario: BatchNorm2D implementation following 5-phase workflow
- Shows: Specialist coordination, parallel worktrees, TDD approach
- Demonstrates: Delegation to appropriate engineer levels

### Example 3: Code Review Workflow

- Scenario: Reviewing PR #300 (Dropout layer)
- Shows: Code Review Orchestrator routing to 5 specialists
- Demonstrates: Consolidated feedback, addressing comments, approval process

### Example 4: Testing Workflow (TDD)

- Scenario: Conv2D implementation with test-driven development
- Shows: Writing failing tests first, minimal implementation, refactoring
- Demonstrates: Proper TDD cycle with specialist guidance

### Example 5: Documentation Workflow

- Scenario: Documenting BatchNorm2D layer
- Shows: API docs, usage examples, code docstrings
- Demonstrates: Documentation in Package phase (parallel with implementation)

### Example 6: Bug Fix Workflow

- Scenario: Conv2D rectangular kernel bug
- Shows: Minimal changes principle, root cause analysis, test-first approach
- Demonstrates: Focused bug fixes without scope creep

### Example 7: Performance Optimization

- Scenario: 2x speedup for Conv2D forward pass
- Shows: Profiling, SIMD optimization, cache tiling, benchmarking
- Demonstrates: Data-driven optimization with correctness validation

### Additional Sections

- **Common Patterns**: Starting work, delegation, 5-phase workflow, GitHub issues, parallel work
- **Troubleshooting**: Agent invocation issues, delegation problems, workflow violations
- **Related Resources**: Links to hierarchy, workflows, delegation rules

Each example includes:

- Initial user prompt (how to ask)
- Agent selection (automatic vs explicit)
- Expected agent responses (what to expect)
- Follow-up steps (how to proceed)
- Expected outputs (deliverables)
- Common pitfalls (what to avoid)

## Related Issues

Part of Wave 5 enhancement from continuous improvement session.
