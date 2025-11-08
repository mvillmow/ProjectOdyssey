# Orchestration Patterns and Delegation Rules

## Overview

This document defines how agents coordinate, delegate tasks, and escalate issues within the 6-level hierarchy. Effective orchestration is critical for the multi-level agent system to function smoothly.

## Core Principles

1. **Clear Scope Boundaries**: Each level has well-defined scope
2. **Explicit Delegation**: Higher levels delegate specific tasks
3. **Horizontal Coordination**: Same-level agents coordinate directly
4. **Vertical Escalation**: Issues bubble up when unresolvable
5. **Status Transparency**: All agents report progress clearly

---

## Delegation Patterns

### Pattern 1: Decomposition Delegation
**When**: Large task needs to be broken into smaller pieces
**How**: Higher level analyzes task → Creates subtasks → Delegates to lower levels

```
Chief Architect Agent
  ↓ Decompose repository into sections
Section Orchestrators (6 sections)
  ↓ Decompose sections into modules
Module Design Agents
  ↓ Decompose modules into components
Component Specialists
  ↓ Decompose components into functions
Implementation Engineers
```

**Example**:
```
Task: "Implement LeNet-5 paper"

Chief Architect:
  - Analyzes paper requirements
  - Delegates to Paper Implementation Orchestrator

Paper Implementation Orchestrator:
  - Breaks into: data prep, model impl, training, eval
  - Delegates each to Module Design Agents

Architecture Design Agent:
  - Designs model architecture
  - Delegates component implementation to Senior Implementation Specialist

Senior Implementation Specialist:
  - Breaks into classes: Conv2D, Pool, Dense layers
  - Delegates each class to Implementation Engineers
```

### Pattern 2: Specialization Delegation
**When**: Task requires specific expertise
**How**: Orchestrator identifies expertise needed → Delegates to specialist agent

```
Section Orchestrator
  ├─> Architecture Design Agent (for architecture tasks)
  ├─> Security Design Agent (for security tasks)
  └─> Integration Design Agent (for integration tasks)
```

**Example**:
```
Task: "Implement secure API authentication"

Section Orchestrator:
  - Identifies security expertise needed
  - Delegates to Security Design Agent

Security Design Agent:
  - Designs authentication approach
  - Delegates to Security Implementation Specialist

Security Implementation Specialist:
  - Implements auth logic
  - Delegates code generation to Implementation Engineer
```

### Pattern 3: Parallel Delegation
**When**: Independent tasks can run simultaneously
**How**: Orchestrator delegates multiple tasks in parallel to different agents

```
Section Orchestrator
  ├─> Module Design Agent A (parallel)
  ├─> Module Design Agent B (parallel)
  └─> Module Design Agent C (parallel)
```

**Example** (5-Phase Workflow):
```
After Plan phase completes:

Component Specialist delegates in parallel to:
  ├─> Test Engineer (create tests)
  ├─> Implementation Engineer (write code)
  └─> Documentation Writer (write docs)

All three work simultaneously in separate worktrees
```

### Pattern 4: Sequential Delegation
**When**: Tasks have dependencies
**How**: Orchestrator delegates tasks in sequence, waiting for completion

```
Section Orchestrator
  Step 1 ↓ Delegate to Agent A
        ↓ Wait for completion
  Step 2 ↓ Delegate to Agent B
        ↓ Wait for completion
  Step 3 ↓ Delegate to Agent C
```

**Example**:
```
Plan Phase must complete before Test/Impl/Package:

1. Section Orchestrator → Architecture Design Agent (Plan)
   ↓ Wait for plan.md completion (local file, not tracked)
2. Section Orchestrator → Component Specialists (Test/Impl/Package)
   ↓ Specialists work in parallel
3. Section Orchestrator → All agents (Cleanup)
```

---

## Coordination Patterns

### Horizontal Coordination (Same Level)

#### Pattern: Peer Review
**Scenario**: Agents review each other's work

```
Implementation Engineer A <──review──> Implementation Engineer B
```

**Process**:
1. Engineer A completes implementation
2. Engineer A requests review from Engineer B
3. Engineer B reviews code, provides feedback
4. Engineer A addresses feedback
5. Both sign off on completion

#### Pattern: Interface Negotiation
**Scenario**: Agents need to agree on shared interfaces

```
Module Design Agent A <──negotiate──> Module Design Agent B
```

**Process**:
1. Agent A designs module A interface
2. Agent B designs module B interface
3. Both identify integration points
4. Negotiate API contracts
5. Document agreed interfaces
6. Proceed with implementation

#### Pattern: Resource Coordination
**Scenario**: Multiple agents need same resource

```
Engineer A ─┐
           ├──> Shared Resource (file, database, etc.)
Engineer B ─┘
```

**Process**:
1. Engineers identify shared resource
2. Coordinate timing to avoid conflicts
3. Use git worktrees for file isolation
4. Communicate through status updates
5. Merge changes carefully

### Vertical Coordination (Across Levels)

#### Pattern: Status Reporting
**Scenario**: Lower level reports progress to higher level

```
Implementation Engineer
  ↓ Status Report
Component Specialist
  ↓ Aggregated Status
Module Design Agent
  ↓ Summary Report
Section Orchestrator
```

**Report Format**:
```markdown
## Status Report

**Agent**: Implementation Engineer - Database Module
**Date**: 2025-11-07
**Phase**: Implementation
**Progress**: 75% complete

### Completed
- User authentication functions
- Database connection pooling
- Error handling

### In Progress
- Query optimization
- Connection retry logic

### Blockers
- None

### Next Steps
- Complete retry logic
- Write unit tests
- Request code review
```

#### Pattern: Specification Cascade
**Scenario**: Higher level provides specifications to lower level

```
Module Design Agent
  ↓ Component Specification
Component Specialist
  ↓ Function Specification
Implementation Engineer
  ↓ Code Implementation
```

**Specification Format**:
```markdown
## Component Specification: UserAuth

**Purpose**: Handle user authentication and session management

**Inputs**:
- username: string
- password: string

**Outputs**:
- auth_token: string
- user_id: int

**Functions Required**:
1. authenticate_user(username, password) -> auth_token
2. validate_token(auth_token) -> bool
3. refresh_token(auth_token) -> new_auth_token

**Error Handling**:
- InvalidCredentials exception
- TokenExpired exception
- DatabaseConnectionError exception

**Performance Requirements**:
- Authentication: < 100ms
- Token validation: < 10ms

**Security Requirements**:
- Password hashing: bcrypt with cost 12
- Token: JWT with HS256
- Session timeout: 30 minutes
```

---

## Escalation Patterns

### Pattern: Blocker Escalation
**When**: Agent cannot proceed due to external blocker
**How**: Escalate to next level up

```
Implementation Engineer (blocked)
  ↓ Escalate blocker
Component Specialist
  ├─> Can resolve? → Resolve and respond
  └─> Cannot resolve? → Escalate to Module Design Agent
```

**Example**:
```
Blocker: "Database schema not defined"

Implementation Engineer:
  - Cannot implement without schema
  - Escalates to Component Specialist

Component Specialist:
  - Recognizes this is architectural issue
  - Escalates to Architecture Design Agent

Architecture Design Agent:
  - Designs database schema
  - Provides specification
  - Issue resolved, work continues
```

### Pattern: Conflict Escalation
**When**: Agents disagree on approach
**How**: Escalate to common superior

```
Agent A (disagrees) ──┐
                      ├──> Common Superior (decides)
Agent B (disagrees) ──┘
```

**Example**:
```
Conflict: Choice of data structure (list vs dict)

Implementation Engineer A: "Use list for performance"
Implementation Engineer B: "Use dict for lookups"

Both escalate to:
  Component Specialist
    - Reviews requirements
    - Analyzes trade-offs
    - Decides: "Use dict, lookups more important"
    - Provides rationale
    - Both engineers implement decision
```

### Pattern: Quality Escalation
**When**: Quality issues detected that violate standards
**How**: Escalate for review and correction

```
Test Engineer (detects failures)
  ↓ Report quality issue
Component Specialist
  ↓ Assigns correction
Implementation Engineer
  ↓ Fixes issues
Test Engineer (verifies fix)
```

---

## Workflow Integration

### 5-Phase Workflow Orchestration

#### Phase 1: Plan (Sequential)
```
Chief Architect
  ↓ Strategic planning
Section Orchestrators
  ↓ Tactical planning
Module Design Agents
  ↓ Component design
Component Specialists
  ↓ Detailed specifications
```

**Completion Criteria**: All plan.md files (local, task-relative) created and reviewed

#### Phase 2-4: Test/Implementation/Packaging (Parallel)
```
Component Specialist
  ├─> Test Specialist → Test Engineers (parallel)
  ├─> Implementation Specialist → Implementation Engineers (parallel)
  └─> Documentation Specialist → Documentation Writers (parallel)
```

**Coordination**: TDD approach - tests and implementation coordinate

#### Phase 5: Cleanup (Sequential)
```
All agents review their own work
  ↓ Identify issues
Component Specialists
  ↓ Aggregate issues
Section Orchestrators
  ↓ Prioritize cleanup
All agents
  ↓ Execute cleanup tasks
```

---

## Git Worktree Coordination

### Worktree Assignment
Each issue = one worktree

```
issue-62-plan-agents/          → Architecture Design Agent
issue-63-test-agents/          → Test Design Specialist
issue-64-impl-agents/          → Implementation Specialists
issue-65-pkg-agents/           → Documentation Specialist
issue-66-cleanup-agents/       → All agents
```

### Cross-Worktree Coordination

#### Scenario: Implementation needs test fixtures from Test worktree

**Option 1: Cherry-pick commits**
```bash
cd worktrees/issue-64-impl-agents
git cherry-pick abc123  # Pick test fixture commit
```

**Option 2: Temporary merge**
```bash
cd worktrees/issue-64-impl-agents
git merge --no-commit issue-63-test-agents
# Use merged state
git reset --hard  # Clean up if needed
```

**Option 3: Coordinate through specifications**
```
Test Engineer: Commits test fixtures to issue-63 branch
Implementation Engineer: Reads specifications (local plan.md or tracked docs) for fixture specs
Implementation Engineer: Creates fixtures independently in issue-64
After both complete: Package Engineer merges both
```

**Note**: plan.md files are task-relative and local (not tracked in git). For team-wide coordination, use tracked documentation in notes/issues/.

---

## Communication Protocols

### Status Updates
**Frequency**: After completing each major task
**Format**: Structured status report (see above)
**Destination**: Direct superior in hierarchy

### Handoffs
**When**: Completing work and passing to next agent
**Format**:
```markdown
## Task Handoff

**From**: [Agent Name]
**To**: [Next Agent Name]
**Date**: [Date]

**Work Completed**:
- [List of completed items]

**Artifacts Produced**:
- [File paths and descriptions]

**Next Steps**:
- [What the next agent should do]

**Notes**:
- [Any important context or caveats]

**Questions for Next Agent**:
- [Any clarifications needed]
```

### Blockers
**When**: Immediately when discovered
**Format**:
```markdown
## Blocker Report

**Agent**: [Your Name]
**Task**: [What you're working on]
**Blocker**: [What's blocking you]
**Impact**: [How this affects timeline]
**Attempted Solutions**: [What you've tried]
**Escalation To**: [Who should resolve this]
**Priority**: [High/Medium/Low]
```

---

## Decision-Making Rules

### Decision Authority by Level

| Level | Can Decide | Must Escalate |
|-------|-----------|---------------|
| 0 | System-wide architecture, technology stack | Strategic business decisions |
| 1 | Section architecture, module organization | Cross-section conflicts |
| 2 | Module design, component interfaces | Section-wide impacts |
| 3 | Component implementation approach | Module-wide changes |
| 4 | Function implementation details | Component-wide refactoring |
| 5 | Code formatting, variable naming | Function-level decisions |

### Escalation Triggers
Escalate when:
1. **Scope Exceeds Authority**: Decision impacts levels above
2. **Resource Conflicts**: Multiple agents need same resource
3. **Timeline Issues**: Work blocked or significantly delayed
4. **Quality Concerns**: Standards violated or quality at risk
5. **Technical Disagreements**: Agents cannot reach consensus
6. **Requirements Unclear**: Specifications incomplete or contradictory

---

## Anti-Patterns (Avoid These)

### ❌ Skipping Levels
**Wrong**: Junior Engineer escalates directly to Chief Architect
**Right**: Junior Engineer → Implementation Engineer → Component Specialist → ... → Chief Architect

### ❌ Micro-Management
**Wrong**: Section Orchestrator specifies function implementations
**Right**: Section Orchestrator specifies requirements, delegates implementation details

### ❌ Working in Silos
**Wrong**: Agents work without communicating, merge conflicts arise
**Right**: Agents coordinate on interfaces, share status, negotiate conflicts

### ❌ Hoarding Information
**Wrong**: Agent completes work without documenting decisions
**Right**: Agent documents rationale, shares learnings, updates specs

### ❌ Premature Optimization
**Wrong**: Junior Engineer refactors entire codebase
**Right**: Junior Engineer implements spec, suggests optimizations to superior

---

## Monitoring and Metrics

### Health Metrics
- **Delegation Depth**: Average levels traversed per task
- **Escalation Rate**: Number of escalations per 100 tasks
- **Cycle Time**: Time from delegation to completion
- **Rework Rate**: Tasks returned for corrections
- **Coordination Overhead**: Time spent on coordination vs execution

### Success Indicators
- Clear task handoffs
- Minimal escalations for trivial issues
- High first-time quality
- Effective parallel execution
- Smooth cross-worktree coordination

---

## Examples

### Example 1: Complete Task Flow

**Task**: "Implement user authentication"

```
1. Chief Architect
   - Recognizes security importance
   - Delegates to Foundation Orchestrator

2. Foundation Orchestrator
   - Assigns to Security Design Agent

3. Security Design Agent (Level 2)
   - Designs authentication approach
   - Creates component specification
   - Delegates to Security Implementation Specialist

4. Security Implementation Specialist (Level 3)
   - Breaks into functions
   - Creates detailed implementation plan
   - Delegates in parallel:
     ├─> Test Engineer (write tests)
     ├─> Implementation Engineer (write code)
     └─> Documentation Writer (write docs)

5. Implementation Engineer (Level 4)
   - Implements authentication functions
   - Coordinates with Test Engineer on TDD
   - Delegates boilerplate to Junior Engineer

6. Junior Engineer (Level 5)
   - Generates function templates
   - Applies code formatters
   - Returns to Implementation Engineer

7. Implementation Engineer
   - Completes implementation
   - Reports to Security Implementation Specialist

8. Security Implementation Specialist
   - Reviews all work (test, impl, docs)
   - Integrates in packaging worktree
   - Reports to Security Design Agent

9. Security Design Agent
   - Validates against design
   - Reports to Foundation Orchestrator

10. Foundation Orchestrator
    - Confirms completion
    - Updates Chief Architect
```

### Example 2: Conflict Resolution

**Conflict**: Test Engineer and Implementation Engineer disagree on function signature

```
1. Test Engineer: "Function should return tuple (success, user_id)"
2. Implementation Engineer: "Function should return user_id or raise exception"

3. Both escalate to Component Specialist

4. Component Specialist:
   - Reviews Python best practices
   - Analyzes use cases
   - Decides: "Raise exception on failure, return user_id on success"
   - Rationale: "Pythonic, clearer error handling"
   - Updates specification

5. Both engineers implement decision
6. Test Engineer updates tests
7. Implementation Engineer updates code
8. Conflict resolved
```

---

## References

- [Agent Hierarchy](./agent-hierarchy.md)
- [Skills Design](./skills-design.md)
- [Worktree Strategy](./worktree-strategy.md)
- [5-Phase Workflow](../../review/README.md)
