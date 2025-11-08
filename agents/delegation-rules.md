# Delegation Rules - Quick Reference

## Core Delegation Rules

### Rule 1: Scope Reduction
Each delegation reduces scope by one level:
```
System → Section → Module → Component → Function → Line
```

### Rule 2: Specification Detail
Each level adds more detail:
```
Strategic goals → Tactical plans → Component specs → Implementation details → Code
```

### Rule 3: Autonomy Increase
Lower levels have more implementation freedom, less strategic freedom

### Rule 4: Review Responsibility
Each level reviews the work of the level below

### Rule 5: Escalation Path
Issues escalate one level up until resolved

### Rule 6: Horizontal Coordination
Same-level agents coordinate when sharing resources or dependencies

## When to Delegate

### Delegate Down When:
- ✅ Task is too detailed for current level
- ✅ Specific expertise required
- ✅ Work can be parallelized
- ✅ Clear specification can be provided

### Escalate Up When:
- ⬆️ Decision exceeds your authority
- ⬆️ Resources needed from higher level
- ⬆️ Blocker cannot be resolved at current level
- ⬆️ Conflicts with other same-level agents

### Coordinate Horizontally When:
- ↔️ Sharing files or resources
- ↔️ Dependencies between components
- ↔️ Interface negotiation needed
- ↔️ Cross-cutting concerns (security, performance)

## Delegation Patterns

### Pattern 1: Sequential Delegation
```
Agent A
  ↓ completes work
  ↓ delegates to B
Agent B
  ↓ completes work
  ↓ delegates to C
Agent C
```

**Use When**: Tasks have strict dependencies

### Pattern 2: Parallel Delegation
```
Orchestrator
  ├─> Agent A (parallel)
  ├─> Agent B (parallel)
  └─> Agent C (parallel)
```

**Use When**: Tasks are independent

### Pattern 3: Fan-Out/Fan-In
```
Orchestrator
  ├─> Agent A ─┐
  ├─> Agent B ─┼─> Integration Agent
  └─> Agent C ─┘
```

**Use When**: Parallel work needs final integration

## Mojo-Specific Delegation

### Language Choice Decisions

**Level 0-1 Decides**:
- Which components use Mojo vs Python
- Overall language strategy
- Interop patterns

**Level 2-3 Implements**:
- Designs Mojo-specific interfaces
- Plans performance-critical paths
- Structures Mojo modules

**Level 4-5 Executes**:
- Writes Mojo code
- Implements Mojo patterns
- Uses Mojo standard library

### Performance-Critical Paths

**Level 2** (Module Design):
- Identifies performance bottlenecks
- Decides where to use Mojo
- Designs SIMD opportunities

**Level 3** (Component Specialist):
- Specifies SIMD operations
- Plans memory layout
- Designs for vectorization

**Level 4** (Engineer):
- Implements SIMD code
- Optimizes memory access
- Uses `@parameter` decorators

## Status Reporting

### Report Frequency
- **Daily**: For active implementation
- **Weekly**: For planning phases
- **On Completion**: Always
- **On Blocker**: Immediately

### Report Template
```markdown
## Status Report

**Agent**: [Your Name]
**Level**: [0-5]
**Date**: [YYYY-MM-DD]
**Phase**: [Plan/Test/Impl/Package/Cleanup]

### Progress: [%]

### Completed
- [Item 1]

### In Progress
- [Item 1]

### Blockers
- [None / Description]

### Next Steps
- [Step 1]
```

## Handoff Protocol

### When Completing Work
1. **Document What Was Done**
2. **List Artifacts Produced** (files, configs, docs)
3. **Specify Next Steps** for receiving agent
4. **Note Any Gotchas** or important context
5. **Request Confirmation** from receiving agent

### Handoff Template
```markdown
## Task Handoff

**From**: [Your Agent Name]
**To**: [Next Agent Name]
**Task**: [Description]

**Completed**:
- [What you did]

**Artifacts**:
- `path/to/file.mojo` - [description]

**Next Steps**:
- [What next agent should do]

**Notes**:
- [Important context]
```

## Escalation Protocol

### Blocker Escalation
```
1. Identify blocker
2. Document:
   - What's blocking you
   - What you've tried
   - Impact on timeline
3. Escalate to immediate superior
4. Superior resolves or escalates further
```

### Conflict Escalation
```
1. Agents attempt to resolve
2. If unresolved, both escalate to common superior
3. Superior reviews, decides, provides rationale
4. Both agents implement decision
```

## Decision Authority

| Level | Can Decide | Must Escalate |
|-------|-----------|---------------|
| 0 | System architecture, tech stack, paper selection | Business strategy |
| 1 | Section organization, cross-module deps | System-wide changes |
| 2 | Module interfaces, component design | Cross-section impacts |
| 3 | Component implementation approach | Module-wide refactoring |
| 4 | Function implementation, algorithm choice | Component restructuring |
| 5 | Variable names, code formatting | Function-level decisions |

## Git Worktree Coordination

### Worktree Assignment
- **One worktree per issue**
- **One agent (or team) per worktree**
- **Clear ownership documented**

### Cross-Worktree Communication
**Option 1**: Cherry-pick commits
```bash
git cherry-pick <commit-hash>
```

**Option 2**: Temporary merge
```bash
git merge --no-commit <branch>
```

**Option 3**: Coordinate via specs
- Document in specifications (local plan.md or tracked notes/issues/)
- Implement independently
- Merge during packaging phase

## Mojo Code Coordination

### When Writing Mojo
**Coordinate on**:
- Struct definitions (shared types)
- Trait implementations (interfaces)
- Memory management patterns
- SIMD vector widths

**Document in specifications**:
- Type signatures
- Performance requirements
- Memory constraints
- Parallelization strategy

**Note**: Use local plan.md files (not tracked in git) for task-relative planning, or tracked documentation in `notes/issues/` for team-wide specifications.

### Mojo Style Consistency
**Level 2-3 Establishes**:
- Naming conventions
- File organization
- Import patterns
- Documentation style

**Level 4-5 Follows**:
- Applies style guide
- Uses consistent patterns
- Maintains conventions

## Quick Decision Tree

```
Can I decide this myself?
  ├─ YES → Decide, document, proceed
  └─ NO → Can my superior decide?
         ├─ YES → Escalate one level
         └─ NO → Escalate higher

Do I need input from peers?
  ├─ YES → Coordinate horizontally first
  └─ NO → Proceed independently

Is this blocked?
  ├─ YES → Can I resolve?
  │       ├─ YES → Resolve and document
  │       └─ NO → Escalate
  └─ NO → Proceed

Should I delegate this?
  ├─ Too detailed for my level? → Delegate down
  ├─ Requires specific expertise? → Delegate to specialist
  ├─ Can run in parallel? → Delegate to multiple agents
  └─ Within my scope? → Handle myself
```

## Anti-Patterns

### ❌ Don't Do This

**Skipping Levels**:
- ❌ Junior Engineer → Chief Architect
- ✅ Junior Engineer → Implementation Engineer → Component Specialist → ...

**Micro-Managing**:
- ❌ Orchestrator specifying variable names
- ✅ Orchestrator specifying requirements

**Working in Silos**:
- ❌ No communication, surprise conflicts
- ✅ Regular status updates, coordinate interfaces

**Hoarding Decisions**:
- ❌ Make all decisions yourself
- ✅ Delegate appropriately, trust hierarchy

## See Also

- [hierarchy.md](hierarchy.md) - Visual hierarchy and levels
- [README.md](README.md) - Overview and quick start
- [/notes/review/orchestration-patterns.md](/notes/review/orchestration-patterns.md) - Detailed patterns
