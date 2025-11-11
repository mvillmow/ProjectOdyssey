# Issues 62-67 Overview - Multi-Level Agent Hierarchy

## Summary

Issues 62-67 represent the foundational work to establish a comprehensive 6-level agent hierarchy for the
ml-odyssey project. This architecture enables sophisticated AI-powered development through hierarchical task
decomposition, specialized agents, and reusable skills.

## Related GitHub Issues

- **Issue 62**: [Plan] Agents - Design and Documentation
- **Issue 63**: [Test] Agents - Write Tests
- **Issue 64**: [Impl] Agents - Implementation
- **Issue 65**: [Package] Agents - Integration and Packaging
- **Issue 66**: [Cleanup] Agents - Refactor and Finalize
- **Issue 67**: [Plan] Tools - Design and Documentation
- **Issues 68-73** (NEW): Skills directory setup (to be created)

## Architecture Overview

### Two Directory Systems

#### 1. Agents System

**`.claude/agents/`** (Working Configurations)

- Operational sub-agent configuration files
- Each agent is an independent AI assistant with its own context
- Follows Claude Code sub-agents conventions
- Version controlled for team sharing

**`agents/`** (Documentation & Reference)

- Team documentation and reference materials
- Configuration templates for creating new agents
- Hierarchy diagrams and delegation rules
- Usage guidelines and best practices

#### 2. Skills System (NEW)

**`.claude/skills/`** (Skills Configurations)

- Reusable capabilities that agents can invoke
- Model-invoked (Claude decides when to use them)
- Organized by tier: Foundational (Tier 1), Domain (Tier 2), Specialized (Tier 3)
- Each skill defined in SKILL.md following Claude Code conventions

### Key Architectural Decisions

1. **Directory Separation**: Operational configs in `.claude/`, documentation in repository root
2. **6-Level Hierarchy**: From meta-orchestrator to junior engineers
3. **Skills as Capabilities**: Separate from sub-agents, used for algorithmic operations
4. **Git Worktree Integration**: Each issue gets its own worktree for isolation
5. **5-Phase Workflow Integration**: Agents map to Plan → Test/Impl/Package → Cleanup phases

## Agent Hierarchy (6 Levels)

### Level 0: Meta-Orchestrator

- **Chief Architect Agent**: System-wide decisions, paper selection, strategic planning

### Level 1: Section Orchestrators

- Foundation, Shared Library, Tooling, Paper Implementation, CI/CD, Agentic Workflows orchestrators
- Manage major repository sections
- Coordinate cross-section dependencies

### Level 2: Module Design Agents

- Architecture Design, Integration Design, Security Design agents
- Design module structure and interfaces
- Break modules into components

### Level 3: Component Specialists

- Implementation, Test, Documentation, Performance, Security specialists
- Handle specific component aspects
- Oversee detailed implementation

### Level 4: Implementation Engineers

- Senior, Standard, Test, Documentation, Performance engineers
- Write code, tests, documentation
- Implement functions and classes

### Level 5: Junior Engineers

- Handle simple, repetitive tasks
- Generate boilerplate code
- Format and lint code

## Skills Taxonomy

### Tier 1: Foundational Skills (Used by All Agents)

- Code analysis (structure, dependencies, complexity)
- Code generation (boilerplate, templates)
- Testing orchestration (run, analyze, report)

### Tier 2: Domain Skills (Specific Agent Types)

- Paper analysis (extract algorithms, architectures)
- ML operations (training, evaluation, datasets)
- Documentation generation (API docs, READMEs)

### Tier 3: Specialized Skills (Few Agents)

- Advanced optimization techniques
- Specialized security analysis
- Domain-specific transformations

## Implementation Strategy

### Phase 1: Plan (Issues 62, 67, 68)

- Update plan.md files with new architecture ✅ COMPLETE
- Document 6-level hierarchy and skills taxonomy (tracked in notes/issues/) ✅ COMPLETE
- Create implementation guidance IN PROGRESS

### Phase 2: Parallel Development (Issues 63-65, 69-71)

- Test: Validate agent configs and skill loading
- Implementation: Create actual .claude/agents/ and .claude/skills/
- Packaging: Integration documentation and setup scripts

### Phase 3: Cleanup (Issues 66, 72)

- Final review and refactoring
- Complete documentation
- Quality assurance

## Git Worktree Strategy

Each issue uses its own worktree for isolation:

- `worktrees/issue-62-plan-agents/` - Plan phase
- `worktrees/issue-63-test-agents/` - Test phase (parallel)
- `worktrees/issue-64-impl-agents/` - Implementation (parallel)
- `worktrees/issue-65-pkg-agents/` - Packaging (parallel)
- `worktrees/issue-66-cleanup-agents/` - Cleanup (sequential)

See [worktree-strategy.md](./worktree-strategy.md) for complete details.

## Documentation Structure

### In Review Directory (`notes/review/` - This Directory)

- `agent-skills-overview.md` - This file (system overview)
- `agent-skills-implementation-summary.md` - Implementation summary and lessons learned
- `agent-architecture-review.md` - Architectural decisions and trade-offs
- `skills-design.md` - Skills taxonomy and design
- `orchestration-patterns.md` - Delegation and coordination rules
- `worktree-strategy.md` - Git worktree workflow

### In Repository Root (`agents/`)

- `README.md` - Team documentation and quick start
- `hierarchy.md` - Visual hierarchy diagram
- `agent-hierarchy.md` - Complete detailed hierarchy specification
- `delegation-rules.md` - Quick reference delegation patterns
- `templates/` - Configuration templates

### In Issue Directories (`notes/issues/`)

- `62/` through `67/` - Individual issue documentation for agents (Plan, Test, Impl, Package, Cleanup, Tools)
- `510/` through `514/` - Individual issue documentation for skills (Plan, Test, Impl, Package, Cleanup)

## Key Distinctions

### Sub-Agents vs Skills

**Sub-Agents**:

- Separate conversation contexts
- Make complex decisions
- Persistent state across invocations
- Example: Architecture Design Agent

**Skills**:

- Run in current context
- Algorithmic/template-based
- Model-invoked automatically
- Example: generate_boilerplate

### agents/ vs .claude/agents/

**agents/** (Repository Root):

- Documentation and reference
- Templates and examples
- Team guidelines
- Not executed directly

**.claude/agents/**:

- Working configurations
- Executed by Claude Code
- Actual operational agents
- Follow Claude Code conventions

## Integration with 5-Phase Workflow

### Plan Phase

- Levels 0-2 active: Orchestrators and designers
- Create specifications for Test/Impl/Package

### Test/Implementation/Packaging Phases (Parallel)

- Levels 3-5 active: Specialists and engineers
- Test: Write and run tests
- Implementation: Build functionality
- Packaging: Integrate artifacts

### Cleanup Phase

- All levels active: Review and refactor
- Fix issues discovered during parallel phases
- Final polish and documentation

## Next Steps

1. ✅ Update plan.md files (Complete)
2. ✅ Create tracked documentation in notes/issues/ (In Progress)
3. Regenerate github_issue.md files (local, from plan.md)
4. Update GitHub issues 62-67
5. Create new GitHub issues 68-73 for skills
6. Begin implementation in respective worktrees

**Note**: plan.md and github_issue.md files are task-relative and NOT tracked in git. Tracked team
documentation is in notes/issues/, notes/review/, and agents/.

## References

- [Claude Code Sub-Agents Documentation](https://code.claude.com/docs/en/sub-agents)
- [Claude Code Skills Documentation](https://code.claude.com/docs/en/skills)
- [Agent Hierarchy Design](./agent-hierarchy.md)
- [Skills Design](./skills-design.md)
- [Orchestration Patterns](./orchestration-patterns.md)
- [Worktree Strategy](./worktree-strategy.md)
