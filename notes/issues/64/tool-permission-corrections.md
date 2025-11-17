# Tool Permission Corrections for Agent System

## Objective

Update tool permissions in all agent configuration files to follow the principle of least privilege,
restricting Bash access to only test and performance specialists/engineers.

## Correction Matrix

### Level 0: Chief Architect (1 agent)

**Principle**: Strategic oversight, no direct execution needed

| Agent | Current Tools | Required Tools | Action |
|-------|--------------|----------------|--------|
| chief-architect | Read,Grep,Glob,Bash,Task | Read,Grep,Glob,Task | Remove Bash |

### Level 1: Section Orchestrators (6 agents)

**Principle**: Coordination only, no writing or execution

| Agent | Current Tools | Required Tools | Action |
|-------|--------------|----------------|--------|
| foundation-orchestrator | Read,Grep,Glob,Bash,Task | Read,Grep,Glob,Task | Remove Bash |
| shared-library-orchestrator | Read,Grep,Glob,Bash,Task | Read,Grep,Glob,Task | Remove Bash |
| tooling-orchestrator | Read,Grep,Glob,Bash,Task | Read,Grep,Glob,Task | Remove Bash |
| papers-orchestrator | Read,Grep,Glob,Bash,Task,WebFetch | Read,Grep,Glob,Task,WebFetch | Remove Bash |
| cicd-orchestrator | Read,Grep,Glob,Bash,Task | Read,Grep,Glob,Task | Remove Bash |
| agentic-workflows-orchestrator | Read,Grep,Glob,Bash,Task,WebFetch | Read,Grep,Glob,Task,WebFetch | Remove Bash |

### Level 2: Module Design & Review Orchestration (4 agents)

**Principle**: Design and coordination with writing, no execution

| Agent | Current Tools | Required Tools | Action |
|-------|--------------|----------------|--------|
| architecture-design | Read,Write,Grep,Glob,Bash,Task | Read,Write,Grep,Glob,Task | Remove Bash |
| integration-design | Read,Write,Grep,Glob,Bash,Task | Read,Write,Grep,Glob,Task | Remove Bash |
| security-design | Read,Write,Grep,Glob,Bash,Task | Read,Write,Grep,Glob,Task | Remove Bash |
| code-review-orchestrator | Read,Grep,Glob,Bash,Task | Read,Grep,Glob,Task | Remove Bash |

### Level 3: Component Specialists (19 agents)

**Principle**: Implementation and review with editing, Bash ONLY for test/performance

#### Keep Bash (2 agents - test/performance specialists)

| Agent | Current Tools | Required Tools | Action |
|-------|--------------|----------------|--------|
| test-specialist | Read,Write,Edit,Bash,Grep,Glob,Task | Read,Write,Edit,Bash,Grep,Glob,Task | No change |
| performance-specialist | Read,Write,Edit,Bash,Grep,Glob,Task | Read,Write,Edit,Bash,Grep,Glob,Task | No change |

#### Remove Bash (17 agents - all other specialists)

| Agent | Current Tools | Required Tools | Action |
|-------|--------------|----------------|--------|
| implementation-specialist | Read,Write,Edit,Grep,Glob,Bash,Task | Read,Write,Edit,Grep,Glob,Task | Remove Bash |
| documentation-specialist | Read,Write,Edit,Grep,Glob,Bash,Task | Read,Write,Edit,Grep,Glob,Task | Remove Bash |
| security-specialist | Read,Write,Edit,Grep,Glob,Bash,Task | Read,Write,Edit,Grep,Glob,Task | Remove Bash |
| blog-writer-specialist | Read,Grep,Glob,Bash,Task | Read,Grep,Glob,Task | Remove Bash |
| algorithm-review-specialist | Read,Grep,Glob,Bash | Read,Grep,Glob | Remove Bash |
| architecture-review-specialist | Read,Grep,Glob,Bash | Read,Grep,Glob | Remove Bash |
| data-engineering-review-specialist | Read,Grep,Glob,Bash | Read,Grep,Glob | Remove Bash |
| dependency-review-specialist | Read,Grep,Glob,Bash | Read,Grep,Glob | Remove Bash |
| documentation-review-specialist | Read,Grep,Glob,Bash | Read,Grep,Glob | Remove Bash |
| implementation-review-specialist | Read,Grep,Glob,Bash | Read,Grep,Glob | Remove Bash |
| mojo-language-review-specialist | Read,Grep,Glob,Bash | Read,Grep,Glob | Remove Bash |
| paper-review-specialist | Read,Grep,Glob,Bash | Read,Grep,Glob | Remove Bash |
| performance-review-specialist | Read,Grep,Glob,Bash | Read,Grep,Glob | Remove Bash |
| research-review-specialist | Read,Grep,Glob,Bash | Read,Grep,Glob | Remove Bash |
| safety-review-specialist | Read,Grep,Glob,Bash | Read,Grep,Glob | Remove Bash |
| security-review-specialist | Read,Grep,Glob,Bash | Read,Grep,Glob | Remove Bash |
| test-review-specialist | Read,Grep,Glob,Bash | Read,Grep,Glob | Remove Bash |

### Level 4: Implementation Engineers (5 agents)

**Principle**: Implementation with editing, Bash ONLY for test/performance

#### Keep Bash (2 agents - test/performance engineers)

| Agent | Current Tools | Required Tools | Action |
|-------|--------------|----------------|--------|
| test-engineer | Read,Write,Edit,Bash,Grep,Glob | Read,Write,Edit,Bash,Grep,Glob | No change |
| performance-engineer | Read,Write,Edit,Bash,Grep,Glob | Read,Write,Edit,Bash,Grep,Glob | No change |

#### Remove Bash (3 agents - implementation engineers)

| Agent | Current Tools | Required Tools | Action |
|-------|--------------|----------------|--------|
| implementation-engineer | Read,Write,Edit,Grep,Glob,Bash | Read,Write,Edit,Grep,Glob | Remove Bash |
| senior-implementation-engineer | Read,Write,Edit,Grep,Glob,Bash | Read,Write,Edit,Grep,Glob | Remove Bash |
| documentation-engineer | Read,Write,Edit,Grep,Glob,Bash | Read,Write,Edit,Grep,Glob | Remove Bash |

### Level 5: Junior Engineers (3 agents)

**Principle**: Simple implementation tasks only, NO Bash

| Agent | Current Tools | Required Tools | Action |
|-------|--------------|----------------|--------|
| junior-implementation-engineer | Read,Write,Edit,Grep,Glob,Bash | Read,Write,Edit,Grep,Glob | Remove Bash |
| junior-documentation-engineer | Read,Write,Edit,Grep,Glob,Bash | Read,Write,Edit,Grep,Glob | Remove Bash |
| junior-test-engineer | Read,Write,Edit,Grep,Glob,Bash | Read,Write,Edit,Grep,Glob | Remove Bash |

## Summary

**Total Agents**: 38
**Agents Requiring Changes**: 33
**Agents Keeping Bash**: 5 (test-specialist, performance-specialist, test-engineer, performance-engineer, and chief-architect retains Task)
**Agents with No Changes**: 4 (test-specialist, performance-specialist, test-engineer, performance-engineer)

## Implementation Steps

1. **Backup**: Verify git status shows clean working tree
2. **Apply Changes**: Update `tools:` line in YAML frontmatter for each agent
3. **Validate**: Run validation script on all updated agents
4. **Test**: Verify agents still load correctly in Claude Code
5. **Commit**: Create commit with clear message documenting changes

## Rationale

Following the principle of least privilege:

- **Bash access** is only needed for agents that must run tests or benchmarks
- **Write access** is only needed for agents that create/modify files
- **Edit access** is only needed for agents that update existing files
- **Task access** is for coordinating sub-agents (orchestrators only)

This reduces the attack surface and prevents accidental misuse of powerful tools.
