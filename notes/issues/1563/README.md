# Issue #1563: Fix Agent Tool Specifications

## Objective

Update all 36 agent configuration files with correct tool specifications to enable agents to run commands (Bash) and delegate to sub-agents (Task).

## Deliverables

- Updated 36 agent YAML frontmatter files with correct `tools:` specifications
- All orchestrators, design agents, specialists, and engineers now have appropriate tools

## Success Criteria

- All 36 agent files updated
- Only `tools:` line modified in YAML frontmatter
- No other content changed
- All files follow specification requirements

## Files Updated

### Priority 1: Orchestrators (8 files)

**Base orchestrators (6 files)** - Added Bash,Task:
- `/home/mvillmow/ml-odyssey/worktrees/issue-1563-fix-agent-tools/.claude/agents/chief-architect.md`
- `/home/mvillmow/ml-odyssey/worktrees/issue-1563-fix-agent-tools/.claude/agents/code-review-orchestrator.md`
- `/home/mvillmow/ml-odyssey/worktrees/issue-1563-fix-agent-tools/.claude/agents/foundation-orchestrator.md`
- `/home/mvillmow/ml-odyssey/worktrees/issue-1563-fix-agent-tools/.claude/agents/shared-library-orchestrator.md`
- `/home/mvillmow/ml-odyssey/worktrees/issue-1563-fix-agent-tools/.claude/agents/tooling-orchestrator.md`
- `/home/mvillmow/ml-odyssey/worktrees/issue-1563-fix-agent-tools/.claude/agents/cicd-orchestrator.md`

**Research orchestrators (2 files)** - Added Bash,Task before WebFetch:
- `/home/mvillmow/ml-odyssey/worktrees/issue-1563-fix-agent-tools/.claude/agents/papers-orchestrator.md`
- `/home/mvillmow/ml-odyssey/worktrees/issue-1563-fix-agent-tools/.claude/agents/agentic-workflows-orchestrator.md`

### Priority 2: Design Agents (3 files)

- `/home/mvillmow/ml-odyssey/worktrees/issue-1563-fix-agent-tools/.claude/agents/architecture-design.md`
- `/home/mvillmow/ml-odyssey/worktrees/issue-1563-fix-agent-tools/.claude/agents/integration-design.md`
- `/home/mvillmow/ml-odyssey/worktrees/issue-1563-fix-agent-tools/.claude/agents/security-design.md`

### Priority 3: Coordination Specialists (3 files)

- `/home/mvillmow/ml-odyssey/worktrees/issue-1563-fix-agent-tools/.claude/agents/implementation-specialist.md`
- `/home/mvillmow/ml-odyssey/worktrees/issue-1563-fix-agent-tools/.claude/agents/documentation-specialist.md`
- `/home/mvillmow/ml-odyssey/worktrees/issue-1563-fix-agent-tools/.claude/agents/security-specialist.md`

### Priority 4: Specialists with Bash (3 files)

**Test/Performance specialists** - Added Task:
- `/home/mvillmow/ml-odyssey/worktrees/issue-1563-fix-agent-tools/.claude/agents/test-specialist.md`
- `/home/mvillmow/ml-odyssey/worktrees/issue-1563-fix-agent-tools/.claude/agents/performance-specialist.md`

**Blog writer specialist** - Added Task:
- `/home/mvillmow/ml-odyssey/worktrees/issue-1563-fix-agent-tools/.claude/agents/blog-writer-specialist.md`

### Priority 5: Review Specialists (13 files)

- `/home/mvillmow/ml-odyssey/worktrees/issue-1563-fix-agent-tools/.claude/agents/algorithm-review-specialist.md`
- `/home/mvillmow/ml-odyssey/worktrees/issue-1563-fix-agent-tools/.claude/agents/architecture-review-specialist.md`
- `/home/mvillmow/ml-odyssey/worktrees/issue-1563-fix-agent-tools/.claude/agents/data-engineering-review-specialist.md`
- `/home/mvillmow/ml-odyssey/worktrees/issue-1563-fix-agent-tools/.claude/agents/dependency-review-specialist.md`
- `/home/mvillmow/ml-odyssey/worktrees/issue-1563-fix-agent-tools/.claude/agents/documentation-review-specialist.md`
- `/home/mvillmow/ml-odyssey/worktrees/issue-1563-fix-agent-tools/.claude/agents/implementation-review-specialist.md`
- `/home/mvillmow/ml-odyssey/worktrees/issue-1563-fix-agent-tools/.claude/agents/mojo-language-review-specialist.md`
- `/home/mvillmow/ml-odyssey/worktrees/issue-1563-fix-agent-tools/.claude/agents/paper-review-specialist.md`
- `/home/mvillmow/ml-odyssey/worktrees/issue-1563-fix-agent-tools/.claude/agents/performance-review-specialist.md`
- `/home/mvillmow/ml-odyssey/worktrees/issue-1563-fix-agent-tools/.claude/agents/research-review-specialist.md`
- `/home/mvillmow/ml-odyssey/worktrees/issue-1563-fix-agent-tools/.claude/agents/safety-review-specialist.md`
- `/home/mvillmow/ml-odyssey/worktrees/issue-1563-fix-agent-tools/.claude/agents/security-review-specialist.md`
- `/home/mvillmow/ml-odyssey/worktrees/issue-1563-fix-agent-tools/.claude/agents/test-review-specialist.md`

### Priority 6: Engineers (6 files)

- `/home/mvillmow/ml-odyssey/worktrees/issue-1563-fix-agent-tools/.claude/agents/implementation-engineer.md`
- `/home/mvillmow/ml-odyssey/worktrees/issue-1563-fix-agent-tools/.claude/agents/senior-implementation-engineer.md`
- `/home/mvillmow/ml-odyssey/worktrees/issue-1563-fix-agent-tools/.claude/agents/junior-implementation-engineer.md`
- `/home/mvillmow/ml-odyssey/worktrees/issue-1563-fix-agent-tools/.claude/agents/documentation-engineer.md`
- `/home/mvillmow/ml-odyssey/worktrees/issue-1563-fix-agent-tools/.claude/agents/junior-documentation-engineer.md`
- `/home/mvillmow/ml-odyssey/worktrees/issue-1563-fix-agent-tools/.claude/agents/junior-test-engineer.md`

## Tool Specifications Summary

### Changes Made

**Priority 1: Orchestrators**
- Standard orchestrators: `Read,Grep,Glob` → `Read,Grep,Glob,Bash,Task`
- Research orchestrators: `Read,Grep,Glob,WebFetch` → `Read,Grep,Glob,Bash,Task,WebFetch`

**Priority 2: Design Agents**
- All design agents: `Read,Write,Grep,Glob` → `Read,Write,Grep,Glob,Bash,Task`

**Priority 3: Coordination Specialists**
- All coordination specialists: `Read,Write,Edit,Grep,Glob` → `Read,Write,Edit,Grep,Glob,Bash,Task`

**Priority 4: Specialists with Bash**
- Test/Performance specialists: `Read,Write,Edit,Bash,Grep,Glob` → `Read,Write,Edit,Bash,Grep,Glob,Task`
- Blog writer specialist: `Read,Grep,Glob,Bash` → `Read,Grep,Glob,Bash,Task`

**Priority 5: Review Specialists**
- All review specialists: `Read,Grep,Glob` → `Read,Grep,Glob,Bash`

**Priority 6: Engineers**
- All engineers: `Read,Write,Edit,Grep,Glob` → `Read,Write,Edit,Grep,Glob,Bash`

## Implementation Notes

- All updates completed in a single pass
- Only YAML frontmatter `tools:` line modified
- No content changes to agent documentation sections
- All files verified with correct tool specifications
