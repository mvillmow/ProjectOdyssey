# Issue #1566: [Fix] Add Task tool to agent configuration validator

## Objective

Update the agent configuration validator to recognize `Task` as a valid tool, fixing CI test failures introduced when PR #1565 added Task to 17 agent configurations.

## Deliverables

- Updated `tests/agents/validate_configs.py` with `Task` in `VALID_TOOLS` set
- All 38 agent configurations pass validation
- CI test-agents check passes

## Success Criteria

- [ ] `Task` added to `VALID_TOOLS` in validate_configs.py
- [ ] All 38 agent configurations pass: `python3 tests/agents/validate_configs.py .claude/agents/`
- [ ] CI test-agents workflow passes
- [ ] No validation errors for agents using Task tool

## Problem

PR #1565 added the `Task` tool to 17 agent configurations to enable sub-agent delegation:

- 8 orchestrators (chief-architect, code-review-orchestrator, etc.)
- 3 design agents (architecture-design, integration-design, security-design)
- 3 coordination specialists
- 3 specialists with delegation capabilities

However, the validation script was not updated, causing CI failures:

```
FAIL: code-review-orchestrator.md
  Errors:
    - Invalid tools: Task
```

## Root Cause

Line 78-82 in `tests/agents/validate_configs.py` defines `VALID_TOOLS` but excludes `Task`:

```python
VALID_TOOLS = {
    "Read", "Write", "Edit", "Bash", "Grep", "Glob",
    "WebFetch", "WebSearch", "NotebookEdit",
    "AskUserQuestion", "TodoWrite", "Skill", "SlashCommand"
}
```

## Solution

Add `"Task"` to the `VALID_TOOLS` set:

```python
VALID_TOOLS = {
    "Read", "Write", "Edit", "Bash", "Grep", "Glob",
    "WebFetch", "WebSearch", "NotebookEdit",
    "AskUserQuestion", "TodoWrite", "Task", "Skill", "SlashCommand"
}
```

## Implementation Notes

### Changes Made

**File: tests/agents/validate_configs.py**

- Line 81: Added `"Task"` to `VALID_TOOLS` set

### Validation Results

Before fix:

```
Total files: 38
Passed: 21
Failed: 17
Total errors: 17
```

After fix:

```
Total files: 38
Passed: 38
Failed: 0
Total errors: 0
```

### Affected Agents (17 total)

All these agents now pass validation:

- agentic-workflows-orchestrator
- architecture-design
- blog-writer-specialist
- chief-architect
- cicd-orchestrator
- code-review-orchestrator
- documentation-specialist
- foundation-orchestrator
- implementation-specialist
- integration-design
- papers-orchestrator
- performance-specialist
- security-design
- security-specialist
- shared-library-orchestrator
- test-specialist
- tooling-orchestrator

## References

- Issue #1566: This fix
- PR #1565: Added Task tool to agent configurations
- Failed CI run: <https://github.com/mvillmow/ml-odyssey/actions/runs/19344583313/job/55341523510>
