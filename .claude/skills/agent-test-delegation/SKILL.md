---
name: agent-test-delegation
description: Test agent delegation patterns to verify proper hierarchy, escalation paths, and coordination between agents. Use when validating agent system design or troubleshooting delegation issues.
---

# Agent Delegation Testing Skill

This skill tests agent delegation patterns to ensure proper hierarchy and coordination.

## When to Use

- User asks to test delegation (e.g., "test agent delegation patterns")
- After modifying agent hierarchy
- Verifying escalation paths
- Troubleshooting delegation issues
- CI/CD validation before merge

## What Gets Tested

### Delegation Chains

Verify agents delegate to correct subordinates:

```text
Chief Architect (L0)
  → Section Orchestrator (L1)
    → Design Agent (L2)
      → Specialist (L3)
        → Engineer (L4)
```

### Escalation Paths

Verify agents can escalate to correct superiors:

```text
Engineer (L4)
  → Specialist (L3)
    → Design Agent (L2)
```

### Circular References

Detect circular delegation:

```text
❌ Agent A → Agent B → Agent A
✅ Proper tree structure
```

## Usage

### Test All Delegation

```bash
# Test all delegation patterns
python3 tests/agents/test_delegation.py .claude/agents/

# Example output:
# ✅ No circular dependencies
# ✅ All delegation targets exist
# ✅ Escalation paths valid
# ✅ Hierarchy levels correct
```

### Test Specific Agent

```bash
# Test delegation from specific agent
./scripts/test_agent_delegation.sh implementation-specialist

# Shows:
# - Who this agent delegates to
# - Who this agent escalates to
# - Delegation chain depth
# - Potential issues
```

### Visualize Delegation

```bash
# Generate delegation tree
./scripts/visualize_delegation.sh

# Output:
# Chief Architect (L0)
# ├── Foundation Orchestrator (L1)
# │   ├── Repository Design Agent (L2)
# │   │   ├── Directory Specialist (L3)
# │   │   └── Config Specialist (L3)
# └── ...
```

## Validation Rules

### 1. Level Hierarchy

Agents must delegate to lower levels:

```yaml
# ✅ Correct
level: 2
delegates_to: ["level-3-agent"]  # Lower level

# ❌ Wrong
level: 3
delegates_to: ["level-2-agent"]  # Higher level
```

### 2. Escalation Direction

Agents must escalate to higher levels:

```yaml
# ✅ Correct
level: 3
escalates_to: ["level-2-agent"]  # Higher level

# ❌ Wrong
level: 3
escalates_to: ["level-4-agent"]  # Lower level
```

### 3. No Circular Delegation

```text
# ❌ Circular
Agent A → Agent B → Agent C → Agent A

# ✅ Tree structure
Agent A → Agent B
Agent A → Agent C
```

### 4. Valid References

All delegation targets must exist:

```yaml
# ✅ Correct
delegates_to: ["existing-agent"]

# ❌ Wrong
delegates_to: ["nonexistent-agent"]
```

## Test Categories

### Structure Tests

- Hierarchy levels correct
- No circular dependencies
- All references valid
- Proper parent-child relationships

### Coverage Tests

- All agents have delegation paths
- All levels represented
- No orphaned agents
- Complete hierarchy tree

### Integration Tests

- Delegation chains work end-to-end
- Escalation paths reachable
- Skip-level delegation documented
- Parallel coordination patterns valid

## Examples

**Test all delegation:**

```bash
python3 tests/agents/test_delegation.py .claude/agents/
```

**Test specific agent:**

```bash
./scripts/test_agent_delegation.sh implementation-specialist
```

**Visualize hierarchy:**

```bash
./scripts/visualize_delegation.sh > delegation_tree.txt
```

**Find circular dependencies:**

```bash
./scripts/find_circular_delegation.sh
```

## Error Messages

### Circular Dependency

```text
❌ Circular delegation detected:
  implementation-specialist → senior-engineer → implementation-specialist
```

**Fix:** Break circular reference by restructuring delegation

### Invalid Level

```text
❌ Invalid delegation:
  Agent 'specialist' (L3) delegates to 'orchestrator' (L1)
  Delegation must be to lower levels
```

**Fix:** Delegate to appropriate level or escalate instead

### Missing Target

```text
❌ Delegation target not found:
  Agent 'specialist' delegates to 'nonexistent-engineer'
```

**Fix:** Create target agent or fix reference name

## Scripts Available

- `scripts/test_agent_delegation.sh` - Test specific agent
- `scripts/visualize_delegation.sh` - Generate delegation tree
- `scripts/find_circular_delegation.sh` - Find circular dependencies
- `tests/agents/test_delegation.py` - Comprehensive delegation tests

## CI Integration

```yaml
- name: Test Agent Delegation
  run: python3 tests/agents/test_delegation.py .claude/agents/
```

## Common Issues

### 1. Skipping Levels

```text
⚠️  Skip-level delegation detected:
  L2 agent delegates to L4 agent (skips L3)
```

**Resolution:** Document skip-level delegation in agent description

### 2. Orphaned Agents

```text
⚠️  Orphaned agent: 'helper-agent'
  No agent delegates to this agent
```

**Resolution:** Add delegation or remove if unused

### 3. Missing Escalation

```text
⚠️  No escalation path for: 'specialist'
```

**Resolution:** Add `escalates_to` field

## Best Practices

1. **Test after changes** - Always test after modifying hierarchy
2. **Visual verification** - Use visualization to understand structure
3. **Document exceptions** - Document skip-level delegation
4. **Keep hierarchy simple** - Avoid overly complex patterns
5. **Validate in CI** - Ensure tests run automatically

See `/agents/delegation-rules.md` for complete delegation guidelines.
