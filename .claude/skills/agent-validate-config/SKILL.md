---
name: agent-validate-config
description: Validate agent YAML frontmatter and configuration files to ensure correct structure, required fields, and proper tool specifications. Use when creating or modifying agent configurations.
---

# Agent Configuration Validation Skill

This skill validates agent configuration files to ensure they meet ML Odyssey's agent system requirements.

## When to Use

- User asks to validate agent configs (e.g., "validate agent configurations")
- After creating new agent configuration
- Before committing agent config changes
- CI/CD validation before merge
- Troubleshooting agent loading issues

## What Gets Validated

### YAML Frontmatter

Required fields:
```yaml
---
name: agent-name
role: Agent Role
level: 0-5
phase: Plan|Test|Implementation|Package|Cleanup
description: Brief description
tools: ["Read", "Write", "Bash"]
delegates_to: ["other-agent"]
escalates_to: ["parent-agent"]
---
```

### Configuration Checks

1. **Syntax** - Valid YAML format
2. **Required fields** - All mandatory fields present
3. **Field values** - Correct types and values
4. **Tool specs** - Valid tool names
5. **Delegation** - Valid agent references
6. **File location** - Correct directory structure

## Usage

### Validate Single Agent

```bash
# Validate specific agent
./scripts/validate_agent.sh .claude/agents/implementation-specialist.md

# Example output:
# ✅ Valid YAML frontmatter
# ✅ All required fields present
# ✅ Tools are valid
# ✅ Delegation targets exist
```

### Validate All Agents

```bash
# Validate all agent configs
python3 tests/agents/validate_configs.py .claude/agents/

# Or use script:
./scripts/validate_all_agents.sh
```

### CI Validation

Runs automatically in CI:
```bash
# .github/workflows/test-agents.yml
python3 tests/agents/validate_configs.py .claude/agents/
```

## Required Fields

### Mandatory

- `name` - Agent identifier (kebab-case)
- `role` - Human-readable role
- `level` - Hierarchy level (0-5)
- `phase` - Development phase
- `description` - Brief description

### Optional

- `tools` - List of tool names
- `delegates_to` - List of delegate agent names
- `escalates_to` - List of parent agent names
- `skills` - List of skill names

## Validation Rules

### Name Format

```yaml
# ✅ Correct
name: implementation-specialist

# ❌ Wrong
name: Implementation Specialist  # No spaces
name: impl_specialist           # No underscores
```

### Level Range

```yaml
# ✅ Correct
level: 3  # 0-5 only

# ❌ Wrong
level: 10  # Out of range
level: "3"  # Must be integer
```

### Phase Values

```yaml
# ✅ Correct
phase: Implementation
phase: Plan
phase: Test

# ❌ Wrong
phase: Development  # Not a valid phase
```

### Tool Names

```yaml
# ✅ Correct
tools: ["Read", "Write", "Bash", "Grep", "Glob"]

# ❌ Wrong
tools: ["FileRead"]  # Invalid tool name
tools: "Read"        # Must be array
```

## Error Messages

### Missing Fields

```text
❌ Validation failed: .claude/agents/agent.md
  Missing required field: 'level'
```

**Fix:** Add missing field to YAML frontmatter

### Invalid Tool

```text
❌ Invalid tool: 'InvalidTool'
  Valid tools: Read, Write, Bash, Grep, Glob
```

**Fix:** Use correct tool name from valid set

### Invalid Reference

```text
❌ Delegation target not found: 'nonexistent-agent'
```

**Fix:** Verify agent name is correct or create referenced agent

## Examples

**Validate specific agent:**
```bash
./scripts/validate_agent.sh .claude/agents/implementation-specialist.md
```

**Validate all agents:**
```bash
python3 tests/agents/validate_configs.py .claude/agents/
```

**Fix validation errors:**
```bash
# Run validation
./scripts/validate_agent.sh .claude/agents/my-agent.md

# Read errors
# Fix frontmatter
# Re-validate
./scripts/validate_agent.sh .claude/agents/my-agent.md
```

## Scripts Available

- `scripts/validate_agent.sh` - Validate single agent
- `scripts/validate_all_agents.sh` - Validate all agents
- `tests/agents/validate_configs.py` - Python validation script

## Integration with Workflow

### Pre-commit Hook

```yaml
- repo: local
  hooks:
    - id: validate-agents
      name: Validate Agent Configs
      entry: python3 tests/agents/validate_configs.py
      language: system
      files: ^\.claude/agents/.*\.md$
```

### CI Pipeline

```yaml
- name: Validate Agent Configurations
  run: python3 tests/agents/validate_configs.py .claude/agents/
```

## Common Errors

### 1. YAML Syntax Error

```text
Error parsing YAML: mapping values are not allowed here
```

**Fix:** Check for proper YAML indentation and syntax

### 2. Missing Delimiter

```text
Error: No YAML frontmatter found
```

**Fix:** Ensure file starts with `---` and ends frontmatter with `---`

### 3. Duplicate Keys

```text
Error: Duplicate key: 'name'
```

**Fix:** Remove duplicate keys in frontmatter

## Best Practices

1. **Validate before commit** - Always validate after changes
2. **Use templates** - Start from agent templates
3. **Check references** - Ensure delegated agents exist
4. **Valid tools only** - Use documented tool names
5. **Correct levels** - Verify level matches hierarchy

See `/agents/templates/` for agent configuration templates.
