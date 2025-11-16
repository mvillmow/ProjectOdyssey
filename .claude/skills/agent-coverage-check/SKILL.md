---
name: agent-coverage-check
description: Check agent configuration coverage across hierarchy levels and phases. Use to ensure complete agent system coverage.
---

# Agent Coverage Check Skill

Verify complete agent system coverage.

## When to Use

- After adding new agents
- Validating agent system
- Finding gaps in coverage
- Ensuring all phases covered

## Coverage Dimensions

### 1. Hierarchy Levels

```bash
# Check all levels have agents
./scripts/check_level_coverage.sh

# Expected:
# L0: Chief Architect
# L1: 6 orchestrators
# L2: Design agents per section
# L3: Specialists per module
# L4: Engineers
# L5: Junior engineers
```

### 2. Phase Coverage

```bash
# Check phase coverage
./scripts/check_phase_coverage.sh

# Expected phases:
# - Plan
# - Test
# - Implementation
# - Package
# - Cleanup
```

### 3. Section Coverage

```bash
# Check section coverage
./scripts/check_section_coverage.sh

# Expected sections:
# - Foundation
# - Shared Library
# - Tooling
# - First Paper
# - CI/CD
# - Agentic Workflows
```

## Reports

```text
Coverage Report
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Hierarchy:
  L0: ✅ 1 agent
  L1: ✅ 6 agents
  L2: ✅ 12 agents
  L3: ✅ 24 agents
  L4: ✅ 3 agents
  L5: ✅ 1 agent

Phases:
  Plan: ✅ Covered
  Test: ✅ Covered
  Implementation: ✅ Covered
  Package: ✅ Covered
  Cleanup: ✅ Covered

Sections:
  Foundation: ✅ Orchestrator + agents
  Shared Library: ✅ Orchestrator + agents
  ...
```

See `agent-validate-config` for validation.
