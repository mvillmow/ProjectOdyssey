---
name: agent-run-orchestrator
description: Run section orchestrators to coordinate multi-component development workflows. Use when starting work on a section or coordinating parallel tasks.
---

# Run Orchestrator Skill

Run section orchestrators to coordinate complex workflows.

## When to Use

- Starting work on new section
- Coordinating multiple components
- Managing parallel development
- Need hierarchical coordination

## Orchestrator Levels

### Level 0: Chief Architect

Top-level strategic decisions and architecture.

### Level 1: Section Orchestrators

Coordinate entire sections:
- Foundation Orchestrator
- Shared Library Orchestrator
- Tooling Orchestrator
- Paper Implementation Orchestrator
- CI/CD Orchestrator
- Agentic Workflows Orchestrator

### Level 2: Design/Module Orchestrators

Coordinate modules within sections.

## Usage

### Start Section Work

```bash
# Invoke orchestrator for section
./scripts/run_orchestrator.sh foundation

# This:
# 1. Loads orchestrator configuration
# 2. Reviews section plan
# 3. Breaks into components
# 4. Delegates to design agents
# 5. Monitors progress
```

### Orchestrator Workflow

1. **Review Plans** - Read section specifications
2. **Break Down** - Divide into manageable components
3. **Delegate** - Assign to appropriate agents
4. **Monitor** - Track progress
5. **Integrate** - Coordinate results
6. **Escalate** - Handle blockers

## Delegation Pattern

```text
Section Orchestrator (L1)
  ↓ delegates to
Module Design Agent (L2)
  ↓ delegates to
Component Specialist (L3)
  ↓ delegates to
Implementation Engineer (L4)
```

## Scripts

- `scripts/run_orchestrator.sh` - Run orchestrator
- `scripts/monitor_section_progress.sh` - Monitor progress

See `/agents/hierarchy.md` for complete orchestrator structure.
