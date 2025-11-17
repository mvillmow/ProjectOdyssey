# Plan: Create Agents Directory and Agent System Documentation

**Parent Plan**: [../plan.md](../plan.md)

**Child Plans**: None (leaf node)

## Overview

This plan establishes the agent system infrastructure for the ml-odyssey project, implementing a comprehensive 6-level hierarchical agent architecture. The system uses two distinct directories:

1. **`.claude/agents/`** - Working sub-agent configuration files (operational agents)
2. **`agents/`** - Documentation, templates, and reference materials (repository root)

The 6-level agent hierarchy spans from meta-orchestration down to junior implementation agents, enabling sophisticated task decomposition and parallel execution while maintaining clear coordination patterns. This agent system integrates with the 5-phase workflow (Plan → Test/Implementation/Package → Cleanup) and leverages git worktrees for isolated agent contexts.

## Inputs

- Project root directory structure (`/home/mvillmow/ml-odyssey/`)
- Existing `.claude/` directory conventions
- Understanding of Claude Code sub-agents capabilities (separate contexts, independent execution)
- Knowledge of the 6-level agent hierarchy:
  - Level 0: Meta-Orchestrator (Chief Architect Agent)
  - Level 1: Section Orchestrators (6 sections: Foundation, Core ML, Advanced ML, Production, Deployment, Maintenance)
  - Level 2: Module Design Agents (Architecture, Integration, Security specialists)
  - Level 3: Component Specialists (Implementation, Test, Documentation, Performance, Security)
  - Level 4: Implementation Engineers (Senior, Standard, Test, Docs, Performance)
  - Level 5: Junior Engineers (Simple tasks, boilerplate generation)
- Understanding of git worktrees for agent context isolation
- 5-phase workflow structure (Plan → Test/Impl/Package → Cleanup)

## Outputs

**`.claude/agents/` directory** containing:

- Sub-agent configuration files for operational agents at each hierarchy level
- Level 0, 1, 2 orchestrator configurations
- Level 3, 4, 5 specialist and implementation agent configurations

**`agents/` directory** (repository root) containing:

- `README.md` - Comprehensive explanation of the agent system, directory purposes, and usage
- `hierarchy.md` - Visual hierarchy diagram and level descriptions
- `delegation-rules.md` - Orchestration patterns and task delegation guidelines
- `templates/` subdirectory with:
  - Configuration templates for each agent level (0-5)
  - Task handoff templates
  - Status reporting templates
- Example configurations demonstrating:
  - Meta-orchestrator setup
  - Section orchestrator configurations
  - Specialist agent configurations
  - Implementation and junior agent configurations
- Integration documentation linking agent system to 5-phase workflow

## Steps

1. **Create directory structure**:
   - Create `.claude/agents/` for working agent configurations
   - Create `agents/` at repository root for documentation
   - Create `agents/templates/` for reusable templates

2. **Write comprehensive README**:
   - Explain distinction between `.claude/agents/` (operational) and `agents/` (documentation)
   - Document the 6-level hierarchy with clear role definitions
   - Clarify sub-agents vs skills (sub-agents = separate contexts; skills = capabilities)
   - Explain git worktree integration for agent context isolation
   - Provide quick-start guide for creating new agents

3. **Document agent hierarchy**:
   - Create `agents/hierarchy.md` with visual diagram
   - Define responsibilities for each level (0-5)
   - Document delegation patterns and escalation paths
   - Specify which levels orchestrate vs execute

4. **Create delegation rules documentation**:
   - Write `agents/delegation-rules.md` with orchestration patterns
   - Define when to delegate up vs down the hierarchy
   - Document coordination mechanisms (status files, git integration)
   - Provide decision trees for task routing

5. **Develop configuration templates**:
   - Create templates for all 6 agent levels in `agents/templates/`
   - Include example prompts, capabilities, and constraints for each level
   - Document required fields and optional customizations
   - Provide template for task handoff formats

6. **Create example configurations**:
   - Provide sample Level 0 meta-orchestrator config
   - Provide sample Level 1 section orchestrator config
   - Provide sample Level 3 specialist config
   - Provide sample Level 4-5 implementation configs

7. **Document workflow integration**:
   - Explain how agents interact with 5-phase workflow
   - Document which agent levels handle which phases
   - Clarify planning vs implementation vs cleanup responsibilities
   - Link to existing workflow documentation

## Acceptance Criteria

- Both `.claude/agents/` and `agents/` directories exist with clear, distinct purposes
- `agents/README.md` comprehensively explains the entire agent system including:
  - Directory separation and purposes
  - Complete 6-level hierarchy
  - Sub-agents vs skills distinction
  - Git worktree integration approach
- `agents/hierarchy.md` provides clear visual representation of all 6 levels with role definitions
- `agents/delegation-rules.md` documents orchestration patterns and task routing logic
- `agents/templates/` contains configuration templates for all 6 agent levels
- Example configurations demonstrate realistic agent setups for multiple hierarchy levels
- Documentation integrates agent system with 5-phase workflow
- All documentation is accurate, concise, and actionable
- Directory structure supports both operational agents and system documentation

## Dependencies

- Completion of base directory structure (`notes/`, `.claude/`)
- Understanding of project organization and workflow phases
- Knowledge of Claude Code sub-agents capabilities

## Risks & Mitigations

**Risk**: Confusion between `.claude/agents/` and `agents/` directories
**Mitigation**: Clear documentation in README explaining distinct purposes; operational vs documentation

**Risk**: Agent hierarchy becomes too complex to manage
**Mitigation**: Start with simple examples; document clear delegation rules; provide templates

**Risk**: Sub-agent configurations incompatible with Claude Code conventions
**Mitigation**: Follow Claude Code documentation; reference official sub-agents guide; test configurations

**Risk**: Agents working in isolated contexts cause coordination issues
**Mitigation**: Document git worktree patterns; establish status file conventions; define clear handoff protocols

## Notes

- **Sub-agents vs Skills**: Sub-agents operate in separate contexts with independent execution, while skills are capabilities/tools available within a single agent's context
- **Git Worktree Integration**: Agents use git worktrees to work in isolated branches, preventing conflicts and enabling parallel execution
- **Claude Code Reference**: Configuration follows Claude Code sub-agents documentation and conventions
- **5-Phase Integration**: Agent levels map to workflow phases:
  - Levels 0-2: Planning and orchestration
  - Levels 3-5: Implementation, testing, packaging
  - All levels: Cleanup phase participation
- **Scalability**: 6-level hierarchy supports projects from small to enterprise scale
- **Documentation-First**: `agents/` directory serves as single source of truth for agent system design; `.claude/agents/` contains runtime configurations
- **Template-Driven**: Templates enable consistent agent creation while allowing customization per use case
- **Status**: This is a foundational plan; agent system will evolve with project needs

## Success Metrics

- Agent system documentation is complete and clear
- Directory structure supports both operational and documentation needs
- Configuration templates accelerate new agent creation
- 6-level hierarchy enables effective task decomposition
- Integration with 5-phase workflow is well-documented
- Example configurations provide clear patterns to follow
- System is ready to support ml-odyssey development workflow
