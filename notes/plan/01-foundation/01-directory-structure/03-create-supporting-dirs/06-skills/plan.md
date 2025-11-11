# Plan: Skills Directory Setup

## Overview

Create the `.claude/skills/` directory to house Claude Code Skills - reusable, autonomous capabilities that extend Claude's functionality through model-invoked patterns. Skills are algorithmic or template-based operations that agents can use (unlike sub-agents which operate in separate conversation contexts). This establishes a three-tier taxonomy of foundational, domain-specific, and specialized skills following Claude Code conventions.

## Parent Plan

[../plan.md](../plan.md)

## Child Plans

None (leaf node)

## Inputs

- Repository root and `.claude/` directory exist
- Understanding of Claude Code Skills system and SKILL.md format
- Skills taxonomy designed (Tier 1: Foundational, Tier 2: Domain, Tier 3: Specialized)
- Knowledge of skills vs sub-agents differences (skills are model-invoked computational patterns; sub-agents are separate decision-making contexts)
- Claude Code Skills conventions and frontmatter requirements

## Outputs

- `.claude/skills/` directory structure with tier-based organization
- Foundational skills (Tier 1) for universal agent use: code analysis, generation, testing patterns
- Domain-specific skills (Tier 2) for specialized agent types: paper analysis, ML operations, documentation generation
- Specialized skills (Tier 3) for narrow use cases
- `SKILL.md` files for each skill following Claude Code frontmatter and structure conventions
- Skills documentation explaining taxonomy, usage patterns, and integration
- Decision matrix documenting when to use skills vs sub-agents

## Steps

1. Create `.claude/skills/` directory structure with tier-based organization (tier-1/, tier-2/, tier-3/)
2. Document skills taxonomy and tier definitions in `.claude/skills/README.md`
3. Create Tier 1 foundational skills with SKILL.md files:
   - Code analysis patterns (structure, dependencies, complexity)
   - Code generation templates (boilerplate, patterns)
   - Testing orchestration (run, analyze, report)
4. Create Tier 2 domain-specific skills with SKILL.md files:
   - Paper analysis (extract algorithms, concepts, architecture)
   - ML operations (dataset prep, training patterns, evaluation)
   - Documentation generation (API docs, README templates)
5. Define Tier 3 specialized skills structure for future narrow-use-case skills
6. Create skills vs sub-agents decision matrix documentation
7. Document skills usage guidelines and integration with agent hierarchy
8. Verify all SKILL.md files follow Claude Code conventions (frontmatter, structure, examples)

## Success Criteria

- [ ] `.claude/skills/` directory exists with tier-based subdirectories
- [ ] Skills organized by tier: tier-1/ (foundational), tier-2/ (domain), tier-3/ (specialized)
- [ ] Each skill has proper `SKILL.md` file with required frontmatter (name, description, parameters, examples)
- [ ] Skills follow Claude Code naming and structure conventions
- [ ] README.md explains skills taxonomy and tier definitions
- [ ] Documentation clearly explains when to use skills vs sub-agents with decision matrix
- [ ] Skills integrate with agent hierarchy (agents know which skills they can invoke)
- [ ] At least 3 Tier 1 skills, 3 Tier 2 skills defined with complete SKILL.md files

## Notes

Skills are fine-grained, reusable capabilities invoked by the model when appropriate. Key distinctions from sub-agents: (1) Skills are computational/algorithmic operations, sub-agents make decisions; (2) Skills run in current context, sub-agents have separate conversation threads; (3) Skills are pattern-based, sub-agents are judgment-based. Use skills for: code generation templates, analysis patterns, test orchestration, data extraction. Use sub-agents for: complex multi-step decisions, research requiring judgment, coordinating multiple tools with planning. Example skills: `generate_boilerplate`, `analyze_code_structure`, `run_test_suite`, `extract_algorithm_from_paper`. Skills should have clear inputs/outputs and deterministic behavior patterns.

## Dependencies

- Repository root exists
- `.claude/` directory structure established
- Agent hierarchy designed (to understand which agents use which skills)

## Estimated Effort

Medium - Requires understanding Claude Code Skills conventions, designing taxonomy, creating multiple SKILL.md files following proper format

## Risk Assessment

Low - Skills are additive and don't affect existing functionality; can iterate on skill definitions
