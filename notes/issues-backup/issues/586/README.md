# Issue #586: [Plan] Skills Directory Setup - Design and Documentation

## Objective

Create the `.claude/skills/` directory structure to house Claude Code Skills - reusable, model-invoked capabilities that extend Claude's functionality through algorithmic and template-based operations. Establish a three-tier taxonomy (foundational, domain-specific, specialized) following Claude Code conventions to enable agents to leverage skills effectively.

## Deliverables

- `.claude/skills/` directory structure with tier-based organization (tier-1/, tier-2/, tier-3/)
- Foundational skills (Tier 1) for universal agent use: code analysis, generation, testing patterns
- Domain-specific skills (Tier 2) for specialized agent types: paper analysis, ML operations, documentation generation
- Specialized skills (Tier 3) directory structure for narrow use cases
- `SKILL.md` files for each skill following Claude Code frontmatter and structure conventions
- Skills documentation (`README.md`) explaining taxonomy, usage patterns, and integration with agent hierarchy
- Decision matrix documenting when to use skills vs sub-agents

## Success Criteria

- [ ] `.claude/skills/` directory exists with tier-based subdirectories
- [ ] Skills organized by tier: tier-1/ (foundational), tier-2/ (domain), tier-3/ (specialized)
- [ ] Each skill has proper `SKILL.md` file with required frontmatter (name, description, parameters, examples)
- [ ] Skills follow Claude Code naming and structure conventions
- [ ] README.md explains skills taxonomy and tier definitions
- [ ] Documentation clearly explains when to use skills vs sub-agents with decision matrix
- [ ] Skills integrate with agent hierarchy (agents know which skills they can invoke)
- [ ] At least 3 Tier 1 skills, 3 Tier 2 skills defined with complete SKILL.md files

## Design Decisions

### 1. Skills vs Sub-Agents Architecture

**Decision**: Skills are model-invoked computational patterns; sub-agents are separate decision-making contexts.

### Rationale

- Skills provide deterministic, algorithmic operations that can be invoked inline during agent execution
- Sub-agents handle complex multi-step decisions requiring judgment and planning in separate conversation threads
- This separation enables efficient reuse of patterns while maintaining clear boundaries for delegation

### Key Distinctions

1. **Execution Context**: Skills run in current context; sub-agents have separate conversation threads
1. **Nature of Work**: Skills are computational/algorithmic; sub-agents make decisions with judgment
1. **Invocation Pattern**: Skills are pattern-based with clear inputs/outputs; sub-agents coordinate multiple tools with planning

### Use Skills For

- Code generation templates and boilerplate creation
- Analysis patterns (structure, dependencies, complexity)
- Test orchestration (run, analyze, report)
- Data extraction from structured sources

### Use Sub-Agents For

- Complex multi-step decisions requiring planning
- Research requiring judgment and context evaluation
- Coordinating multiple tools with workflow orchestration
- Tasks requiring separate conversation context

### 2. Three-Tier Taxonomy

**Decision**: Organize skills into three tiers based on scope and specialization.

### Tier 1 - Foundational Skills

- Universal capabilities usable by all agents
- Examples: code analysis, code generation templates, testing patterns
- Target: At least 3 foundational skills in initial implementation

### Tier 2 - Domain-Specific Skills

- Specialized capabilities for specific agent types
- Examples: paper analysis, ML operations, documentation generation
- Target: At least 3 domain-specific skills in initial implementation

### Tier 3 - Specialized Skills

- Narrow use-case skills for specific scenarios
- Initial implementation: directory structure only (populated as needed)

### Rationale

- Clear organization enables agents to quickly find appropriate skills
- Tier-based structure scales as more skills are added
- Matches agent hierarchy levels (specialists use domain skills, all agents use foundational)

### 3. SKILL.md Format and Conventions

**Decision**: All skills follow Claude Code SKILL.md format with required frontmatter.

### Required Components

- Frontmatter: name, description, tier, parameters, returns
- Clear examples demonstrating usage patterns
- Input/output specifications
- Integration guidance for agents

### Rationale

- Standardized format enables consistent skill discovery and usage
- Frontmatter metadata supports automated tooling and skill validation
- Examples reduce learning curve for agents adopting new skills

### 4. Skills-Agent Integration

**Decision**: Document which agent types can invoke which skills in both skill definitions and agent configurations.

### Integration Points

- Skills reference compatible agent types in SKILL.md
- Agent configurations list available skills for that role
- README.md provides mapping between agent hierarchy and skill tiers

### Rationale

- Bidirectional references prevent skill misuse
- Clear mappings enable efficient skill discovery
- Supports validation during agent execution

### 5. Decision Matrix for Skills vs Sub-Agents

**Decision**: Create explicit decision matrix documenting selection criteria.

### Matrix Criteria

- Deterministic vs judgment-based work
- Single-context vs multi-context execution
- Pattern-based vs planning-based workflow
- Immediate invocation vs delegated execution

### Rationale

- Reduces ambiguity when designing new capabilities
- Ensures consistent architectural patterns
- Guides future skill and sub-agent development

## References

### Source Plan

- [notes/plan/01-foundation/01-directory-structure/03-create-supporting-dirs/06-skills/plan.md](notes/plan/01-foundation/01-directory-structure/03-create-supporting-dirs/06-skills/plan.md)

### Related Issues

- [#587: [Test] Skills Directory Setup - Write Tests](https://github.com/mvillmow/ml-odyssey/issues/587)
- [#588: [Impl] Skills Directory Setup - Implementation](https://github.com/mvillmow/ml-odyssey/issues/588)
- [#589: [Package] Skills Directory Setup - Integration and Packaging](https://github.com/mvillmow/ml-odyssey/issues/589)
- [#590: [Cleanup] Skills Directory Setup - Refactor and Finalize](https://github.com/mvillmow/ml-odyssey/issues/590)

### Related Documentation

- [CLAUDE.md - Agent Hierarchy](CLAUDE.md#working-with-agents)
- [agents/README.md - Agent Quick Start](agents/README.md)
- [agents/hierarchy.md - Visual Agent Hierarchy](agents/hierarchy.md)

## Implementation Notes

### Phase 1: Planning (Current)

### Completed

- Analyzed source plan and component requirements
- Documented design decisions for skills vs sub-agents architecture
- Defined three-tier taxonomy structure
- Established SKILL.md format requirements
- Created decision matrix criteria

### Next Steps

- Implementation phase (#588) will create directory structure and initial SKILL.md files
- Test phase (#587) will validate skill invocation patterns and frontmatter parsing
- Packaging phase (#589) will integrate skills with agent configurations
- Cleanup phase (#590) will refine documentation and validate skill examples

### Key Design Principles

1. **Skills are Deterministic**: Each skill should have predictable behavior with clear inputs/outputs
1. **Skills are Reusable**: Design skills for use across multiple agents and scenarios
1. **Skills are Discoverable**: Clear naming, organization, and documentation enable easy discovery
1. **Skills are Documented**: Every skill includes examples and integration guidance

### Example Skills (Planned)

### Tier 1 - Foundational

- `analyze_code_structure` - Analyze code dependencies, complexity, and structure
- `generate_boilerplate` - Generate standard code templates and patterns
- `run_test_suite` - Orchestrate test execution and result analysis

### Tier 2 - Domain-Specific

- `extract_algorithm_from_paper` - Parse research papers to extract algorithms and concepts
- `prepare_ml_dataset` - Standardize dataset preparation workflows
- `generate_api_docs` - Create API reference documentation from code

### Tier 3 - Specialized

- Directory structure reserved for future narrow-use-case skills

### Dependencies

### Requires

- Repository root exists
- `.claude/` directory structure established
- Agent hierarchy designed and documented

### Enables

- Agents can leverage skills for efficient pattern-based operations
- Reduced code duplication across agent implementations
- Faster development through reusable skill templates

### Risk Mitigation

**Risk**: Skills may be overused when sub-agents are more appropriate

**Mitigation**: Clear decision matrix and documentation of selection criteria

**Risk**: Skill taxonomy may not scale with future requirements

**Mitigation**: Three-tier structure provides flexibility; specialized tier handles edge cases

**Risk**: SKILL.md format may diverge from Claude Code conventions

**Mitigation**: Follow official conventions; validate against Claude Code documentation

## Estimated Effort

**Medium** - Requires understanding Claude Code Skills conventions, designing taxonomy, creating multiple SKILL.md files following proper format

## Timeline

- **Planning Phase**: Issue #586 (current)
- **Test Phase**: Issue #587 (parallel after planning)
- **Implementation Phase**: Issue #588 (parallel after planning)
- **Packaging Phase**: Issue #589 (parallel after planning)
- **Cleanup Phase**: Issue #590 (sequential after parallel phases)
