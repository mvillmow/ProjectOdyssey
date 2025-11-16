# Issue #591: [Plan] Create Supporting Directories - Design and Documentation

## Objective

Create additional supporting directories that provide infrastructure for the repository, including benchmarks, documentation, agents, development tools, configuration files, and Claude Code skills directories. This planning phase will define detailed specifications, design the architecture, document API contracts, and create comprehensive design documentation to guide implementation.

## Deliverables

- **benchmarks/** directory for performance testing
- **docs/** directory for detailed documentation
- **agents/** directory for AI agent configurations (repository root - documentation and templates)
- **.claude/agents/** directory for operational agent configurations
- **tools/** directory for development utilities
- **configs/** directory for configuration files
- **.claude/skills/** directory for Claude Code Skills
- README files in each directory explaining purpose and structure
- Documentation for each directory's organization and usage

## Success Criteria

- [ ] All supporting directories exist at repository root (or under .claude/ as appropriate)
- [ ] Each directory has a README explaining its purpose and structure
- [ ] Directory structure is organized and logical
- [ ] All directories are ready for content
- [ ] Skills directory (.claude/skills/) exists and is configured with tier-based organization
- [ ] Clear distinction documented between .claude/agents/ (operational) and agents/ (documentation)
- [ ] Configuration templates created for agents (all 6 levels)
- [ ] Skills taxonomy documented (Tier 1: Foundational, Tier 2: Domain, Tier 3: Specialized)

## Design Decisions

### 1. Directory Organization Strategy

**Decision**: Create six supporting directories with distinct purposes, organized by location (repository root vs .claude/)

**Rationale**:

- **Repository root directories** (benchmarks/, docs/, agents/, tools/, configs/) contain user-facing content
- **.claude/ directories** (.claude/agents/, .claude/skills/) contain Claude Code operational configurations
- Separates documentation/templates from operational configurations
- Follows Claude Code conventions for .claude/ directory usage

### 2. Agents Directory Dual Structure

**Decision**: Use two separate directories for agent-related content

**Structure**:

- **agents/** (repository root): Documentation, templates, hierarchy diagrams, delegation rules
- **.claude/agents/**: Operational agent configuration files for the 6-level hierarchy

**Rationale**:

- Prevents confusion between documentation and operational configs
- Keeps documentation version-controlled and easily accessible
- Operational configs in .claude/ follow Claude Code conventions
- Supports both reference materials and working configurations

**6-Level Agent Hierarchy**:

- Level 0: Meta-Orchestrator (Chief Architect Agent)
- Level 1: Section Orchestrators (Foundation, Core ML, Advanced ML, Production, Deployment, Maintenance)
- Level 2: Module Design Agents (Architecture, Integration, Security specialists)
- Level 3: Component Specialists (Implementation, Test, Documentation, Performance, Security)
- Level 4: Implementation Engineers (Senior, Standard, Test, Docs, Performance)
- Level 5: Junior Engineers (Simple tasks, boilerplate generation)

### 3. Skills Tier-Based Taxonomy

**Decision**: Organize skills into three tiers by scope and specialization

**Taxonomy**:

- **Tier 1 (Foundational)**: Universal skills for all agents (code analysis, generation, testing patterns)
- **Tier 2 (Domain-Specific)**: Specialized skills for specific agent types (paper analysis, ML operations, documentation generation)
- **Tier 3 (Specialized)**: Narrow use-case skills for specific scenarios

**Rationale**:

- Clear organization by scope and reusability
- Easy for agents to discover relevant skills
- Supports skill evolution from specialized to foundational
- Aligns with agent hierarchy levels (foundational skills for all, specialized for specific agents)

### 4. Skills vs Sub-Agents Distinction

**Decision**: Document clear decision matrix for when to use skills vs sub-agents

**Key Differences**:

- **Skills**: Model-invoked computational/algorithmic operations running in current context
- **Sub-Agents**: Separate conversation contexts making independent decisions

**Use Skills For**:

- Code generation templates
- Analysis patterns
- Test orchestration
- Data extraction
- Deterministic operations

**Use Sub-Agents For**:

- Complex multi-step decisions
- Research requiring judgment
- Coordinating multiple tools with planning
- Tasks needing separate context

### 5. Tools vs Scripts Separation

**Decision**: Create tools/ directory distinct from existing scripts/ directory

**Distinction**:

- **scripts/**: Automation scripts (create_issues.py, regenerate_github_issues.py)
- **tools/**: Development utilities (CLI tools, code generators, paper implementation helpers)

**Rationale**:

- Scripts automate repository management
- Tools assist in ML/AI development workflow
- Clear separation prevents confusion
- Supports future tooling expansion (paper scaffolding, testing utilities, benchmarking tools)

### 6. Configuration Organization

**Decision**: Use configs/ directory for shared configurations across paper implementations

**Structure**:

- Subdirectories for different config types (training, models, experiments)
- YAML or TOML format for human readability
- Example configurations and templates

**Rationale**:

- Promotes consistency across paper implementations
- Enables configuration reuse
- Clear naming and organization
- Separates configuration from code

### 7. Documentation Structure

**Decision**: Use docs/ directory for comprehensive project documentation, supplementing README

**Structure**:

- Subdirectories for guides, API docs, and architecture
- Templates and examples
- Clear organization by documentation type and audience

**Rationale**:

- README provides quick overview
- docs/ provides detailed information
- Organized by type and audience
- Supports documentation growth

### 8. Benchmarking Approach

**Decision**: Create benchmarks/ directory with reproducible performance testing structure

**Requirements**:

- Clear instructions for running benchmarks
- Environment specifications
- Organized structure for scripts and results
- Guidelines for adding new benchmarks

**Rationale**:

- Enables performance comparison across implementations
- Tracks performance improvements over time
- Reproducible results with clear documentation
- Supports ML research requirements

## References

### Source Plan

[notes/plan/01-foundation/01-directory-structure/03-create-supporting-dirs/plan.md](/home/mvillmow/ml-odyssey-manual/notes/plan/01-foundation/01-directory-structure/03-create-supporting-dirs/plan.md)

### Child Plans

1. [Benchmarks Directory](../../../plan/01-foundation/01-directory-structure/03-create-supporting-dirs/01-benchmarks/plan.md)
2. [Docs Directory](../../../plan/01-foundation/01-directory-structure/03-create-supporting-dirs/02-docs/plan.md)
3. [Agents Directory](../../../plan/01-foundation/01-directory-structure/03-create-supporting-dirs/03-agents/plan.md)
4. [Tools Directory](../../../plan/01-foundation/01-directory-structure/03-create-supporting-dirs/04-tools/plan.md)
5. [Configs Directory](../../../plan/01-foundation/01-directory-structure/03-create-supporting-dirs/05-configs/plan.md)
6. [Skills Directory](../../../plan/01-foundation/01-directory-structure/03-create-supporting-dirs/06-skills/plan.md)

### Related Issues

- Issue #592: [Test] Create Supporting Directories
- Issue #593: [Impl] Create Supporting Directories
- Issue #594: [Package] Create Supporting Directories
- Issue #595: [Cleanup] Create Supporting Directories

### Comprehensive Documentation

- [Agent Architecture Review](/home/mvillmow/ml-odyssey-manual/notes/review/agent-architecture-review.md)
- [Skills Design](/home/mvillmow/ml-odyssey-manual/notes/review/skills-design.md)
- [Agent Hierarchy](/home/mvillmow/ml-odyssey-manual/agents/agent-hierarchy.md)
- [Delegation Rules](/home/mvillmow/ml-odyssey-manual/agents/delegation-rules.md)

## Implementation Notes

*This section will be populated during Test, Implementation, Package, and Cleanup phases with findings, decisions, and technical details discovered during execution.*

### Testing Phase Notes

(To be filled during issue #592)

### Implementation Phase Notes

(To be filled during issue #593)

### Packaging Phase Notes

(To be filled during issue #594)

### Cleanup Phase Notes

(To be filled during issue #595)
