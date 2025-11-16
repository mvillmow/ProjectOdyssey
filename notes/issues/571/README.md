# Issue #571: [Plan] Create Agents Directory and Agent System Documentation

## Objective

Establish the agent system infrastructure for ml-odyssey, implementing a comprehensive 6-level hierarchical agent architecture with two distinct directories: `.claude/agents/` for operational agent configurations and `agents/` for documentation, templates, and reference materials. This foundation enables sophisticated task decomposition, parallel execution, and clear coordination patterns integrated with the 5-phase workflow.

## Deliverables

### `.claude/agents/` Directory (Operational Configurations)

- Sub-agent configuration files for operational agents at each hierarchy level
- Level 0, 1, 2 orchestrator configurations
- Level 3, 4, 5 specialist and implementation agent configurations

### `agents/` Directory (Documentation & Templates)

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

## Success Criteria

- [ ] Both `.claude/agents/` and `agents/` directories exist with clear, distinct purposes
- [ ] `agents/README.md` comprehensively explains the entire agent system including:
  - [ ] Directory separation and purposes
  - [ ] Complete 6-level hierarchy
  - [ ] Sub-agents vs skills distinction
  - [ ] Git worktree integration approach
- [ ] `agents/hierarchy.md` provides clear visual representation of all 6 levels with role definitions
- [ ] `agents/delegation-rules.md` documents orchestration patterns and task routing logic
- [ ] `agents/templates/` contains configuration templates for all 6 agent levels
- [ ] Example configurations demonstrate realistic agent setups for multiple hierarchy levels
- [ ] Documentation integrates agent system with 5-phase workflow
- [ ] All documentation is accurate, concise, and actionable
- [ ] Directory structure supports both operational agents and system documentation

## Design Decisions

### 1. Two-Directory Architecture

**Decision**: Separate operational configurations (`.claude/agents/`) from documentation (`agents/`)

**Rationale**:

- Operational configs are working files used by Claude Code during execution
- Documentation, templates, and examples are reference materials for developers
- Clear separation prevents confusion about file purposes
- Aligns with existing `.claude/` convention for operational files
- Keeps repository root organized with high-level documentation

### 2. Six-Level Agent Hierarchy

**Decision**: Implement a 6-level hierarchical architecture (Level 0 to Level 5)

**Hierarchy Structure**:

- **Level 0**: Meta-Orchestrator (Chief Architect Agent) - Strategic decisions and architecture
- **Level 1**: Section Orchestrators (6 sections: Foundation, Core ML, Advanced ML, Production, Deployment, Maintenance)
- **Level 2**: Module Design Agents (Architecture, Integration, Security specialists)
- **Level 3**: Component Specialists (Implementation, Test, Documentation, Performance, Security)
- **Level 4**: Implementation Engineers (Senior, Standard, Test, Docs, Performance)
- **Level 5**: Junior Engineers (Simple tasks, boilerplate generation)

**Rationale**:

- Enables sophisticated task decomposition from strategic to tactical
- Supports parallel execution at appropriate levels
- Provides clear escalation paths for decision-making
- Scales from small to enterprise-sized projects
- Matches typical organizational hierarchies developers understand
- Separates orchestration (Levels 0-2) from execution (Levels 3-5)

### 3. Sub-Agents vs Skills Distinction

**Decision**: Clearly distinguish between sub-agents (separate contexts) and skills (capabilities within context)

**Sub-agents**:

- Operate in separate execution contexts
- Have independent state and memory
- Can work in parallel without interference
- Use git worktrees for isolated branches
- Defined in configuration files

**Skills**:

- Capabilities/tools available within a single agent's context
- Shared state and memory with parent agent
- Execute sequentially within agent workflow
- No separate context or isolation
- Invoked as part of agent's task execution

**Rationale**:

- Prevents confusion about when to use each approach
- Clarifies coordination mechanisms needed
- Guides architecture decisions (isolation vs integration)
- Aligns with Claude Code capabilities and limitations

### 4. Git Worktree Integration

**Decision**: Leverage git worktrees for agent context isolation

**Implementation**:

- Each agent can work in its own git worktree
- Agents work on isolated branches to prevent conflicts
- Enables true parallel execution without interference
- Status files and handoff protocols coordinate between agents

**Rationale**:

- Prevents merge conflicts from parallel work
- Provides natural isolation boundaries
- Supports independent testing and validation
- Aligns with git best practices for parallel development
- Enables rollback of individual agent work without affecting others

### 5. Integration with 5-Phase Workflow

**Decision**: Map agent levels to workflow phases explicitly

**Mapping**:

- **Planning Phase**: Primarily Levels 0-2 (orchestrators and designers)
- **Test/Implementation/Package Phases**: Primarily Levels 3-5 (specialists and engineers)
- **Cleanup Phase**: All levels participate (review and refinement)

**Rationale**:

- Aligns agent capabilities with workflow requirements
- Clarifies which agents handle which phases
- Enables efficient task routing
- Supports existing TDD and parallel execution patterns
- Maintains separation of concerns (design vs implementation)

### 6. Template-Driven Configuration

**Decision**: Provide configuration templates for all agent levels

**Templates Include**:

- Required fields for each level
- Example prompts and capabilities
- Constraint definitions
- Task handoff formats
- Status reporting structures

**Rationale**:

- Accelerates new agent creation
- Ensures consistency across configurations
- Reduces errors from missing required fields
- Provides clear patterns to follow
- Supports customization while maintaining standards
- Serves as living documentation of best practices

### 7. Documentation-First Approach

**Decision**: Treat `agents/` directory as single source of truth for agent system design

**Priority**:

1. Write comprehensive documentation first
2. Create templates based on documented patterns
3. Build example configurations following templates
4. Deploy operational configs to `.claude/agents/`

**Rationale**:

- Ensures design is well-thought-out before implementation
- Creates reference material for all stakeholders
- Enables review and feedback on design before coding
- Supports onboarding and knowledge transfer
- Aligns with documentation-specialist role

## References

### Source Plan

[/notes/plan/01-foundation/01-directory-structure/03-create-supporting-dirs/03-agents/plan.md](../../../plan/01-foundation/01-directory-structure/03-create-supporting-dirs/03-agents/plan.md)

### Related Issues

- **Issue #572**: [Test] Create Agents Directory and Agent System Documentation - Testing
- **Issue #573**: [Impl] Create Agents Directory and Agent System Documentation - Implementation
- **Issue #574**: [Package] Create Agents Directory and Agent System Documentation - Packaging
- **Issue #575**: [Cleanup] Create Agents Directory and Agent System Documentation - Cleanup

### Related Documentation

- `/CLAUDE.md` - Project-level Claude Code instructions
- `/agents/README.md` - Agent system overview (to be created)
- `/agents/hierarchy.md` - Agent hierarchy diagram (to be created)
- `/agents/delegation-rules.md` - Orchestration patterns (to be created)
- `/notes/review/` - Comprehensive specifications directory

## Implementation Notes

_(This section will be populated during Test, Implementation, Package, and Cleanup phases)_

### Key Considerations

- **Claude Code Compatibility**: All configurations must follow Claude Code sub-agents documentation and conventions
- **Scalability**: Design must support project growth from simple to enterprise scale
- **Evolution**: Agent system is foundational but will evolve with project needs
- **Coordination**: Clear handoff protocols needed between isolated agent contexts
- **Status Tracking**: Status files or git integration required for agent coordination

### Dependencies

- Base directory structure (`notes/`, `.claude/`) must exist
- Understanding of project organization and workflow phases
- Knowledge of Claude Code sub-agents capabilities
- Git worktree setup for agent isolation

### Risks & Mitigations

| Risk | Mitigation |
|------|------------|
| Confusion between `.claude/agents/` and `agents/` directories | Clear documentation in README explaining distinct purposes |
| Agent hierarchy becomes too complex to manage | Start with simple examples; document clear delegation rules; provide templates |
| Sub-agent configurations incompatible with Claude Code | Follow Claude Code documentation; reference official sub-agents guide; test configurations |
| Agents in isolated contexts cause coordination issues | Document git worktree patterns; establish status file conventions; define clear handoff protocols |
