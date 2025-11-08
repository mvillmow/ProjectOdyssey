# Agent Hierarchy - Complete Specification

## Overview

This document defines the complete 6-level agent hierarchy for the ml-odyssey project. Each level has
distinct responsibilities, scope, and delegation patterns.

## Hierarchy Diagram

```text
Level 0: Meta-Orchestrator
    │
    ├─> Chief Architect Agent
    │       │
    │       ▼
Level 1: Section Orchestrators
    │
    ├─> Foundation Orchestrator
    ├─> Shared Library Orchestrator
    ├─> Tooling Orchestrator
    ├─> Paper Implementation Orchestrator
    ├─> CI/CD Orchestrator
    └─> Agentic Workflows Orchestrator
            │
            ▼
Level 2: Module Design Agents
    │
    ├─> Architecture Design Agent
    ├─> Integration Design Agent
    └─> Security Design Agent
            │
            ▼
Level 3: Component Specialists
    │
    ├─> Senior Implementation Specialist
    ├─> Test Design Specialist
    ├─> Documentation Specialist
    ├─> Performance Specialist
    └─> Security Implementation Specialist
            │
            ▼
Level 4: Implementation Engineers
    │
    ├─> Senior Implementation Engineer
    ├─> Implementation Engineer
    ├─> Test Engineer
    ├─> Documentation Writer
    └─> Performance Engineer
            │
            ▼
Level 5: Junior Engineers
    │
    ├─> Junior Implementation Engineer
    ├─> Junior Test Engineer
    └─> Junior Documentation Engineer
```

---

## Level 0: Meta-Orchestrator

### Chief Architect Agent

**Scope**: Entire repository ecosystem

**Responsibilities**:

- Select which AI research papers to implement
- Define repository-wide architectural patterns
- Establish coding standards and conventions
- Coordinate across all 6 major sections
- Resolve conflicts between section orchestrators
- Make technology stack decisions
- Monitor overall project health

**Inputs**:

- Research papers
- User requirements
- Project goals
- Industry best practices

**Outputs**:

- High-level roadmap
- Architectural decision records (ADRs)
- Section assignments
- Technology selection documents
- Cross-section dependency graphs

**Delegates To**: Section Orchestrators (Level 1)

**Coordinates With**: External stakeholders, repository owners

**Decision Scope**: System-wide (multiple sections)

**Workflow Phase**: Primarily Plan phase, oversight in all phases

**Configuration File**: `.claude/agents/chief-architect.md`

---

## Level 1: Section Orchestrators

### Foundation Orchestrator

**Scope**: Section 01-foundation

**Responsibilities**:

- Coordinate directory structure creation
- Manage configuration file setup
- Oversee initial documentation
- Ensure foundation is ready before other sections proceed

**Delegates To**: Module Design Agents

**Artifacts**: Foundation completion report, configuration baselines

### Shared Library Orchestrator

**Scope**: Section 02-shared-library

**Responsibilities**:

- Design shared component architecture
- Coordinate core operations, training utilities, data utilities
- Ensure API consistency across modules
- Manage backward compatibility

**Delegates To**: Module Design Agents

**Artifacts**: API documentation, shared library release notes

### Tooling Orchestrator

**Scope**: Section 03-tooling

**Responsibilities**:

- Coordinate tooling development
- Ensure tools integrate with workflow
- Manage CLI interfaces and automation scripts

**Delegates To**: Module Design Agents

**Artifacts**: Tool documentation, automation scripts

### Paper Implementation Orchestrator

**Scope**: Section 04-first-paper and future papers

**Responsibilities**:

- Analyze research paper requirements
- Design paper-specific architecture
- Coordinate data preparation, model implementation, training, evaluation
- Ensure paper implementation follows repository patterns

**Delegates To**: Module Design Agents

**Artifacts**: Paper implementation report, evaluation results

### CI/CD Orchestrator

**Scope**: Section 05-ci-cd

**Responsibilities**:

- Design CI/CD pipeline architecture
- Coordinate testing infrastructure, deployment processes, monitoring
- Ensure quality gates are effective

**Delegates To**: Module Design Agents

**Artifacts**: Pipeline configurations, quality metrics

### Agentic Workflows Orchestrator

**Scope**: Section 06-agentic-workflows

**Responsibilities**:

- Design agent system architecture
- Coordinate research assistant, code review agent, documentation agent
- Ensure agents follow Claude best practices

**Delegates To**: Module Design Agents

**Artifacts**: Agent configurations, prompt templates

**Configuration Files**: `.claude/agents/foundation-orchestrator.md`, etc.

---

## Level 2: Module Design Agents

### Architecture Design Agent

**Scope**: Module-level architecture

**Responsibilities**:

- Break down module into components
- Define component interfaces and contracts
- Design data flow within module
- Identify reusable patterns
- Create module architecture documents

**Inputs**: Section requirements from orchestrator

**Outputs**: Component specifications, interface definitions

**Delegates To**: Component Specialists (Level 3)

**Coordinates With**: Other Module Design Agents for cross-module dependencies

**Workflow Phase**: Plan phase

**Configuration File**: `.claude/agents/architecture-design.md`

### Integration Design Agent

**Scope**: Module-level integration

**Responsibilities**:

- Design integration points between components
- Define module-level APIs
- Create integration test plans
- Manage module dependencies

**Delegates To**: Component Specialists

**Artifacts**: Integration diagrams, API specifications

**Configuration File**: `.claude/agents/integration-design.md`

### Security Design Agent

**Scope**: Module-level security

**Responsibilities**:

- Threat modeling for module
- Define security requirements
- Design authentication/authorization if needed
- Review for security vulnerabilities

**Delegates To**: Security Implementation Specialist (Level 3)

**Artifacts**: Threat models, security requirements

**Configuration File**: `.claude/agents/security-design.md`

---

## Level 3: Component Specialists

### Senior Implementation Specialist

**Scope**: Complex components

**Responsibilities**:

- Break component into functions/classes
- Design component architecture
- Create detailed implementation plan
- Review code quality

**Delegates To**: Implementation Engineers (Level 4)

**Coordinates With**: Test Specialist, Documentation Specialist

**Artifacts**: Component design docs, code review reports

**Workflow Phase**: Plan, Implementation, Cleanup

**Configuration File**: `.claude/agents/senior-implementation-specialist.md`

### Test Design Specialist

**Scope**: Component-level testing

**Responsibilities**:

- Create test plan for component
- Define test cases (unit, integration, edge cases)
- Design test fixtures and mocks
- Specify coverage requirements

**Delegates To**: Test Engineers (Level 4)

**Artifacts**: Test plans, test case specifications

**Workflow Phase**: Plan, Test

**Configuration File**: `.claude/agents/test-design-specialist.md`

### Documentation Specialist

**Scope**: Component-level documentation

**Responsibilities**:

- Write component README
- Document APIs and interfaces
- Create usage examples
- Write tutorials if needed

**Delegates To**: Documentation Writers (Level 4)

**Artifacts**: READMEs, API docs, tutorials

**Workflow Phase**: Plan, Packaging, Cleanup

**Configuration File**: `.claude/agents/documentation-specialist.md`

### Performance Specialist

**Scope**: Component-level performance

**Responsibilities**:

- Define performance requirements
- Design benchmarks
- Identify optimization opportunities
- Profile and analyze performance

**Delegates To**: Performance Engineers (Level 4)

**Artifacts**: Benchmark results, performance reports

**Workflow Phase**: Plan, Implementation, Cleanup

**Configuration File**: `.claude/agents/performance-specialist.md`

### Security Implementation Specialist

**Scope**: Component-level security implementation

**Responsibilities**:

- Implement security requirements
- Code security best practices
- Perform security testing
- Fix vulnerabilities

**Delegates To**: Implementation Engineers (Level 4)

**Artifacts**: Security test results, vulnerability reports

**Workflow Phase**: Plan, Implementation, Test, Cleanup

**Configuration File**: `.claude/agents/security-implementation-specialist.md`

---

## Level 4: Implementation Engineers

### Senior Implementation Engineer

**Scope**: Complex functions/classes

**Responsibilities**:

- Write implementation code
- Follow coding standards
- Implement error handling
- Write inline documentation
- Optimize algorithms

**Inputs**: Detailed specifications from specialists

**Outputs**: Implementation code, unit tests

**Delegates To**: Junior Engineers for simple tasks

**Coordinates With**: Test Engineers for TDD

**Workflow Phase**: Implementation

**Skills Used**: code_generation, refactoring, optimization

**Configuration File**: `.claude/agents/senior-implementation-engineer.md`

### Implementation Engineer

**Scope**: Standard functions/classes

**Responsibilities**:

- Write implementation code
- Follow coding patterns
- Write basic tests
- Document code

**Delegates To**: Junior Engineers for repetitive tasks

**Artifacts**: Source code files, unit tests

**Workflow Phase**: Implementation

**Skills Used**: code_generation, testing, documentation

**Configuration File**: `.claude/agents/implementation-engineer.md`

### Test Engineer

**Scope**: Test implementation

**Responsibilities**:

- Implement unit tests
- Implement integration tests
- Create test fixtures
- Maintain test suite
- Fix failing tests

**Coordinates With**: Implementation Engineers

**Artifacts**: Test files, test reports

**Workflow Phase**: Test

**Skills Used**: test_generation, test_execution, coverage_analysis

**Configuration File**: `.claude/agents/test-engineer.md`

### Documentation Writer

**Scope**: Documentation writing

**Responsibilities**:

- Write docstrings
- Create code examples
- Write README sections
- Update documentation as code changes

**Artifacts**: Documentation files, docstrings

**Workflow Phase**: Packaging

**Skills Used**: documentation_generation, example_extraction

**Configuration File**: `.claude/agents/documentation-writer.md`

### Performance Engineer

**Scope**: Performance implementation

**Responsibilities**:

- Write benchmark code
- Profile code execution
- Implement optimizations
- Verify performance improvements

**Artifacts**: Benchmark code, profiling results

**Workflow Phase**: Implementation, Cleanup

**Skills Used**: profiling, benchmarking, optimization

**Configuration File**: `.claude/agents/performance-engineer.md`

---

## Level 5: Junior Engineers

### Junior Implementation Engineer

**Scope**: Simple functions, boilerplate code

**Responsibilities**:

- Write simple functions
- Generate boilerplate code
- Apply code templates
- Format code
- Run linters

**Inputs**: Clear, detailed instructions

**Outputs**: Simple code implementations

**No Delegation**: Lowest level of hierarchy

**Workflow Phase**: Implementation

**Skills Used**: boilerplate_generation, code_formatting, linting

**Configuration File**: `.claude/agents/junior-implementation-engineer.md`

### Junior Test Engineer

**Scope**: Simple test cases

**Responsibilities**:

- Write simple unit tests
- Generate test boilerplate
- Update existing tests
- Run test suites

**Artifacts**: Basic test implementations

**Workflow Phase**: Test

**Skills Used**: test_generation, test_execution

**Configuration File**: `.claude/agents/junior-test-engineer.md`

### Junior Documentation Engineer

**Scope**: Simple documentation

**Responsibilities**:

- Fill in docstring templates
- Format documentation
- Generate changelog entries
- Update simple README sections

**Artifacts**: Basic documentation

**Workflow Phase**: Packaging

**Skills Used**: documentation_generation, formatting

**Configuration File**: `.claude/agents/junior-documentation-engineer.md`

---

## Delegation Rules

### Rule 1: Scope Reduction

Each delegation reduces scope by one level of abstraction:

- System → Section → Module → Component → Function → Line

### Rule 2: Specification Detail

Each level adds more detail to specifications:

- Strategic goals → Tactical plans → Component specs → Implementation details → Code

### Rule 3: Autonomy Increase

Lower levels have more implementation autonomy but less strategic freedom

### Rule 4: Review Responsibility

Each level reviews work of the level below

### Rule 5: Escalation Path

Issues escalate one level up until resolved

### Rule 6: Coordination Requirements

Agents coordinate horizontally when sharing resources or dependencies

---

## Agent Configuration Template

```text
---
name: agent-name
description: Brief description of when to use this agent
tools: Read,Write,Edit,Bash,Grep,Glob
model: sonnet
---

# Agent Name

## Role

[Agent's role in the hierarchy]

## Responsibilities

- Responsibility 1
- Responsibility 2

## Scope

[What this agent handles]

## Delegation

Delegates to: [Lower level agents]
Coordinates with: [Same level agents]

## Workflow Phase

[Which phases this agent participates in]

## Skills Used

- skill_name_1
- skill_name_2

## Instructions

[Detailed instructions for this agent]

## Examples

[Example tasks this agent handles]

## Constraints

[What this agent should NOT do]
```

---

## Mapping to Organizational Models

### Traditional Hierarchy

- Level 0 = CTO/VP Engineering
- Level 1 = Engineering Managers
- Level 2 = Principal/Staff Engineers
- Level 3 = Senior Engineers
- Level 4 = Engineers
- Level 5 = Junior Engineers/Interns

### Spotify Model

- Tribes = Level 1 (Section Orchestrators)
- Squads = Level 2 (Module Design Agents)
- Chapters = Cross-cutting specialists
- Guilds = Skills shared across agents

---

## Integration with 5-Phase Workflow

| Phase | Active Levels | Focus |
|-------|---------------|-------|
| Plan | 0, 1, 2, 3 | Orchestrators and designers create specifications |
| Test | 3, 4, 5 | Specialists and engineers write tests |
| Implementation | 3, 4, 5 | Specialists and engineers build functionality |
| Packaging | 3, 4, 5 | Specialists and engineers integrate artifacts |
| Cleanup | All | All levels review and refactor their work |

---

## Next Steps

1. Create configuration files for each agent in `.claude/agents/`
2. Create templates in `agents/templates/`
3. Test agent delegation patterns
4. Document specific workflows
5. Train team on agent usage

## References

- [Claude Code Sub-Agents](https://code.claude.com/docs/en/sub-agents)
- [Orchestration Patterns](./orchestration-patterns.md)
- [Skills Design](./skills-design.md)
