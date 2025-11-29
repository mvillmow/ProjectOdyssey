# Agent Hierarchy - Visual Diagram and Quick Reference

## Hierarchy Diagram

```text
┌─────────────────────────────────────────────────────────────┐
│                    Level 0: Meta-Orchestrator                │
│                   Chief Architect Agent                      │
│         (System-wide decisions, paper selection)             │
└───────────────────────────┬─────────────────────────────────┘
                            │
        ┌───────────────────┼───────────────────┐
        ▼                   ▼                   ▼
┌──────────────────┐ ┌──────────────────┐ ┌──────────────────┐
│   Level 1:       │ │   Level 1:       │ │   Level 1:       │
│   Foundation     │ │ Shared Library   │ │    Tooling       │
│  Orchestrator    │ │  Orchestrator    │ │  Orchestrator    │
└────────┬─────────┘ └────────┬─────────┘ └────────┬─────────┘
         │                    │                     │
┌────────┴────────┐  ┌────────┴────────┐  ┌────────┴────────┐
│  Level 1: Paper │  │  Level 1: CI/CD │  │ Level 1: Agentic│
│ Implementation  │  │  Orchestrator   │  │    Workflows    │
│  Orchestrator   │  │                 │  │  Orchestrator   │
└────────┬────────┘  └────────┬────────┘  └────────┬────────┘
         │                    │                     │
         └────────────────────┼─────────────────────┘
                              ▼
            ┌─────────────────────────────────────┐
            │   Level 2: Design & Review Agents   │
            ├─────────────────────────────────────┤
            │  • Architecture Design Agent        │
            │  • Integration Design Agent         │
            │  • Security Design Agent            │
            │  • Code Review Orchestrator         │
            └──────────────────┬──────────────────┘
                               │
                               ▼
            ┌─────────────────────────────────────┐
            │   Level 3: Specialists (24 agents)   │
            ├──────────────────────────────────────┤
            │  • Implementation Specialist         │
            │  • Test Specialist                   │
            │  • Documentation Specialist          │
            │  • Performance Specialist            │
            │  • Blog Writer Specialist            │
            │  • 13 Code Review Specialists        │
            │  • 4 Additional Specialists          │
            └──────────────────┬──────────────────┘
                               │
                               ▼
            ┌─────────────────────────────────────┐
            │   Level 4: Engineers (6 agents)      │
            ├──────────────────────────────────────┤
            │  • Senior Implementation Engineer    │
            │  • Implementation Engineer           │
            │  • Test Engineer                     │
            │  • Documentation Engineer            │
            │  • Performance Engineer              │
            │  • Log Analyzer                      │
            └──────────────────┬──────────────────┘
                               │
                               ▼
            ┌─────────────────────────────────────┐
            │      Level 5: Junior Engineers       │
            ├──────────────────────────────────────┤
            │  • Junior Implementation Engineer    │
            │  • Junior Test Engineer              │
            │  • Junior Documentation Engineer     │
            └──────────────────────────────────────┘
```text

## Level Summaries

### Level 0: Meta-Orchestrator

- **Agents**: 1 (Chief Architect)
- **Scope**: Entire repository
- **Decisions**: Strategic (paper selection, tech stack, architecture)
- **Phase**: Primarily Plan
- **Language Context**: Makes Mojo vs Python decisions for different components

### Level 1: Section Orchestrators

- **Agents**: 6 (one per major section)
- **Scope**: Repository sections
- **Decisions**: Tactical (module organization, dependencies)
- **Phase**: Plan
- **Language Context**: Coordinates Mojo implementation across sections

### Level 2: Module Design & Review Agents

- **Agents**: 4 total (3 design agents + 1 review orchestrator)
  - Design Agents: Architecture, Integration, Security
  - Code Review Orchestrator: Routes PR changes to 13 specialist reviewers
- **Scope**: Modules within sections and overall PR review coordination
- **Decisions**: Module structure, interfaces, security, and code review routing
- **Phase**: Plan (design) and Cleanup (code review)
- **Language Context**: Designs Mojo module structures, leverages Mojo features (SIMD, traits, structs); coordinates review across all code dimensions

### Level 3: Specialists & Review Specialists

- **Agents**: 24 total (11 implementation/execution specialists + 13 code review specialists)
- **Scope**: Components within modules and PR review dimensions
- **Decisions**: Component implementation approach and code review assessment
- **Phase**: Plan, Test, Implementation, Package, Cleanup
- **Language Context**: Chooses Mojo patterns (fn vs def, struct vs class, SIMD usage); reviews for language correctness, safety, and idioms
- **Package Phase**: Design packaging strategy, specify .mojopkg requirements, plan CI/CD workflows
- **Review Specialties**: Implementation, documentation, test, security, safety, Mojo language, performance, algorithm, architecture, data engineering, paper, research, dependency review

### Level 4: Implementation Engineers

- **Agents**: 6 (Senior Implementation, Implementation, Test, Documentation, Performance, Log Analyzer)
- **Scope**: Functions and classes
- **Decisions**: Implementation details
- **Phase**: Test, Implementation, Package
- **Language Context**: Writes Mojo code, uses Mojo standard library, implements algorithms
- **Package Phase**: Build .mojopkg files, create distribution archives, implement packaging scripts

### Level 5: Junior Engineers

- **Agents**: 3 types (Implementation, Test, Documentation)
- **Scope**: Simple functions, boilerplate
- **Decisions**: None (follows instructions)
- **Phase**: Test, Implementation, Package
- **Language Context**: Generates Mojo boilerplate, applies formatting
- **Package Phase**: Run package builds, verify installations, execute packaging commands

## Mojo-Specific Considerations

### Language Expertise by Level

**Level 0-2** (Architects and Designers):

- Deep understanding of Mojo vs Python trade-offs
- Knowledge of Mojo compilation model
- Familiarity with SIMD operations and performance characteristics
- Understanding of MAX platform integration

**Level 3** (Specialists):

- Proficiency in Mojo syntax and idioms
- Knowledge of Mojo traits, structs, and memory management
- Understanding of `fn` vs `def`, owned vs borrowed, etc.
- Ability to design Mojo-specific patterns

**Level 4-5** (Engineers):

- Hands-on Mojo coding ability
- Familiarity with Mojo standard library
- Ability to write performance-critical code
- Knowledge of Mojo testing frameworks

### Mojo-Python Hybrid Considerations

- **Level 0-1**: Decide which components use Mojo vs Python
  - Mojo: Performance-critical ML operations (training, inference)
  - Python: Scripting, tooling, data loading, visualization

- **Level 2-3**: Design interop between Mojo and Python
  - Use Mojo for tensor operations
  - Use Python for data preprocessing
  - Design clean interfaces between languages

- **Level 4-5**: Implement with appropriate language
  - Follow architecture decisions on language choice
  - Ensure proper type annotations
  - Handle conversions between Python and Mojo

## Delegation Flow

### Top-Down (Task Decomposition)

```text
Paper Selection (Level 0)
    ↓
Section Planning (Level 1)
    ↓
Module Design (Level 2)
    ↓
Component Specification (Level 3)
    ↓
Function Implementation (Level 4)
    ↓
Boilerplate Generation (Level 5)
```text

### Bottom-Up (Status Reporting)

```text
Code Metrics (Level 5)
    ↑
Component Health (Level 4)
    ↑
Module Stability (Level 3)
    ↑
Section Status (Level 2)
    ↑
Project Health (Level 1)
    ↑
Strategic Alignment (Level 0)
```text

## Agent Count

| Level | Name | Current Count |
|-------|------|---|
| 0     | Meta-Orchestrator | 1 |
| 1     | Section Orchestrators | 6 |
| 2     | Module Design & Review Orchestrators | 4 |
| 3     | Specialists (Implementation + Code Review) | 24 |
| 4     | Implementation Engineers | 6 |
| 5     | Junior Engineers | 3 |
| **Total** | **All Agents** | **44** |

**Level 3 Breakdown:**
- Implementation/Execution Specialists: 11 (implementation, test, documentation, performance, security, blog writer, numerical stability, test flakiness, PR cleanup, mojo syntax validator, CI failure analyzer)
- Code Review Specialists: 13 (implementation, documentation, test, security, safety, mojo language, performance, algorithm, architecture, data engineering, paper, research, dependency)

*Historical Note: Initial planning estimated 23 agent types. Actual implementation has expanded to 44 specialized agents to handle emerging needs (blog writing, CI analysis, numerical stability, test flakiness, PR cleanup, mojo validation).*

## Quick Reference

### When to Use Each Level

**Use Level 0** when:

- Selecting which research paper to implement
- Making system-wide architectural decisions
- Resolving cross-section conflicts

**Use Level 1** when:

- Planning a major repository section
- Coordinating multiple modules
- Managing section dependencies

**Use Level 2** when:

- Designing module architecture
- Defining component interfaces
- Planning security or integration

**Use Level 3** when:

- Specifying component implementation
- Planning tests for a component
- Designing performance optimization strategy

**Use Level 4** when:

- Writing Mojo code for functions/classes
- Implementing tests
- Writing documentation

**Use Level 5** when:

- Generating Mojo boilerplate
- Formatting code
- Simple documentation tasks

## Coordination Rules

1. **Delegate Down**: When task is too detailed for current level
1. **Escalate Up**: When decision exceeds current authority
1. **Coordinate Laterally**: When sharing resources or dependencies
1. **Report Status**: Keep superior informed of progress
1. **Document Decisions**: Capture rationale for future reference

## See Also

- [README.md](README.md) - Overview and quick start
- [delegation-rules.md](delegation-rules.md) - Detailed coordination patterns
- [templates/](templates/) - Agent configuration templates
- [agent-hierarchy.md](agent-hierarchy.md) - Complete detailed specification
- [/notes/review/orchestration-patterns.md](../notes/review/orchestration-patterns.md) - Orchestration patterns
- [Mojo Documentation](https://docs.modular.com/mojo/manual/) - Mojo language reference
