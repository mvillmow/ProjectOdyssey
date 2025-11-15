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
            │      Level 2: Module Design Agents  │
            ├─────────────────────────────────────┤
            │  • Architecture Design Agent        │
            │  • Integration Design Agent         │
            │  • Security Design Agent            │
            └──────────────────┬──────────────────┘
                               │
                               ▼
            ┌─────────────────────────────────────┐
            │   Level 3: Component Specialists     │
            ├──────────────────────────────────────┤
            │  • Senior Implementation Specialist  │
            │  • Test Design Specialist            │
            │  • Documentation Specialist          │
            │  • Performance Specialist            │
            │  • Security Implementation Specialist│
            └──────────────────┬──────────────────┘
                               │
                               ▼
            ┌─────────────────────────────────────┐
            │   Level 4: Implementation Engineers  │
            ├──────────────────────────────────────┤
            │  • Senior Implementation Engineer    │
            │  • Implementation Engineer           │
            │  • Test Engineer                     │
            │  • Documentation Writer              │
            │  • Performance Engineer              │
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
```

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

### Level 2: Module Design Agents

- **Agents**: 3 core types (Architecture, Integration, Security)
- **Scope**: Modules within sections
- **Decisions**: Module structure, interfaces, security
- **Phase**: Plan
- **Language Context**: Designs Mojo module structures, leverages Mojo features (SIMD, traits, structs)

### Level 3: Component Specialists

- **Agents**: 5 types (Implementation, Test, Docs, Performance, Security)
- **Scope**: Components within modules
- **Decisions**: Component implementation approach
- **Phase**: Plan, Test, Implementation, Package
- **Language Context**: Chooses Mojo patterns (fn vs def, struct vs class, SIMD usage)
- **Package Phase**: Design packaging strategy, specify .mojopkg requirements, plan CI/CD workflows

### Level 4: Implementation Engineers

- **Agents**: 5 types (Senior, Standard, Test, Docs, Performance)
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
```

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
```

## Agent Count Estimates

| Level | Types | Per Paper | Total (6 sections) |
|-------|-------|-----------|-------------------|
| 0     | 1     | 1         | 1                 |
| 1     | 6     | 6         | 6                 |
| 2     | 3     | 18        | 18                |
| 3     | 5     | 30        | 30                |
| 4     | 5     | 50        | 50                |
| 5     | 3     | 30        | 30                |
| **Total** | **23 types** | **135** | **135** |

*Note: These are estimates. Actual count depends on project complexity.*

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
2. **Escalate Up**: When decision exceeds current authority
3. **Coordinate Laterally**: When sharing resources or dependencies
4. **Report Status**: Keep superior informed of progress
5. **Document Decisions**: Capture rationale for future reference

## See Also

- [README.md](README.md) - Overview and quick start
- [delegation-rules.md](delegation-rules.md) - Detailed coordination patterns
- [templates/](templates/) - Agent configuration templates
- [agent-hierarchy.md](agent-hierarchy.md) - Complete detailed specification
- [/notes/review/orchestration-patterns.md](/notes/review/orchestration-patterns.md) - Orchestration patterns
- [Mojo Documentation](https://docs.modular.com/mojo/manual/) - Mojo language reference
