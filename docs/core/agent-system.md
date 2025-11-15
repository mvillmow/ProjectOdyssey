# Agent System Guide

Understanding and using ML Odyssey's hierarchical agent system for development.

## Overview

ML Odyssey uses a **hierarchical agent system** powered by Claude to coordinate development work. This system enables
efficient, high-quality development through specialized agents that handle different aspects of the project.

This document provides an overview for developers. For complete agent documentation, see `/agents/`.

## Why Use Agents?

The agent system provides:

- **Specialization**: Each agent is expert in their domain
- **Consistency**: Standardized approaches across the codebase
- **Quality**: Built-in review and validation
- **Efficiency**: Parallel work on independent tasks
- **Documentation**: Automatic tracking of decisions and progress

## Agent Hierarchy

ML Odyssey uses a 6-level hierarchy:

```text
Level 0: Chief Architect
    ↓
Level 1: Orchestrators (Section)
    ↓
Level 2: Design Agents & Orchestrators (Module)
    ↓
Level 3: Specialists (Component)
    ↓
Level 4: Engineers (Implementation)
    ↓
Level 5: Junior Engineers (Simple tasks)
```

### Level 0: Chief Architect

**Role**: Strategic decisions and architectural oversight

**Responsibilities**:

- System-wide architecture decisions
- Technology selection
- Cross-section coordination
- Architectural reviews

**When to consult**: Major architectural changes, technology choices, cross-cutting concerns

### Level 1: Orchestrators

**Role**: Coordinate work within a major section

**Sections**:

- **Foundation Orchestrator**: Repository structure, configuration
- **Shared Library Orchestrator**: Core reusable components
- **Tooling Orchestrator**: Development and testing tools
- **First Paper Orchestrator**: LeNet-5 implementation
- **CI/CD Orchestrator**: Continuous integration pipelines
- **Agentic Workflows Orchestrator**: Agent system development

**Responsibilities**:

- Plan section-level work
- Coordinate module-level agents
- Ensure section consistency
- Track section progress

**When to use**: Starting work on a new section, coordinating multiple modules

### Level 2: Design Agents & Orchestrators

**Role**: Module-level design and coordination

**Agents**:

- **Design Agent**: Module architecture and specifications
- **Code Review Orchestrator**: Coordinate code reviews
- **Module Orchestrators**: Coordinate module implementation

**Responsibilities**:

- Design module architecture
- Create specifications
- Coordinate implementation
- Review code quality

**When to use**: Designing a new module, coordinating module development

### Level 3: Specialists

**Role**: Component-level expertise

**Agents**:

- **Implementation Specialist**: Component implementation
- **Test Specialist**: Testing strategy and implementation
- **Documentation Specialist**: Component documentation
- **Integration Specialist**: Component integration
- **Performance Specialist**: Optimization and profiling

**Responsibilities**:

- Design components
- Coordinate engineers
- Ensure quality
- Review deliverables

**When to use**: Implementing a component, writing tests, creating documentation

### Level 4: Engineers

**Role**: Hands-on implementation

**Agents**:

- **Mojo Engineer**: Mojo code implementation
- **Test Engineer**: Test implementation
- **Documentation Engineer**: Documentation writing

**Responsibilities**:

- Write code
- Implement tests
- Create documentation
- Fix bugs

**When to use**: Implementing specific features, writing tests, creating docs

### Level 5: Junior Engineers

**Role**: Simple, well-defined tasks

**Agents**:

- **Junior Mojo Engineer**: Simple Mojo implementations
- **Junior Test Engineer**: Basic test cases
- **Junior Documentation Engineer**: Simple documentation

**Responsibilities**:

- Implement simple features
- Write basic tests
- Update documentation

**When to use**: Simple, repetitive tasks with clear specifications

## 5-Phase Development Workflow

Every component follows a structured workflow:

```text
Plan → [Test | Implementation | Packaging] → Cleanup
```

### Phase 1: Plan

**Agent**: Design Agent or Specialist (depending on scope)

**Outputs**:

- Component specification
- Architecture decisions
- Implementation plan
- Test plan

**Example**:

```bash
# Issue #39: [Plan] Data Module
- Design module architecture
- Specify dataset interfaces
- Plan data loader implementation
- Define test strategy
```

### Phase 2: Test (Parallel)

**Agent**: Test Specialist → Test Engineer

**Outputs**:

- Test suite structure
- Unit tests
- Integration tests
- Test documentation

**Example**:

```bash
# Issue #40: [Test] Data Module
- Implement dataset tests
- Implement loader tests
- Implement transform tests
- Integration tests
```

### Phase 3: Implementation (Parallel)

**Agent**: Implementation Specialist → Mojo Engineer

**Outputs**:

- Module implementation
- Core functionality
- API implementation
- Code documentation

**Example**:

```bash
# Issue #41: [Impl] Data Module
- Implement TensorDataset
- Implement BatchLoader
- Implement transforms
- Implement samplers
```

### Phase 4: Packaging (Parallel)

**Agent**: Integration Specialist → Mojo Engineer

**Outputs**:

- Module integration
- Examples
- Documentation
- Benchmarks

**Example**:

```bash
# Issue #42: [Package] Data Module
- Integrate with shared library
- Create usage examples
- Write user documentation
- Performance benchmarks
```

### Phase 5: Cleanup

**Agent**: Code Review Orchestrator → Specialists

**Outputs**:

- Code refactoring
- Documentation updates
- Performance optimization
- Final review

**Example**:

```bash
# Issue #43: [Cleanup] Data Module
- Refactor based on review
- Update documentation
- Optimize performance
- Final quality check
```

## Using Agents in Practice

### Starting a New Feature

#### Step 1: Identify the right orchestrator

```bash
# Example: Adding a new paper implementation
# Use: First Paper Orchestrator
```

#### Step 2: Create GitHub issue

```bash
gh issue create \
  --title "[Plan] AlexNet Implementation" \
  --label "planning" \
  --body "Plan implementation of AlexNet paper"
```

#### Step 3: Let orchestrator delegate

The orchestrator will:

1. Break down work into components
2. Create child issues for each phase
3. Delegate to appropriate specialists
4. Track progress

### Implementing a Component

#### Step 1: Specialist creates specification

```markdown
# Issue #XX: [Plan] Conv2D Layer

## Specification

- Forward pass with SIMD optimization
- Backward pass with gradient computation
- Support for padding, stride, dilation
- Comprehensive tests

## Architecture

[Detailed design...]
```

#### Step 2: Engineers implement in parallel

- **Test Engineer**: Writes tests from specification
- **Mojo Engineer**: Implements functionality
- **Documentation Engineer**: Creates documentation

#### Step 3: Specialist reviews and integrates

- Reviews all implementations
- Ensures consistency
- Runs tests
- Merges work

### Code Review Workflow

#### Step 1: Create Pull Request

```bash
# Link PR to issue
gh pr create --issue XX
```

#### Step 2: Code Review Orchestrator assigns reviewers

Reviewers based on:

- Code Review Specialist: Architecture review
- Implementation Specialist: Code quality
- Test Specialist: Test coverage

#### Step 3: Address feedback

```bash
# Reply to each review comment
gh api repos/OWNER/REPO/pulls/PR/comments/COMMENT_ID/replies \
  --method POST \
  -f body="✅ Fixed - [description]"
```

#### Step 4: Merge after approval

## Agent Communication

### Documentation Flow

All agent work is documented in `/notes/issues/<issue-number>/README.md`:

```markdown
# Issue #XX: [Phase] Component

## Objective

What this issue accomplishes

## Deliverables

- List of outputs

## Success Criteria

- Checklist of completion

## Implementation Notes

- Findings during implementation
- Decisions made
- Challenges encountered
```

### Escalation

Agents escalate when they encounter:

- **Ambiguity**: Unclear requirements
- **Scope Change**: Work outside original scope
- **Technical Blockers**: Can't proceed without help
- **Design Questions**: Architecture decisions needed

**Escalation path**:

```text
Engineer → Specialist → Design Agent → Orchestrator → Chief Architect
```

### Delegation

Agents delegate work down the hierarchy:

```text
Orchestrator
    ↓ (delegates module)
Design Agent
    ↓ (delegates components)
Specialist
    ↓ (delegates implementation)
Engineer
    ↓ (delegates simple tasks)
Junior Engineer
```

## Common Workflows

### Workflow 1: New Module

```text
1. Orchestrator: Plan module structure
2. Design Agent: Create specifications
3. Specialists: Design components (parallel)
4. Engineers: Implement (parallel)
5. Code Review Orchestrator: Review and integrate
```

### Workflow 2: Bug Fix

```text
1. Engineer: Identify root cause
2. Specialist: Review and approve fix
3. Test Engineer: Add regression test
4. Code Review: Quick review
5. Merge
```

### Workflow 3: Performance Optimization

```text
1. Performance Specialist: Profile and identify bottlenecks
2. Implementation Specialist: Design optimization
3. Mojo Engineer: Implement optimization
4. Test Engineer: Verify correctness maintained
5. Performance Specialist: Benchmark and validate
```

### Workflow 4: Documentation

```text
1. Documentation Specialist: Plan documentation
2. Documentation Engineer: Write docs
3. Implementation Specialist: Technical review
4. Documentation Specialist: Final review
```

## Best Practices

### DO

- ✅ Start with the right orchestrator for your section
- ✅ Let agents delegate naturally (don't skip levels)
- ✅ Document all decisions in `/notes/issues/`
- ✅ Link all PRs to issues
- ✅ Reply to all review comments
- ✅ Follow the 5-phase workflow

### DON'T

- ❌ Skip hierarchy levels (no skip-level delegation)
- ❌ Work outside your agent's scope
- ❌ Create PRs without linked issues
- ❌ Ignore review feedback
- ❌ Duplicate work across issues

## Troubleshooting

### Not Sure Which Agent to Use?

1. **Check scope**: Section? Module? Component?
2. **Check phase**: Plan? Test? Implementation?
3. **Check hierarchy**: Start at orchestrator, let them delegate

### Agent Seems Stuck?

**Common causes**:

- Missing inputs (escalate to get them)
- Unclear requirements (escalate for clarification)
- Technical blocker (escalate for help)

**Solution**: Escalate up one level

### Work Overlapping?

**Cause**: Usually improper delegation or scope creep

**Solution**:

1. Check issue objectives
2. Verify no duplicate issues exist
3. Clarify scope with orchestrator

## Agent System Documentation

For complete documentation, see:

- **`/agents/README.md`** - Quick start guide
- **`/agents/hierarchy.md`** - Visual hierarchy diagram
- **`/agents/agent-hierarchy.md`** - Complete agent specifications
- **`/agents/delegation-rules.md`** - Delegation patterns
- **`/agents/templates/`** - Agent configuration templates

### Agent Configuration

Agents are configured in `.claude/agents/`:

```text
.claude/agents/
├── orchestrators/
│   ├── shared-library-orchestrator.md
│   ├── first-paper-orchestrator.md
│   └── ...
├── specialists/
│   ├── implementation-specialist.md
│   ├── test-specialist.md
│   └── ...
└── engineers/
    ├── mojo-engineer.md
    ├── test-engineer.md
    └── ...
```

## Learning More

### For New Contributors

1. **Read**: `/agents/README.md` - Quick introduction
2. **Review**: Recent GitHub issues to see agents in action
3. **Start small**: Pick a "good first issue" assigned to Junior Engineer
4. **Ask questions**: GitHub Discussions or issue comments

### For Experienced Developers

1. **Read**: `/agents/agent-hierarchy.md` - Complete specifications
2. **Study**: `/notes/review/orchestration-patterns.md` - Advanced patterns
3. **Practice**: Take on Specialist-level work
4. **Contribute**: Help improve agent system itself

## Next Steps

- **[Workflow](workflow.md)** - Detailed 5-phase workflow explanation
- **[Project Structure](project-structure.md)** - Where agent outputs go
- **[Testing Strategy](testing-strategy.md)** - How tests fit into agent workflow
- **[Configuration](configuration.md)** - Agent configuration files

## Related Documentation

- [Agent README](/agents/README.md) - Quick start
- [Agent Hierarchy](/agents/hierarchy.md) - Visual diagram
- [Delegation Rules](/agents/delegation-rules.md) - Coordination patterns
- [Orchestration Patterns](/notes/review/orchestration-patterns.md) - Advanced patterns
