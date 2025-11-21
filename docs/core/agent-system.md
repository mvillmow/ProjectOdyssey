# Agent System User Guide

## Overview

The ML Odyssey project uses a **multi-level agent system** to organize development work. Agents are AI-powered
decision-makers that coordinate code review, planning, implementation, testing, and documentation across the entire
project.

**Key Idea**: Instead of one Claude instance handling everything, specialized agents at different levels work together
like a software engineering team:

- **Architects** (Levels 0-2) make strategic and design decisions
- **Specialists** (Level 3) plan components and coordinate aspects
- **Engineers** (Levels 4-5) write code, tests, and documentation

## When to Use Agents

### Use Agents For

- **Planning features** - Use Section Orchestrators to break down work
- **Code review** - Code Review Orchestrator routes to specialized reviewers
- **Implementing components** - Implementation Engineers handle coding
- **Writing tests** - Test Engineers create comprehensive test suites
- **Creating documentation** - Documentation Engineers write docstrings and guides
- **Performance optimization** - Performance Engineers profile and optimize

### Don't Use Agents For

- Quick syntax questions (ask Claude directly)
- One-off script fixes (use Implementation Engineers if part of codebase)
- External tool issues (escalate to environment team)

## Agent Levels at a Glance

| Level | Role | Scope | What They Do |
| --- | --- | --- | --- |
| 1 | Section Orchestrators | Major sections | Break down sections into modules, manage dependencies |
| 2 | Module Designers | Modules | Design interfaces, security, integration |
| 3 | Specialists | Components | Plan implementation, tests, docs, performance |
| 4 | Engineers | Functions/classes | Write code, tests, documentation |
| 5 | Junior Engineers | Simple tasks | Boilerplate, formatting, simple implementations |

## Available Agents by Category

### Architecture & Planning

- **Chief Architect** - Paper selection, system design decisions
- **Section Orchestrators** (6) - Foundation, Shared Library, Tooling, Papers, CI/CD, Workflows
- **Architecture Design Agent** - Module structure and interfaces
- **Integration Design Agent** - Cross-module APIs and Python-Mojo interop
- **Security Design Agent** - Threat modeling and security requirements

### Implementation & Coding

- **Senior Implementation Engineer** - Complex, performance-critical code
- **Implementation Engineer** - Standard functions and classes
- **Junior Implementation Engineer** - Boilerplate and simple functions

### Testing

- **Test Specialist** - Test planning and strategy
- **Test Engineer** - Unit and integration tests
- **Junior Test Engineer** - Simple tests and test fixtures

### Documentation

- **Documentation Specialist** - Component READMEs and API design
- **Documentation Engineer** - Docstrings, examples, updates
- **Junior Documentation Engineer** - Docstring templates, formatting

### Specialized Work

- **Performance Specialist** - Performance requirements and strategy
- **Performance Engineer** - Benchmark code and optimization
- **Security Specialist** - Security implementation and testing

## How to Use Agents

### Automatic Invocation (Recommended)

Claude will automatically invoke the right agent when you describe a task:

```text
```text

User: "Design the architecture for the neural network module"
→ Architecture Design Agent invokes automatically

```text

### Explicit Invocation

You can explicitly request an agent:

```text

```text

User: "Use the Implementation Engineer to code the forward pass function"
→ Implementation Engineer invokes explicitly

```text

### Communication Pattern

When working with agents:

1. **Describe what you need** - "I need tests for the tensor operations module"
1. **Agent takes action** - Reviews context, coordinates work, implements solution
1. **Agent reports status** - Shows progress, asks clarifying questions
1. **You review output** - Check documentation, tests, code in `/notes/issues/<issue-number>/`

## Quick Reference: Which Agent to Use

### I need to

- **Design a feature** → Architecture Design Agent
- **Plan component work** → Component Specialist (appropriate type)
- **Write a function** → Implementation Engineer
- **Test code** → Test Engineer
- **Write documentation** → Documentation Engineer
- **Optimize performance** → Performance Engineer
- **Review a pull request** → Code Review Orchestrator
- **Fix security issue** → Security Specialist
- **Coordinate a section** → Section Orchestrator

## Where Agents Document Work

All agent work is documented in:

```text
```text

/notes/issues/<issue-number>/README.md

```text

This directory contains:

- Implementation decisions
- Design choices
- Examples and usage patterns
- Links to comprehensive docs in `/agents/` and `/notes/review/`

## Understanding the Workflow

Agents work through a **5-phase workflow**:

1. **Plan** (Sequential) - Architects and designers create specifications
2. **Test** (Parallel) - Test Engineers write test suites
3. **Implement** (Parallel) - Implementation Engineers write code
4. **Package** (Parallel) - Create distributable artifacts (`.mojopkg` files, archives)
5. **Cleanup** (Sequential) - All levels review and refactor

Each GitHub issue focuses on one phase for a specific component.

## Comprehensive Documentation

For detailed information, see:

- **[/agents/README.md](../../../../../../agents/README.md)** - Complete agent system overview
- **[/agents/hierarchy.md](../../../../../../agents/hierarchy.md)** - Visual hierarchy diagram
- **[/agents/delegation-rules.md](../../../../../../agents/delegation-rules.md)** - How agents coordinate
- **[/notes/review/README.md](../../../../../../notes/review/README.md)** - Architectural decisions and detailed specs

## Key Principles

1. **Trust the hierarchy** - Agents know their scope and limitations
2. **Let agents delegate** - Don't micromanage; let orchestrators coordinate
3. **Communicate clearly** - Describe what you need, not how to do it
4. **Escalate blockers** - Don't stay stuck; agents escalate when needed
5. **Document decisions** - Agent work is documented for team reference

## Getting Started

To work with agents effectively:

1. **Read the overview** - Understand the 6-level hierarchy
2. **Pick an agent** - Match the work to appropriate level
3. **Describe your task** - Give context and requirements
4. **Review the output** - Check `/notes/issues/<issue-number>/`
5. **Link to comprehensive docs** - Find detailed specs in `/agents/` and `/notes/review/`

## Still Have Questions

- Check `/agents/README.md` for complete overview
- Review `/agents/hierarchy.md` for visual reference
- See `/notes/review/orchestration-patterns.md` for coordination details
- Ask in your team channel

---

**Last Updated**: 2025-11-17 | **Status**: Production
