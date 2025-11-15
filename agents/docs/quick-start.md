# Quick Start Guide - Agent System

## 5-Minute Introduction

Welcome to the ML Odyssey agent system! This guide will get you started with using our 6-level hierarchical agent
system for Mojo-based AI research implementations.

### What Are Agents?

Agents are AI assistants organized in a hierarchy, each with specific roles and responsibilities. Think of them as
a team of specialists:

- **High levels** (0-2): Strategic planners and architects
- **Mid levels** (3): Component specialists who coordinate implementation
- **Low levels** (4-5): Engineers who write code, tests, and documentation

### How It Works

When you give a task to the system:

1. The right agent is automatically selected based on your request
2. That agent analyzes the task and delegates to lower-level agents
3. Agents coordinate horizontally and report progress vertically
4. Work happens in parallel worktrees when possible
5. Results are integrated and delivered

## Table of Contents

- [How to Invoke Agents](#how-to-invoke-agents)
- [Common Use Cases](#common-use-cases)
- [Troubleshooting Quick Reference](#troubleshooting-quick-reference)
- [Next Steps](#next-steps)

## How to Invoke Agents

### Automatic Invocation (Recommended)

Simply describe what you want in natural language. Claude Code will automatically invoke the appropriate agent:

```text
"Design the architecture for the LeNet-5 implementation"
→ Automatically invokes Architecture Design Agent

"Write tests for the convolution layer"
→ Automatically invokes Test Engineer

"Optimize the matrix multiplication performance"
→ Automatically invokes Performance Engineer
```

**Why automatic?** Each agent has a carefully crafted description that helps Claude recognize when to use it.

### Explicit Invocation

You can also explicitly name the agent you want:

```text
"Use the chief architect agent to evaluate which paper to implement next"

"Have the senior implementation engineer write the SIMD-optimized convolution"

"Ask the documentation specialist to create a tutorial"
```

**When to use explicit?** When you know exactly which specialist you need, or when automatic invocation selected
the wrong agent.

### Checking Available Agents

View all agents in the catalog:

```bash
# List all agent configuration files
ls -la .claude/agents/

# Read a specific agent's configuration
cat .claude/agents/chief-architect.md
```

## Common Use Cases

### 1. Implementing a New Feature

**Scenario**: You need to add a new neural network layer type

**How to invoke**:

```text
"Implement a batch normalization layer for our Mojo neural network library"
```

**What happens**:

1. Chief Architect analyzes requirements
2. Shared Library Orchestrator coordinates the module
3. Architecture Design Agent designs the layer structure
4. Implementation Specialist creates detailed specs
5. Senior Implementation Engineer writes Mojo code with SIMD
6. Test Engineer creates comprehensive tests
7. Documentation Writer adds API docs

### 2. Fixing a Bug

**Scenario**: Tests are failing in the training loop

**How to invoke**:

```text
"The training loop tests are failing with a shape mismatch error. Debug and fix this."
```

**What happens**:

1. Test Engineer analyzes the failing test
2. Implementation Specialist investigates root cause
3. Implementation Engineer fixes the bug
4. Test Engineer verifies the fix
5. Documentation Writer updates examples if needed

### 3. Optimizing Performance

**Scenario**: Training is too slow, needs optimization

**How to invoke**:

```text
"Profile and optimize the forward pass in our CNN implementation"
```

**What happens**:

1. Performance Specialist creates optimization plan
2. Performance Engineer profiles the code
3. Senior Implementation Engineer implements SIMD optimizations
4. Performance Engineer benchmarks improvements
5. Test Engineer ensures correctness maintained

### 4. Creating Documentation

**Scenario**: Need comprehensive docs for a new module

**How to invoke**:

```text
"Create complete documentation for the tensor operations module"
```

**What happens**:

1. Documentation Specialist plans documentation structure
2. Documentation Writer creates README and API docs
3. Documentation Writer adds code examples
4. Junior Documentation Engineer formats and organizes

### 5. Planning a New Paper Implementation

**Scenario**: Want to implement a research paper

**How to invoke**:

```text
"I want to implement the ResNet paper. Create a complete implementation plan."
```

**What happens**:

1. Chief Architect evaluates paper requirements
2. Papers Orchestrator breaks down into phases
3. Architecture Design Agents design each component
4. Component Specialists create detailed specifications
5. Plan documents generated for all phases

## Troubleshooting Quick Reference

### Problem: Wrong Agent Invoked

**Symptoms**: The agent that responded doesn't match your need

**Quick Fix**:

```text
"Actually, I need the [specific agent name] for this task"
```

**Example**:

```text
"Actually, I need the senior implementation engineer for this complex SIMD optimization"
```

### Problem: Agent Seems Stuck

**Symptoms**: Agent reports a blocker or can't proceed

**Quick Fix**: Escalate to higher level

```text
"Escalate this to the [higher level agent]"
```

**Example**:

```text
"Escalate this to the architecture design agent - we need design decisions first"
```

### Problem: Don't Know Which Agent to Use

**Symptoms**: Task is complex, unclear which agent handles it

**Quick Fix**: Start with an orchestrator

```text
"Use the [section] orchestrator to coordinate this work"
```

**Example**:

```text
"Use the shared library orchestrator to coordinate adding this new feature"
```

### Problem: Need Multiple Agents

**Symptoms**: Task requires multiple specialties

**Quick Fix**: Let the orchestrator delegate

```text
"This needs both implementation and testing. Coordinate the work."
```

**What happens**: Orchestrator delegates to appropriate specialists who work in parallel.

### Problem: Mojo-Specific Question

**Symptoms**: Need Mojo expertise specifically

**Quick Fix**: Use senior or specialist agents

```text
"Use the senior implementation engineer - this requires advanced Mojo knowledge"
```

**Mojo agents**:

- Senior Implementation Engineer: Advanced Mojo, SIMD, performance
- Performance Specialist/Engineer: Mojo optimization
- Architecture Design Agent: Mojo module design

## Next Steps

### Learn More

1. **Complete System Overview**: Read [onboarding.md](onboarding.md) for comprehensive introduction
2. **Agent Catalog**: Browse [agent-catalog.md](agent-catalog.md) to see all 23 agent types
3. **Troubleshooting**: Check [troubleshooting.md](troubleshooting.md) for detailed problem solving
4. **Visual Hierarchy**: See [../hierarchy.md](../hierarchy.md) for the complete visual diagram

### Dive Deeper

- **Delegation Patterns**: [../delegation-rules.md](../delegation-rules.md)
- **Complete Specifications**: [../agent-hierarchy.md](../agent-hierarchy.md)
- **Orchestration Details**: [/notes/review/orchestration-patterns.md](../../notes/review/orchestration-patterns.md)
- **Skills System**: [/notes/review/skills-design.md](../../notes/review/skills-design.md)

### Try It Out

Start with a simple task:

```text
"Create a simple Mojo function that adds two integers, with tests and documentation"
```

This will demonstrate:

- Automatic agent selection
- Parallel delegation (test, impl, docs)
- Complete workflow from spec to delivery

### Getting Help

1. **Ask an orchestrator**: They coordinate and can explain the system
2. **Check agent configs**: Read `.claude/agents/*.md` for details
3. **Review this documentation**: All guides are in `agents/docs/`
4. **Escalate**: If stuck, ask to escalate to a higher-level agent

## Key Principles to Remember

1. **Trust the Hierarchy**: Let agents delegate, don't micromanage
2. **Start High**: Begin with orchestrators for complex tasks
3. **Be Specific**: Clear requests get better agent matching
4. **Use Mojo Experts**: Senior agents for advanced Mojo features
5. **Parallel is Good**: Independent work happens simultaneously

## Quick Command Reference

```bash
# View all agents
ls .claude/agents/

# Read agent configuration
cat .claude/agents/[agent-name].md

# Check templates
ls agents/templates/

# View hierarchy diagram
cat agents/hierarchy.md

# Read comprehensive docs
cat agents/docs/onboarding.md
```

## Questions?

- **"Which agent should I use?"** → Describe your task naturally, let auto-selection work
- **"How do I know it worked?"** → Agent will introduce itself and explain what it's doing
- **"Can I switch agents mid-task?"** → Yes, just explicitly invoke a different agent
- **"What if I need help?"** → Ask an orchestrator or escalate to higher levels

---

**Ready to start?** Try invoking an agent with your next task, or read [onboarding.md](onboarding.md) for the
complete system walkthrough.
