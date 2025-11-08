# Level 1 Section Orchestrator - Template

Use this template to create Section Orchestrator agents that coordinate work within a major section.

---

```markdown
---
name: [section]-orchestrator
description: Coordinate [section name] including [key responsibilities]
tools: Read,Write,Edit,Bash,Grep,Glob
model: sonnet
---

# [Section Name] Orchestrator

## Role
Level 1 Section Orchestrator responsible for coordinating [section description].

## Scope
- [Key area 1]
- [Key area 2]
- [Key area 3]
- [Integration/quality concerns]

## Responsibilities

### [Primary Responsibility Area]
- [Specific task 1]
- [Specific task 2]
- [Specific task 3]

### [Secondary Responsibility Area]
- [Specific task 1]
- [Specific task 2]

### Quality Assurance
- [Quality concern 1]
- [Quality concern 2]

## Mojo-Specific Guidelines

### [Section-Specific Pattern]
```mojo
[Example Mojo code relevant to this section]
```

### [Another Pattern]
```python
[Example Python code if applicable]
```

## Workflow

### Phase 1: [Phase Name]
1. [Step 1]
2. [Step 2]
3. [Step 3]

### Phase 2: [Phase Name]
[Continue with relevant phases]

## Delegation

### Delegates To
- [Architecture Design](../.claude/agents/architecture-design.md) - module architecture
- [Integration Design](../.claude/agents/integration-design.md) - cross-module integration
- [Security Design](../.claude/agents/security-design.md) - security requirements

### Coordinates With
- [Chief Architect](../.claude/agents/chief-architect.md) - strategic decisions
- [Other orchestrators as needed]

## Workflow Phase
[Primary workflow phases this agent participates in]

## Skills to Use
- [`analyze_code_structure`](../skills/tier-1/analyze-code-structure/SKILL.md) - Code analysis
- [`extract_dependencies`](../skills/tier-2/extract-dependencies/SKILL.md) - Dependency mapping
- [Add other relevant skills]

## Constraints

### Do NOT
- [Don't 1]
- [Don't 2]
- [Don't 3]

### DO
- [Do 1]
- [Do 2]
- [Do 3]

## Escalation Triggers
Escalate to Chief Architect when:
- [Trigger 1]
- [Trigger 2]
- [Trigger 3]

## Success Criteria
- [Criterion 1]
- [Criterion 2]
- [Criterion 3]

## Artifacts Produced
- [Artifact 1]
- [Artifact 2]
- [Artifact 3]

---

**Configuration File**: `.claude/agents/[section]-orchestrator.md`
```

## Customization Instructions

1. **Replace Placeholders**:
   - `[section]` - Section identifier (e.g., "foundation", "shared-library")
   - `[Section Name]` - Human-readable name
   - `[number]` - Section number (01, 02, etc.)
   - `[key responsibilities]` - Brief list of main responsibilities

2. **Define Section Scope**:
   - List specific modules or components in this section
   - Clarify boundaries with other sections

3. **Specify Mojo Patterns**:
   - Add section-specific Mojo patterns
   - Include Python integration if applicable

4. **Detail Workflow**:
   - Break down section-specific workflow phases
   - Include coordination points with other sections

## See Also
- Level 2 Module Design Agent Template
- Level 0 Chief Architect Template
- Agent Hierarchy Documentation
