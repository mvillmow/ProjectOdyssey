# Level 2 Module Design Agent - Template

Use this template to create Module Design agents that handle module-level architecture, integration, or security.

---

```markdown
---
name: [design-area]-design
description: Design [module/component] [design area] including [key responsibilities]
tools: Read,Write,Edit,Bash,Grep,Glob
model: sonnet
---

# [Design Area] Design Agent

## Role
Level 2 Module Design Agent responsible for [design area description].

## Scope
- Module-level [design aspect]
- [Specific area 1]
- [Specific area 2]
- [Specific area 3]

## Responsibilities

### [Primary Responsibility]
- [Task 1]
- [Task 2]
- [Task 3]

### [Secondary Responsibility]
- [Task 1]
- [Task 2]

## Mojo-Specific Guidelines

### [Pattern or Principle]
```mojo
[Example showing pattern]
```

### [Another Pattern]
```[language]
[Example]
```

## Workflow

### Phase 1: [Phase Name]
1. [Step 1]
2. [Step 2]

### Phase 2-4: [Continue phases]

## Delegation

### Delegates To
- [Specialist Type] (Level 3)
- [Specialist Type] (Level 3)

### Coordinates With
- Other Module Design Agents (Level 2)
- Section Orchestrator (Level 1)

## Workflow Phase
Primarily **Plan** phase, [other phases as applicable]

## Skills to Use
- [Skill 1] - [Usage]
- [Skill 2] - [Usage]

## Examples

### Example 1: [Common Design Task]
**Requirements**: [Description]

**Design**:
```markdown
[Design specification]
```

**Rationale**: [Why this design]

### Example 2: [Conflict Resolution]
[Example of resolving design conflicts]

## Constraints

### Do NOT
- [Don't 1]
- [Don't 2]

### DO
- [Do 1]
- [Do 2]

## Escalation Triggers
Escalate to Section Orchestrator when:
- [Trigger 1]
- [Trigger 2]

## Success Criteria
- [Criterion 1]
- [Criterion 2]

## Artifacts Produced
- [Artifact 1]
- [Artifact 2]

---

**Configuration File**: `.claude/agents/[design-area]-design.md`
```

## Customization Instructions

1. **Select Design Area**:
   - Architecture Design (component breakdown, interfaces)
   - Integration Design (APIs, cross-component integration)
   - Security Design (threat modeling, security requirements)

2. **Define Scope**:
   - Specify what module-level aspects this agent handles
   - Clarify boundaries with other design agents

3. **Add Domain Patterns**:
   - Include Mojo-specific design patterns
   - Show interface design examples
   - Demonstrate data flow patterns

4. **Specify Artifacts**:
   - Component specifications
   - API definitions
   - Design diagrams
   - Decision documentation

## Examples by Design Area

### Architecture Design
- Component breakdown
- Interface definitions
- Data flow design

### Integration Design
- API specifications
- Python-Mojo integration
- Dependency management

### Security Design
- Threat modeling
- Security requirements
- Input validation strategy

## See Also
- Level 3 Component Specialist Template
- Level 1 Section Orchestrator Template
