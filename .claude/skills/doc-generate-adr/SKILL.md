---
name: doc-generate-adr
description: Generate Architecture Decision Records (ADRs) to document significant architectural and design decisions. Use when making important technical decisions that need documentation.
---

# Generate ADR Skill

This skill generates Architecture Decision Records (ADRs) for technical decisions.

## When to Use

- User asks to create ADR (e.g., "document the language selection decision")
- Making significant architectural decisions
- Choosing between technical alternatives
- Documenting design trade-offs
- Recording rationale for future reference

## What is an ADR?

An ADR documents:

- **Context** - The issue or decision to be made
- **Decision** - What was decided
- **Rationale** - Why this decision was made
- **Consequences** - Impact of the decision
- **Alternatives** - Options considered

## ADR Format

```markdown
# ADR-XXX: Title

**Status**: Accepted | Proposed | Deprecated | Superseded

**Date**: YYYY-MM-DD

**Deciders**: Names or roles

## Context

What is the issue we're facing? What factors are we considering?

## Decision

What is the decision we're making?

## Rationale

Why are we making this decision? What are the key reasons?

## Consequences

### Positive

- Benefit 1
- Benefit 2

### Negative

- Drawback 1
- Drawback 2

### Neutral

- Other impact 1

## Alternatives Considered

### Alternative 1

Description and why not chosen.

### Alternative 2

Description and why not chosen.

## References

- Links to related docs
- Evidence or research
```

## Usage

### Generate ADR

```bash
# Create new ADR
./scripts/create_adr.sh "Language Selection for Tooling"

# This creates:
# notes/review/adr/ADR-XXX-language-selection-tooling.md
```

### Number Assignment

ADRs are numbered sequentially:

- ADR-001 - First decision
- ADR-002 - Second decision
- ADR-XXX - Next available number

```bash
# Script automatically finds next number
./scripts/create_adr.sh "Decision Title"
# Creates: ADR-003-decision-title.md
```

## ADR Examples

### Example 1: Language Selection

```markdown
# ADR-001: Mojo First with Python for Automation

**Status**: Accepted

**Date**: 2024-11-15

## Context

Need to choose implementation languages for ML implementations and
automation tooling.

## Decision

Use Mojo for ML/AI implementations, Python for automation when
technical limitations require it.

## Rationale

- Mojo: Performance, type safety, SIMD for ML
- Python: Subprocess handling, regex, mature libraries for tooling

## Consequences

### Positive
- Best performance for ML workloads
- Pragmatic automation with Python

### Negative
- Need to maintain two languages
- Team must know both languages

## Alternatives Considered

### Python for Everything
Rejected: Poor performance for ML workloads

### Mojo for Everything
Rejected: Current limitations in subprocess/regex
```

### Example 2: Testing Strategy

```markdown
# ADR-002: Test-Driven Development

**Status**: Accepted

## Decision

Adopt TDD (write tests before implementation) for all new code.

## Rationale

- Catches bugs early
- Ensures testable design
- Serves as documentation
- Faster debugging

## Consequences

### Positive
- Higher code quality
- Better test coverage
- Clearer requirements

### Negative
- Slower initial development
- Learning curve for team
```

## ADR Workflow

### 1. Identify Decision

```bash
# Questions that need ADRs:
# - What language/framework to use?
# - How to structure the code?
# - What architecture pattern?
# - How to handle errors?
# - What testing strategy?
```

### 2. Research Alternatives

```bash
# Gather evidence:
# - Performance benchmarks
# - Community feedback
# - Documentation quality
# - Team experience
```

### 3. Create ADR

```bash
./scripts/create_adr.sh "Decision Title"
# Edit the generated template
```

### 4. Review and Approve

```bash
# Get team review
# Update status to "Accepted"
# Commit to repository
```

## ADR Status Lifecycle

- **Proposed** - Under consideration
- **Accepted** - Decision made and active
- **Deprecated** - No longer recommended
- **Superseded** - Replaced by newer ADR

## Storage Location

ADRs are stored in:

```text
notes/review/adr/
├── ADR-001-language-selection.md
├── ADR-002-testing-strategy.md
├── ADR-003-package-structure.md
└── README.md (index of all ADRs)
```

## Error Handling

- **Missing context**: Add more background information
- **Unclear decision**: Make decision more specific
- **Missing alternatives**: Document at least 2 alternatives
- **No consequences**: Think through positive and negative impacts

## Examples

**Create ADR:**

```bash
./scripts/create_adr.sh "Mojo Memory Management Strategy"
```

**List all ADRs:**

```bash
./scripts/list_adrs.sh
```

**Update ADR status:**

```bash
./scripts/update_adr_status.sh ADR-001 "Superseded"
```

## Scripts Available

- `scripts/create_adr.sh` - Create new ADR
- `scripts/list_adrs.sh` - List all ADRs
- `scripts/update_adr_status.sh` - Update ADR status
- `scripts/find_adr.sh` - Search ADRs

## Templates

- `templates/adr_template.md` - Standard ADR template

## Best Practices

1. **Keep focused** - One decision per ADR
2. **Be specific** - Clear, actionable decisions
3. **Document alternatives** - Show what was considered
4. **Update status** - Keep status current
5. **Link evidence** - Include benchmarks, research
6. **Review regularly** - Revisit as technology evolves

See existing ADRs in `/notes/review/adr/` for examples.
