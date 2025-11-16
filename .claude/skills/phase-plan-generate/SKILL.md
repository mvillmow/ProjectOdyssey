---
name: phase-plan-generate
description: Generate comprehensive plan documentation following the 9-section Template 1 format for ML Odyssey planning phase. Use when creating new plan.md files or when asked to generate planning documentation.
---

# Plan Generation Skill

This skill generates plan documentation following ML Odyssey's standard 9-section format (Template 1).

## When to Use

- User asks to create a plan (e.g., "generate a plan for X")
- Starting a new component or subsection
- Creating planning phase documentation
- Need to follow Template 1 format

## Plan Format (Template 1)

All plan.md files must include these 9 sections:

1. **Title** - Component name
2. **Overview** - Brief description (2-3 sentences)
3. **Parent Plan** - Link to parent or "None (top-level)"
4. **Child Plans** - Links to child plans or "None (leaf node)"
5. **Inputs** - Prerequisites and dependencies
6. **Outputs** - Deliverables and artifacts
7. **Steps** - Ordered implementation steps
8. **Success Criteria** - Measurable completion criteria
9. **Notes** - Additional context and considerations

## Usage

### Generate Plan

```bash
# Generate plan from template
./scripts/generate_plan.sh <component-name> <parent-path>

# Example: Create plan for new subsection
./scripts/generate_plan.sh "authentication-module" "notes/plan/02-shared-library"
```

### Key Principles

- **Use relative paths** for links (e.g., `../plan.md`)
- **Maintain hierarchy** (parent/child relationships)
- **Be specific** in Inputs/Outputs
- **Measurable** Success Criteria
- **Actionable** Steps

## Plan Structure

```markdown
# Component Name

## Overview

Brief description of the component (2-3 sentences).

## Parent Plan

[../plan.md](../plan.md)

## Child Plans

- [child1/plan.md](child1/plan.md)
- [child2/plan.md](child2/plan.md)

Or: "None (leaf node)" for level 4

## Inputs

- Prerequisite 1
- Prerequisite 2

## Outputs

- Deliverable 1
- Deliverable 2

## Steps

1. Step 1
2. Step 2
3. Step 3

## Success Criteria

- [ ] Criterion 1
- [ ] Criterion 2

## Notes

Additional context, considerations, or references.
```

## After Creating Plan

1. **Update parent plan's** "Child Plans" section
2. **Regenerate GitHub issues**: `python3 scripts/regenerate_github_issues.py --section <section>`
3. **Verify links** work correctly (relative paths)
4. **Test issue creation** (dry-run first)

## Error Handling

- **Missing sections**: All 9 sections required
- **Invalid links**: Use relative paths only
- **Hierarchy broken**: Verify parent/child relationships
- **Success criteria not measurable**: Add specific checkboxes

## Examples

**Generate new component plan:**
```bash
./scripts/generate_plan.sh "tensor-operations" "notes/plan/02-shared-library/01-core"
```

**Validate existing plan:**
```bash
./scripts/validate_plan.sh "notes/plan/02-shared-library/01-core/plan.md"
```

## Scripts Available

- `scripts/generate_plan.sh` - Generate plan from template
- `scripts/validate_plan.sh` - Validate plan format
- `scripts/update_parent_links.sh` - Update parent plan child links

## Templates

- `templates/plan_template.md` - Complete Template 1 format
- `templates/subsection_plan.md` - Subsection-level template
- `templates/component_plan.md` - Component-level template

See CLAUDE.md section "Plan File Format (Template 1)" for complete specification.
