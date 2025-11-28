---
name: doc-issue-readme
description: Generate issue-specific documentation in /notes/issues/<number>/README.md. Use when starting work on an issue to create proper documentation structure.
mcp_fallback: none
category: doc
---

# Issue README Generation Skill

Create issue-specific documentation following ML Odyssey standards.

## When to Use

- Starting work on a GitHub issue
- Need structured documentation for issue tracking
- Tracking implementation progress
- Consolidating findings and decisions

## Quick Reference

```bash
mkdir -p notes/issues/42
cp .claude/skills/doc-issue-readme/templates/issue_readme.md notes/issues/42/README.md
# Edit to fill in details
```

## README Format

```markdown
# Issue #XX: [Phase] Component Name

## Objective
What this issue accomplishes (1-2 sentences)

## Deliverables
- Specific files/changes
- Measurable outputs

## Success Criteria
- [ ] Criterion 1
- [ ] Criterion 2

## References
- Links to /agents/ and /notes/review/ documentation
- Related issues

## Implementation Notes
Discoveries during implementation (fill as work progresses)

## Testing Notes
Test results and coverage (if applicable)

## Review Feedback
PR review feedback and resolutions
```

## Workflow

1. **Create directory** - `mkdir -p notes/issues/<number>`
2. **Create README** - Copy template or fetch issue details
3. **Populate sections** - Fill Objective, Deliverables, Success Criteria
4. **Add references** - Link to shared docs (don't duplicate)
5. **Track progress** - Update notes as work continues
6. **Update before PR** - Verify all sections complete

## Documentation Rules

### DO

- Keep issue-specific
- Link to comprehensive docs (don't duplicate)
- Update as work progresses
- Be specific and measurable

### DON'T

- Duplicate comprehensive docs from /notes/review/
- Create shared specifications here
- Leave sections empty long-term
- Forget to update during work

## Common Sections

### Objective

Good: "Implement tensor operations (add, multiply, matmul) with SIMD"
Bad: "Work on tensors"

### Deliverables

Good: Specific file paths and outputs
Bad: "Various improvements"

### Success Criteria

Must be measurable checkboxes - "Tests passing >90% coverage"

## Error Handling

| Issue | Fix |
|-------|-----|
| Missing Objective | Add specific 1-2 sentence description |
| Vague Deliverables | List exact files and outputs |
| Non-measurable criteria | Add percentages, counts, or concrete outcomes |
| Duplicated docs | Link instead, or move to /notes/review/ |

## References

- See `/notes/issues/*/README.md` for examples
- Documentation rules: CLAUDE.md
- Skill: `doc-validate-markdown` for formatting
