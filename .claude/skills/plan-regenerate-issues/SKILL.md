---
name: plan-regenerate-issues
description: Regenerate github_issue.md files from plan.md after modifying plans. Use after editing plan.md files to update issue descriptions.
category: plan
---

# Regenerate GitHub Issues Skill

Regenerate issue descriptions from updated plans.

## When to Use

- After editing plan.md files
- Plan structure changes
- Need to update issue descriptions
- Before creating new issues

## Quick Reference

```bash
# Regenerate all issues
python3 scripts/regenerate_github_issues.py

# Regenerate specific section
python3 scripts/regenerate_github_issues.py --section 01-foundation

# Regenerate single component
./scripts/regenerate_single_issue.sh notes/plan/01-foundation/plan.md
```

## Workflow

```bash
# 1. Edit plan.md
vim notes/plan/01-foundation/plan.md

# 2. Regenerate github_issue.md
python3 scripts/regenerate_github_issues.py --section 01-foundation

# 3. Review changes
git diff notes/plan/01-foundation/github_issue.md

# 4. Commit both files
git add notes/plan/01-foundation/plan.md
git add notes/plan/01-foundation/github_issue.md
git commit -m "docs: update foundation plan"
```

## Important Rules

**Never edit github_issue.md manually**

- Source of truth is plan.md
- Always regenerate after plan changes
- Regeneration preserves issue numbers

## Script Behavior

The script:

- Extracts title from plan.md
- Formats sections for GitHub
- Preserves issue numbers if already created
- Updates labels based on phase
- Maintains hierarchy links
- Validates output format

## Regeneration Options

### All Issues

```bash
python3 scripts/regenerate_github_issues.py
# Processes all plans in notes/plan/
# Updates all github_issue.md files
```

### Section Only

```bash
python3 scripts/regenerate_github_issues.py --section 02-shared-library
# Only processes plans in that section
```

### Single Component

```bash
./scripts/regenerate_single_issue.sh notes/plan/01-foundation/02-structure/plan.md
# Regenerates single github_issue.md
```

## Validation After Regeneration

```bash
# Validate format
./scripts/validate_issue_format.sh notes/plan/01-foundation/github_issue.md

# Test issue creation (dry-run)
./scripts/create_single_component_issues.py notes/plan/01-foundation/github_issue.md --dry-run

# Check for errors
./scripts/check_issue_format.sh
```

## Error Handling

| Error | Fix |
|-------|-----|
| "Missing plan.md" | Create plan.md in component directory |
| "Invalid title" | Ensure title is `# Component Name` |
| "Broken links" | Fix relative paths in plan.md |
| "Bad format" | Verify all 9 sections present |

## Scripts Available

- `scripts/regenerate_github_issues.py` - Regenerate all issues
- `scripts/regenerate_single_issue.sh` - Regenerate one issue
- `scripts/validate_issue_format.sh` - Validate format
- `scripts/check_issue_format.sh` - Check all issues

## References

- Plan format: CLAUDE.md
- Related skill: `plan-validate-structure` for validation
- Related skill: `plan-create-component` for creating plans
- Script docs: `scripts/README.md`
