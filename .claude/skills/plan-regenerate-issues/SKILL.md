---
name: plan-regenerate-issues
description: Regenerate github_issue.md files from plan.md after modifying plans. Use after editing plan.md files to update issue descriptions.
---

# Regenerate GitHub Issues Skill

Regenerate github_issue.md files from plan.md.

## When to Use

- After editing plan.md files
- Plan structure changes
- Need to update issue descriptions
- Before creating new issues

## Usage

### Regenerate All Issues

```bash
# Regenerate all github_issue.md files
python3 scripts/regenerate_github_issues.py

# This:
# 1. Finds all plan.md files
# 2. Generates github_issue.md for each
# 3. Preserves issue numbers if already created
```

### Regenerate Section

```bash
# Regenerate specific section
python3 scripts/regenerate_github_issues.py --section 01-foundation

# Only processes plans in that section
```

### Regenerate Single Component

```bash
# Regenerate specific component
./scripts/regenerate_single_issue.sh notes/plan/01-foundation/plan.md
```

## Important Rules

- **NEVER edit github_issue.md manually** - Always regenerate
- **Edit plan.md** - Source of truth
- **Regenerate after changes** - Keep in sync
- **Check diffs** - Review changes before committing

## Workflow

```bash
# 1. Edit plan.md
vim notes/plan/01-foundation/plan.md

# 2. Regenerate issues
python3 scripts/regenerate_github_issues.py --section 01-foundation

# 3. Review changes
git diff notes/plan/01-foundation/github_issue.md

# 4. Commit both files
git add notes/plan/01-foundation/plan.md
git add notes/plan/01-foundation/github_issue.md
git commit -m "docs: update foundation plan"
```

## Script Behavior

The script:

- Extracts title from plan.md
- Formats sections for GitHub
- Preserves issue numbers
- Updates labels
- Maintains hierarchy links

## Validation

After regenerating:

```bash
# Validate issue format
./scripts/validate_issue_format.sh notes/plan/01-foundation/github_issue.md

# Test issue creation (dry-run)
./scripts/create_single_component_issues.py notes/plan/01-foundation/github_issue.md --dry-run
```

## Scripts

- `scripts/regenerate_github_issues.py` - Regenerate all
- `scripts/regenerate_single_issue.sh` - Regenerate one
- `scripts/validate_issue_format.sh` - Validate format

See `scripts/README.md` for complete documentation.
