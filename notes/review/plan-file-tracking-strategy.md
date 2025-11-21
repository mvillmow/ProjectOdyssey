# Plan File Tracking Strategy

**Status**: Active
**Last Updated**: 2025-11-21
**Purpose**: Define the relationship between `plan.md` and `github_issue.md` files

## Overview

ML Odyssey uses a dual-file system for planning: `plan.md` for comprehensive planning and
`github_issue.md` for GitHub issue creation. This document clarifies their relationship and workflow.

## File Roles

### plan.md - Source of Truth

**Purpose**: Comprehensive planning document for human readers and planning decisions

**Location**: `notes/plan/<section>/<subsection>/<component>/plan.md`

**Format**: Template 1 (9-section format)

**Sections**:

1. Title
1. Overview
1. Parent Plan
1. Child Plans
1. Inputs
1. Outputs
1. Steps
1. Success Criteria
1. Notes

**Characteristics**:

- Human-readable planning format
- Detailed context and rationale
- Links to parent/child plans
- Manually edited by developers
- Source of truth for all planning decisions

### github_issue.md - Generated Derivative

**Purpose**: GitHub-formatted issue descriptions generated from plan.md

**Location**: Same directory as plan.md

**Format**: GitHub-flavored Markdown optimized for issue creation

**Sections** (per phase):

- Title
- Labels
- Description
- Success Criteria
- Dependencies

**Characteristics**:

- Generated automatically from plan.md
- GitHub-optimized formatting
- Split into 5 phase-specific issues
- Never manually edited
- Regenerated whenever plan.md changes

## Workflow

### 1. Creating a New Component Plan

```bash
# Step 1: Create directory structure
mkdir -p notes/plan/02-shared-library/03-new-component

# Step 2: Create plan.md following Template 1 format
# - Fill in all 9 sections
# - Link to parent plan
# - Define inputs/outputs/steps

# Step 3: Generate github_issue.md
python3 scripts/regenerate_github_issues.py --path notes/plan/02-shared-library/03-new-component

# Step 4: Create GitHub issues
python3 scripts/create_issues.py --file notes/plan/02-shared-library/03-new-component/github_issue.md
```

### 2. Updating an Existing Plan

```bash
# Step 1: Edit plan.md
# - Make changes to any section
# - Update steps, success criteria, etc.

# Step 2: Regenerate github_issue.md
python3 scripts/regenerate_github_issues.py --path notes/plan/02-shared-library/03-new-component

# Step 3: Update GitHub issues (if already created)
# - Option A: Close old issues and create new ones
# - Option B: Manually update issues on GitHub (for minor changes)
```

### 3. Regenerating All Issue Files

```bash
# Regenerate for entire section
python3 scripts/regenerate_github_issues.py --section 02-shared-library

# Regenerate for entire repository
python3 scripts/regenerate_github_issues.py
```

## Best Practices

### DO

- ✅ **Edit plan.md only** - All changes start with plan.md
- ✅ **Regenerate after changes** - Always regenerate github_issue.md after editing plan.md
- ✅ **Use scripts** - Use regenerate_github_issues.py for consistency
- ✅ **Follow Template 1** - Maintain the 9-section format
- ✅ **Link hierarchically** - Parent/child plan links must be correct
- ✅ **Commit both files** - Commit plan.md and regenerated github_issue.md together

### DON'T

- ❌ **Never edit github_issue.md manually** - It will be overwritten
- ❌ **Don't skip regeneration** - Out-of-sync files cause confusion
- ❌ **Don't create issues directly** - Always regenerate github_issue.md first
- ❌ **Don't duplicate content** - plan.md is the single source of truth
- ❌ **Don't skip sections** - All 9 Template 1 sections are required

## File Relationship Diagram

```text
plan.md (Source of Truth)
    ↓
    | Edited by developers
    | Template 1 format (9 sections)
    | Comprehensive planning
    ↓
regenerate_github_issues.py
    ↓
    | Reads plan.md
    | Generates 5 phase-specific issues
    | GitHub-optimized formatting
    ↓
github_issue.md (Generated)
    ↓
    | Plan Issue
    | Test Issue
    | Implementation Issue
    | Packaging Issue
    | Cleanup Issue
    ↓
create_issues.py
    ↓
    | Creates GitHub issues via gh CLI
    | One issue per phase
    ↓
GitHub Issues
```

## Synchronization Rules

### When plan.md Changes

1. **Immediately regenerate** github_issue.md
1. **Review changes** in github_issue.md
1. **Commit together** - plan.md and github_issue.md in same commit
1. **Update issues** if already created on GitHub

### When github_issue.md Changes

**Never manually change github_issue.md!** If you see incorrect content:

1. Edit plan.md to fix the source
1. Regenerate github_issue.md
1. Commit both files

### Handling Conflicts

If plan.md and github_issue.md are out of sync:

1. **Trust plan.md** - It's the source of truth
1. **Regenerate** github_issue.md from plan.md
1. **Discard** manual changes to github_issue.md
1. **Update** GitHub issues if needed

## Examples

### Example 1: Adding a New Step

**Scenario**: Need to add a new step to component implementation

**Workflow**:

```bash
# 1. Edit plan.md
vim notes/plan/02-shared-library/03-component/plan.md
# Add new step to ## Steps section

# 2. Regenerate github_issue.md
python3 scripts/regenerate_github_issues.py --path notes/plan/02-shared-library/03-component

# 3. Commit changes
git add notes/plan/02-shared-library/03-component/
git commit -m "feat(plan): add new step to component implementation"

# 4. Update GitHub issues (if already created)
# Manually update the issue description on GitHub, or close and recreate
```

### Example 2: Changing Success Criteria

**Scenario**: Success criteria need refinement

**Workflow**:

```bash
# 1. Edit plan.md
vim notes/plan/02-shared-library/03-component/plan.md
# Update ## Success Criteria section

# 2. Regenerate github_issue.md
python3 scripts/regenerate_github_issues.py --path notes/plan/02-shared-library/03-component

# 3. Review changes
git diff notes/plan/02-shared-library/03-component/github_issue.md

# 4. Commit
git add notes/plan/02-shared-library/03-component/
git commit -m "docs(plan): refine success criteria for component"
```

### Example 3: Creating Issues for New Section

**Scenario**: Created plans for entire new section, ready to create issues

**Workflow**:

```bash
# 1. Regenerate all github_issue.md files in section
python3 scripts/regenerate_github_issues.py --section 03-new-section

# 2. Verify regeneration succeeded
git diff notes/plan/03-new-section/

# 3. Create all GitHub issues
python3 scripts/create_issues.py --section 03-new-section

# 4. Commit state files
git add logs/.issue_creation_state_*.json
git commit -m "chore: create issues for 03-new-section"
```

## Troubleshooting

### Issue: github_issue.md missing

**Solution**: Regenerate from plan.md

```bash
python3 scripts/regenerate_github_issues.py --path <component-path>
```

### Issue: github_issue.md outdated

**Solution**: Regenerate from plan.md

```bash
python3 scripts/regenerate_github_issues.py --path <component-path>
```

### Issue: Content doesn't match

**Problem**: github_issue.md shows different content than expected from plan.md

**Solution**:

1. Check plan.md is correct
1. Regenerate github_issue.md
1. If still incorrect, check regeneration script

### Issue: Created issues wrong

**Problem**: GitHub issues created with incorrect content

**Solution**:

1. Close incorrect issues
1. Fix plan.md
1. Regenerate github_issue.md
1. Create new issues

## Script Reference

### regenerate_github_issues.py

**Purpose**: Generate github_issue.md files from plan.md

**Usage**:

```bash
# Regenerate single component
python3 scripts/regenerate_github_issues.py --path notes/plan/02-shared-library/03-component

# Regenerate entire section
python3 scripts/regenerate_github_issues.py --section 02-shared-library

# Regenerate everything
python3 scripts/regenerate_github_issues.py
```

### create_issues.py

**Purpose**: Create GitHub issues from github_issue.md files

**Usage**:

```bash
# Dry run (preview)
python3 scripts/create_issues.py --dry-run

# Create from single file
python3 scripts/create_issues.py --file notes/plan/.../github_issue.md

# Create for entire section
python3 scripts/create_issues.py --section 02-shared-library

# Create all issues
python3 scripts/create_issues.py
```

## Integration with Development Workflow

### Planning Phase

1. Create plan.md (Template 1 format)
1. Generate github_issue.md
1. Review generated issues
1. Create GitHub issues

### Implementation Phase

1. Reference GitHub issue for task details
1. If changes needed, edit plan.md
1. Regenerate github_issue.md
1. Update GitHub issue

### Review Phase

1. Review against GitHub issue success criteria
1. If criteria change during review, update plan.md
1. Regenerate and update GitHub issue

## Version Control

### Commit Guidelines

**DO commit together**:

- plan.md and github_issue.md (when regenerated)
- All files in component directory

**Commit message format**:

```text
feat(plan): add new component for X

- Created plan.md with 9 sections
- Generated github_issue.md (5 phase issues)
- Linked to parent plan in 02-shared-library
```

### Git Workflow

```bash
# After editing plan.md and regenerating
git add notes/plan/<section>/<subsection>/<component>/
git commit -m "docs(plan): update component X with Y changes"
```

## Summary

- **plan.md**: Source of truth, manually edited
- **github_issue.md**: Generated derivative, never manually edited
- **Always regenerate** github_issue.md after editing plan.md
- **Commit together** to keep files synchronized
- **Use scripts** for consistency and correctness

---

**Related Documentation**:

- Template 1 format specification
- Issue creation guide
- 5-phase development workflow
- Hierarchical planning structure
