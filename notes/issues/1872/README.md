# Issue #1872: Create Plan File Tracking Strategy Documentation

## Overview

Document the relationship and workflow between `plan.md` and `github_issue.md` files to clarify their roles and prevent confusion.

## Problem

Developers may be unclear about:

- Which file is the source of truth
- When to edit which file
- How the files are synchronized
- The workflow for updating plans

## Proposed Content

Create `notes/review/plan-file-tracking-strategy.md` with:

### Sections

1. **Overview** - Purpose of dual-file system
1. **File Roles**
   - `plan.md`: Source of truth, comprehensive planning
   - `github_issue.md`: Generated derivative, GitHub-formatted
1. **Workflow**
   - Edit plan.md only
   - Regenerate github_issue.md using scripts
   - Create/update GitHub issues from github_issue.md
1. **Best Practices**
   - Never edit github_issue.md manually
   - Always regenerate after plan.md changes
   - Use scripts/regenerate_github_issues.py
1. **Examples** - Real workflow scenarios

## Implementation

### Created Documentation

**File**: `notes/review/plan-file-tracking-strategy.md` (500+ lines)

### Content Structure

1. **Overview** - Purpose of dual-file system
1. **File Roles**
   - plan.md: Source of truth, Template 1 format
   - github_issue.md: Generated derivative, never manually edited

1. **Workflow** - Three complete workflows
   - Creating new component plan
   - Updating existing plan
   - Regenerating all issue files

1. **Best Practices** - DO/DON'T lists
   - 6 best practices to follow
   - 6 anti-patterns to avoid

1. **File Relationship Diagram** - Visual flow
1. **Synchronization Rules** - When/how to sync files
1. **Examples** - Three detailed scenarios
   - Adding new step
   - Changing success criteria
   - Creating issues for new section

1. **Troubleshooting** - Common issues and solutions
1. **Script Reference** - Complete usage guide
1. **Integration** - Development workflow integration
1. **Version Control** - Commit guidelines

### Key Points Documented

- ✅ plan.md is source of truth (manually edited)
- ✅ github_issue.md is generated (never manually edited)
- ✅ Always regenerate after editing plan.md
- ✅ Commit both files together
- ✅ Use regenerate_github_issues.py script
- ✅ Template 1 format (9 sections) required

### Usage Examples Included

```bash
# Create new component
mkdir -p notes/plan/02-shared-library/03-new-component
# Edit plan.md
python3 scripts/regenerate_github_issues.py --path <component-path>
python3 scripts/create_issues.py --file <github_issue.md>

# Update existing plan
# Edit plan.md
python3 scripts/regenerate_github_issues.py --path <component-path>
# Update GitHub issues if needed
```

### Documentation Quality

- Comprehensive (500+ lines)
- Practical examples (3 complete workflows)
- Clear diagrams (visual file relationship)
- Troubleshooting guide (4 common issues)
- Script reference (complete usage)
- Best practices (12 guidelines)

## Status

**COMPLETED** ✅ - Comprehensive documentation created

## Related Issues

Part of Wave 4 architecture improvements from continuous improvement session.
