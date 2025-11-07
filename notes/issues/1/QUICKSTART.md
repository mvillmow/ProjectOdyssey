# Quick Start: Update All GitHub Issues

## TL;DR

Run this one command:

```bash
cd /home/mvillmow/ml-odyssey && python3 simple_update.py
```

Done! All 331 `github_issue.md` files will be updated with detailed issue bodies.

## What This Does

- Updates ALL 331 `github_issue.md` files across the entire plan structure
- Replaces generic "See plan.md" bodies with detailed, specific task lists
- Generates 5 detailed issue types for each component: Plan, Test, Implementation, Packaging, Cleanup
- Extracts component-specific information from plan.md files
- Takes approximately 10-30 seconds to complete

## Example: Before and After

### Before (Old Format)
```markdown
**Plan Issue**:
- Title: [Plan] Component Name - Design and Documentation
- Body: See plan: [plan.md](plan.md)
```

### After (New Format)
```markdown
**Title**: [Plan] Foundation - Directory Structure - Create Papers Dir - Design and Documentation

**Body**:
```
## Overview
Set up the papers directory structure which will contain individual paper implementations...

## Planning Tasks
- [ ] Review parent plan and understand context
- [ ] Research best practices and approaches
- [ ] Define detailed implementation strategy
- [ ] Document design decisions and rationale
- [ ] Identify dependencies and prerequisites
- [ ] Create architectural diagrams if needed
- [ ] Define success criteria and acceptance tests
- [ ] Document edge cases and considerations

## Deliverables
- Detailed implementation plan
- Architecture documentation
- Design decisions documented in plan.md
- Success criteria defined

## Reference
See detailed plan: notes/plan/01-foundation/01-directory-structure/01-create-papers-dir/plan.md
```
```

## Verification

After running, verify it worked:

```bash
# Check one of the updated files
cat notes/plan/03-tooling/github_issue.md

# Verify all files were updated
grep -l "## Planning Tasks" notes/plan/**/github_issue.md | wc -l
# Should output: 331
```

## More Info

- **Full Instructions**: See `UPDATE_INSTRUCTIONS.md`
- **Detailed Summary**: See `UPDATE_SUMMARY.md`
- **Update Script**: `simple_update.py`

## Questions?

The script is fully automated and handles all edge cases. Just run it and you're done!
