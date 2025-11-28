# Documentation Rules

Shared documentation rules for all agents. Reference this file instead of duplicating.

## Output Location

**All outputs must go to `/notes/issues/{issue-number}/README.md`**

## Before Starting Work

1. **Verify GitHub issue number** is provided
2. **Check if `/notes/issues/{issue-number}/` exists**
3. **If directory doesn't exist**: Create it with README.md
4. **If no issue number provided**: STOP and escalate - request issue creation first

## What Goes Where

| Content | Location |
|---------|----------|
| Issue-specific work | `/notes/issues/{N}/README.md` |
| Architectural decisions | `/notes/review/` |
| Team guides | `/agents/` |
| Comprehensive specs | `/notes/review/` |

## Rules

- Write ALL findings, decisions, and outputs to issue directory
- Link to comprehensive docs in `/notes/review/` and `/agents/` (don't duplicate)
- Keep issue-specific content focused and concise
- Do NOT write documentation outside issue directory
- Do NOT duplicate comprehensive documentation from other locations

See [CLAUDE.md](../../CLAUDE.md#documentation-rules) for complete documentation organization.
