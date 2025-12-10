# Review Specialist Common Sections

This template provides common sections for all review specialists to eliminate duplication.

## Output Location (Include Verbatim in All Specialists)

**CRITICAL**: All review feedback MUST be posted directly to the GitHub pull request using
`gh pr review` (GitHub CLI only). **NEVER** write reviews to local files or `notes/review/`.

## Feedback Format (Standard Template)

```markdown
[EMOJI] [SEVERITY]: [Issue summary] - Fix all N occurrences

Locations:
- file.mojo:42: [brief description]

Fix: [2-3 line solution]

See: [link to docs]
```

Severity: 🔴 CRITICAL (must fix), 🟠 MAJOR (should fix), 🟡 MINOR (nice to have), 🔵 INFO (informational)

## Example Review Structure

**Issue**: [Problem statement]

**Feedback**:
🟠 MAJOR: [Summary of issue]

**Solution**: [Code fix or remediation steps]

```mojo
# Example corrected code
fn example() -> Int:
    return 42
```

## Coordination Pattern

**Coordinates With**:

- [Code Review Orchestrator](../code-review-orchestrator.md) - Receives review assignments
- [Related Specialist] - Cross-functional collaboration on shared concerns

**Escalates To**:

- [Code Review Orchestrator](../code-review-orchestrator.md) - Issues outside specialist scope

## GitHub CLI Usage

Use `gh pr review` for all review feedback:

```bash
# Post review feedback
gh pr review <pr-number> --comment --body "$(cat <<'EOF'
## [Specialist Name] Review

[Feedback content]
EOF
)"

# Approve PR
gh pr review <pr-number> --approve --body "✅ [Specialist] review passed"

# Request changes
gh pr review <pr-number> --request-changes --body "$(cat <<'EOF'
## [Specialist] Review - Changes Requested

[Detailed feedback]
EOF
)"
```

---

*This template eliminates ~20 lines of duplication per specialist × 14 specialists = 280 lines saved*
