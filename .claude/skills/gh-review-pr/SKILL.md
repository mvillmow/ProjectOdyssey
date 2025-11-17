---
name: gh-review-pr
description: Comprehensively review a pull request including code changes, CI status, test coverage, and adherence to project standards. Use when asked to review a PR or when evaluating pull requests.
---

# GitHub PR Review Skill

This skill provides comprehensive pull request review capabilities following ML Odyssey project standards.

## When to Use

- User asks to review a PR (e.g., "review PR #123")
- Evaluating pull request quality
- Checking adherence to coding standards
- Validating CI status and test coverage

## Review Process

### 1. Get PR Information

```bash
# Get PR details
gh pr view <pr-number>

# Get PR diff
gh pr diff <pr-number>

# Check PR status
gh pr checks <pr-number>
```

### 2. Review Checklist

Use the review checklist from reference.md to ensure thorough evaluation:

- [ ] Code changes follow project standards
- [ ] CI passes all checks
- [ ] Tests are present and passing
- [ ] PR is linked to an issue
- [ ] Commit messages follow conventional commits
- [ ] No security vulnerabilities introduced
- [ ] Documentation updated if needed
- [ ] No unintended files included

### 3. Analyze Code Changes

Focus on:

- **Quality**: Clean, readable, maintainable code
- **Standards**: Follows CLAUDE.md guidelines
- **Security**: No vulnerabilities or unsafe patterns
- **Tests**: Adequate coverage for changes
- **Documentation**: Updated as needed

### 4. Generate Review

Use the review comment template from templates/review_comment.md to provide structured feedback.

## Error Handling

- If PR number is invalid: Report error and ask for correct number
- If gh CLI fails: Check authentication with `gh auth status`
- If CI is pending: Note that review is based on current state

## Output Format

Provide review as markdown with sections:

1. **Summary** - Overall assessment
2. **Strengths** - What's done well
3. **Issues** - Problems that must be fixed
4. **Suggestions** - Optional improvements
5. **Verdict** - Approve / Request Changes / Comment

## Examples

**Good usage:**

- "Review PR #42"
- "Check if PR #15 is ready to merge"
- "Evaluate the quality of pull request 7"

**Scripts Available:**

- `scripts/check_pr_status.sh` - Check CI and review status
- `scripts/analyze_changes.sh` - Analyze code changes
- `scripts/validate_tests.sh` - Validate test coverage

See reference.md for detailed review standards and criteria.
