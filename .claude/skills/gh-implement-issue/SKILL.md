---
name: gh-implement-issue
description: "End-to-end implementation workflow for a GitHub issue from planning through PR creation. Use when starting work on an issue from scratch."
category: github
mcp_fallback: github
---

# Implement GitHub Issue

Complete workflow for implementing a GitHub issue from start to finish.

## When to Use

- Starting work on a new issue
- Need structured workflow from branch to PR
- Want to follow best practices end-to-end
- Working on assigned GitHub issue

## Quick Reference

```bash
# 1. Fetch issue and create branch
gh issue view <issue>
git checkout -b <issue>-<description>

# 2. Implement with TDD
# - Write tests first
# - Implement code
# - Run tests: mojo test tests/

# 3. Quality checks
pre-commit run --all-files
mojo format src/**/*.mojo

# 4. Commit and PR
git add . && git commit -m "feat: description

Closes #<issue>"
git push -u origin <branch>
gh pr create --issue <issue>
```

## Workflow

1. **Read issue**: Understand requirements and acceptance criteria
2. **Create branch**: `git checkout -b <issue>-<description>`
3. **Write tests first**: TDD approach - tests drive implementation
4. **Implement code**: Build functionality to pass tests
5. **Quality check**: Format code and run pre-commit
6. **Commit**: Create focused commit with issue reference
7. **Push and PR**: Create PR linked to issue
8. **Monitor CI**: Verify all checks pass

## Branch Naming Convention

Format: `<issue-number>-<description>`

Examples:

- `42-add-tensor-ops`
- `73-fix-memory-leak`
- `105-update-docs`

## Commit Message Format

Follow conventional commits:

```text
type(scope): Brief description

Detailed explanation of changes.

Closes #<issue-number>
```

Types: `feat`, `fix`, `docs`, `refactor`, `test`, `chore`

## Code Quality Checklist

Before creating PR:

- [ ] Issue requirements met
- [ ] Tests written and passing
- [ ] Code formatted (mojo format)
- [ ] Pre-commit hooks pass
- [ ] No warnings or unused variables
- [ ] Documentation updated
- [ ] Commit messages follow convention

## Error Handling

| Problem | Solution |
|---------|----------|
| Issue not found | Verify issue number |
| Branch exists | Use different name or delete old branch |
| Tests fail | Fix code before creating PR |
| CI fails | Address issues before merge |

## Documentation Requirements

Every issue should have `/notes/issues/<issue>/README.md`:

```markdown
# Issue #42: Add Tensor Operations

## Objective
What this issue accomplishes

## Deliverables
- List of files/changes

## Success Criteria
- [ ] Tests passing
- [ ] Documentation complete

## Implementation Notes
- Notes discovered during work
```

## References

- See CLAUDE.md for complete development workflow
- See CLAUDE.md for Mojo syntax standards
- See CLAUDE.md for zero-warnings policy
