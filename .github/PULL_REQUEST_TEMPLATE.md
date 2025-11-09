## Description

<!-- Provide a clear, concise description of what this PR does -->

## Related Issues

<!-- Link to related issues using #issue-number -->

Closes #
Related to #

## Type of Change

<!-- Check all that apply -->

- [ ] Bug fix (non-breaking change that fixes an issue)
- [ ] New feature (non-breaking change that adds functionality)
- [ ] Breaking change (fix or feature that would cause existing functionality to not work as expected)
- [ ] Documentation update
- [ ] Infrastructure/tooling improvement
- [ ] Paper implementation
- [ ] Agent system enhancement
- [ ] Refactoring (no functional changes)

## Development Phase

<!-- Check the primary phase this PR addresses -->

- [ ] Plan (design, specifications, documentation)
- [ ] Test (test implementation, TDD)
- [ ] Implementation (core functionality)
- [ ] Packaging (integration, deployment)
- [ ] Cleanup (refactoring, finalization)

## Component

<!-- Check all components affected -->

- [ ] Foundation (repository structure, configuration)
- [ ] Shared Library (core operations, training utilities)
- [ ] Tooling (CLI, scripts, automation)
- [ ] Papers (research implementations)
- [ ] CI/CD (testing, deployment pipelines)
- [ ] Agents (agentic workflows)
- [ ] Documentation

## Testing

### Test Coverage

- [ ] Unit tests added/updated
- [ ] Integration tests added/updated
- [ ] All tests pass locally
- [ ] Test coverage is adequate (>80% for new code)

### Test Commands

<!-- Provide commands to run tests -->

```bash
# Example
pixi run pytest tests/test_new_feature.py -v
```

## Code Quality

### Pre-commit Checks

- [ ] All pre-commit hooks pass
- [ ] Code follows project conventions (.clinerules)
- [ ] Mojo code formatted with `mojo format` (if applicable)
- [ ] Markdown files pass linting
- [ ] No trailing whitespace or mixed line endings

### Code Review

- [ ] Code is self-documenting with clear names
- [ ] Complex logic has explanatory comments
- [ ] Functions have docstrings (Python/Mojo)
- [ ] Error handling is comprehensive
- [ ] Type hints are used (Python/Mojo)

## Documentation

- [ ] README updated (if needed)
- [ ] API documentation updated
- [ ] CHANGELOG updated (if applicable)
- [ ] Code examples provided for new features
- [ ] Agent documentation updated (if relevant)

## Mojo-Specific (if applicable)

- [ ] Performance-critical code uses `fn` instead of `def`
- [ ] SIMD operations used where appropriate
- [ ] Memory safety verified (ownership, lifetimes)
- [ ] Trait implementations are correct
- [ ] Python interop is clean and documented

## Agent System (if applicable)

- [ ] Agent configuration follows templates
- [ ] Delegation patterns are clear
- [ ] Agent description triggers appropriate invocation
- [ ] Skills are used appropriately
- [ ] Escalation triggers documented

## Breaking Changes

<!-- Describe any breaking changes and migration path -->

**Breaking Changes**: Yes / No

<!-- If yes, describe: -->

### What breaks

### Migration path

### Deprecation timeline (if applicable)

## Performance Impact

<!-- Describe any performance implications -->

- [ ] No significant performance impact
- [ ] Performance improved (describe below)
- [ ] Performance impact assessed and acceptable (describe below)

**Performance Notes**:

## Security Considerations

<!-- Address any security implications -->

- [ ] No security implications
- [ ] Security review completed
- [ ] Input validation added/verified
- [ ] No sensitive data exposed

**Security Notes**:

## Deployment Notes

<!-- Any special deployment considerations -->

- [ ] No special deployment steps needed
- [ ] Database migrations required
- [ ] Configuration changes required
- [ ] Dependencies added/updated

**Deployment Instructions**:

## Screenshots / Examples (if applicable)

<!-- Add screenshots, terminal output, or usage examples -->

```bash
# Example usage
```

## Checklist

### Required (all must be checked)

- [ ] I have read the contributing guidelines
- [ ] My code follows the project's style guidelines
- [ ] I have performed a self-review of my code
- [ ] I have commented my code, particularly in hard-to-understand areas
- [ ] My changes generate no new warnings
- [ ] I have added tests that prove my fix is effective or that my feature works
- [ ] New and existing unit tests pass locally with my changes
- [ ] Any dependent changes have been merged and published

### Recommended

- [ ] I have updated the documentation accordingly
- [ ] I have added this to the CHANGELOG (if applicable)
- [ ] I have tested this on multiple platforms (if relevant)

## Additional Context

<!-- Add any other context about the PR here -->

## For Reviewers

<!-- Highlight areas that need special attention -->

**Focus areas for review**:

**Questions for reviewers**:

---

**Review Assignment**: <!-- @mention relevant reviewers or teams -->
