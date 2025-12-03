# PR Review Standards and Criteria

## ML Odyssey Review Standards

### Code Quality Criteria

### Mojo Code

- Uses `fn` for performance-critical functions
- Proper ownership and borrowing patterns
- Type annotations present
- SIMD optimizations where appropriate
- Memory safety verified

### Python Code

- Type hints on all functions
- Clear docstrings
- Follows PEP 8
- ADR-001 header if using Python for automation

### Commit Message Standards

Follow conventional commits:

```text
<type>(<scope>): <subject>

<body>

<footer>
```text

Types: feat, fix, docs, refactor, test, chore

### File Organization

- Changes in appropriate directories
- No changes to unrelated files
- Proper file naming conventions
- Tests in tests/ directory

### Documentation Requirements

- README updated if public API changed
- ADR created for architectural decisions
- Issue-specific docs posted to GitHub issue comments
- No duplicate documentation

### Testing Requirements

- New features have tests
- Tests follow TDD practices
- All tests pass in CI
- Coverage maintained or improved

### Security Checklist

- No hardcoded credentials
- No command injection vulnerabilities
- Input validation present
- No SQL injection risks
- Dependencies are secure

### Performance Considerations

- No obvious performance regressions
- Efficient algorithms used
- SIMD optimizations for Mojo hot paths
- Proper memory management

## Review Severity Levels

### Critical (Must Fix):

- Security vulnerabilities
- Breaking changes without migration
- Tests failing
- CI failures

### Major (Should Fix):

- Code quality issues
- Missing tests
- Incomplete documentation
- Performance regressions

### Minor (Nice to Have):

- Code style improvements
- Refactoring opportunities
- Additional tests
- Documentation enhancements

## Common Issues

### Anti-Patterns to Watch For

1. **Scope Creep**: Changes beyond issue scope
1. **Missing Tests**: New code without tests
1. **Broken Links**: Documentation with broken references
1. **Duplicate Code**: Copy-paste instead of abstraction
1. **Magic Numbers**: Unexplained constants
1. **Poor Naming**: Unclear variable/function names
1. **Long Functions**: Functions > 50 lines
1. **Deep Nesting**: Nesting > 3 levels

### Language-Specific Issues

### Mojo:

- Using `def` instead of `fn` for performance code
- Missing type annotations
- Unsafe memory operations
- No SIMD optimizations in hot paths

### Python:

- Missing type hints
- No docstrings
- Using Python for ML code (should use Mojo)

## Positive Patterns to Recognize

- Clear, focused commits
- Comprehensive test coverage
- Well-documented code
- Proper error handling
- Efficient algorithms
- Good separation of concerns
- Reusable abstractions

## Review Response Templates

### Approval

```markdown
✅ **Approved**

Great work! This PR:
- [List strengths]

Ready to merge.
```text

### Request Changes

```markdown
🔧 **Changes Requested**

Issues that need to be addressed:
- [List critical/major issues]

Please fix these issues and I'll review again.
```text

### Comments Only

```markdown
💬 **Comments**

Overall looks good. Some suggestions:
- [List minor suggestions]

These are optional improvements, not blockers.
```text
