# Issue #691: [Plan] Write Standards - Design and Documentation

## Objective

Define coding standards, style guidelines, and best practices for the repository to ensure code consistency, quality, and maintainability across all contributions. This planning phase creates comprehensive specifications for the standards section in CONTRIBUTING.md.

## Deliverables

- Standards section in CONTRIBUTING.md
- Code style guidelines (Mojo and Python)
- Documentation standards
- Testing requirements
- Commit message conventions

## Success Criteria

- [ ] Standards are clearly defined and practical
- [ ] Code style expectations are documented
- [ ] Testing requirements are specified
- [ ] Examples illustrate good practices
- [ ] Standards align with tool configurations

## Design Decisions

### 1. Standards Philosophy

**Decision**: Focus on practical standards that improve code quality without being overly restrictive.

### Rationale

- Reduce friction for contributors while maintaining quality
- Standards should serve the project's goals, not hinder development
- Provide clear rationale for important standards to help contributors understand the "why"

### Alternatives Considered

- Extremely strict standards - rejected as potentially discouraging contributions
- Minimal standards - rejected as insufficient for ensuring code quality at scale

### 2. Code Style Approach

**Decision**: Use automated tooling (mojo format, black, pre-commit) to enforce style.

### Rationale

- Automation eliminates bikeshedding over style choices
- Pre-commit hooks catch issues before review
- Contributors get immediate feedback
- Maintainers focus on design, not formatting

### Implementation Details

- `mojo format` for Mojo code (already configured)
- `black` for Python code (already configured)
- Pre-commit hooks enforce on commit (already in place)
- CI validates in workflows

### 3. Documentation Standards

**Decision**: Markdown-based with automated linting via markdownlint-cli2.

### Rationale

- Consistent formatting improves readability
- Linting catches common issues (missing languages, line length)
- Standards align with existing .markdownlint.yaml configuration

### Key Rules

- Code blocks must specify language
- Proper spacing around blocks/lists/headings
- 120-character line limit (with exceptions for URLs)

### 4. Testing Requirements

**Decision**: Test-Driven Development (TDD) as the primary approach.

### Rationale

- Tests guide implementation design
- Higher confidence in changes
- Better test coverage
- Catches regressions early

### Coverage Expectations

- Core functionality: >80% coverage
- Test happy paths, error cases, and edge cases
- Integration tests for critical workflows

### 5. Commit Message Conventions

**Decision**: Use conventional commits format.

### Rationale

- Clear categorization of changes (feat, fix, docs, etc.)
- Enables automated changelog generation
- Improves git log readability
- Industry standard practice

### Format

```text
<type>(<scope>): <description>

Examples:
feat(tensor): Add matrix multiplication
fix(scripts): Correct parsing issue
docs(contributing): Update testing guidelines
```text

### 6. Mojo-Specific Guidelines

**Decision**: Emphasize modern Mojo patterns (fn, owned/borrowed, SIMD, struct).

### Rationale

- Mojo is the primary language for ML/AI implementations
- Modern patterns provide better performance and type safety
- Clear guidance helps contributors write idiomatic Mojo

### Key Principles

- Prefer `fn` over `def` for better performance
- Use explicit memory management (`owned`, `borrowed`)
- Leverage SIMD for performance-critical code
- Use `struct` over `class` when possible

### 7. Python Usage Policy

**Decision**: Python for automation only, with documented justification (see ADR-001).

### Rationale

- Mojo is the primary language target
- Python acceptable for tooling with technical limitations
- Must document why Python is required

### Allowed Use Cases

- Subprocess output capture (Mojo v0.25.7 limitation)
- Regex-heavy processing (no Mojo regex support)
- GitHub API interaction

## Architecture Considerations

### Integration with Existing Tools

The standards build upon existing tool configurations:

1. **Pre-commit hooks** (.pre-commit-config.yaml)
   - mojo format
   - markdownlint-cli2
   - trailing-whitespace, end-of-file-fixer
   - check-yaml, check-added-large-files

1. **Python tools** (pyproject.toml)
   - black (code formatting)
   - ruff (linting)
   - pytest (testing)

1. **CI/CD** (.github/workflows/)
   - pre-commit.yml enforces standards
   - test-agents.yml validates agent configs

### Documentation Organization

Standards documentation follows the 3-location pattern:

1. **CONTRIBUTING.md** - Primary standards reference for contributors
1. **/notes/review/** - Architectural decisions (ADR-001 language selection)
1. **/notes/issues/691/** - Planning documentation (this file)

### Scope Boundaries

### In Scope

- Code style for Mojo and Python
- Documentation formatting standards
- Testing requirements and TDD approach
- Commit message conventions
- Pre-commit hook usage

### Out of Scope

- Architectural patterns (covered in /notes/review/)
- Agent hierarchy and delegation (covered in /agents/)
- Development workflow (covered in workflow section)
- PR process details (covered in PR process section)

## Implementation Steps

The implementation phase (#693) will:

1. Document code style guidelines (refer to black, ruff, mojo format configs)
1. Define documentation standards for code and APIs
1. Specify testing requirements and coverage expectations
1. Establish commit message conventions
1. Provide examples of good practices

Each step includes practical examples and links to tool configurations.

## References

### Source Plan

- [notes/plan/01-foundation/03-initial-documentation/02-contributing/02-write-standards/plan.md](notes/plan/01-foundation/03-initial-documentation/02-contributing/02-write-standards/plan.md)

### Parent Context

- [notes/plan/01-foundation/03-initial-documentation/02-contributing/plan.md](notes/plan/01-foundation/03-initial-documentation/02-contributing/plan.md) - CONTRIBUTING.md component plan

### Related Issues

- #692 - [Test] Write Standards
- #693 - [Implementation] Write Standards
- #694 - [Package] Write Standards
- #695 - [Cleanup] Write Standards

### Related Documentation

- [CONTRIBUTING.md](CONTRIBUTING.md) - Current contributing guidelines (target file)
- [CLAUDE.md](CLAUDE.md) - Project conventions and standards
- [ADR-001](notes/review/adr/ADR-001-language-selection-tooling.md) - Language selection strategy
- [.pre-commit-config.yaml](.pre-commit-config.yaml) - Pre-commit hook configuration
- [.markdownlint.yaml](.markdownlint.yaml) - Markdown linting rules

### Development Principles

The standards align with core development principles (from CLAUDE.md):

1. **KISS** - Keep standards simple and practical
1. **YAGNI** - Don't add unnecessary restrictions
1. **TDD** - Test-driven development as primary approach
1. **DRY** - Avoid duplicating standards across docs
1. **POLA** - Standards should not surprise contributors

## Implementation Notes

*This section will be populated during the implementation phase (#693) with findings, decisions, and challenges encountered.*
