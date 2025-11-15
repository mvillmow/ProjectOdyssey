# Issue #178: [Plan] Write Standards - Design and Documentation

## Objective

Define comprehensive coding standards, style guidelines, and best practices for the repository to ensure code consistency, quality, and maintainability across all contributions. This planning phase will establish the specifications and design for the Standards section of CONTRIBUTING.md.

## Deliverables

This planning phase will produce:

1. **Standards Architecture Document** - Comprehensive design for coding standards
   - Code style guidelines (Mojo and Python)
   - Documentation standards
   - Testing requirements and coverage expectations
   - Commit message conventions
   - Examples illustrating best practices

2. **Tool Configuration Alignment** - Mapping standards to existing tool configurations
   - Pre-commit hooks (.pre-commit-config.yaml)
   - Python tools (pyproject.toml: ruff, mypy, pytest)
   - Markdown linting (.markdownlint.json)
   - Mojo formatter (when enabled)

3. **Implementation Specifications** - Detailed requirements for follow-up phases
   - Test phase (#692): Standards validation and testing requirements
   - Implementation phase (#693): Standards documentation structure
   - Packaging phase (#694): Integration with other CONTRIBUTING.md sections

## Success Criteria

- [ ] Standards architecture is comprehensive and practical
- [ ] All code style expectations are clearly defined with rationale
- [ ] Testing requirements and coverage expectations are specified
- [ ] Documentation standards align with markdown linting rules
- [ ] Commit message conventions are documented with examples
- [ ] Standards align with existing tool configurations
- [ ] Examples illustrate good practices for each standard
- [ ] Design provides clear specifications for implementation phase
- [ ] Testing requirements are defined for validation

## References

### Source Plans

- [Parent Plan](../../../../plan/01-foundation/03-initial-documentation/02-contributing/plan.md)
- [Component Plan](../../../../plan/01-foundation/03-initial-documentation/02-contributing/02-write-standards/plan.md)

### Related Documentation

- [CLAUDE.md](../../../../CLAUDE.md) - Current coding standards and markdown guidelines
- [Existing CONTRIBUTING.md](../../../../CONTRIBUTING.md) - Current contribution guidelines
- [5-Phase Development Workflow](../../../review/workflow.md)
- [Agent Hierarchy](../../../../agents/hierarchy.md)

### Tool Configurations

- [.pre-commit-config.yaml](../../../../.pre-commit-config.yaml) - Pre-commit hooks
- [pyproject.toml](../../../../pyproject.toml) - Python tooling (ruff, mypy, pytest)
- [.markdownlint.json](../../../../.markdownlint.json) - Markdown linting rules

### Related Issues

- Issue #691: [Plan] Write Standards - Design and Documentation (duplicate/original)
- Issue #692: [Test] Write Standards - Write Tests
- Issue #693: [Impl] Write Standards - Implementation
- Issue #694: [Package] Write Standards - Integration and Packaging
- Issue #695: [Cleanup] Write Standards - Refactor and Finalize

## Implementation Notes

### Standards Architecture Design

#### 1. Code Style Standards

**Mojo Code Style**:
- **Function Declaration**: Prefer `fn` over `def` for better performance and type safety
- **Memory Management**: Use `owned` and `borrowed` parameters for explicit memory management
- **Performance**: Leverage SIMD operations for performance-critical code
- **Data Structures**: Use `struct` over `class` when possible
- **Documentation**: Add comprehensive docstrings to all public APIs using Google style
- **Formatting**: Auto-format using `mojo format` (currently disabled due to bug #5573)

**Python Code Style** (for automation scripts):
- **Standards**: Follow PEP 8 guidelines
- **Type Safety**: Use type hints for all function parameters and return values
- **Documentation**: Write clear docstrings using Google style
- **Formatting**: Use `ruff` for linting (line-length: 120, target: py311)
- **Type Checking**: Use `mypy` with strict settings (disallow_untyped_defs: true)
- **Error Handling**: Comprehensive error handling with logging

**Rationale**:
- Mojo-first approach aligns with project goals for ML/AI implementations
- Type safety catches errors at compile time
- Consistent formatting reduces review burden
- Documentation standards improve code discoverability

#### 2. Documentation Standards

**Markdown Standards**:
- Code blocks must have language specified (` ```python ` not ` ``` `)
- Code blocks surrounded by blank lines (before and after)
- Lists surrounded by blank lines
- Headings surrounded by blank lines
- Lines ≤ 120 characters (except URLs in links)
- Use relative links when possible
- Files must end with newline
- No trailing whitespace

**Code Documentation**:
- All public APIs must have docstrings
- Use Google style for docstrings
- Include: description, Args, Returns, Raises (if applicable)
- Provide usage examples for complex functions
- Document design decisions and rationale

**Rationale**:
- Markdown standards align with markdownlint-cli2 rules
- Consistent documentation improves onboarding
- Examples reduce misuse of APIs
- Design rationale helps future contributors

#### 3. Testing Requirements

**Test Coverage**:
- Aim for >80% code coverage on core functionality
- Test both happy paths and error cases
- Include edge cases and boundary conditions
- Use descriptive test names: `test_function_name_with_scenario`
- Include docstrings explaining what is being tested

**Test-Driven Development**:
- Write tests before implementation whenever possible
- Tests should guide implementation design
- Use pytest framework with configured options
- Run tests with coverage reporting

**Test Organization**:
- Tests in `tests/` directory mirroring module structure
- Use pytest conventions (test_*.py, test_*, Test* classes)
- Separate unit tests, integration tests, and benchmarks

**Rationale**:
- TDD improves code design and reduces bugs
- High coverage provides confidence in changes
- Clear organization makes tests easy to find and run

#### 4. Commit Message Conventions

**Format**: Follow conventional commits specification

```text
<type>(<scope>): <description>

[optional body]

[optional footer]
```

**Types**:
- `feat` - New feature
- `fix` - Bug fix
- `docs` - Documentation only changes
- `refactor` - Code change that neither fixes a bug nor adds a feature
- `test` - Adding missing tests or correcting existing tests
- `chore` - Changes to build process or auxiliary tools
- `perf` - Performance improvements
- `style` - Code style changes (formatting, missing semi-colons, etc.)
- `ci` - Changes to CI configuration files and scripts

**Examples**:
```text
feat(neural): Add convolutional layer implementation
fix(training): Correct gradient calculation in backprop
docs(readme): Update installation instructions
refactor(tensor): Simplify SIMD kernel implementation
test(layers): Add edge case tests for pooling layer
```

**Rationale**:
- Conventional commits enable automated changelog generation
- Clear types make commit history scannable
- Scope helps identify affected components
- Standard format improves git log readability

#### 5. Best Practices Examples

**Mojo Example - Good Practice**:
```mojo
fn sigmoid(x: Float32) -> Float32:
    """Apply sigmoid activation function.

    Args:
        x: Input value

    Returns:
        Sigmoid of x: 1 / (1 + e^(-x))
    """
    return 1.0 / (1.0 + exp(-x))

fn sigmoid_simd[simd_width: Int](x: SIMD[DType.float32, simd_width]) -> SIMD[DType.float32, simd_width]:
    """Apply sigmoid activation using SIMD for performance.

    Parameters:
        simd_width: SIMD vector width

    Args:
        x: Input SIMD vector

    Returns:
        Sigmoid applied element-wise to vector
    """
    return 1.0 / (1.0 + exp(-x))
```

**Python Example - Good Practice**:
```python
#!/usr/bin/env python3
"""
Script to validate agent configurations.

Usage:
    python scripts/validate_agents.py [agent_dir]
"""

from pathlib import Path
from typing import List, Dict, Optional
import logging

logger = logging.getLogger(__name__)

def validate_agent_config(config_path: Path) -> bool:
    """Validate agent configuration file.

    Args:
        config_path: Path to agent configuration file

    Returns:
        True if configuration is valid

    Raises:
        ValueError: If configuration is invalid
    """
    if not config_path.exists():
        raise ValueError(f"Configuration file not found: {config_path}")

    # Validation logic here
    return True
```

**Documentation Example - Good Practice**:
```markdown
## Installation

To install the package:

```bash
pip install ml-odyssey
```

## Usage

Use the library like this:

- Step 1: Import the module
- Step 2: Configure parameters
- Step 3: Run the model

Example:

```python
from ml_odyssey import Model

model = Model(layers=[64, 32, 10])
model.train(data, epochs=100)
```
```

### Tool Configuration Alignment

#### Pre-commit Hooks

Current configuration in `.pre-commit-config.yaml`:

1. **mojo-format** - Currently disabled due to bug #5573
   - Standards should document expected Mojo formatting
   - Note temporary use of manual formatting
   - Plan to re-enable when bug is fixed

2. **markdownlint-cli2** - Active
   - Standards align with .markdownlint.json rules
   - Excludes notes/ directories from strict linting
   - Configuration: MD013 (line-length: 120)

3. **General file checks** - Active
   - trailing-whitespace: Remove trailing whitespace
   - end-of-file-fixer: Ensure newline at EOF
   - check-yaml: Validate YAML syntax
   - check-added-large-files: Prevent files >1MB
   - mixed-line-ending: Fix line endings

#### Python Tools (pyproject.toml)

1. **pytest** - Test framework
   - testpaths: ["tests"]
   - Coverage reporting (term, xml, html)
   - Standards should specify >80% coverage target

2. **ruff** - Python linter
   - line-length: 120
   - target-version: py311
   - Standards should reference ruff for linting

3. **mypy** - Type checker
   - python_version: 3.11
   - disallow_untyped_defs: true
   - Standards should require type hints

#### Markdown Linting (.markdownlint.json)

Standards should explicitly reference:
- MD031: Code blocks must be surrounded by blank lines
- MD040: Code blocks must specify language
- MD032: Lists must be surrounded by blank lines
- MD022: Headings must be surrounded by blank lines
- MD013: Line length limit (120 characters)

### Specifications for Follow-up Phases

#### Test Phase (#692)

The test phase should validate:

1. **Standards Completeness**
   - All required sections are present
   - Examples are correct and runnable
   - Tool configurations are referenced accurately

2. **Standards Consistency**
   - Standards don't conflict with tool configurations
   - Examples follow the documented standards
   - Existing CONTRIBUTING.md aligns with standards

3. **Standards Practicality**
   - Standards are achievable for contributors
   - Examples demonstrate real-world use cases
   - Standards don't impose unnecessary restrictions

**Test Deliverables**:
- Validation checklist for standards content
- Test cases for example code snippets
- Alignment verification with tool configs
- User testing with sample contributions

#### Implementation Phase (#693)

The implementation phase should create:

1. **Standards Section Structure**
   ```markdown
   ## Code Style Guidelines
   ### Mojo Code Style
   ### Python Code Style
   ### Documentation Style

   ## Testing Guidelines
   ### Writing Tests
   ### Test Coverage

   ## Commit Message Conventions
   ### Format
   ### Types
   ### Examples

   ## Best Practices
   ### Mojo Examples
   ### Python Examples
   ### Documentation Examples
   ```

2. **Content Requirements**
   - Clear, concise descriptions of each standard
   - Rationale for important standards (why, not just what)
   - Correct, runnable code examples
   - Links to tool configurations
   - Common pitfalls and how to avoid them

3. **Integration Points**
   - Links to related sections (workflow, PR process)
   - References to tool documentation
   - Pointers to comprehensive guides in /agents/ and /notes/review/

**Implementation Deliverables**:
- Standards section content for CONTRIBUTING.md
- Example code snippets (verified to work)
- Links to tool configurations
- Cross-references to related documentation

#### Packaging Phase (#694)

The packaging phase should integrate:

1. **Standards with Workflow Section**
   - Standards enforcement in development workflow
   - Pre-commit hooks as part of workflow
   - Testing standards in TDD workflow

2. **Standards with PR Process**
   - Code review checklist aligned with standards
   - CI/CD checks enforcing standards
   - Review comment templates for standard violations

3. **Standards with Existing Content**
   - Merge with current CONTRIBUTING.md
   - Resolve conflicts or duplications
   - Ensure consistent voice and style

**Packaging Deliverables**:
- Integrated CONTRIBUTING.md with all sections
- Resolved conflicts with existing content
- Consistent documentation style throughout
- Functional cross-references and links

### Design Decisions

#### Decision 1: Standards Location

**Options**:
1. Separate STANDARDS.md file
2. Section in CONTRIBUTING.md
3. Distributed across multiple files

**Decision**: Section in CONTRIBUTING.md

**Rationale**:
- Single file makes it easier for contributors to find guidelines
- Standards are directly related to contribution process
- CONTRIBUTING.md is the standard location for contribution guidelines
- Reduces documentation fragmentation

#### Decision 2: Example Format

**Options**:
1. Inline examples in standards text
2. Separate examples file
3. References to existing code

**Decision**: Inline examples with references to tool configs

**Rationale**:
- Inline examples make standards immediately actionable
- References to tool configs provide verification
- Existing code examples may change or violate standards
- Inline approach is common in contribution guidelines

#### Decision 3: Tool Configuration References

**Options**:
1. Duplicate tool settings in standards
2. Reference tool configs with links
3. Omit tool details entirely

**Decision**: Reference tool configs with explanatory context

**Rationale**:
- Avoid duplication (single source of truth)
- Provide context for why tools are configured as they are
- Links enable contributors to verify settings
- Balance between completeness and maintainability

#### Decision 4: Standards Strictness

**Options**:
1. Very strict with no exceptions
2. Guidelines with flexibility
3. Minimal standards only

**Decision**: Clear standards with documented exceptions

**Rationale**:
- Strict standards reduce ambiguity
- Documented exceptions handle edge cases
- Clear rationale helps contributors understand importance
- Balance consistency with practical needs

### Open Questions

1. **Mojo Formatting**: When bug #5573 is fixed, should we immediately enable mojo-format?
   - Recommendation: Yes, but announce with deprecation period for manual formatting

2. **Coverage Target**: Should we enforce 80% coverage in CI or just recommend it?
   - Recommendation: Start with recommendation, move to enforcement after foundation is stable

3. **Documentation Examples**: Should we include negative examples (what NOT to do)?
   - Recommendation: Yes, for common pitfalls; helps prevent mistakes

4. **Tool Versions**: Should we specify minimum versions for development tools?
   - Recommendation: Yes, document in prerequisites section

### Next Steps

1. **Review and Validate** this design with stakeholders
2. **Create Test Phase Specifications** (Issue #692)
   - Develop validation checklist
   - Create test cases for examples
   - Define alignment verification process

3. **Prepare Implementation Phase** (Issue #693)
   - Draft standards content
   - Write and verify example code
   - Create cross-reference map

4. **Plan Integration Strategy** (Issue #694)
   - Map standards to workflow section
   - Map standards to PR process section
   - Identify conflicts with existing content

## Status

- **Phase**: Planning
- **Status**: In Progress
- **Created**: 2025-11-15
- **Last Updated**: 2025-11-15

## Notes

This planning document establishes the comprehensive design for coding standards that will guide all contributions to ML Odyssey. The standards balance clarity and flexibility, align with existing tool configurations, and provide practical examples. The design supports a phased implementation approach with clear specifications for testing, implementation, and packaging phases.

Key principles:
- **Clarity over brevity** - Be explicit about expectations
- **Rationale over rules** - Explain why standards matter
- **Examples over theory** - Show good practices in action
- **Alignment over duplication** - Reference tools, don't duplicate
- **Practical over perfect** - Standards should be achievable
