# Contributing to ML Odyssey

Thank you for your interest in contributing to ML Odyssey! We welcome contributions from everyone and are grateful
for your help. This document provides guidelines and instructions for contributing to the project.

## Development Setup

### Prerequisites

- Python 3.7 or higher
- Mojo compiler (v0.25.7 or later)
- Git

### Environment Setup with Pixi

We use [Pixi](https://pixi.sh/) for environment management. This ensures everyone uses the same dependencies.

```bash

# Install Pixi (if not already installed)
# Visit https://pixi.sh/ for installation instructions

# Create and activate the development environment
pixi shell

```text

### Verify Your Setup

```bash

# Check Mojo installation
mojo --version

# Check Python installation
python3 --version

# Verify pre-commit hooks are installed
pre-commit install

```text

## Running Tests

We follow Test-Driven Development (TDD) principles. Tests should be written before implementation whenever possible.

```bash

# Run all tests
pixi run test

# Run tests for a specific module
pixi run test ml_odyssey/module_name/tests

# Run tests with verbose output
pixi run test --verbose

# Run tests with coverage
pixi run test --cov=ml_odyssey --cov-report=html

```text

## Code Style Guidelines

### Mojo Code Style

We use `mojo format` for consistent code formatting. Pre-commit hooks will automatically run this on staged files.

### Key principles for Mojo code

- Prefer `fn` over `def` for better performance and type safety
- Use `owned` and `borrowed` parameters for explicit memory management
- Leverage SIMD operations for performance-critical code
- Use `struct` over `class` when possible
- Add comprehensive docstrings to all public APIs

### Example

```mojo

fn add(x: Int32, y: Int32) -> Int32:
    """Add two integers.

    Args:
        x: First integer
        y: Second integer

    Returns:
        Sum of x and y
    """
    return x + y

```text

### Python Code Style

For automation scripts that require Python:

- Follow [PEP 8](https://peps.python.org/pep-0008/)
- Use type hints for all function parameters and return values
- Write clear docstrings using the Google style
- Use `black` for code formatting (included in pre-commit)

### Example

```python

def calculate_mean(values: list[float]) -> float:
    """Calculate the mean of a list of values.

    Args:
        values: List of numeric values

    Returns:
        Mean of the values

    Raises:
        ValueError: If values list is empty
    """
    if not values:
        raise ValueError("Cannot calculate mean of empty list")
    return sum(values) / len(values)

```text

### Documentation Style

All documentation files follow markdown standards and must pass `markdownlint-cli2` linting.

### Key rules

- Code blocks must have a language specified (` ```python ` not ` ``` `)
- Code blocks must be surrounded by blank lines (before and after)
- Lists must be surrounded by blank lines
- Headings must be surrounded by blank lines
- Lines should not exceed 120 characters
- Use relative links when possible

### Example

```markdown

## Installation

To install the package:

```bash

pip install ml-odyssey

```text
## Usage

Use the library like this:

- Step 1: Import
- Step 2: Configure
- Step 3: Run

```text

(No additional configuration needed - Pixi manages everything)

```text
### Pre-commit Hooks

Pre-commit hooks automatically check code quality before commits. They run:

- `mojo format` - Auto-format Mojo code
- `markdownlint-cli2` - Lint markdown files
- `trailing-whitespace` - Remove trailing whitespace
- `end-of-file-fixer` - Ensure files end with newline
- `check-yaml` - Validate YAML syntax

```bash

# Install pre-commit hooks (one-time setup)

pre-commit install

# Run hooks on all files

pre-commit run --all-files

# Run hooks on staged files only

pre-commit run

# Skip hooks (use sparingly, only when necessary)

git commit --no-verify

```text
## Pull Request Process

### Before You Start

1. Check if an issue exists for your work (create one if needed)
2. Create a branch from `main` using the naming convention: `<issue-number>-<description>`
   (e.g., `42-add-convolution-layer`)
3. Write tests first following TDD principles
4. Implement your changes
5. Run tests and pre-commit hooks locally

### Creating Your Pull Request

1. Push your branch to the repository
2. Create a pull request using GitHub CLI:

```bash

gh pr create --issue <issue-number>

```text
This automatically links your PR to the issue.

Fill out the PR template with:

- Clear title describing the change
- Description of what changed and why
- Reference to the linked issue
- Any testing notes

### Code Review

1. Address review comments promptly
2. For each review comment, make the requested change
3. Reply to EACH review comment individually:

```bash

# Get review comment IDs

gh api repos/OWNER/REPO/pulls/PR_NUMBER/comments \
  --jq '.[] | {id: .id, path: .path, body: .body}'

# Reply to a specific comment

gh api repos/OWNER/REPO/pulls/PR_NUMBER/comments/COMMENT_ID/replies \
  --method POST \
  -f body="✅ Fixed - [brief description]"

```text
Keep responses short and start with ✅ to indicate the issue is resolved.

After making changes, verify CI passes before requesting re-review.

### Merging

Once approved and CI passes:

1. Ensure your branch is up to date with `main`
2. Squash commits if appropriate (maintainers can do this)
3. Merge using the GitHub web interface or CLI
4. Delete the feature branch after merging

## Reporting Issues

### Bug Reports

When reporting a bug, include:

- Clear title describing the issue
- Steps to reproduce the problem
- Expected behavior
- Actual behavior
- Environment details (Mojo version, OS, Python version)
- Relevant code snippets or logs

### Feature Requests

When requesting a feature:

- Clear title describing the feature
- Problem it solves or value it provides
- Proposed solution (if you have one)
- Any alternatives you've considered

## Documentation

Comprehensive documentation lives in two locations:

### Team Documentation (`/agents/`)

Quick start guides and visual references for team onboarding.

### Issue-Specific Documentation (GitHub Issues)

Implementation notes and decisions are posted directly to GitHub issues as comments.
See `.claude/shared/github-issue-workflow.md` for the workflow.

When writing documentation:

1. Follow the markdown standards above
2. Include code examples with proper syntax highlighting
3. Link to related documentation
4. Keep language clear and concise

## Testing Guidelines

We follow Test-Driven Development principles.

### Writing Tests

1. Write tests before implementation
2. Tests should be in a `tests/` directory mirroring the module structure
3. Use descriptive test names: `test_function_name_with_scenario`
4. Include docstrings explaining what is being tested

**Example**:

```python

def test_add_positive_integers():
    """Test adding two positive integers."""
    assert add(2, 3) == 5

def test_add_negative_integers():
    """Test adding two negative integers."""
    assert add(-2, -3) == -5

```text
### Test Coverage

- Aim for high code coverage on core functionality (>80%)
- Test both happy paths and error cases
- Include edge cases

## Questions

If you have questions:

1. Check existing documentation in `/notes/review/` and `/agents/`
2. Search existing GitHub issues
3. Create a new discussion or issue with your question
4. Contact the maintainers

## Additional Resources

- [Project Architecture](../../CLAUDE.md)
- [Development Workflow](../../docs/dev/workflow.md)
- [5-Phase Development Process](../../notes/review/workflow.md)
- [Mojo Documentation](https://docs.modular.com/mojo/manual/)
- [ML Odyssey README](../../README.md)

## Code of Conduct

Please review our [Code of Conduct](CODE_OF_CONDUCT.md) before contributing. We are committed to providing a
welcoming and inclusive environment for all contributors.

---

Thank you for contributing to ML Odyssey! Your effort helps advance AI research and education.
