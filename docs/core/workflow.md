# Development Workflow

ML Odyssey's complete development workflow including the 5-phase process, git workflow, and best practices.

## Overview

ML Odyssey follows a structured, hierarchical development workflow that ensures quality, consistency, and
maintainability. This guide covers the entire development lifecycle from planning to deployment.

## 5-Phase Development Process

Every component follows this workflow:

```text
Phase 1: Plan
    ↓
Phase 2: Test ──┐
Phase 3: Impl ──┤ (Parallel)
Phase 4: Package┘
    ↓
Phase 5: Cleanup
```

### Phase Dependencies

- **Plan must complete** before other phases start
- **Test, Implementation, Packaging run in parallel** after Plan
- **Cleanup runs** after all parallel phases complete

### Phase 1: Plan

**Purpose**: Design and document before implementation

**Outputs**:

- Component specification
- Architecture decisions
- API design
- Test plan
- Implementation plan

**Example Issue**: `#57 [Plan] Docs`

**Deliverables**:

```markdown
## Planning Deliverables

- `/notes/issues/57/README.md` - Issue documentation
- Architecture diagrams
- API specifications
- Test strategy
- Success criteria
```

**Completion Criteria**:

- [ ] Specification complete and reviewed
- [ ] Architecture documented
- [ ] APIs defined
- [ ] Test plan created
- [ ] Dependencies identified

### Phase 2: Test (Parallel)

**Purpose**: Write tests following TDD principles

**Outputs**:

- Test suite structure
- Unit tests (with stubs)
- Integration tests
- Test documentation

**Example Issue**: `#58 [Test] Docs`

**Deliverables**:

```mojo
# Test structure created
tests/foundation/docs/
├── test_structure.py
├── test_markdown_lint.py
├── test_links.py
└── conftest.py
```

**Completion Criteria**:

- [ ] Test structure created
- [ ] Unit tests written (may use stubs)
- [ ] Integration tests written
- [ ] Tests documented
- [ ] Tests pass (with stubs)

### Phase 3: Implementation (Parallel)

**Purpose**: Build the functionality

**Outputs**:

- Module implementation
- Core functionality
- API implementation
- Code documentation

**Example Issue**: `#59 [Impl] Docs`

**Deliverables**:

```text
docs/
├── index.md
├── getting-started/
├── core/
├── advanced/
└── dev/
```

**Completion Criteria**:

- [ ] All components implemented
- [ ] Code documented
- [ ] Code formatted (`mojo format`)
- [ ] Tests pass
- [ ] Code reviewed

### Phase 4: Packaging (Parallel)

**Purpose**: Integration and user-facing packaging

**Outputs**:

- Module integration
- Examples
- User documentation
- Benchmarks

**Example Issue**: `#60 [Package] Docs`

**Deliverables**:

```text
- Integration with mkdocs
- Example documentation
- User guides
- Navigation structure
```

**Completion Criteria**:

- [ ] Module integrated
- [ ] Examples working
- [ ] Documentation complete
- [ ] Benchmarks run
- [ ] Ready for users

### Phase 5: Cleanup (Sequential)

**Purpose**: Refine and finalize

**Outputs**:

- Code refactoring
- Documentation updates
- Performance optimization
- Final review

**Example Issue**: `#61 [Cleanup] Docs`

**Tasks**:

```text
- Address review feedback
- Refactor based on learnings
- Update documentation
- Final quality check
```

**Completion Criteria**:

- [ ] All review feedback addressed
- [ ] Code refactored
- [ ] Documentation updated
- [ ] Performance acceptable
- [ ] Ready to merge

## Git Workflow

### Branch Strategy

ML Odyssey uses a simple branch strategy:

- **`main`**: Production-ready code
- **Feature branches**: One per GitHub issue

### Branch Naming

Format: `<issue-number>-<phase>-<component>`

**Examples**:

```bash
# Good branch names
34-impl-training          # Issue #34, Implementation phase
39-impl-data              # Issue #39, Implementation phase
57-plan-docs              # Issue #57, Planning phase
58-test-docs              # Issue #58, Testing phase

# Bad branch names
fix-bug                   # No issue number
feature-new-model         # No issue number
improvement               # Too vague
```

### Git Worktrees

Use git worktrees for parallel development:

```bash
# Create main repository
cd ml-odyssey

# Create worktree for issue #39
git worktree add ../worktrees/issue-39-impl-data 39-impl-data

# Work in worktree
cd ../worktrees/issue-39-impl-data
# Make changes...

# Create another worktree for issue #40 (parallel work)
git worktree add ../worktrees/issue-40-test-data 40-test-data

# List worktrees
git worktree list

# Remove worktree when done
git worktree remove ../worktrees/issue-39-impl-data
```

**Benefits**:

- Work on multiple issues simultaneously
- No branch switching (preserves state)
- Clean separation of work
- No merge conflicts from switching

### Commit Message Format

Follow conventional commits:

```text
<type>(<scope>): <description>

[optional body]

[optional footer]
```

**Types**:

- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation
- `test`: Test changes
- `refactor`: Code refactoring
- `perf`: Performance improvement
- `chore`: Build/tooling changes

**Examples**:

```bash
# Good commit messages
git commit -m "feat(data): implement TensorDataset"
git commit -m "test(data): add BatchLoader tests"
git commit -m "docs(core): add shared library guide"
git commit -m "fix(training): correct gradient accumulation"
git commit -m "refactor(core): simplify layer interface"

# Commit with body
git commit -m "feat(training): add learning rate scheduler

Implements cosine annealing scheduler as described in the paper.
Includes tests and documentation.

Closes #123"
```

### Pull Request Workflow

#### Step 1: Ensure tests pass locally

```bash
# Format code
pixi run mojo format shared/

# Run tests
pixi run pytest tests/

# Run pre-commit checks
pre-commit run --all-files
```

#### Step 2: Commit and push

```bash
git add .
git commit -m "feat(data): implement data module"
git push origin 39-impl-data
```

#### Step 3: Create Pull Request

```bash
# Link to issue (REQUIRED)
gh pr create --issue 39

# Or specify details manually
gh pr create \
  --title "feat(data): Implement data module" \
  --body "Closes #39" \
  --label "implementation"
```

#### Step 4: Address review feedback

```bash
# Make changes based on review
vim shared/data/datasets.mojo

# Commit fixes
git commit -m "fix(data): address review feedback on TensorDataset"
git push

# Reply to review comments (REQUIRED)
gh api repos/OWNER/REPO/pulls/PR/comments/COMMENT_ID/replies \
  --method POST \
  -f body="✅ Fixed - Updated TensorDataset to use borrowed parameters"
```

#### Step 5: Merge

```bash
# After approval, merge (usually squash merge)
gh pr merge --squash --delete-branch
```

## Issue Management

### Creating Issues

Issues are created from hierarchical plans:

```bash
# Create issues for a component
python3 scripts/create_issues.py --section 02-shared-library
```

Each component gets 5 issues (one per phase):

- `[Plan] Component Name`
- `[Test] Component Name`
- `[Impl] Component Name`
- `[Package] Component Name`
- `[Cleanup] Component Name`

### Issue Labels

Standard labels indicate phase and type:

- `planning` - Planning phase (light purple)
- `testing` - Testing phase (yellow)
- `tdd` - Test-driven development (yellow)
- `implementation` - Implementation phase (dark blue)
- `packaging` - Packaging phase (light green)
- `integration` - Integration tasks (light green)
- `cleanup` - Cleanup phase (red)
- `documentation` - Documentation work (blue)

### Issue Workflow

1. **Assigned**: Agent assigned to issue
2. **In Progress**: Work started, branch created
3. **PR Created**: Pull request created and linked
4. **Review**: Code review in progress
5. **Merged**: PR merged, issue closed

## Code Review Process

### Review Checklist

Reviewers check:

- [ ] **Functionality**: Code works as intended
- [ ] **Tests**: Adequate test coverage
- [ ] **Documentation**: Public APIs documented
- [ ] **Style**: Follows Mojo patterns
- [ ] **Performance**: No obvious performance issues
- [ ] **Scope**: Changes match issue requirements

### Review Comments

**Requesting changes**:

```markdown
## Code Structure

The forward pass could be simplified by extracting the convolution logic:

```mojo

fn forward(borrowed self, borrowed input: Tensor) -> Tensor:
    return self.apply_conv(input)

fn apply_conv(borrowed self, borrowed input: Tensor) -> Tensor:
    # Convolution logic here
```

**Approving**:

```markdown

LGTM! Nice implementation of the data loader.

A few minor suggestions but nothing blocking:

- Consider adding docstring examples
- Could benefit from type aliases for readability

Approved ✅

```

### Addressing Feedback

**Respond to EVERY comment**:

```bash

# Get review comment IDs

gh api repos/OWNER/REPO/pulls/PR/comments

# Reply to each comment

gh api repos/OWNER/REPO/pulls/PR/comments/COMMENT_ID/replies \
  --method POST \
  -f body="✅ Fixed - Extracted convolution logic to apply_conv method"

```

## Development Environment

### Local Setup

```bash

# Clone repository

git clone https://github.com/mvillmow/ml-odyssey.git
cd ml-odyssey

# Install dependencies

pixi install

# Set up pre-commit hooks

pre-commit install

# Verify setup

pixi run mojo --version
pixi run pytest --version

```

### IDE Configuration

**VS Code** (`.vscode/settings.json`):

```json

{
  "mojo.format.onSave": true,
  "editor.formatOnSave": true,
  "python.linting.enabled": true,
  "python.testing.pytestEnabled": true
}

```

### Environment Variables

Set in `.env` (not committed):

```bash

# Optional: Override defaults

MOJO_PYTHON_VERSION=3.10
ML_ODYSSEY_DATA_DIR=/path/to/datasets
ML_ODYSSEY_LOG_LEVEL=DEBUG

```

## Testing Workflow

### TDD Cycle

1. **Write test** (red)
2. **Implement minimum code** to pass (green)
3. **Refactor** (clean)
4. **Repeat**

#### Example

```mojo

# Step 1: Write failing test

fn test_linear_forward():
    var layer = Linear(10, 5)  # Doesn't exist yet
    var input = Tensor.randn(8, 10)
    var output = layer.forward(input)
    assert_equal(output.shape, [8, 5])

# Step 2: Implement minimum to pass

struct Linear:
    fn forward(borrowed self, borrowed input: Tensor) -> Tensor:
        return Tensor.zeros(input.shape[0], 5)  # Stub

# Step 3: Refactor to real implementation

struct Linear:
    var weight: Tensor
    var bias: Tensor

    fn forward(borrowed self, borrowed input: Tensor) -> Tensor:
        return input @ self.weight.T + self.bias

```

#### Running Tests

```bash

# All tests

pixi run pytest tests/

# Specific module

pixi run pytest tests/shared/data/

# Specific test

pixi run pytest tests/shared/data/test_datasets.mojo::test_tensor_dataset

# With coverage

pixi run pytest tests/ --cov=shared

# Fast tests only

pixi run pytest tests/ -m "not slow"

```

## Documentation Workflow

### Documentation Structure

All documentation goes to appropriate location:

- **Issue-specific**: `/notes/issues/<issue-number>/README.md`
- **User docs**: `/docs/`
- **Team docs**: `/agents/`
- **Architecture decisions**: `/notes/review/`

### Writing Documentation

#### Step 1: Create issue documentation

```markdown

# Issue #XX: [Phase] Component

## Objective

Brief description

## Deliverables

- List of outputs

## Implementation Notes

Notes discovered during work

```

#### Step 2: Create user documentation

```markdown

# Component Name

User-facing guide for the component

## Quick Example

[Working code example]

## API Reference

[Complete API docs]

```

#### Step 3: Update README links

```markdown

- [Component Guide](docs/core/component.md)

```

## CI/CD Integration

### Automated Checks

Every PR triggers:

1. **Pre-commit hooks**: Code formatting, linting
2. **Tests**: Full test suite
3. **Agent validation**: Agent config validation
4. **Build**: Verify code compiles

### CI Configuration

Located in `.github/workflows/`:

- `test.yml` - Run test suite
- `pre-commit.yml` - Code quality checks
- `test-agents.yml` - Agent validation

## Best Practices

### DO

- ✅ Follow the 5-phase workflow
- ✅ Write tests before implementation (TDD)
- ✅ Link all PRs to issues
- ✅ Use git worktrees for parallel work
- ✅ Commit frequently with clear messages
- ✅ Reply to all review comments
- ✅ Document decisions in issue README

### DON'T

- ❌ Skip planning phase
- ❌ Create PRs without linked issues
- ❌ Work outside issue scope
- ❌ Ignore test failures
- ❌ Skip code review
- ❌ Commit directly to main
- ❌ Merge without approval

## Troubleshooting

### Tests Failing Locally

```bash

# Format code

pixi run mojo format shared/

# Clear cache

rm -rf .pytest_cache __pycache__

# Reinstall dependencies

pixi install --force-reinstall

# Run single test for debugging

pixi run pytest tests/path/to/test.mojo::test_name -v -s

```

### PR Not Linked to Issue

```bash

# Check if linked

gh pr view PR --json body

# Add link to PR description

gh pr edit PR --body "Closes #XX"

# Or add comment

gh pr comment PR --body "Closes #XX"

```

### Merge Conflicts

```bash

# Rebase on main

git fetch origin
git rebase origin/main

# Resolve conflicts

vim conflicted_file.mojo
git add conflicted_file.mojo
git rebase --continue

# Force push (be careful!)

git push --force-with-lease

```

## Quick Reference

### Common Commands

```bash

# Create worktree

git worktree add ../worktrees/issue-XX-phase-component XX-phase-component

# Create PR

gh pr create --issue XX

# Run tests

pixi run pytest tests/

# Format code

pixi run mojo format shared/

# Pre-commit checks

pre-commit run --all-files

# View issue

gh issue view XX

# Reply to review

gh api repos/OWNER/REPO/pulls/PR/comments/ID/replies -f body="✅ Fixed"

```

### Phase Checklist

- [ ] **Plan**: Specification complete and reviewed
- [ ] **Test**: Tests written and passing (with stubs if needed)
- [ ] **Implementation**: Code complete, tests passing, reviewed
- [ ] **Packaging**: Integrated, documented, examples working
- [ ] **Cleanup**: Refactored, optimized, final review complete

## Next Steps

- **[Agent System](agent-system.md)** - Understanding agent coordination
- **[Testing Strategy](testing-strategy.md)** - Comprehensive testing approach
- **[Project Structure](project-structure.md)** - Repository organization
- **[Configuration](configuration.md)** - Environment configuration

## Related Documentation

- [Git Worktree Strategy](/notes/review/worktree-strategy.md) - Advanced worktree usage
- [Orchestration Patterns](/notes/review/orchestration-patterns.md) - Agent coordination
- [Contributing Guide](/CONTRIBUTING.md) - Contribution guidelines
