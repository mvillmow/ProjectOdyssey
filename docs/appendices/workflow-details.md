# Development Workflow Details

Complete reference for ML Odyssey's development workflow with comprehensive examples, troubleshooting, and
advanced patterns.

> **Quick Reference**: For a concise overview, see [Development Workflow](../core/workflow.md).

## Table of Contents

- [5-Phase Process Deep Dive](#5-phase-process-deep-dive)
- [Git Workflow Advanced](#git-workflow-advanced)
- [Pull Request Best Practices](#pull-request-best-practices)
- [Issue Management Details](#issue-management-details)
- [Code Review Deep Dive](#code-review-deep-dive)
- [Testing Workflow Details](#testing-workflow-details)
- [Documentation Workflow Details](#documentation-workflow-details)
- [CI/CD Integration Details](#cicd-details)
- [Advanced Troubleshooting](#advanced-troubleshooting)
- [Complete Examples](#complete-examples)

## 5-Phase Process Deep Dive

### Phase 1: Planning - Complete Walkthrough

**Step-by-step planning process:**

```bash
# 1. Create issue documentation directory
mkdir -p /notes/issues/57

# 2. Create initial README.md
cat > /notes/issues/57/README.md << 'EOF'
# Issue #57: [Plan] Docs

## Objective

Design comprehensive documentation structure for ML Odyssey.

## Deliverables

- Documentation structure plan
- Content organization strategy
- MkDocs configuration
- Writing guidelines

## Implementation Notes

(To be filled during planning)
EOF

# 3. Research and document architecture
# Add architecture diagrams, API specifications, etc.

# 4. Create specification documents
# Add to /notes/review/ for comprehensive specs
```

**Planning deliverables checklist:**

- [ ] Component specification with clear scope
- [ ] Architecture diagrams (use mermaid, plantuml, or ascii art)
- [ ] API design with all public interfaces
- [ ] Test strategy with coverage targets
- [ ] Success criteria with measurable outcomes
- [ ] Dependencies identified with versions
- [ ] Performance requirements specified
- [ ] Security considerations documented

### Phase 2: Testing - TDD in Practice

**Complete TDD cycle example:**

```mojo
# Step 1: Write failing test (RED)
fn test_linear_layer_forward():
    """Test linear layer forward pass."""
    var layer = Linear(input_size=10, output_size=5)  # Doesn't exist yet!
    var input = Tensor.randn(8, 10)
    var output = layer.forward(input)

    # Expected behavior
    assert_equal(output.shape[0], 8)   # Batch size preserved
    assert_equal(output.shape[1], 5)   # Output size correct
    assert_true(output.dtype == Float32)

# Run test - should FAIL
# pixi run pytest tests/test_linear.mojo::test_linear_layer_forward

# Step 2: Implement minimum code to pass (GREEN)
struct Linear:
    var weight: Tensor
    var bias: Tensor

    fn __init__(inout self, input_size: Int, output_size: Int):
        self.weight = Tensor.randn(output_size, input_size)
        self.bias = Tensor.zeros(output_size)

    fn forward(borrowed self, borrowed input: Tensor) -> Tensor:
        # Minimum implementation to pass test
        return input @ self.weight.T + self.bias

# Run test again - should PASS
# pixi run pytest tests/test_linear.mojo::test_linear_layer_forward

# Step 3: Refactor (REFACTOR)
# Add error checking, optimize, document
fn forward(borrowed self, borrowed input: Tensor) raises -> Tensor:
    """
    Forward pass through linear layer.

    Args:
        input: Input tensor of shape [batch_size, input_size]

    Returns:
        Output tensor of shape [batch_size, output_size]

    Raises:
        ValueError: If input shape doesn't match layer input_size
    """
    if input.shape[1] != self.weight.shape[1]:
        raise ValueError("Input size mismatch")

    # Optimized matrix multiplication
    var result = matmul_simd(input, self.weight.T)
    result = result + self.bias.broadcast_to(result.shape)
    return result

# Run test again - should still PASS
# pixi run pytest tests/test_linear.mojo::test_linear_layer_forward
```

**Test organization patterns:**

```text
tests/
├── shared/
│   ├── core/
│   │   ├── test_layers.mojo      # Layer tests
│   │   ├── test_activations.mojo # Activation function tests
│   │   └── test_tensor_ops.mojo  # Tensor operation tests
│   ├── training/
│   │   ├── test_optimizers.mojo  # Optimizer tests
│   │   └── test_losses.mojo      # Loss function tests
│   └── data/
│       ├── test_datasets.mojo    # Dataset tests
│       └── test_loaders.mojo     # Data loader tests
├── integration/
│   ├── test_end_to_end.mojo     # Full pipeline tests
│   └── test_paper_reproduction.mojo
└── conftest.py                   # Shared fixtures
```

### Phase 3: Implementation - Code Organization

**Module implementation pattern:**

```mojo
# File: shared/core/layers.mojo

"""
Neural network layers with automatic gradient computation.

This module provides common layer types:
- Linear: Fully connected layer
- Conv2D: 2D convolutional layer
- MaxPool2D: Max pooling layer
"""

from shared.core.types import Tensor
from shared.core.ops import matmul, conv2d

# 1. Define interfaces first
trait Layer:
    """Base trait for all neural network layers."""
    fn forward(inout self, borrowed input: Tensor) -> Tensor:
        """Forward pass through the layer."""
        ...

    fn parameters(inout self) -> List[Tensor]:
        """Get trainable parameters."""
        ...

# 2. Implement concrete types
struct Linear(Layer):
    """Fully connected linear layer."""
    var weight: Tensor
    var bias: Tensor

    fn __init__(inout self, input_size: Int, output_size: Int):
        """Initialize layer with random weights."""
        self.weight = Tensor.randn(output_size, input_size) * sqrt(2.0 / input_size)
        self.bias = Tensor.zeros(output_size)

    fn forward(inout self, borrowed input: Tensor) -> Tensor:
        """Compute output = input @ weight.T + bias."""
        return matmul(input, self.weight.T) + self.bias

    fn parameters(inout self) -> List[Tensor]:
        """Return [weight, bias]."""
        return [self.weight, self.bias]

# 3. Export public API
alias __all__ = ["Layer", "Linear", "Conv2D", "MaxPool2D"]
```

**Code quality checklist:**

- [ ] All public functions have docstrings
- [ ] Type hints on all function signatures
- [ ] Error handling for invalid inputs
- [ ] SIMD optimization where applicable
- [ ] Memory safety (proper use of owned/borrowed)
- [ ] Follows Mojo best practices (fn over def, struct over class)
- [ ] Code formatted with `mojo format`
- [ ] No compiler warnings

### Phase 4: Packaging - Integration Steps

**Integration checklist:**

```bash
# 1. Verify module structure
tree shared/
# shared/
# ├── __init__.mojo          # Re-export public APIs
# ├── core/
# │   ├── __init__.mojo      # Export core types
# │   ├── layers.mojo
# │   └── activations.mojo
# ├── training/
# │   └── ...
# └── data/
#     └── ...

# 2. Create examples
mkdir -p examples/core/
cat > examples/core/linear_example.mojo << 'EOF'
from shared.core import Linear
from shared.core.types import Tensor

fn main():
    # Create layer
    var layer = Linear(input_size=784, output_size=10)

    # Forward pass
    var input = Tensor.randn(32, 784)
    var output = layer.forward(input)

    print("Output shape:", output.shape)  # [32, 10]
EOF

# 3. Test examples work
pixi run mojo examples/core/linear_example.mojo

# 4. Update documentation
# Add usage examples to docs/core/shared-library.md

# 5. Create benchmarks
mkdir -p benchmarks/
cat > benchmarks/bench_linear.mojo << 'EOF'
from shared.core import Linear
from shared.utils import Profiler

fn benchmark_linear():
    var profiler = Profiler()
    var layer = Linear(1024, 1024)
    var input = Tensor.randn(128, 1024)

    with profiler.section("forward"):
        for _ in range(100):
            _ = layer.forward(input)

    profiler.print_summary()
EOF

# 6. Run benchmarks
pixi run mojo benchmarks/bench_linear.mojo
```

### Phase 5: Cleanup - Refactoring Guide

**Refactoring patterns:**

```mojo
# BEFORE: Repetitive code
fn forward_conv1(borrowed self, borrowed x: Tensor) -> Tensor:
    var out = self.conv1.forward(x)
    out = relu(out)
    out = self.pool1.forward(out)
    return out

fn forward_conv2(borrowed self, borrowed x: Tensor) -> Tensor:
    var out = self.conv2.forward(x)
    out = relu(out)
    out = self.pool2.forward(out)
    return out

# AFTER: Extract common pattern
fn conv_block(borrowed self, borrowed x: Tensor,
              borrowed conv: Conv2D, borrowed pool: MaxPool2D) -> Tensor:
    """Apply convolution -> ReLU -> pooling block."""
    var out = conv.forward(x)
    out = relu(out)
    out = pool.forward(out)
    return out

fn forward(borrowed self, borrowed x: Tensor) -> Tensor:
    var x = self.conv_block(x, self.conv1, self.pool1)
    var x = self.conv_block(x, self.conv2, self.pool2)
    return x
```

**Performance optimization:**

```mojo
# BEFORE: Naive implementation
fn matmul_naive(borrowed a: Tensor, borrowed b: Tensor) -> Tensor:
    var result = Tensor.zeros(a.shape[0], b.shape[1])
    for i in range(a.shape[0]):
        for j in range(b.shape[1]):
            for k in range(a.shape[1]):
                result[i, j] += a[i, k] * b[k, j]
    return result

# AFTER: SIMD optimized
fn matmul_simd(borrowed a: Tensor, borrowed b: Tensor) -> Tensor:
    var result = Tensor.zeros(a.shape[0], b.shape[1])

    @parameter
    fn simd_inner[simd_width: Int](i: Int, j: Int):
        var sum = SIMD[DType.float32, simd_width](0)
        for k in range(0, a.shape[1], simd_width):
            var a_vec = a.load[simd_width](i, k)
            var b_vec = b.load[simd_width](k, j)
            sum += a_vec * b_vec
        result[i, j] = sum.reduce_add()

    # Vectorize loops
    vectorize[simd_inner, target_width](a.shape[0], b.shape[1])
    return result
```

## Git Workflow Advanced

### Worktree Management

**Complete worktree workflow:**

```bash
# Initial setup
cd ~/ml-odyssey
git worktree add ../worktrees/issue-39-impl-data 39-impl-data

# Work in worktree
cd ../worktrees/issue-39-impl-data
# Make changes...
git add .
git commit -m "feat(data): implement TensorDataset"
git push origin 39-impl-data

# Meanwhile, work on another issue in parallel
cd ~/ml-odyssey
git worktree add ../worktrees/issue-40-test-data 40-test-data
cd ../worktrees/issue-40-test-data
# Work on tests in parallel...

# List all worktrees
git worktree list
# /home/user/ml-odyssey                     abc1234 [main]
# /home/user/worktrees/issue-39-impl-data  def5678 [39-impl-data]
# /home/user/worktrees/issue-40-test-data  ghi9012 [40-test-data]

# When done with issue #39
cd /home/user/worktrees/issue-39-impl-data
# Ensure everything is committed
git status
# Create PR
gh pr create --issue 39

# After PR is merged, clean up worktree
cd ~/ml-odyssey
git worktree remove ../worktrees/issue-39-impl-data
git branch -d 39-impl-data  # Delete local branch
git fetch --prune          # Clean up remote tracking
```

**Worktree best practices:**

- One worktree per issue/feature
- Keep worktrees in separate directory (not nested in main repo)
- Clean up worktrees after PRs are merged
- Use absolute paths for worktree commands
- Don't create worktrees for quick fixes (use main repo)

### Advanced Git Commands

**Interactive rebase for clean history:**

```bash
# Squash commits before PR
git rebase -i main

# In editor, change 'pick' to 'squash' for commits to combine:
# pick abc1234 feat(data): add TensorDataset
# squash def5678 fix typo
# squash ghi9012 add docstring
# pick jkl3456 feat(data): add BatchLoader

# Force push (use with caution!)
git push --force-with-lease

# Cherry-pick commits from another branch
git cherry-pick abc1234
```

**Stash management:**

```bash
# Stash changes with message
git stash push -m "WIP: refactoring layer interface"

# List stashes
git stash list
# stash@{0}: On 39-impl-data: WIP: refactoring layer interface
# stash@{1}: On main: experimental feature

# Apply specific stash
git stash apply stash@{0}

# Apply and delete stash
git stash pop

# Create branch from stash
git stash branch experimental-feature stash@{1}
```

## Pull Request Best Practices

### PR Description Template

```markdown
## Summary

Brief description of changes (1-2 sentences).

## Changes

- **Added**: New TensorDataset class for in-memory data
- **Modified**: BatchLoader to support custom samplers
- **Fixed**: Memory leak in data loader iteration
- **Removed**: Deprecated DataIterator class

## Testing

- [ ] All tests pass locally
- [ ] Added tests for new functionality
- [ ] Updated documentation
- [ ] Ran benchmarks (no performance regression)

## Checklist

- [ ] Code follows Mojo style guide
- [ ] Docstrings added for all public APIs
- [ ] Pre-commit hooks pass
- [ ] Linked to issue #39

## Screenshots/Examples

```mojo
# Example usage
var dataset = TensorDataset(X_train, y_train)
var loader = BatchLoader(dataset, batch_size=32)

for batch in loader:
    var inputs, targets = batch
    # Process batch...
```

## Performance Impact

| Operation | Before | After | Change |
|-----------|--------|-------|--------|
| Batch loading | 5.2ms | 4.8ms | -7.7% ✅ |
| Dataset iteration | 0.8ms | 0.8ms | 0.0% |

## Related Issues

Closes #39
Related to #34, #38

## Review Notes

Focus on:
- Memory safety in data loader iteration
- SIMD optimizations in batch collation
```

### Addressing Review Feedback

**Complete feedback resolution workflow:**

```bash
# 1. Get all review comments
gh api repos/mvillmow/ml-odyssey/pulls/1588/comments \
  --jq '.[] | {id: .id, path: .path, line: .line, body: .body}'

# Output:
# {
#   "id": 123456,
#   "path": "shared/data/datasets.mojo",
#   "line": 45,
#   "body": "Consider using borrowed parameter here for better performance"
# }

# 2. Make the requested change
vim shared/data/datasets.mojo
# Change: fn __getitem__(inout self, idx: Int) -> Tensor:
# To:     fn __getitem__(borrowed self, idx: Int) -> Tensor:

# 3. Commit the fix
git add shared/data/datasets.mojo
git commit -m "fix(data): use borrowed parameter in __getitem__ for performance"

# 4. Push changes
git push

# 5. Reply to the review comment
gh api repos/mvillmow/ml-odyssey/pulls/1588/comments/123456/replies \
  --method POST \
  -f body="✅ Fixed - Changed to borrowed parameter in __getitem__ method"

# 6. Verify reply was posted
gh api repos/mvillmow/ml-odyssey/pulls/1588/comments \
  --jq '.[] | select(.in_reply_to_id == 123456) | {body: .body}'

# 7. Wait for CI to complete
sleep 30
gh pr checks 1588

# Output:
# All checks have passed
# ✓ test (2m 34s)
# ✓ pre-commit (1m 12s)
# ✓ validate-agents (45s)
```

## Issue Management Details

### Issue Templates

**Planning phase issue template:**

```markdown
# [Plan] Component Name

## Overview

Brief description of what will be planned.

## Deliverables

- [ ] Component specification document
- [ ] Architecture diagrams
- [ ] API design
- [ ] Test strategy
- [ ] Implementation plan

## Dependencies

- Issue #XX must be complete first
- Requires research on topic Y

## Timeline

Estimated: 2 days

## Acceptance Criteria

- [ ] Specification reviewed and approved
- [ ] All questions answered in planning doc
- [ ] Test strategy covers all scenarios
- [ ] Implementation plan is clear and actionable
```

**Implementation phase issue template:**

```markdown
# [Impl] Component Name

## Overview

Implement the component according to the specification from Issue #XX.

## Specification

Link to planning doc: /notes/issues/XX/README.md

## Deliverables

- [ ] Module implementation
- [ ] Unit tests (from Test phase)
- [ ] Integration tests
- [ ] Documentation
- [ ] Examples

## Implementation Notes

(Fill in during implementation)

## Acceptance Criteria

- [ ] All tests pass
- [ ] Code coverage ≥ 90%
- [ ] Code reviewed and approved
- [ ] Documentation complete
- [ ] Examples working
```

## Code Review Deep Dive

### Review Process Details

**Reviewer responsibilities:**

1. **Understand context**: Read linked issue and planning docs
2. **Check functionality**: Does the code do what it's supposed to?
3. **Test coverage**: Are there adequate tests?
4. **Code quality**: Does it follow best practices?
5. **Performance**: Are there obvious performance issues?
6. **Documentation**: Are public APIs documented?
7. **Security**: Are there any security concerns?

**Review checklist (detailed):**

```markdown
## Functionality
- [ ] Code matches specification from planning phase
- [ ] Edge cases handled
- [ ] Error handling appropriate
- [ ] No hardcoded values (use config)

## Testing
- [ ] Unit tests cover all public methods
- [ ] Integration tests for component interaction
- [ ] Edge cases tested
- [ ] Tests are deterministic (no flaky tests)
- [ ] Code coverage ≥ 90%

## Code Quality
- [ ] Follows Mojo style guide (fn, struct, borrowed/owned)
- [ ] No compiler warnings
- [ ] Type hints on all functions
- [ ] Docstrings for all public APIs
- [ ] Meaningful variable names
- [ ] Functions are focused (single responsibility)
- [ ] No code duplication

## Performance
- [ ] SIMD used where appropriate
- [ ] No unnecessary allocations
- [ ] Efficient algorithms (correct complexity)
- [ ] No performance regressions (benchmarks)

## Documentation
- [ ] Module-level docstring
- [ ] Function docstrings with parameters and returns
- [ ] Examples in docstrings for complex functions
- [ ] Updated user documentation
- [ ] Changelog updated

## Security
- [ ] Input validation
- [ ] No buffer overflows
- [ ] Safe type conversions
- [ ] No unsafe memory access
```

## Advanced Troubleshooting

### Debugging Test Failures

**Step-by-step debugging:**

```bash
# 1. Reproduce failure locally
pixi run pytest tests/test_failing.mojo -v -s

# 2. Run single test with maximum verbosity
pixi run pytest tests/test_failing.mojo::test_specific_case -vv -s --tb=long

# 3. Add debug prints
# In test file:
fn test_specific_case():
    var layer = Linear(10, 5)
    print("Layer created:", layer)

    var input = Tensor.randn(8, 10)
    print("Input shape:", input.shape)
    print("Input data:", input)

    var output = layer.forward(input)
    print("Output shape:", output.shape)
    print("Output data:", output)

    assert_equal(output.shape, [8, 5])

# 4. Use Python debugger for test infrastructure
pixi run pytest tests/test_failing.mojo --pdb
# Will drop into debugger on failure

# 5. Check for environment issues
python -c "import sys; print(sys.version)"
pixi run mojo --version
pixi list

# 6. Clear caches and retry
rm -rf .pytest_cache __pycache__
pixi clean
pixi install
pixi run pytest tests/test_failing.mojo
```

### Resolving Merge Conflicts

**Conflict resolution workflow:**

```bash
# 1. Update main branch
git fetch origin
git checkout main
git pull

# 2. Attempt rebase
git checkout 39-impl-data
git rebase main

# Output:
# CONFLICT (content): Merge conflict in shared/data/datasets.mojo
# error: could not apply abc1234... feat(data): implement TensorDataset

# 3. Examine conflict
git status
# On branch 39-impl-data
# You are currently rebasing branch '39-impl-data' on 'def5678'.
# (fix conflicts and then run "git rebase --continue")
#
# Unmerged paths:
#   both modified:   shared/data/datasets.mojo

# 4. Open file and resolve conflict
vim shared/data/datasets.mojo

# File shows:
# <<<<<<< HEAD (main branch)
# struct TensorDataset:
#     var data: List[Tensor]
# =======
# struct TensorDataset:
#     var X: Tensor
#     var y: Tensor
# >>>>>>> abc1234... feat(data): implement TensorDataset

# Resolve to:
# struct TensorDataset:
#     var X: Tensor
#     var y: Tensor
#     # Updated to match new interface from main

# 5. Mark as resolved
git add shared/data/datasets.mojo

# 6. Continue rebase
git rebase --continue

# 7. Test after resolution
pixi run pytest tests/

# 8. Force push (rebase rewrites history)
git push --force-with-lease
```

### CI/CD Troubleshooting

**Debugging CI failures:**

```bash
# 1. Check CI logs
gh pr checks 1588
# Failing check: test
gh run view $(gh run list --branch 39-impl-data --limit 1 --json databaseId --jq '.[0].databaseId')

# 2. Download logs
gh run download $(gh run list --branch 39-impl-data --limit 1 --json databaseId --jq '.[0].databaseId')

# 3. Reproduce CI environment locally with Act
act -j test

# 4. Or use Docker to match CI environment exactly
docker run -it --rm -v $(pwd):/workspace ubuntu:latest /bin/bash
cd /workspace
# Install dependencies and run tests

# 5. Check for CI-specific issues
# - Environment variables
# - File permissions
# - Network access
# - Timing issues

# 6. Fix and push
git add .
git commit -m "fix(ci): resolve environment-specific test failure"
git push
```

## Complete Examples

### Example: End-to-End Feature Development

**Scenario**: Implement a new optimizer (AdamW)

```bash
# PHASE 1: PLANNING (Issue #150)
# Create planning doc
mkdir -p notes/issues/150
cat > notes/issues/150/README.md << 'EOF'
# Issue #150: [Plan] AdamW Optimizer

## Objective

Plan implementation of AdamW optimizer (Adam with decoupled weight decay).

## Specification

Based on "Decoupled Weight Decay Regularization" (Loshchilov & Hutter, 2019)

### Algorithm
1. Update biased first moment: m_t = β1 * m_{t-1} + (1 - β1) * g_t
2. Update biased second moment: v_t = β2 * v_{t-1} + (1 - β2) * g_t²
3. Correct bias: m̂_t = m_t / (1 - β1^t), v̂_t = v_t / (1 - β2^t)
4. Update parameters: θ_t = θ_{t-1} - α * (m̂_t / (√v̂_t + ε) + λ * θ_{t-1})

### API Design

```mojo
struct AdamW(Optimizer):
    var learning_rate: Float64
    var beta1: Float64
    var beta2: Float64
    var epsilon: Float64
    var weight_decay: Float64
    var t: Int  # Time step

    fn __init__(inout self, learning_rate: Float64 = 0.001,
                beta1: Float64 = 0.9, beta2: Float64 = 0.999,
                epsilon: Float64 = 1e-8, weight_decay: Float64 = 0.01):
        ...

    fn step(inout self, inout param: Tensor, borrowed grad: Tensor):
        """Update parameter using AdamW algorithm."""
        ...
```

## Deliverables

- [x] Algorithm specification
- [x] API design
- [x] Test plan
- [ ] Implementation plan
EOF

# Complete planning...
# Issue #150 closes when planning is approved

# PHASE 2: TESTING (Issue #151 - runs in parallel with Impl)
# Write tests first
cat > tests/shared/training/test_adamw.mojo << 'EOF'
from testing import assert_equal, assert_close
from shared.training import AdamW
from shared.core.types import Tensor

fn test_adamw_init():
    """Test AdamW initialization."""
    var opt = AdamW(learning_rate=0.001)
    assert_equal(opt.learning_rate, 0.001)
    assert_equal(opt.beta1, 0.9)
    assert_equal(opt.beta2, 0.999)

fn test_adamw_step():
    """Test single optimization step."""
    var opt = AdamW(learning_rate=0.1, weight_decay=0.01)
    var param = Tensor([1.0, 2.0, 3.0])
    var grad = Tensor([0.1, 0.2, 0.3])

    opt.step(param, grad)

    # Parameters should decrease (gradient descent)
    assert_true(param[0] < 1.0)
    assert_true(param[1] < 2.0)
    assert_true(param[2] < 3.0)

fn test_adamw_convergence():
    """Test AdamW converges on simple problem."""
    var opt = AdamW(learning_rate=0.01)
    var param = Tensor([10.0])  # Start far from optimum

    # Optimize: minimize (param - 2)^2
    for _ in range(100):
        var grad = 2 * (param - 2.0)  # Gradient
        opt.step(param, grad)

    # Should converge to 2.0
    assert_close(param[0], 2.0, rtol=0.01)
EOF

# Tests fail initially (implementation doesn't exist)
# pixi run pytest tests/shared/training/test_adamw.mojo  # FAILS ❌

# PHASE 3: IMPLEMENTATION (Issue #152 - parallel with Test)
cat > shared/training/adamw.mojo << 'EOF'
"""AdamW optimizer implementation."""

from shared.core.types import Tensor
from memory import memset_zero
from math import sqrt, pow

struct AdamW:
    """
    AdamW optimizer with decoupled weight decay.

    Reference: "Decoupled Weight Decay Regularization"
    Loshchilov & Hutter, ICLR 2019
    """
    var learning_rate: Float64
    var beta1: Float64
    var beta2: Float64
    var epsilon: Float64
    var weight_decay: Float64
    var t: Int

    # State variables (stored per parameter)
    var m: Dict[Int, Tensor]  # First moment
    var v: Dict[Int, Tensor]  # Second moment

    fn __init__(inout self, learning_rate: Float64 = 0.001,
                beta1: Float64 = 0.9, beta2: Float64 = 0.999,
                epsilon: Float64 = 1e-8, weight_decay: Float64 = 0.01):
        """Initialize AdamW optimizer."""
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.weight_decay = weight_decay
        self.t = 0
        self.m = Dict[Int, Tensor]()
        self.v = Dict[Int, Tensor]()

    fn step(inout self, inout param: Tensor, borrowed grad: Tensor):
        """
        Perform single optimization step.

        Args:
            param: Parameter to update (modified in-place)
            grad: Gradient of loss w.r.t. parameter
        """
        self.t += 1
        var param_id = id(param)

        # Initialize moments if needed
        if param_id not in self.m:
            self.m[param_id] = Tensor.zeros_like(param)
            self.v[param_id] = Tensor.zeros_like(param)

        var m = self.m[param_id]
        var v = self.v[param_id]

        # Update biased first moment
        m = self.beta1 * m + (1.0 - self.beta1) * grad

        # Update biased second moment
        v = self.beta2 * v + (1.0 - self.beta2) * grad * grad

        # Bias correction
        var m_hat = m / (1.0 - pow(self.beta1, self.t))
        var v_hat = v / (1.0 - pow(self.beta2, self.t))

        # Update parameters with decoupled weight decay
        param = param - self.learning_rate * (
            m_hat / (sqrt(v_hat) + self.epsilon) +
            self.weight_decay * param
        )

        # Store updated moments
        self.m[param_id] = m
        self.v[param_id] = v
EOF

# Now tests pass
# pixi run pytest tests/shared/training/test_adamw.mojo  # PASSES ✅

# PHASE 4: PACKAGING (Issue #153 - parallel with Impl)
# Create example
cat > examples/training/adamw_example.mojo << 'EOF'
"""Example of using AdamW optimizer."""

from shared.training import AdamW
from shared.core import Linear
from shared.data import TensorDataset, BatchLoader

fn main():
    # Create model
    var model = Linear(input_size=784, output_size=10)

    # Create optimizer
    var optimizer = AdamW(
        learning_rate=0.001,
        weight_decay=0.01
    )

    # Training loop
    for epoch in range(10):
        for batch in train_loader:
            var inputs, targets = batch

            # Forward pass
            var outputs = model.forward(inputs)
            var loss = cross_entropy_loss(outputs, targets)

            # Backward pass
            loss.backward()

            # Update parameters
            for param in model.parameters():
                optimizer.step(param, param.grad)
                param.grad.zero_()

        print("Epoch", epoch, "complete")
EOF

# Test example works
# pixi run mojo examples/training/adamw_example.mojo

# Update documentation
# Add to docs/core/shared-library.md

# PHASE 5: CLEANUP (Issue #154 - after parallel phases complete)
# Review feedback: "Add learning rate scheduling support"
# Refactor to support dynamic learning rate
# ...
# Address all review comments
# Final optimization and polish

# Create PR and merge
gh pr create --issue 152 --title "feat(training): implement AdamW optimizer" \
  --body "Implements AdamW optimizer with decoupled weight decay. Closes #152"
```

## Testing Workflow Details

(See core documentation for testing strategy details)

## Documentation Workflow Details

(See core documentation for documentation standards)

## CI/CD Details

(See dev documentation for CI/CD pipeline details)

This appendix provides complete details for the development workflow. For quick reference,
see [Development Workflow](../core/workflow.md).
