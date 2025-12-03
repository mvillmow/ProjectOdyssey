# Agent System Usage Examples

<!-- markdownlint-disable MD051 -->

## Table of Contents

- [Overview](#overview)
- [Example 1: Planning a New Feature](#example-1-planning-a-new-feature)
- [Example 2: Implementing a Component](#example-2-implementing-a-component)
- [Example 3: Code Review Workflow](#example-3-code-review-workflow)
- [Example 4: Testing Workflow (TDD)](#example-4-testing-workflow-tdd)
- [Example 5: Documentation Workflow](#example-5-documentation-workflow)
- [Example 6: Bug Fix Workflow](#example-6-bug-fix-workflow)
- [Example 7: Performance Optimization](#example-7-performance-optimization)
- [Common Patterns](#common-patterns)
- [Troubleshooting](#troubleshooting)

## Overview

This guide provides practical examples of using the ML Odyssey agent system for common development tasks. Each example
shows:

- The initial user prompt/request
- Which agent(s) are invoked
- Expected agent responses and outputs
- Follow-up steps and coordination
- Common pitfalls to avoid

**Note**: Agents can be invoked automatically (Claude recognizes the task) or explicitly (you specify the agent).

## Example 1: Planning a New Feature

### Scenario

You want to add a new activation function (GELU) to the shared library.

### Initial Prompt

```text
I want to add GELU activation to the shared library. Can you help me plan this?
```text

### Agent Selection

**Automatic Invocation**: Chief Architect or Shared Library Orchestrator will be triggered based on "shared library"
context.

**Explicit Invocation** (recommended for clarity):

```text
Use the shared library orchestrator to plan adding GELU activation to the shared library.
```text

### Expected Agent Response

```text
Shared Library Orchestrator:

I'll help you plan the GELU activation addition. Let me break this down into components:

**Analysis:**
- GELU is a common activation function used in transformers and modern architectures
- Belongs in: src/core_ops/activations/
- Dependencies: Core tensor operations (already implemented)

**Plan:**
1. Create GitHub issue #XXX: [Plan] GELU Activation
2. Design component structure (Level 2: Architecture Design Agent)
3. Create parallel issues for Test/Implementation/Package phases
4. Follow 5-phase workflow

**Component Breakdown:**
- GELU struct with forward and backward methods
- SIMD-optimized implementation
- Unit tests comparing with PyTorch
- API documentation and usage examples

**Timeline Estimate:** 1-2 weeks

Would you like me to:
a) Create the GitHub issue and detailed plan?
b) Delegate to Architecture Design Agent for detailed design?
c) Both (recommended)?
```text

### Follow-up Steps

**Option 1**: Accept recommendation and proceed

```text
Yes, please create the issue and delegate to the design agent.
```text

**Option 2**: Request more details first

```text
Before creating the issue, can you show me the proposed API and file structure?
```text

### Expected Outputs

1. **GitHub Issue Created**: Issue #XXX with full specification
1. **Plan Document**: Posted as comments on GitHub issue #XXX with objectives, deliverables, success criteria
1. **Design Spec**: Component architecture from Architecture Design Agent (posted to issue)
1. **Child Issues**: Test (#XXX+1), Implementation (#XXX+2), Package (#XXX+3), Cleanup (#XXX+4)

### Common Pitfalls

- ❌ Skipping the planning phase and jumping to implementation
- ❌ Not creating GitHub issues (breaks traceability)
- ❌ Trying to implement in the planning phase
- ✅ Let orchestrators break down work before starting implementation

## Example 2: Implementing a Component

### Scenario

You have a design spec for BatchNorm2D and want to implement it.

### Initial Prompt

```text
I have the design spec for BatchNorm2D in issue #150. Let's implement it following the 5-phase workflow.
```text

### Agent Selection

**Automatic**: Implementation Specialist will be triggered for "implement" context.

**Explicit**:

```text
Use the implementation specialist to coordinate BatchNorm2D implementation from issue #150.
```text

### Expected Agent Response

```text
Implementation Specialist:

I'll coordinate the BatchNorm2D implementation. Let me review the spec from issue #150...

**Spec Summary:**
- BatchNorm2D struct with training/inference modes
- Methods: __init__, forward, forward_train, backward
- SIMD optimization required for _normalize_simd
- Dependencies: Tensor operations from core_ops

**Implementation Strategy:**

Phase 1 (Plan): ✅ Already complete (issue #150)

Phase 2-4 (Parallel):
- Test (Issue #151): Test Engineer writes tests first (TDD)
- Implementation (Issue #152): Senior + Standard Engineers implement
- Package (Issue #153): Documentation Engineer writes docs

**Delegation:**
- SIMD optimization (_normalize_simd) → Senior Implementation Engineer
- Standard methods (forward, backward) → Implementation Engineer
- Helper functions → Junior Implementation Engineer

**Git Worktrees:**
```bash
git worktree add worktrees/issue-151-test-batchnorm -b 151-test-batchnorm
git worktree add worktrees/issue-152-impl-batchnorm -b 152-impl-batchnorm
git worktree add worktrees/issue-153-pkg-batchnorm -b 153-pkg-batchnorm
```text

Ready to proceed?

```text
[User agrees]
```text

### Follow-up Steps

**Step 1**: Start Test Phase (TDD)

```text
Yes, let's start with the test phase. Use the test engineer to write tests for BatchNorm2D.
```text

**Step 2**: Monitor parallel execution

```text
Implementation Specialist:

Test Engineer is working on tests in issue #151.
Senior Implementation Engineer is implementing _normalize_simd in issue #152.
Documentation Engineer is drafting API docs in issue #153.

I'll notify you when all phases are ready for cleanup.
```text

**Step 3**: Coordinate integration

```text
What's the status of the parallel work?
```text

### Expected Outputs

1. **Test Files**: `tests/test_batchnorm.mojo` with comprehensive test cases
1. **Implementation**: `src/core_ops/layers/batchnorm.mojo` with SIMD optimization
1. **Documentation**: `docs/api/batchnorm.md` with usage examples
1. **Three PRs**: One for each parallel phase, all linked to parent issue

### Common Pitfalls

- ❌ Implementing before writing tests (breaks TDD)
- ❌ Working in a single branch instead of parallel worktrees
- ❌ Not delegating SIMD code to Senior Engineer
- ✅ Follow the 5-phase workflow strictly
- ✅ Use git worktrees for parallel work

## Example 3: Code Review Workflow

### Scenario

You submitted PR #300 adding a Dropout layer and need a review.

### Initial Prompt

```text
I've submitted PR #300 with a Dropout layer implementation. Can you review it?
```text

### Agent Selection

**Automatic**: Code Review Orchestrator will be triggered for "review" context.

**Explicit**:

```text
Use the code review orchestrator to review PR #300.
```text

### Expected Agent Response

```text
Code Review Orchestrator:

I'll coordinate the review of PR #300. Let me analyze the changes...

**PR Analysis:**
- Files changed: src/core_ops/layers/dropout.mojo, tests/test_dropout.mojo, docs/api/dropout.md
- Lines changed: +250 -0
- Complexity: Medium (new layer with randomness)

**Review Routing:**
I'll delegate to these specialists:

1. Implementation Review Specialist - Code correctness and Mojo idioms
2. Mojo Language Review Specialist - Memory management, fn/def usage
3. Test Review Specialist - Test coverage and quality
4. Documentation Review Specialist - API docs and examples
5. Security Review Specialist - Random number generation security

**Timeline:** Reviews will complete in parallel, expect feedback within 2-4 hours.
```text

### Follow-up Steps

**Step 1**: Wait for reviews to complete

```text
Code Review Orchestrator:

All reviews complete. Here's the consolidated feedback:

**Implementation Review Specialist:**
- ✅ Code structure looks good
- ❌ Request: Add error handling for invalid dropout rate (p < 0 or p > 1)

**Mojo Language Review Specialist:**
- ✅ Good use of `fn` for performance-critical code
- ✅ Proper memory management with `owned` parameters

**Test Review Specialist:**
- ✅ Good coverage of forward pass
- ❌ Missing: Test for inference mode (dropout should be disabled)

**Documentation Review Specialist:**
- ✅ Clear API documentation
- ❌ Request: Add performance benchmark comparison with PyTorch

**Security Review Specialist:**
- ⚠️  Warning: Random seed should be configurable for reproducibility
- ✅ No other security concerns

**Summary:** 2 required changes, 1 recommended improvement. Please address feedback.
```text

**Step 2**: Address feedback

```text
I've addressed all the feedback:
- Added error handling for invalid p
- Added inference mode test
- Added performance benchmark
- Made random seed configurable

Ready for re-review?
```text

**Step 3**: Re-review and approval

```text
Code Review Orchestrator:

All specialists have re-reviewed and approved:
- Implementation Review Specialist: ✅ Approved
- Mojo Language Review Specialist: ✅ Approved
- Test Review Specialist: ✅ Approved
- Documentation Review Specialist: ✅ Approved
- Security Review Specialist: ✅ Approved

**Recommendation:** Ready to merge.
```text

### Expected Outputs

1. **Review Comments**: Consolidated feedback from 5 specialists
1. **Updated PR**: All feedback addressed
1. **Approval**: All specialists approve
1. **Merge**: PR merged to main

### Common Pitfalls

- ❌ Submitting PR without self-review
- ❌ Arguing with reviewers instead of clarifying
- ❌ Ignoring feedback (breaks team collaboration)
- ✅ Address all feedback thoroughly
- ✅ Ask questions if feedback is unclear

## Example 4: Testing Workflow (TDD)

### Scenario

You want to implement a Conv2D layer using test-driven development.

### Initial Prompt

```text
I want to implement Conv2D using TDD. Let's write tests first.
```text

### Agent Selection

**Automatic**: Test Specialist will be triggered for "TDD" context.

**Explicit**:

```text
Use the test specialist to plan TDD for Conv2D implementation.
```text

### Expected Agent Response

```text
Test Specialist:

Excellent! TDD is the right approach for Conv2D. Let me create a comprehensive test plan.

**Test Strategy:**

1. **Unit Tests** (test individual methods):
   - test_conv2d_initialization
   - test_conv2d_forward_shape
   - test_conv2d_forward_correctness
   - test_conv2d_backward_gradient
   - test_conv2d_parameter_update

2. **Integration Tests** (test with other layers):
   - test_conv2d_in_sequential_model
   - test_conv2d_with_batchnorm
   - test_conv2d_with_pooling

3. **Performance Tests**:
   - test_conv2d_simd_speedup
   - benchmark_conv2d_vs_pytorch

4. **Edge Cases**:
   - test_conv2d_single_channel
   - test_conv2d_rectangular_kernel
   - test_conv2d_padding_modes

**TDD Workflow:**
1. Write failing tests (all tests should fail initially)
2. Implement minimum code to pass tests
3. Refactor while keeping tests passing
4. Add SIMD optimization last

**Delegation:** Test Engineer will implement tests.

Ready to start?
```text

### Follow-up Steps

**Step 1**: Write failing tests

```text
Yes, let's write the tests. Use the test engineer to implement the test plan.
```text

```text
Test Engineer:

I've written tests in tests/test_conv2d.mojo:

fn test_conv2d_forward_shape():
    """Test Conv2D output shape is correct."""
    var conv = Conv2D(in_channels=3, out_channels=64, kernel_size=3)
    var input = Tensor[DType.float32](1, 3, 32, 32)
    var output = conv.forward(input)

    # Expected: (1, 64, 30, 30) with no padding
    assert_equal(output.shape[0], 1)
    assert_equal(output.shape[1], 64)
    assert_equal(output.shape[2], 30)
    assert_equal(output.shape[3], 30)

fn test_conv2d_forward_correctness():
    """Test Conv2D produces correct values."""
    # Known input and kernel
    var conv = Conv2D(in_channels=1, out_channels=1, kernel_size=3)
    # Set kernel to identity-like pattern
    conv.weight.fill(0.0)
    conv.weight[0, 0, 1, 1] = 1.0  # Center pixel

    var input = Tensor[DType.float32](1, 1, 5, 5)
    input.fill(5.0)

    var output = conv.forward(input)
    # Center should be ~5.0
    assert_close(output[0, 0, 1, 1], 5.0, atol=0.01)

All tests currently FAIL (no implementation yet).
This is expected for TDD!

Next: Implement Conv2D to make tests pass.
```text

**Step 2**: Implement to pass tests

```text
Use the implementation engineer to implement Conv2D to pass the tests.
```text

**Step 3**: Refactor and optimize

```text
Tests are passing! Now use the senior implementation engineer to add SIMD optimization.
```text

### Expected Outputs

1. **Test Suite**: Comprehensive tests written before implementation
1. **Initial Implementation**: Basic implementation that passes tests
1. **Optimized Implementation**: SIMD-optimized version (tests still pass)
1. **Confidence**: High confidence in correctness due to TDD

### Common Pitfalls

- ❌ Writing implementation before tests
- ❌ Writing tests that don't actually test the behavior
- ❌ Optimizing before tests pass
- ✅ Always write tests first
- ✅ Make tests fail initially to verify they work

## Example 5: Documentation Workflow

### Scenario

You need to document the newly implemented BatchNorm2D layer.

### Initial Prompt

```text
I need comprehensive documentation for the BatchNorm2D layer we just implemented.
```text

### Agent Selection

**Automatic**: Documentation Specialist will be triggered.

**Explicit**:

```text
Use the documentation specialist to create docs for BatchNorm2D.
```text

### Expected Agent Response

```text
Documentation Specialist:

I'll create comprehensive documentation for BatchNorm2D. Let me plan the docs structure:

**Documentation Plan:**

1. **API Reference** (docs/api/batchnorm.md):
   - Class overview
   - Constructor parameters
   - Methods (forward, forward_train, backward)
   - Mathematical background
   - Performance characteristics

2. **Usage Examples** (docs/guides/using-batchnorm.md):
   - Basic usage (training and inference)
   - Integration with models
   - Best practices
   - Common pitfalls

3. **Code Documentation** (docstrings):
   - Update BatchNorm2D struct docstring
   - Document all methods
   - Add parameter descriptions

**Delegation:** Documentation Engineer will write the docs.

**Timeline:** 2-3 days for complete documentation.

Ready to proceed?
```text

### Follow-up Steps

**Step 1**: Generate API docs

```text
Yes, start with the API reference. Use the documentation engineer to write it.
```text

```text
Documentation Engineer:

I've created docs/api/batchnorm.md:

# BatchNorm2D

## Overview

Batch Normalization normalizes inputs across the batch dimension, reducing internal
covariate shift and enabling higher learning rates.

## Mathematical Background

For input x, batch normalization computes:

    y = γ * (x - μ) / √(σ² + ε) + β

Where:
- μ: batch mean
- σ²: batch variance
- γ: learned scale parameter
- β: learned shift parameter
- ε: numerical stability constant

## API Reference

### Constructor

```mojo
BatchNorm2D(
    num_features: Int,
    eps: Float32 = 1e-5,
    momentum: Float32 = 0.1
)
```text

[... full API docs ...]

Would you like me to continue with usage examples?

```text
[End of API docs]
```text

**Step 2**: Add usage examples

```text
Yes, add comprehensive usage examples.
```text

**Step 3**: Update code docstrings

```text
Finally, update the docstrings in the batchnorm.mojo source file.
```text

### Expected Outputs

1. **API Documentation**: Complete reference in `docs/api/batchnorm.md`
1. **Usage Guide**: Practical examples in `docs/guides/using-batchnorm.md`
1. **Code Docstrings**: Updated source code documentation
1. **Updated README**: `src/core_ops/README.md` mentions BatchNorm2D

### Common Pitfalls

- ❌ Writing docs after implementation is complete (docs are stale)
- ❌ Copying docstrings from PyTorch without Mojo-specific details
- ❌ Missing usage examples
- ✅ Write docs in parallel with implementation (Package phase)
- ✅ Include Mojo-specific patterns (fn, owned, SIMD)

## Example 6: Bug Fix Workflow

### Scenario

Conv2D produces incorrect output for rectangular kernels (3x5).

### Initial Prompt

```text
Bug: Conv2D fails with rectangular kernels. 3x5 kernel produces wrong output shape.
```text

### Agent Selection

**Automatic**: Implementation Specialist will handle bug analysis.

**Explicit**:

```text
Use the implementation specialist to analyze and fix the Conv2D rectangular kernel bug.
```text

### Expected Agent Response

```text
Implementation Specialist:

I'll analyze the rectangular kernel bug. Let me investigate...

**Analysis:**
- Reviewed Conv2D implementation in src/core_ops/layers/conv2d.mojo
- Root cause: kernel_size parameter assumes square kernels
- Line 45: `for ky in range(self.kernel_size):` assumes height = width

**Fix Plan:**
1. Update Conv2D struct to have kernel_height and kernel_width
2. Fix _apply_kernel_simd to use separate dimensions
3. Update tests to cover rectangular kernels

**Minimal Fix Strategy (following minimal changes principle):**
- Change 1: Add kernel_height, kernel_width fields
- Change 2: Update __init__ to accept both or single kernel_size
- Change 3: Fix loop in _apply_kernel_simd (2 lines)

**Timeline:** 2-4 hours for fix + tests

**Delegation:**
- Test Engineer: Write failing test for rectangular kernels
- Implementation Engineer: Apply the fix
- Test Engineer: Verify fix works

Ready to proceed?
```text

### Follow-up Steps

**Step 1**: Write failing test first

```text
Yes, start with a failing test. Use the test engineer to add a test for rectangular kernels.
```text

```text
Test Engineer:

Added test in tests/test_conv2d.mojo:

fn test_conv2d_rectangular_kernel():
    """Test Conv2D with non-square kernel."""
    var conv = Conv2D(in_channels=1, out_channels=1, kernel_height=3, kernel_width=5)
    var input = Tensor[DType.float32](1, 1, 28, 28)
    var output = conv.forward(input)

    # Expected output shape: (1, 1, 26, 24)
    # height: 28 - 3 + 1 = 26
    # width: 28 - 5 + 1 = 24
    assert_equal(output.shape[2], 26)
    assert_equal(output.shape[3], 24)  # Currently FAILS ❌

Test result: FAILED (as expected)
```text

**Step 2**: Apply the fix

```text
Good. Now use the implementation engineer to fix the bug.
```text

```text
Implementation Engineer:

Fixed in src/core_ops/layers/conv2d.mojo:

# Changed struct definition:
struct Conv2D:
    var kernel_height: Int
    var kernel_width: Int
    # ... other fields

# Updated __init__:
fn __init__(inout self, in_channels: Int, out_channels: Int,
            kernel_height: Int, kernel_width: Int):
    self.kernel_height = kernel_height
    self.kernel_width = kernel_width
    # ...

# Fixed loop in _apply_kernel_simd:
for ky in range(self.kernel_height):  # Was: kernel_size
    for kx in range(self.kernel_width):  # Was: kernel_size
        # ...

Test result: PASSED ✅

All existing tests still pass ✅
```text

**Step 3**: Cleanup and finalize

```text
Great! Add edge case tests and update documentation.
```text

### Expected Outputs

1. **Bug Fix**: Conv2D now supports rectangular kernels
1. **Tests**: New test for rectangular kernels + edge cases
1. **Documentation**: Updated to mention rectangular kernel support
1. **PR**: Small, focused PR fixing just this bug

### Common Pitfalls

- ❌ Making broad refactors when fixing a bug
- ❌ Not writing a failing test first
- ❌ Fixing without understanding root cause
- ✅ Minimal changes only (KISS principle)
- ✅ Test first, then fix

## Example 7: Performance Optimization

### Scenario

Conv2D forward pass is slower than expected. Need 2x speedup.

### Initial Prompt

```text
Conv2D is too slow. Can we optimize it to be 2x faster?
```text

### Agent Selection

**Automatic**: Performance Specialist will be triggered.

**Explicit**:

```text
Use the performance specialist to optimize Conv2D forward pass for 2x speedup.
```text

### Expected Agent Response

```text
Performance Specialist:

I'll analyze Conv2D performance and create an optimization plan.

**Current Performance:**
- Profiled Conv2D forward pass
- Bottleneck: Inner convolution loop (60% of execution time)
- Input: 32x3x224x224, Kernel: 3x3, takes 85ms

**Optimization Opportunities:**
1. Better SIMD vectorization (current: partial, target: full)
2. Cache tiling for large inputs (current: none)
3. Loop reordering (current: suboptimal)
4. Function inlining (current: some)

**Optimization Plan:**
1. Profile baseline performance
2. Optimize SIMD vectorization (expected: 1.5x speedup)
3. Add cache tiling (expected: 1.3x additional speedup)
4. Benchmark and validate correctness

**Target:** 2.0x speedup (85ms → 42ms)

**Delegation:**
- Performance Engineer: Implement optimizations
- Test Engineer: Verify correctness maintained

**Timeline:** 1-2 weeks

Ready to proceed?
```text

### Follow-up Steps

**Step 1**: Baseline benchmarks

```text
Yes, start with baseline benchmarks. Use the performance engineer to profile current performance.
```text

```text
Performance Engineer:

Created baseline benchmarks in benchmarks/bench_conv2d.mojo:

Benchmark Results (32x3x224x224 input, 64x3x3x3 kernel):
- Forward pass: 85.3ms
- Memory bandwidth: 12.4 GB/s
- SIMD utilization: 45%

Baseline established. Ready for optimization.
```text

**Step 2**: Optimize SIMD

```text
Good. Now optimize the SIMD vectorization.
```text

```text
Performance Engineer:

Optimized SIMD in _apply_kernel_simd:

Before:
    for h in range(height):
        for w in range(width):
            result[h, w] = compute_pixel(h, w)  # Scalar

After:
    @parameter
    fn vectorized_compute[simd_width: Int](h: Int, w: Int):
        let val = input.load[simd_width](h, w)
        let conv = kernel.dot[simd_width](val)
        output.store[simd_width](h, w, conv)

    for h in range(height):
        vectorize[vectorized_compute, simdwidthof[DType.float32]()](width)

Benchmark Results:
- Forward pass: 52.1ms (1.64x speedup) ✅
- SIMD utilization: 87%

All tests still pass ✅
```text

**Step 3**: Add cache tiling

```text
Great progress! Now add cache tiling to hit the 2x target.
```text

```text
Performance Engineer:

Added cache tiling (8x8 tiles) in forward method:

alias tile_size = 8

for h_tile in range(0, height, tile_size):
    for w_tile in range(0, width, tile_size):
        # Process 8x8 tile for better cache locality
        process_tile_simd[tile_size](h_tile, w_tile)

Benchmark Results:
- Forward pass: 39.2ms (2.18x speedup) ✅✅
- Cache hit rate: 94% (was 78%)

TARGET ACHIEVED: 2.18x speedup (target was 2.0x)

All tests still pass ✅
```text

### Expected Outputs

1. **Optimized Code**: Conv2D with SIMD + tiling optimizations
1. **Benchmarks**: 2.18x speedup validated
1. **Correctness**: All tests pass (no regressions)
1. **Documentation**: Performance characteristics updated

### Common Pitfalls

- ❌ Optimizing without profiling (guessing bottlenecks)
- ❌ Breaking correctness for performance
- ❌ Premature optimization
- ✅ Profile first, then optimize
- ✅ Validate correctness after every optimization

## Common Patterns

### Pattern 1: Starting New Work

**Always start with an orchestrator:**

```text
Use the [section] orchestrator to plan [new feature/component].
```text

**Don't start with engineers:**

```text
❌ Use the implementation engineer to add GELU activation.
```text

### Pattern 2: Delegating Complex Work

**Let specialists delegate to engineers:**

```text
✅ Use the implementation specialist to coordinate SIMD optimization.
```text

**Don't skip levels:**

```text
❌ Use the junior engineer to optimize SIMD code.
```text

### Pattern 3: Following 5-Phase Workflow

**Always go through phases:**

```text
1. Plan → Design spec
2. Test → Write tests (TDD)
3. Implementation → Write code
4. Package → Write docs
5. Cleanup → Refactor and polish
```text

**Don't skip phases:**

```text
❌ Implement without planning
❌ Document after everything is done
❌ Skip cleanup phase
```text

### Pattern 4: Creating GitHub Issues

**Always create issues for trackability:**

```text
✅ Create issue #XXX for [component/feature]
✅ Link PRs to issues
✅ Update issue status
```text

**Don't work without issues:**

```text
❌ Just implement without tracking
```text

### Pattern 5: Parallel Work

**Use git worktrees for parallel phases:**

```bash
git worktree add worktrees/issue-X-test -b X-test
git worktree add worktrees/issue-Y-impl -b Y-impl
git worktree add worktrees/issue-Z-pkg -b Z-pkg
```text

**Don't do parallel work in one branch:**

```text
❌ Mixing test/impl/docs in single branch
```text

## Troubleshooting

### Issue: Agent not invoked automatically

**Symptom**: You ask for something but Claude doesn't invoke the right agent.

**Solution**:

```text
# Be explicit:
Use the [agent name] to [task].

# Example:
Use the test specialist to plan tests for Conv2D.
```text

### Issue: Agent delegates to wrong level

**Symptom**: Orchestrator delegates to junior engineer for complex task.

**Solution**:

```text
# Specify the right level:
Use the senior implementation engineer for SIMD optimization.

# Or let specialist decide:
Use the implementation specialist to delegate this work appropriately.
```text

### Issue: Work not following 5-phase workflow

**Symptom**: Implementation happening in plan phase, or docs skipped entirely.

**Solution**:

```text
# Enforce workflow explicitly:
Let's follow the 5-phase workflow:
1. Plan phase complete ✅
2. Now start Test phase - use the test engineer
3. Then Implementation phase (parallel with Test)
4. Then Package phase (documentation)
5. Finally Cleanup phase
```text

### Issue: No GitHub issues created

**Symptom**: Work is happening but not tracked in GitHub.

**Solution**:

```text
# Always request issue creation:
Before we start, create a GitHub issue for this work.

# Or:
Use the [orchestrator] to plan this and create the necessary GitHub issues.
```text

### Issue: Changes too broad (not minimal)

**Symptom**: Agent wants to refactor everything when fixing a small bug.

**Solution**:

```text
# Enforce minimal changes:
Apply the minimal changes principle - only fix the bug, don't refactor.

# Or:
What's the smallest change that fixes this issue?
```text

## Related Resources

- [Agent Hierarchy](../hierarchy.md) - Complete agent reference
- [Workflows Guide](./workflows.md) - Detailed workflow examples
- [5-Phase Integration](./5-phase-integration.md) - Workflow explanation
- [Delegation Rules](../delegation-rules.md) - Coordination patterns
- [Git Worktree Guide](./git-worktree-guide.md) - Using worktrees
