# Autograd - Automatic Differentiation Framework

## Overview

Autograd provides automatic differentiation capabilities for the ML Odyssey platform, enabling neural network training without manual gradient computation. This subsection implements a computation graph tracking system (gradient tape) that records forward operations and automatically computes gradients via the chain rule, leveraging the existing 27 backward pass functions in ExTensor.

The implementation follows a tape-based approach similar to PyTorch's autograd, where operations are recorded during the forward pass and replayed in reverse during backward propagation.

## Parent Plan

[../plan.md](../plan.md) - Shared Library

## Child Plans

- [01-computation-graph/plan.md](01-computation-graph/plan.md) - Computation graph data structures
- [02-gradient-tape/plan.md](02-gradient-tape/plan.md) - Operation recording and replay system
- [03-variable/plan.md](03-variable/plan.md) - Autograd-enabled tensor wrapper (Variable)
- [04-backward-engine/plan.md](04-backward-engine/plan.md) - Automatic gradient computation engine
- [05-loss-functions/plan.md](05-loss-functions/plan.md) - Differentiable loss functions
- [06-optimizers/plan.md](06-optimizers/plan.md) - Gradient-based optimization algorithms
- [07-integration/plan.md](07-integration/plan.md) - Integration with existing ExTensor backward passes

## Inputs

**Prerequisites:**

- ExTensor core operations (arithmetic, matrix, reduction, elementwise)
- 27 backward pass functions (add_backward, multiply_backward, matmul_backward, etc.)
- Activation functions with backward passes (ReLU, sigmoid, tanh, softmax, etc.)
- Gradient container types (GradientPair, GradientTriple)
- Gradient checking utilities for validation
- Backward pass documentation (`docs/backward-passes/README.md`)

**Design Constraints:**

- Mojo v0.25.7 limitations (no tuple returns, use struct-based gradients per ADR-002)
- YAGNI principle - implement minimal complete API first
- KISS principle - keep the design simple and maintainable
- TDD approach - write tests before implementation

## Outputs

**Core Components:**

- `shared/autograd/graph.mojo` - Computation graph node and edge structures
- `shared/autograd/tape.mojo` - Gradient tape for recording operations
- `shared/autograd/variable.mojo` - Variable type wrapping ExTensor with grad tracking
- `shared/autograd/engine.mojo` - Backward pass execution engine
- `shared/autograd/functions.mojo` - Autograd-wrapped operations
- `shared/core/loss.mojo` - Loss functions (MSE, cross-entropy, BCE)
- `shared/training/optimizers.mojo` - SGD, Adam, RMSprop optimizers

**Testing:**

- Unit tests for computation graph (`tests/autograd/test_graph.mojo`)
- Unit tests for gradient tape (`tests/autograd/test_tape.mojo`)
- Unit tests for Variable operations (`tests/autograd/test_variable.mojo`)
- Integration tests comparing autograd vs manual gradients
- End-to-end training examples validating correctness

**Documentation:**

- Autograd architecture design (`notes/review/autograd-architecture.md`)
- Usage guide with examples (`docs/autograd/README.md`)
- API reference for Variable and autograd functions

## Steps

**Phase 1: Planning (MUST complete first)**

1. Design computation graph architecture (nodes, edges, topological ordering)
2. Design gradient tape recording mechanism (operation IDs, forward/backward mapping)
3. Design Variable API (creation, operations, backward(), grad access)
4. Document architecture decisions in `/notes/review/autograd-architecture.md`
5. Create detailed plan.md files for each child component

**Phase 2: Testing (parallel after planning)**

1. Write tests for computation graph (node creation, edge tracking, topological sort)
2. Write tests for gradient tape (record/replay, operation mapping)
3. Write tests for Variable (creation, arithmetic, matrix ops, grad accumulation)
4. Write integration tests (compare autograd vs manual backward passes)
5. Create gradient validation suite using numerical differentiation

**Phase 3: Implementation (parallel after planning)**

1. Implement computation graph data structures
2. Implement gradient tape recording system
3. Implement Variable type wrapping ExTensor
4. Implement backward engine (topological sort, chain rule application)
5. Wrap existing backward passes in autograd API
6. Implement missing loss functions
7. Implement optimizers (SGD, Adam)

**Phase 4: Package (parallel after planning)**

1. Create autograd module package (`shared/autograd/__init__.mojo`)
2. Export public API (Variable, functions, optimizers)
3. Create installation documentation
4. Build `.mojopkg` distribution package

**Phase 5: Cleanup (after parallel phases)**

1. Refactor based on test feedback
2. Optimize memory usage (gradient tape cleanup)
3. Add performance benchmarks
4. Review and finalize documentation
5. Address any technical debt

## Success Criteria

**Functional Requirements:**

- [ ] Variable type supports all ExTensor operations with gradient tracking
- [ ] Backward engine correctly computes gradients via chain rule
- [ ] Autograd gradients match manual backward passes (numerical validation)
- [ ] Loss functions are differentiable and integrate with autograd
- [ ] Optimizers (SGD, Adam) successfully update model parameters
- [ ] End-to-end training example trains a simple MLP to >95% accuracy on toy dataset

**Code Quality:**

- [ ] All tests pass with >90% coverage
- [ ] Code follows Mojo best practices (memory safety, SIMD where applicable)
- [ ] Documentation is comprehensive and includes usage examples
- [ ] Pre-commit hooks pass (mojo format, markdown linting)
- [ ] CI/CD pipeline validates autograd functionality

**Performance:**

- [ ] Autograd overhead <10% compared to manual backprop
- [ ] Memory usage scales linearly with computation graph size
- [ ] Gradient tape cleanup prevents memory leaks

## Notes

**Design Philosophy:**

This autograd implementation follows the **tape-based** approach (eager execution) rather than static graph compilation:

- **Pros**: Easier to debug, more flexible, Pythonic API
- **Cons**: Slightly higher memory overhead, no graph optimization
- **Rationale**: Matches PyTorch ergonomics, simpler first implementation (YAGNI)

**Key Architectural Decisions:**

1. **Variable vs Tensor**: Variable wraps ExTensor and adds `requires_grad` flag and `grad` storage
2. **Operation Recording**: Each forward operation records (op_type, inputs, output, backward_fn)
3. **Gradient Accumulation**: Multiple backward passes to same Variable accumulate gradients
4. **Tape Lifecycle**: Created per backward() call, destroyed after to free memory

**Integration with Existing Code:**

The existing 27 backward pass functions (`add_backward`, `matmul_backward`, etc.) become the building blocks for autograd. The gradient tape simply needs to:

1. Record which operation was performed
2. Store input/output references
3. Call the appropriate backward function during `.backward()`

This means **minimal changes** to existing code - we're building on top, not rewriting.

**Deferred Features (YAGNI):**

- Higher-order gradients (grad of grad)
- Graph optimization and fusion
- GPU/distributed autograd
- Checkpointing for memory efficiency
- Custom autograd Function API

**References:**

- PyTorch Autograd: <https://docs.pytorch.org/tutorials/beginner/blitz/autograd_tutorial.html>
- Micrograd (minimal autograd): <https://github.com/karpathy/micrograd>
- Existing backward passes: `/home/user/ml-odyssey/shared/core/` (arithmetic.mojo, matrix.mojo, etc.)
- Gradient documentation: `/home/user/ml-odyssey/docs/backward-passes/README.md`
- ADR-002: Gradient struct return types
