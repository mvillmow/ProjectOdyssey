# Issue #253: [Plan] Activations - Design and Documentation

## Objective

Design and document the architecture for common activation functions used in neural networks, including ReLU family
(ReLU, Leaky ReLU, PReLU), sigmoid and tanh, and modern activations like softmax and GELU. These functions introduce
non-linearity essential for deep learning models.

## Deliverables

- ReLU variants (ReLU, Leaky ReLU, PReLU) for sparse activation
- Sigmoid and tanh functions for bounded outputs
- Softmax for probability distributions and GELU for smooth activation
- Comprehensive design documentation with API contracts and numerical stability considerations
- Architecture specifications for all activation function implementations

## Success Criteria

- [ ] All activations produce correct outputs
- [ ] Implementations are numerically stable
- [ ] Functions handle edge cases gracefully
- [ ] All child plans (#254-257) are completed successfully
- [ ] Design documentation is comprehensive and actionable
- [ ] API contracts and interfaces are clearly defined

## Design Decisions

### Architecture Overview

The activation functions module is organized into three logical groups based on functionality and use cases:

1. **ReLU Family** - Sparse activation functions with different negative slope behaviors
1. **Sigmoid/Tanh** - Bounded output functions for gates and normalization
1. **Softmax/GELU** - Modern functions for classification and smooth non-linearity

### Component Design

#### 1. ReLU Family (`01-relu-family`)

**Purpose**: Provide sparse activation with variants for different gradient preservation needs.

### Components

- **ReLU**: Standard rectified linear unit (`max(0, x)`)
  - Simplest implementation
  - Zero gradient for negative values (dead neuron problem)
  - Most computationally efficient

- **Leaky ReLU**: ReLU with small negative slope (`max(alpha*x, x)`)
  - Configurable leak parameter (typically 0.01)
  - Prevents dead neurons by preserving small negative gradients
  - Fixed slope parameter

- **PReLU**: Parametric ReLU with learnable slope
  - Learnable alpha parameter per channel
  - Requires parameter storage and updates
  - Most flexible but more complex

### Key Design Considerations

- All variants share similar structure (element-wise operations)
- Gradient computation differs for each variant
- PReLU requires parameter management (storage, initialization, updates)
- Edge cases: zeros, very large values, NaN/Inf handling

#### 2. Sigmoid/Tanh (`02-sigmoid-tanh`)

**Purpose**: Provide bounded activation functions for gates, normalization, and RNN architectures.

### Components

- **Sigmoid**: Maps inputs to (0, 1) range
  - Formula: `1 / (1 + exp(-x))`
  - Used for binary classification and gate mechanisms
  - Numerical stability critical for extreme inputs

- **Tanh**: Maps inputs to (-1, 1) range
  - Formula: `(exp(x) - exp(-x)) / (exp(x) + exp(-x))`
  - Used in RNNs and architectures requiring centered outputs
  - Related to sigmoid: `tanh(x) = 2*sigmoid(2*x) - 1`

### Key Design Considerations

- **Numerical Stability is Critical**:
  - Large positive inputs cause overflow in `exp(x)`
  - Large negative inputs cause underflow in `exp(-x)`
  - Need clipping or log-sum-exp tricks

- **Implementation Strategies**:
  - Sigmoid: Use stable formulation with input clipping
  - Tanh: Either use relationship to sigmoid or stable exponential form
  - Test extensively with extreme values (±100, ±1000)

- **Gradient Handling**:
  - Sigmoid gradient: `sigmoid(x) * (1 - sigmoid(x))`
  - Tanh gradient: `1 - tanh(x)^2`
  - Both suffer from vanishing gradients for extreme inputs

#### 3. Softmax/GELU (`03-softmax-gelu`)

**Purpose**: Provide modern activation functions for classification and smooth non-linearity.

### Components

- **Softmax**: Converts logits to probability distribution
  - Formula: `exp(x) / sum(exp(x))`
  - Outputs sum to 1.0 along specified axis
  - Essential for multi-class classification
  - Requires axis parameter for reduction

- **GELU**: Gaussian Error Linear Unit for smooth activation
  - Formula: `x * Phi(x)` where Phi is Gaussian CDF
  - Used in transformers (BERT, GPT)
  - Can use exact formula (with erf function) or approximation

### Key Design Considerations

- **Softmax Numerical Stability**:
  - Subtract max value before exponential: `exp(x - max(x))`
  - Prevents overflow for large logits
  - Preserves mathematical correctness (numerator and denominator scale equally)

- **Softmax Axis Support**:
  - Must support reduction along arbitrary axis
  - Common cases: last axis (batch classification), middle axes (attention)
  - Tensor operations must preserve other dimensions

- **GELU Implementation Choice**:
  - Exact: `x * 0.5 * (1 + erf(x / sqrt(2)))`
  - Approximation: `0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))`
  - Trade-off: accuracy vs computational cost
  - Recommendation: Start with approximation (used in BERT/GPT), add exact if needed

### Cross-Cutting Concerns

#### Numerical Stability

All activation functions must handle:

- **Large positive values**: Prevent overflow in exponentials
- **Large negative values**: Prevent underflow
- **Zero values**: Ensure defined behavior
- **NaN/Inf propagation**: Either handle gracefully or fail fast

### Testing Strategy

- Test with values in normal range: [-10, 10]
- Test with extreme values: [-1000, 1000]
- Test with special values: 0, NaN, Inf, -Inf
- Verify output bounds are maintained

#### Gradient Computation

All functions need backward pass support:

- **Forward pass**: Store intermediate values needed for gradients
- **Backward pass**: Compute gradients efficiently
- **Gradient checks**: Numerical gradient verification in tests

### Implementation Pattern

```mojo
fn forward(x: Tensor) -> Tensor:
    # Compute activation
    # Store values needed for backward pass
    return output

fn backward(grad_output: Tensor) -> Tensor:
    # Use stored values to compute gradient
    return grad_input
```text

#### API Design

### Consistent Interface

All activation functions should follow a uniform API pattern:

```mojo
struct ActivationFunction:
    # Configuration parameters (if any)
    fn __init__(inout self, ...): pass

    # Forward pass
    fn forward(self, x: Tensor) -> Tensor: pass

    # Backward pass
    fn backward(self, grad_output: Tensor) -> Tensor: pass
```text

**Functional Interface** (for stateless activations):

```mojo
fn relu(x: Tensor) -> Tensor: pass
fn sigmoid(x: Tensor) -> Tensor: pass
# etc
```text

### Design Questions to Resolve

1. Should we use struct-based (OOP) or functional API?
   - Recommendation: Both - structs for stateful (PReLU), functions for stateless

1. How to handle gradient storage?
   - Option A: Store in activation struct (memory overhead)
   - Option B: Return tuple (activation, cache) for backward pass
   - Recommendation: Start with Option B for explicit control

1. Should activations be in-place or allocate new tensors?
   - Trade-off: Memory efficiency vs safety
   - Recommendation: Start with allocation, add in-place variants later

#### Performance Considerations

**Optimization Opportunities** (defer to implementation phase):

- SIMD vectorization for element-wise operations
- Loop fusion for composite operations (e.g., GELU approximation)
- Memory layout optimization (contiguous access patterns)
- Lazy evaluation for gradient computation chains

**Priority**: Correctness first, then numerical stability, then performance.

### Testing Strategy

Each activation function requires:

1. **Correctness Tests**:
   - Known input/output pairs
   - Mathematical properties (e.g., softmax sums to 1)
   - Boundary conditions

1. **Numerical Stability Tests**:
   - Extreme values (±1000)
   - Special values (0, NaN, Inf)
   - Verify no overflow/underflow

1. **Gradient Tests**:
   - Numerical gradient checking
   - Gradient flow verification
   - Edge case gradients

1. **Performance Tests** (optional in initial implementation):
   - Benchmark against reference implementations
   - Verify SIMD utilization

### Dependencies

**From Tensor Operations** (prerequisite):

- Element-wise arithmetic operations
- Exponential and logarithm functions
- Reduction operations (sum, max)
- Axis-based operations for softmax

**Provides To** (downstream consumers):

- Forward pass activation in neural network layers
- Gradient computation for backpropagation
- Building blocks for complex architectures

## References

### Source Plan

- [notes/plan/02-shared-library/01-core-operations/02-activations/plan.md](../../../plan/02-shared-library/01-core-operations/02-activations/plan.md)

### Child Plans

- [ReLU Family](../../../plan/02-shared-library/01-core-operations/02-activations/01-relu-family/plan.md)
- [Sigmoid/Tanh](../../../plan/02-shared-library/01-core-operations/02-activations/02-sigmoid-tanh/plan.md)
- [Softmax/GELU](../../../plan/02-shared-library/01-core-operations/02-activations/03-softmax-gelu/plan.md)

### Parent Plan

- [Core Operations](../../../plan/02-shared-library/01-core-operations/plan.md)

### Related Issues

- Issue #254: [Test] Activations - Test Suite Development
- Issue #255: [Impl] Activations - Implementation
- Issue #256: [Package] Activations - Integration and Packaging
- Issue #257: [Cleanup] Activations - Refactoring and Finalization

### Additional Documentation

See comprehensive documentation in:

- `/agents/` - Agent hierarchy and delegation rules
- `/notes/review/` - Architectural decisions and design patterns
- `/notes/plan/02-shared-library/` - Shared library component plans

## Implementation Notes

This section will be populated during subsequent phases with:

- Findings discovered during test development
- Implementation challenges and solutions
- Performance benchmarks and optimization results
- Integration issues and resolutions
- Refactoring decisions and rationale

---

**Status**: Planning phase complete - ready for parallel test, implementation, and packaging phases.

### Next Steps

1. Test Phase (#254): Develop comprehensive test suite following TDD principles
1. Implementation Phase (#255): Implement activation functions with numerical stability
1. Packaging Phase (#256): Integrate into shared library and document APIs
1. Cleanup Phase (#257): Refactor based on findings from parallel phases
