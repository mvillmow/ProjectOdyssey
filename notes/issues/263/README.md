# Issue #263: [Plan] Kaiming He - Design and Documentation

## Objective

Design and document the Kaiming (He) initialization method for neural network weights, specifically optimized for ReLU activations. This initialization method scales weights based on the number of input/output units and accounts for ReLU zeroing half the activations to prevent gradient problems in deep networks.

## Deliverables

- Kaiming uniform initialization: U(-sqrt(6/fan), sqrt(6/fan))
- Kaiming normal initialization: N(0, sqrt(2/fan))
- Properly scaled weight tensors for ReLU networks
- Comprehensive design documentation including API specifications and mathematical formulations

## Success Criteria

- [ ] Weights have correct variance for ReLU activations (Var(W) = 2/fan)
- [ ] Both fan_in and fan_out modes work correctly
- [ ] Uniform and normal variants use proper scaling factors
- [ ] Random seed produces reproducible initialization
- [ ] API design is consistent with Xavier/Glorot initializer
- [ ] Mathematical formulations are documented and verified
- [ ] Design decisions are clearly documented

## Design Decisions

### Initialization Variants

**Kaiming Uniform Distribution:**

- Formula: `W ~ U(-limit, limit)` where `limit = sqrt(6/fan)`
- Theoretical basis: Uniform distribution with variance = 2/fan
- Use case: When uniform weight distribution is preferred

**Kaiming Normal Distribution:**

- Formula: `W ~ N(0, std)` where `std = sqrt(2/fan)`
- Theoretical basis: Normal distribution with variance = 2/fan
- Use case: When Gaussian weight distribution is preferred (more common)

### Fan Mode Selection

**fan_in mode:**

- Uses number of input units for variance calculation
- Preserves variance during forward pass
- Default choice for most layers

**fan_out mode:**

- Uses number of output units for variance calculation
- Preserves variance during backward pass
- Useful for specific network architectures

### Mathematical Foundation

Kaiming initialization is based on the following principles:

1. **ReLU Activation Impact:** ReLU zeros out approximately half of the activations (negative values)
2. **Variance Preservation:** To maintain signal variance through layers, use Var(W) = 2/fan
3. **Scaling Factor:** The factor of 2 (vs 1 in Xavier) compensates for ReLU's zero-out effect

**Derivation:**

- For linear layer: `y = Wx + b`
- With ReLU activation: `E[ReLU(x)²] ≈ (1/2)E[x²]` (half of activations zeroed)
- To preserve variance: `Var(y) = Var(x)` requires `Var(W) = 2/n_in`

### API Design

**Function Signatures (to be implemented):**

```mojo
fn kaiming_uniform[fan_mode: String = "fan_in"](
    shape: TensorShape,
    seed: Optional[Int] = None
) -> Tensor:
    """
    Kaiming uniform initialization.

    Args:
        shape: Output tensor shape (rows, cols) or (in_features, out_features)
        fan_mode: Either "fan_in" or "fan_out" for variance calculation
        seed: Random seed for reproducibility

    Returns:
        Tensor with weights sampled from U(-limit, limit)
        where limit = sqrt(6/fan)
    """
    pass

fn kaiming_normal[fan_mode: String = "fan_in"](
    shape: TensorShape,
    seed: Optional[Int] = None
) -> Tensor:
    """
    Kaiming normal initialization.

    Args:
        shape: Output tensor shape (rows, cols) or (in_features, out_features)
        fan_mode: Either "fan_in" or "fan_out" for variance calculation
        seed: Random seed for reproducibility

    Returns:
        Tensor with weights sampled from N(0, std)
        where std = sqrt(2/fan)
    """
    pass
```

### Fan Calculation

```mojo
fn calculate_fan(shape: TensorShape, mode: String) -> Int:
    """
    Calculate fan_in or fan_out from tensor shape.

    For 2D tensors (rows, cols):
        - fan_in = cols (number of input features)
        - fan_out = rows (number of output features)
    """
    if mode == "fan_in":
        return shape[1]  # Input features
    elif mode == "fan_out":
        return shape[0]  # Output features
    else:
        raise ValueError("mode must be 'fan_in' or 'fan_out'")
```

### Consistency with Xavier/Glorot

To maintain API consistency across initializers:

1. **Naming Convention:** `kaiming_uniform()` and `kaiming_normal()` match Xavier naming
2. **Parameter Order:** shape, seed pattern consistent across initializers
3. **Fan Mode:** Both Xavier and Kaiming support fan_in/fan_out selection
4. **Return Type:** All initializers return Tensor objects

### Implementation Strategy

**Phase 1: Core Functions**

1. Implement fan calculation utility
2. Implement Kaiming uniform with correct scaling
3. Implement Kaiming normal with correct variance

**Phase 2: Reproducibility**

1. Ensure random seed handling works correctly
2. Verify initialization is deterministic with same seed
3. Test variance of generated weights

**Phase 3: Validation**

1. Verify mathematical correctness (variance = 2/fan)
2. Test both fan modes produce correct statistics
3. Compare with PyTorch/TensorFlow implementations

### Testing Strategy

**Statistical Tests:**

- Verify mean ≈ 0 for both uniform and normal variants
- Verify variance ≈ 2/fan for both variants
- Test with different layer sizes (small, medium, large)

**Reproducibility Tests:**

- Same seed produces identical initialization
- Different seeds produce different initialization
- Verify across multiple runs

**Edge Cases:**

- Single neuron (fan = 1)
- Very wide layers (fan >> 1)
- Very deep networks (many sequential initializations)

### Dependencies

**Required Components:**

- Random number generation (uniform and normal distributions)
- Tensor data structure
- Basic math operations (sqrt)
- Optional type support

**Integration Points:**

- Will be used by layer initialization in neural network modules
- Must work with existing Tensor API
- Should integrate with Xavier/Glorot initializer in same module

## References

**Source Plan:**

- [Kaiming He Plan](/home/mvillmow/ml-odyssey-manual/notes/plan/02-shared-library/01-core-operations/03-initializers/02-kaiming-he/plan.md)
- [Initializers Parent Plan](/home/mvillmow/ml-odyssey-manual/notes/plan/02-shared-library/01-core-operations/03-initializers/plan.md)

**Related Issues:**

- Issue #264: [Test] Kaiming He - Test Implementation
- Issue #265: [Impl] Kaiming He - Implementation
- Issue #266: [Package] Kaiming He - Integration and Packaging
- Issue #267: [Cleanup] Kaiming He - Finalization

**Research Papers:**

- He et al. (2015): "Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification"
- Original paper introducing Kaiming initialization for ReLU networks

**Related Components:**

- Xavier/Glorot Initializer (sibling component)
- Uniform/Normal Initializer (sibling component)
- Random number generation utilities

## Implementation Notes

(This section will be populated during Test, Implementation, and Packaging phases)

### Discoveries During Development

(To be filled as work progresses)

### Design Refinements

(To be filled as work progresses)

### Technical Challenges

(To be filled as work progresses)
