# Issue #393: [Plan] Shuffling - Design and Documentation

## Objective

Design and document a data shuffling component that randomizes sample order across training epochs, preventing the model from learning spurious patterns based on data order and improving generalization by providing varied training sequences each epoch.

## Deliverables

- Randomized sample indices implementation
- Reproducible shuffling with seed control
- Per-epoch shuffle variation mechanism
- Option to disable shuffling for validation/test sets
- Comprehensive API documentation
- Design specifications for implementation phase

## Success Criteria

- [ ] Shuffling randomizes sample order effectively
- [ ] Same seed produces identical shuffle results (reproducibility)
- [ ] Different epochs produce different shuffle orders
- [ ] Shuffle can be disabled for validation/test sets
- [ ] API contracts and interfaces are clearly documented
- [ ] Design handles distributed training scenarios

## Design Decisions

### 1. Random Number Generation

**Decision**: Use Mojo's RNG with configurable seed support.

### Rationale

- Ensures reproducibility across runs when using the same seed
- Allows for deterministic debugging and validation
- Provides foundation for distributed training consistency

### Implementation Notes

- Seed should be configurable at initialization
- Default seed should be documented for repeatability

### 2. Epoch-Based Shuffle Variation

**Decision**: Generate new shuffle order each epoch using epoch-based seed derivation.

### Rationale

- Prevents model from memorizing sample order
- Provides varied training sequences for better generalization
- Maintains reproducibility when base seed is fixed

### Implementation Approach

- Derive per-epoch seed from base seed + epoch number
- Ensures different epochs produce different orders
- Maintains reproducibility for specific base seed + epoch combinations

### 3. Validation Set Handling

**Decision**: Provide shuffle enable/disable flag, defaulting to disabled for validation/test sets.

### Rationale

- Validation/test sets should maintain consistent order for fair evaluation
- Training sets benefit from randomization
- Explicit flag provides clear control and prevents errors

### API Design

- Boolean `shuffle` parameter in data loader configuration
- Clear documentation of when to enable/disable
- Separate handling for train vs. validation/test modes

### 4. Distributed Training Consistency

**Decision**: Ensure shuffle produces consistent results across distributed workers.

### Rationale

- Each worker must see samples in the same global order
- Prevents data duplication or gaps in distributed scenarios
- Critical for correct gradient aggregation

### Considerations

- Workers must use synchronized seeds
- Index shuffling must be deterministic across workers
- Need coordination mechanism for distributed setups

### 5. Index-Based vs. Sample-Based Shuffling

**Decision**: Shuffle indices rather than actual data samples.

### Rationale

- More memory efficient (indices are small)
- Allows lazy loading of actual samples
- Decouples shuffle from data loading logic
- Enables efficient distributed data partitioning

### Benefits

- Minimal memory overhead
- Compatible with large datasets
- Flexible for various data loading strategies

## API Contract

### Shuffle Function Signature

```mojo
fn shuffle_indices(
    indices: List[Int],
    seed: Int,
    epoch: Int = 0,
    enabled: Bool = True
) -> List[Int]:
    """
    Shuffle dataset indices for training.

    Args:
        indices: Original ordered indices
        seed: Base random seed for reproducibility
        epoch: Current epoch number (for variation)
        enabled: Whether to perform shuffle (False for validation)

    Returns:
        Shuffled indices (or original if disabled)
    """
```text

### Configuration Parameters

- `base_seed: Int` - Base random seed for reproducibility
- `shuffle_train: Bool` - Enable shuffle for training data (default: True)
- `shuffle_val: Bool` - Enable shuffle for validation data (default: False)
- `shuffle_test: Bool` - Enable shuffle for test data (default: False)

### Reproducibility Guarantees

1. **Same seed + same epoch = same shuffle**: Given identical `seed` and `epoch` values, the shuffle produces identical index ordering
1. **Different epochs = different shuffle**: For the same `seed` but different `epoch` values, shuffle produces different orderings
1. **Disabled shuffle = original order**: When `enabled=False`, returns indices unchanged

## Architecture

### Component Structure

```text
shuffling/
├── shuffle.mojo           # Core shuffle implementation
├── rng.mojo              # Random number generation utilities
└── tests/
    ├── test_shuffle.mojo     # Shuffle functionality tests
    └── test_reproducibility.mojo  # Seed/epoch reproducibility tests
```text

### Key Components

1. **Shuffle Engine**
   - Fisher-Yates shuffle algorithm for uniform randomness
   - Epoch-based seed derivation: `derived_seed = base_seed + epoch * large_prime`
   - Efficient in-place index shuffling

1. **RNG Wrapper**
   - Encapsulates Mojo's random number generator
   - Provides seedable interface
   - Ensures deterministic behavior

1. **Configuration Interface**
   - Shuffle enable/disable controls
   - Seed management
   - Epoch tracking

### Data Flow

```text
Input: [indices, seed, epoch, enabled]
    |
    v
Check enabled flag
    |
    +---> If disabled: Return original indices
    |
    v
Derive epoch-specific seed
    |
    v
Initialize RNG with derived seed
    |
    v
Apply Fisher-Yates shuffle
    |
    v
Output: Shuffled indices
```text

## Integration Points

### DataLoader Integration

- DataLoader calls shuffle function at epoch start
- Passes current epoch number for variation
- Respects shuffle configuration flags
- Uses shuffled indices for batch sampling

### Distributed Training Integration

- Master process generates shuffle indices
- Broadcasts shuffle order to all workers
- Each worker applies same shuffle to its partition
- Ensures global consistency across distributed setup

## Testing Strategy

### Unit Tests

1. **Shuffle Correctness**
   - Verify all indices present in output
   - Verify no duplicate indices
   - Verify order is different from input (when enabled)

1. **Reproducibility**
   - Same seed + epoch produces identical results
   - Different seeds produce different results
   - Different epochs produce different results

1. **Edge Cases**
   - Empty index list
   - Single-element list
   - Very large index lists

1. **Disable Flag**
   - Disabled shuffle returns original order
   - Enabled shuffle modifies order

### Integration Tests

1. **DataLoader Integration**
   - Shuffle integrates correctly with batch sampling
   - Epoch transitions produce new shuffle orders
   - Validation mode respects shuffle=False

1. **Multi-Epoch Consistency**
   - Same configuration produces different orders per epoch
   - Reproducible across multiple runs with same seed

## Performance Considerations

### Time Complexity

- Fisher-Yates shuffle: O(n) time complexity
- RNG seed derivation: O(1)
- Overall: O(n) per epoch

### Space Complexity

- In-place shuffling: O(1) auxiliary space
- Index list: O(n) (required for output)

### Optimization Opportunities

- SIMD operations for index swapping (if beneficial)
- Memory layout optimization for cache efficiency
- Batch seed derivation for distributed scenarios

## Security & Reproducibility

### Seed Management

- Document recommended seed ranges
- Warn against predictable seeds (e.g., 0, 1, 2)
- Provide default seed generation guidance

### Reproducibility Best Practices

- Document exact RNG algorithm used
- Version shuffle implementation for reproducibility
- Provide seed logging for experiment tracking

## References

### Source Plan

[notes/plan/02-shared-library/03-data-utils/02-data-loader/02-shuffling/plan.md](notes/plan/02-shared-library/03-data-utils/02-data-loader/02-shuffling/plan.md)

### Parent Component

[notes/plan/02-shared-library/03-data-utils/02-data-loader/plan.md](notes/plan/02-shared-library/03-data-utils/02-data-loader/plan.md)

### Related Issues

- Issue #393: [Plan] Shuffling - Design and Documentation (this issue)
- Issue #394: [Test] Shuffling - Test Development
- Issue #395: [Impl] Shuffling - Implementation
- Issue #396: [Package] Shuffling - Integration and Packaging
- Issue #397: [Cleanup] Shuffling - Refactoring and Finalization

### Comprehensive Documentation

- [Agent Hierarchy](agents/agent-hierarchy.md)
- [5-Phase Development Workflow](notes/review/README.md)
- [Delegation Rules](agents/delegation-rules.md)

## Implementation Notes

(This section will be populated during subsequent phases with findings, challenges, and decisions made during implementation)

### Open Questions

1. Which specific RNG algorithm should Mojo use (platform-specific or portable)?
1. What is the optimal prime number for epoch-seed derivation?
1. Should shuffle support custom shuffle algorithms (beyond Fisher-Yates)?
1. How to handle very large datasets that don't fit in memory?

### Future Enhancements

- Support for stratified shuffling (maintaining class balance in batches)
- Support for block shuffling (shuffle blocks of samples, preserve local order)
- Integration with checkpointing for mid-epoch recovery
- Advanced distributed shuffle strategies for massive datasets
