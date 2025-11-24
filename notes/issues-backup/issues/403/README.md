# Issue #403: [Plan] Data Loader - Design and Documentation

## Objective

Design and document a data loader for efficient batch processing of datasets that bridges datasets and training loops by
providing batching, shuffling, and iteration support for accessing batches sequentially.

## Deliverables

- Batching mechanism for grouping samples into mini-batches
- Shuffling support for randomized sample order across epochs
- Iterator interface for sequential batch access with Python for-loop compatibility
- Comprehensive API specification and design documentation
- Architecture documentation covering all child components

## Success Criteria

- [ ] Batching correctly groups samples by batch size with proper handling of partial batches
- [ ] Shuffling randomizes sample order when enabled with reproducible seed control
- [ ] Iterator provides clean batch access following Python iterator protocol
- [ ] All design decisions are documented with rationale
- [ ] API contracts are clearly defined for all interfaces
- [ ] Edge cases are identified and handling strategies documented
- [ ] Related issues #404-407 are updated with planning documentation reference

## Design Decisions

### Architecture Overview

The Data Loader component consists of three core subsystems:

1. **Batching** (`01-batching/`) - Groups individual dataset samples into mini-batches
1. **Shuffling** (`02-shuffling/`) - Randomizes sample order for better generalization
1. **Iteration** (`03-iteration/`) - Provides Python iterator protocol for batch traversal

### Key Design Principles

#### 1. Simple Sequential Foundation

**Decision**: Start with simple sequential batching before adding complexity.

**Rationale**: Following KISS principle - establish correct basic behavior before optimization.

### Implementation Strategy

- Begin with fixed-size batches in dataset order
- Add shuffling as second layer
- Ensure iteration protocol works before advanced features

#### 2. Partial Batch Handling

**Decision**: Provide two modes for handling final partial batch - drop or include.

**Rationale**: Different training scenarios require different handling:

- Drop mode: Ensures consistent batch size (important for some architectures)
- Include mode: Uses all data (important for small datasets)

### API Design

```python
DataLoader(dataset, batch_size=32, drop_last=False)
```text

#### 3. Reproducible Shuffling

**Decision**: Use configurable random seed with per-epoch variation.

**Rationale**: Enables reproducibility while providing different orders each epoch.

### Implementation Strategy

- Accept base random seed in constructor
- Derive epoch-specific seed: `epoch_seed = base_seed + epoch_number`
- Same base seed + same epoch = same shuffle order
- Different epochs automatically get different orders

#### 4. Shuffle Disable for Validation

**Decision**: Provide shuffle flag to disable randomization.

**Rationale**: Validation and test sets should maintain consistent order for reproducibility.

### API Design

```python
# Training loader - shuffled
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, seed=42)

# Validation loader - sequential
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
```text

#### 5. Collate Function for Custom Batching

**Decision**: Support custom collate functions for flexible batch assembly.

**Rationale**: Different data types require different batching strategies:

- Images: Stack along new batch dimension
- Variable-length sequences: Pad to max length in batch
- Complex structures: Custom combination logic

### Default Behavior

```python
def default_collate(batch):
    """Stack tensors along new batch dimension"""
    return stack(batch, axis=0)
```text

### Custom Collate Example

```python
def pad_collate(batch):
    """Pad sequences to max length in batch"""
    max_len = max(len(item) for item in batch)
    return [pad(item, max_len) for item in batch]

loader = DataLoader(dataset, batch_size=32, collate_fn=pad_collate)
```text

#### 6. Python Iterator Protocol

**Decision**: Follow standard Python iterator protocol with `__iter__` and `__next__`.

**Rationale**: Enables natural Python for-loop syntax and compatibility with existing tools.

### Protocol Implementation

```python
class DataLoader:
    def __iter__(self):
        """Return iterator instance"""
        self.reset()  # Reset state for new epoch
        return self

    def __next__(self):
        """Yield next batch or raise StopIteration"""
        if self.has_next():
            return self.get_next_batch()
        else:
            raise StopIteration
```text

### Usage

```python
# Natural for-loop syntax
for batch in data_loader:
    train_on_batch(batch)

# Multiple epochs
for epoch in range(num_epochs):
    for batch in data_loader:  # Automatically resets
        train_on_batch(batch)
```text

#### 7. State Management for Resumption

**Decision**: Track iteration state to support training resumption.

**Rationale**: Long training runs may need checkpointing and resumption.

### State to Track

- Current batch index
- Current epoch number
- RNG state for shuffle reproducibility

### API Design

```python
# Save state
state = loader.state_dict()

# Resume from state
loader.load_state_dict(state)
```text

### Component Specifications

#### Batching Component (Issue #408-411)

**Purpose**: Group individual samples into mini-batches for efficient training.

### Key Features

- Fixed batch size with optional variable final batch
- Custom collate functions for flexible batch assembly
- Support for variable-length sequences
- Proper handling of different data types

### Edge Cases

- Dataset size not divisible by batch size → Handle with drop_last flag
- Single sample in final batch → Include or drop based on configuration
- Variable-length sequences → Use padding or custom collate
- Empty dataset → Raise error early with clear message

#### Shuffling Component (Issue #412-415)

**Purpose**: Randomize sample order to prevent learning spurious patterns.

### Key Features

- Reproducible shuffling with seed control
- Per-epoch variation with automatic seed derivation
- Enable/disable flag for validation sets
- Compatible with distributed training

### Edge Cases

- Same seed across epochs → Use epoch-based seed derivation
- Validation/test sets → Disable shuffling with flag
- Distributed training → Ensure consistent shuffle across workers
- Very large datasets → Use index-based shuffling (memory efficient)

#### Iteration Component (Issue #416-419)

**Purpose**: Provide standard Python iterator interface for batch traversal.

### Key Features

- Python iterator protocol (`__iter__`, `__next__`)
- Automatic reset for multiple epochs
- StopIteration at epoch boundaries
- State tracking for resumption

### Edge Cases

- Empty dataset → Immediate StopIteration
- Single batch → One iteration then StopIteration
- Multiple concurrent iterators → Each maintains separate state
- Resumption mid-epoch → Restore exact position

### API Contract

#### Constructor

```python
DataLoader(
    dataset: Dataset,           # Dataset implementing __len__ and __getitem__
    batch_size: int = 1,        # Number of samples per batch
    shuffle: bool = False,      # Whether to shuffle samples
    seed: Optional[int] = None, # Random seed for reproducibility
    drop_last: bool = False,    # Drop final partial batch
    collate_fn: Optional[Callable] = None  # Custom batch assembly
)
```text

#### Methods

```python
def __iter__(self) -> Iterator:
    """Return iterator instance, reset state for new epoch"""

def __next__(self) -> Batch:
    """Yield next batch or raise StopIteration at epoch end"""

def __len__(self) -> int:
    """Return number of batches per epoch"""

def reset(self) -> None:
    """Reset iteration state for new epoch"""

def state_dict(self) -> dict:
    """Return current state for checkpointing"""

def load_state_dict(self, state: dict) -> None:
    """Restore state from checkpoint"""
```text

#### Properties

```python
@property
def dataset(self) -> Dataset:
    """Underlying dataset"""

@property
def batch_size(self) -> int:
    """Number of samples per batch"""

@property
def num_batches(self) -> int:
    """Total batches per epoch"""
```text

### Implementation Phases

This planning phase (#403) produces specifications for:

1. **Test Phase** (#404) - Test suite covering all edge cases
1. **Implementation Phase** (#405) - Core data loader implementation
1. **Packaging Phase** (#406) - Integration with data utils module
1. **Cleanup Phase** (#407) - Refactoring and optimization

### Integration with Data Utils

The Data Loader integrates with the broader Data Utils module:

### Dependencies

- **Base Dataset** (#372-383) - Required interface with `__len__` and `__getitem__`
- **Tensor Operations** - For stacking and batching tensors
- **RNG Utilities** - For reproducible shuffling

### Provides to Downstream

- **Training Loops** - Efficient batch iteration
- **Augmentations** (#420-439) - Batched augmentation support
- **Distributed Training** - Consistent data distribution

### Performance Considerations

### Memory Efficiency

- Use index-based shuffling (shuffle indices, not data)
- Lazy batch assembly (create batches on-demand)
- Avoid copying dataset into loader

### Computational Efficiency

- Cache shuffled indices per epoch
- Minimize allocations in hot paths
- Consider prefetching for I/O-bound datasets (future enhancement)

### Design Trade-offs

- Simplicity vs. features: Start simple, add features incrementally
- Memory vs. speed: Prefer memory efficiency for initial implementation
- Flexibility vs. performance: Provide escape hatches (custom collate) without sacrificing common case performance

### Error Handling

### Validation at Construction

- `batch_size > 0`: Raise ValueError if non-positive
- `dataset` implements required interface: Raise TypeError if missing `__len__` or `__getitem__`
- `seed` is valid: Raise ValueError if negative

### Runtime Errors

- Empty dataset: Raise ValueError with clear message
- Corrupted data: Propagate exception from dataset
- Invalid state dict: Raise ValueError with details

### Testing Strategy

**Unit Tests** (Component-level):

- Batching: Correct batch sizes, partial batch handling
- Shuffling: Reproducibility, per-epoch variation
- Iteration: Protocol compliance, state management

**Integration Tests** (Cross-component):

- Batching + Shuffling: Correct randomized batches
- Iteration + State: Proper resumption behavior
- Full DataLoader: End-to-end workflows

### Edge Case Tests

- Empty datasets
- Single-sample datasets
- Datasets not divisible by batch size
- Very large batch sizes (larger than dataset)
- State save/load during iteration

## References

### Source Documentation

- **Source Plan**: [notes/plan/02-shared-library/03-data-utils/02-data-loader/plan.md](../../plan/02-shared-library/03-data-utils/02-data-loader/plan.md)
- **Parent Module**: [notes/plan/02-shared-library/03-data-utils/plan.md](../../plan/02-shared-library/03-data-utils/plan.md)

### Child Components

- **Batching**: [notes/plan/02-shared-library/03-data-utils/02-data-loader/01-batching/plan.md](../../plan/02-shared-library/03-data-utils/02-data-loader/01-batching/plan.md)
- **Shuffling**: [notes/plan/02-shared-library/03-data-utils/02-data-loader/02-shuffling/plan.md](../../plan/02-shared-library/03-data-utils/02-data-loader/02-shuffling/plan.md)
- **Iteration**: [notes/plan/02-shared-library/03-data-utils/02-data-loader/03-iteration/plan.md](../../plan/02-shared-library/03-data-utils/02-data-loader/03-iteration/plan.md)

### Related Issues

- **Test Phase**: #404 - Test suite implementation
- **Implementation Phase**: #405 - Core data loader implementation
- **Packaging Phase**: #406 - Module integration
- **Cleanup Phase**: #407 - Refactoring and optimization
- **Batching Component**: #408 (Plan), #409 (Test), #410 (Impl), #411 (Package)
- **Shuffling Component**: #412 (Plan), #413 (Test), #414 (Impl), #415 (Package)
- **Iteration Component**: #416 (Plan), #417 (Test), #418 (Impl), #419 (Package)
- **Base Dataset Dependency**: #372-383
- **Augmentations Downstream**: #420-439

### Architecture Documentation

- **Agent Hierarchy**: [agents/hierarchy.md](../../../agents/hierarchy.md)
- **Delegation Rules**: [agents/delegation-rules.md](../../../agents/delegation-rules.md)
- **5-Phase Workflow**: [notes/review/README.md](../../review/README.md)

## Implementation Notes

*This section will be populated during subsequent phases (Test, Implementation, Packaging, Cleanup) with findings,
challenges, and decisions made during development.*

### Planned Additions During Development

- Performance benchmarks and optimization opportunities
- Integration challenges with base dataset interface
- Edge cases discovered during testing
- API refinements based on usage patterns
- Distributed training considerations (if applicable)
