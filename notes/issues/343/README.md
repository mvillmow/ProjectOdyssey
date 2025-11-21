# Issue #343: [Plan] Checkpointing - Design and Documentation

## Objective

Design and document a checkpointing callback system that enables saving and restoring complete training state (model
weights, optimizer state, scheduler state, and training progress) to support training resumption after interruptions
and preservation of best model versions.

## Deliverables

- Comprehensive specification for checkpointing callback architecture
- API contracts and interfaces for state collection and restoration
- Design documentation for checkpoint file format and metadata structure
- Checkpoint retention strategies (periodic, best-only, rolling window)
- Documentation for best model tracking by validation metrics
- Integration guidelines with the callback system and training loop

## Success Criteria

- [ ] Checkpointing specification captures all stateful components (model, optimizer, scheduler, RNG state)
- [ ] Checkpoint loading design ensures exact training state restoration
- [ ] Best model tracking mechanism is clearly defined with metric-based comparison
- [ ] Checkpoint cleanup strategy is documented with configurable retention policies
- [ ] API contracts are well-defined and version-aware for compatibility
- [ ] Design follows callback system patterns and maintains loose coupling

## Design Decisions

### 1. Checkpoint State Components

The checkpointing system must capture all components necessary for exact training resumption:

- **Model State**: Complete model parameters and architecture information
- **Optimizer State**: Momentum buffers, adaptive learning rate parameters, step counters
- **Scheduler State**: Current learning rate, epoch/step counters, warmup/cooldown state
- **RNG State**: Random number generator state for reproducibility
- **Training Progress**: Current epoch, global step, best metric value
- **Metadata**: Timestamp, training configuration, metric history

**Rationale**: Incomplete state capture would prevent exact resumption and could lead to training degradation or
non-reproducible results.

### 2. Checkpoint File Format

Checkpoints will use a versioned format with the following structure:

```text
checkpoint_epoch_N.ckpt
├── version: str (format version)
├── metadata: dict (timestamp, metrics, config)
├── model_state: dict (parameters)
├── optimizer_state: dict (optimizer internals)
├── scheduler_state: dict (scheduler internals)
└── rng_state: dict (random state)
```text

**Rationale**: Versioned format enables backward compatibility as the system evolves. Structured organization makes
checkpoint inspection and debugging easier.

### 3. Checkpoint Naming and Organization

Checkpoint files will follow a consistent naming scheme:

- **Periodic Checkpoints**: `checkpoint_epoch_{epoch:04d}.ckpt`
- **Best Model**: `best_model_{metric_name}.ckpt`
- **Latest Checkpoint**: `latest.ckpt` (symlink or copy)

**Rationale**: Clear naming enables easy identification of checkpoint type and training progress. Symlinks reduce
storage overhead while maintaining easy access to latest state.

### 4. Trigger Conditions

Support multiple checkpoint triggering strategies:

- **Periodic**: Save every N epochs or M steps
- **Best Metric**: Save when validation metric improves
- **Combined**: Periodic checkpoints + best model preservation
- **Manual**: API for explicit checkpoint creation

**Rationale**: Different use cases require different checkpoint strategies. Research experiments may need frequent
checkpoints, while production training may only need best models.

### 5. Checkpoint Retention Policies

Implement configurable cleanup strategies:

- **Keep All**: No automatic deletion
- **Keep Last N**: Rolling window of N most recent checkpoints
- **Keep Best K**: Preserve K best models by metric
- **Keep Best + Latest**: Preserve best model and most recent checkpoint
- **Custom**: User-defined retention logic

**Rationale**: Training runs generate many checkpoints that consume storage. Automatic cleanup balances recovery
capability with storage efficiency.

### 6. Best Model Tracking

Track best model using:

- **Primary Metric**: Single metric for comparison (e.g., validation loss, accuracy)
- **Comparison Mode**: Minimize or maximize metric
- **Tie-Breaking**: Use epoch number (earlier is better) when metrics are equal
- **Metric History**: Track top-K metric values for analysis

**Rationale**: Clear best model definition enables reproducible model selection. Metric history provides insight into
training dynamics.

### 7. Integration with Callback System

Checkpointing will integrate with the callback system through standard hooks:

- `on_epoch_end`: Trigger periodic and metric-based checkpoints
- `on_train_end`: Save final checkpoint
- `on_exception`: Emergency checkpoint before crash

**Rationale**: Leveraging existing callback hooks maintains consistency with the broader callback architecture and
avoids tight coupling with trainer implementation.

### 8. Error Handling and Recovery

Implement robust error handling:

- **Atomic Writes**: Write to temporary file, then rename to prevent corruption
- **Validation**: Verify checkpoint integrity after writing
- **Fallback**: Keep previous checkpoint if new save fails
- **Logging**: Clear error messages for debugging

**Rationale**: Checkpoint corruption or save failures should not compromise training or lose previous valid
checkpoints.

### 9. Performance Considerations

Optimize checkpoint performance:

- **Async Saving**: Option to save checkpoints asynchronously to avoid blocking training
- **Compression**: Optional compression for large models
- **Incremental Checkpoints**: Save only changed parameters (future enhancement)
- **Fast Loading**: Optimize checkpoint loading for quick resumption

**Rationale**: Checkpoint operations should minimize impact on training throughput, especially for large models.

### 10. API Design

Provide clean, intuitive API:

```python
# Example API (conceptual - will be implemented in Mojo)
checkpoint_callback = CheckpointCallback(
    checkpoint_dir="./checkpoints",
    save_frequency=1,  # Save every epoch
    save_best=True,
    monitor="val_loss",
    mode="min",
    keep_last_n=3,
    async_save=True
)

# Explicit save
checkpoint_callback.save_checkpoint(trainer, epoch=10, metrics={"val_loss": 0.5})

# Load and resume
state = checkpoint_callback.load_checkpoint("checkpoint_epoch_0010.ckpt")
trainer.restore_state(state)
```text

**Rationale**: Simple, explicit API reduces cognitive load and makes common use cases straightforward while supporting
advanced scenarios.

## References

### Source Plan

- [Checkpointing Plan](notes/plan/02-shared-library/02-training-utils/03-callbacks/01-checkpointing/plan.md)
- [Callbacks System Plan](notes/plan/02-shared-library/02-training-utils/03-callbacks/plan.md)

### Related Issues

- Issue #344: [Test] Checkpointing - Write Tests
- Issue #345: [Impl] Checkpointing - Implementation
- Issue #346: [Package] Checkpointing - Integration and Packaging
- Issue #347: [Cleanup] Checkpointing - Refactor and Finalize

### Related Components

- Base Callback Interface (dependency)
- Trainer Interface (integration point)
- Validation Loop (metric source)
- File I/O Utilities (checkpoint storage)

## Implementation Notes

(This section will be populated during implementation phases with findings, challenges, and solutions discovered while
building the checkpointing system.)
