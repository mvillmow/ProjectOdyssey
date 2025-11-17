# Checkpointing

## Overview

Implement checkpointing callback for saving and restoring complete training state. This includes model weights, optimizer state, scheduler state, and training progress. Checkpointing enables training resumption after interruptions and preserves best model versions.

## Parent Plan

[../plan.md](../plan.md)

## Child Plans

None (leaf node)

## Inputs

- Model, optimizer, and scheduler states
- Checkpoint directory and naming scheme
- Trigger conditions (every N epochs, best metric)
- Maximum checkpoints to keep

## Outputs

- Saved checkpoint files with complete state
- Checkpoint metadata (epoch, metrics, timestamp)
- Functionality to load and resume from checkpoint
- Automatic cleanup of old checkpoints

## Steps

1. Implement state collection from all components
2. Create checkpoint saving with metadata
3. Add checkpoint loading and state restoration
4. Support multiple checkpoint retention strategies
5. Enable best model tracking by metric

## Success Criteria

- [ ] Checkpoints capture complete training state
- [ ] Loading restores training exactly
- [ ] Best model tracking works correctly
- [ ] Old checkpoints clean up as configured

## Notes

Save all stateful components: model, optimizer, scheduler, RNG state. Include metadata: epoch, step, metrics. Support saving best by validation metric. Make checkpoint format version-aware for compatibility.
