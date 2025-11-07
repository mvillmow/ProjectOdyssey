# Training Loop

## Overview
Implement the core training loop that iterates over training data, performs forward passes, computes losses, executes backpropagation, and updates model weights. This is the heart of the training process, coordinating all components to improve model performance.

## Parent Plan
[../plan.md](../plan.md)

## Child Plans
None (leaf node)

## Inputs
- Model to train
- Training data loader
- Loss function and optimizer
- Training configuration (epochs, logging frequency)

## Outputs
- Trained model with updated weights
- Training metrics (loss, accuracy per batch/epoch)
- Training state for resumption
- Callback invocations at key points

## Steps
1. Implement batch iteration over training data
2. Create forward pass with loss computation
3. Add backward pass with gradient computation
4. Integrate optimizer for weight updates
5. Add callback hooks and metric tracking

## Success Criteria
- [ ] Loop correctly iterates over all training data
- [ ] Forward and backward passes work correctly
- [ ] Weights update according to optimizer
- [ ] Metrics track accurately
- [ ] Callbacks fire at appropriate times

## Notes
Keep the loop straightforward and readable. Handle edge cases like empty batches. Ensure proper gradient zeroing between batches. Make logging configurable but informative.
