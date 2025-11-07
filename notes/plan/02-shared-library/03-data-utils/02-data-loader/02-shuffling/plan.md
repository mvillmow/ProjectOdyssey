# Shuffling

## Overview
Implement data shuffling to randomize sample order across training epochs. Shuffling prevents the model from learning spurious patterns based on data order and improves generalization by providing varied training sequences each epoch.

## Parent Plan
[../plan.md](../plan.md)

## Child Plans
None (leaf node)

## Inputs
- Dataset indices or samples
- Random seed for reproducibility
- Shuffle flag (enable/disable)
- Epoch number for seed variation

## Outputs
- Randomized sample indices
- Reproducible shuffling with seed control
- Per-epoch shuffle variation
- Option to disable for validation

## Steps
1. Implement index shuffling with random seed
2. Support per-epoch shuffle variation
3. Add shuffle enable/disable flag
4. Ensure reproducibility with seed control

## Success Criteria
- [ ] Shuffling randomizes sample order
- [ ] Same seed produces same shuffle
- [ ] Different epochs produce different orders
- [ ] Shuffle can be disabled for validation

## Notes
Use RNG with configurable seed. Generate new order each epoch using epoch-based seed. Validation/test sets should not shuffle. Ensure shuffle works with distributed training (consistent across workers).
