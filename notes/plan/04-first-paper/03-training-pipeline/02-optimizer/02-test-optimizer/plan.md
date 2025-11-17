# Test Optimizer

## Overview

Write tests to verify the optimizer correctly updates parameters in the direction that reduces loss.

## Parent Plan

[Parent](../plan.md)

## Child Plans

None (leaf node)

## Inputs

- Test optimizer on simple function
- Verify parameter updates
- Check convergence

## Outputs

- Completed test optimizer

## Steps

1. Test optimizer on simple function
2. Verify parameter updates
3. Check convergence

## Success Criteria

- [ ] Optimizer tests pass
- [ ] Parameters move toward minimum
- [ ] Loss decreases over steps
- [ ] Tests are comprehensive

## Notes

- Test on simple convex function (x^2)
- Verify parameters approach minimum
- Test with different learning rates
- Check update direction
