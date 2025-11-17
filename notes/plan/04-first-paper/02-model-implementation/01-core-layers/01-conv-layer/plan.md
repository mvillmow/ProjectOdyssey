# Conv Layer

## Overview

Implement a 2D convolutional layer that performs convolution operations on 2D images, supporting configurable kernel sizes, padding, and stride.

## Parent Plan

[Parent](../plan.md)

## Child Plans

None (leaf node)

## Inputs

- Implement 2D convolution operation
- Support padding and stride parameters
- Test with known inputs

## Outputs

- Completed conv layer

## Steps

1. Implement 2D convolution operation
2. Support padding and stride parameters
3. Test with known inputs

## Success Criteria

- [ ] Conv layer performs correct 2D convolution
- [ ] Supports configurable parameters
- [ ] Handles batched inputs
- [ ] Output shapes are correct
- [ ] Tests pass

## Notes

- Implement naive algorithm (loops)
- Support padding: valid, same
- Support stride parameter
- Input: (batch, channels, height, width)
- Output: (batch, out_channels, out_height, out_width)
