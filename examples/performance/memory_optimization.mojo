"""Example: Performance - Memory Optimization

This example demonstrates in-place operations and buffer reuse.

Usage:
    pixi run mojo run examples/performance/memory_optimization.mojo

See documentation: docs/advanced/performance.md
"""

from algorithm import vectorize
from sys.info import simdwidthof
from shared.core.types import Tensor


# Bad: Creates temporary tensors
fn bad_update(weights: Tensor, grad: Tensor, lr: Float64) -> Tensor:
    """Inefficient update with allocations."""
    var scaled_grad = grad * lr  # Allocation 1
    return weights - scaled_grad  # Allocation 2


# Good: In-place update
fn good_update(inout weights: Tensor, borrowed grad: Tensor, lr: Float64):
    """Efficient in-place update."""
    weights -= lr * grad  # No allocation


# Best: Fused in-place operation with SIMD
fn best_update(inout weights: Tensor, borrowed grad: Tensor, lr: Float64):
    """Fused in-place update with SIMD."""
    alias width = simdwidthof[DType.float32]()

    @parameter
    fn fused_update[w: Int](idx: Int):
        var w_val = weights.load[w](idx)
        var g = grad.load[w](idx)
        weights.store[w](idx, w_val - lr * g)

    vectorize[width, fused_update](weights.size())


struct EfficientConv2D:
    """Conv2D with preallocated buffers for memory efficiency."""
    var weight: Tensor
    var im2col_buffer: Tensor  # Reused buffer

    fn __init__(mut self, in_channels: Int, out_channels: Int, kernel_size: Int):
        self.weight = Tensor.randn(out_channels, in_channels, kernel_size, kernel_size)
        # Preallocate maximum buffer size
        self.im2col_buffer = Tensor.zeros(1024, in_channels * kernel_size * kernel_size, DType.float32)

    fn forward(mut self, input: Tensor) -> Tensor:
        """Forward pass reusing buffer instead of allocating."""
        im2col(input, self.im2col_buffer)
        return self.im2col_buffer @ self.weight.reshape(-1)


fn main() raises:
    """Demonstrate memory optimization."""

    print("=" * 50)
    print("Memory Optimization Examples")
    print("=" * 50)

    # Example 1: Compare update methods
    print("\n1. Weight Update Methods")
    var weights1 = Tensor.randn(1000, 1000)
    var weights2 = Tensor.randn(1000, 1000)
    var weights3 = Tensor.randn(1000, 1000)
    var grad = Tensor.randn(1000, 1000)
    var lr = 0.01

    print("Bad update (2 allocations)...")
    var result1 = bad_update(weights1, grad, lr)
    print("Complete!")

    print("Good update (in-place, no allocation)...")
    good_update(weights2, grad, lr)
    print("Complete!")

    print("Best update (fused SIMD, no allocation)...")
    best_update(weights3, grad, lr)
    print("Complete!")

    # Example 2: Buffer reuse
    print("\n2. Buffer Reuse in Conv2D")
    var conv = EfficientConv2D(3, 64, 3)
    var input = Tensor.randn(32, 3, 224, 224)
    print("Processing batch with reused buffer...")
    var output = conv.forward(input)
    print("Output shape:", output.shape())
    print("Buffer reused - no allocations!")

    print("\n" + "=" * 50)
    print("Memory optimization examples complete!")
    print("=" * 50)
