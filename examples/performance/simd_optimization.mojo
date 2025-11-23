"""Example: Performance - SIMD Optimization

This example demonstrates SIMD vectorization for performance.

Usage:
    pixi run mojo run examples/performance/simd_optimization.mojo

See documentation: docs/advanced/performance.md
"""

from algorithm import vectorize
from sys.info import simdwidthof
from shared.core.types import Tensor


fn relu_simd(inout tensor: Tensor):
    """ReLU activation with SIMD."""
    alias simd_width = simdwidthof[DType.float32]()

    @parameter
    fn vectorized_relu[width: Int](idx: Int):
        var val = tensor.data.simd_load[width](idx)
        tensor.data.simd_store[width](idx, max(val, 0.0))

    vectorize[simd_width, vectorized_relu](tensor.size())


fn batch_norm_simd(inout input: Tensor, borrowed mean: Tensor, borrowed var: Tensor):
    """Batch normalization with SIMD."""
    alias width = simdwidthof[DType.float32]()

    @parameter
    fn vectorized[w: Int](idx: Int):
        var x = input.load[w](idx)
        var m = mean.load[w](idx)
        var v = var.load[w](idx)
        var normalized = (x - m) / sqrt(v + 1e-5)
        input.store[w](idx, normalized)

    vectorize[width, vectorized](input.size())


fn matmul_simd(borrowed a: Tensor, borrowed b: Tensor) -> Tensor:
    """Optimized matrix multiplication with SIMD."""
    var m = a.shape[0]
    var n = b.shape[1]
    var k = a.shape[1]

    var result = Tensor.zeros(m, n, DType.float32)

    alias simd_width = simdwidthof[DType.float32]()

    for i in range(m):
        for j in range(n):
            var sum = SIMD[DType.float32, simd_width](0.0)

            @parameter
            fn dot_product[width: Int](idx: Int):
                var a_vec = a.load[width](i * k + idx)
                var b_vec = b.load[width](idx * n + j)
                sum += a_vec * b_vec

            vectorize[simd_width, dot_product](k)

            result[i, j] = sum.reduce_add()

    return result


fn main() raises:
    """Demonstrate SIMD optimization."""

    print("=" * 50)
    print("SIMD Optimization Examples")
    print("=" * 50)

    # Example 1: SIMD ReLU
    print("\n1. SIMD ReLU Activation")
    var tensor = Tensor.randn(1000)
    print("Processing", tensor.size(), "elements...")
    relu_simd(tensor)
    print("SIMD ReLU complete!")

    # Example 2: SIMD Batch Normalization
    print("\n2. SIMD Batch Normalization")
    var input_data = Tensor.randn(128, 256)
    var mean = Tensor.zeros(256, DType.float32)
    var variance = Tensor.ones(256, DType.float32)
    print("Normalizing tensor of shape", input_data.shape, "...")
    batch_norm_simd(input_data, mean, variance)
    print("SIMD batch norm complete!")

    # Example 3: SIMD Matrix Multiplication
    print("\n3. SIMD Matrix Multiplication")
    var a = Tensor.randn(100, 200)
    var b = Tensor.randn(200, 150)
    print("Multiplying", a.shape, "@", b.shape, "...")
    var c = matmul_simd(a, b)
    print("Result shape:", c.shape)
    print("SIMD matmul complete!")

    print("\n" + "=" * 50)
    print("All SIMD optimizations complete!")
    print("=" * 50)
