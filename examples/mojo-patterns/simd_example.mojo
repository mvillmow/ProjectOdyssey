"""Example: Mojo Patterns - SIMD Optimization

This example demonstrates vectorized operations using SIMD.

Usage:
    pixi run mojo run examples/mojo-patterns/simd_example.mojo

See documentation: docs/core/mojo-patterns.md
"""

from algorithm import vectorize
from sys.info import simdwidthof
from shared.core.types import Tensor


fn relu_simd(inout tensor: Tensor):
    """ReLU activation using SIMD."""
    alias simd_width = simdwidthof[DType.float32]()

    @parameter
    fn vectorized_relu[width: Int](idx: Int):
        var val = tensor.data.simd_load[width](idx)
        tensor.data.simd_store[width](idx, max(val, 0.0))

    vectorize[simd_width, vectorized_relu](tensor.size())


fn matmul_simd(borrowed a: Tensor, borrowed b: Tensor) -> Tensor:
    """Matrix multiplication using SIMD."""
    var result = Tensor.zeros(a.shape[0], b.shape[1], DType.float32)

    alias simd_width = simdwidthof[DType.float32]()

    for i in range(a.shape[0]):
        for j in range(b.shape[1]):
            @parameter
            fn dot_product[width: Int](k: Int):
                var a_vec = a.data.simd_load[width](i * a.shape[1] + k)
                var b_vec = b.data.simd_load[width](k * b.shape[1] + j)
                result[i, j] += (a_vec * b_vec).reduce_add()

            vectorize[simd_width, dot_product](a.shape[1])

    return result


fn main() raises:
    """Demonstrate SIMD optimization."""

    # Example 1: SIMD ReLU
    var tensor = Tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
    print("Before ReLU:", tensor)
    relu_simd(tensor)
    print("After ReLU:", tensor)

    # Example 2: SIMD Matrix Multiplication
    var a = Tensor.randn(10, 20)
    var b = Tensor.randn(20, 15)
    var c = matmul_simd(a, b)
    print("\nMatrix multiplication result shape:", c.shape)

    print("\nSIMD example complete!")
