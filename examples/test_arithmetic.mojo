"""Simple arithmetic test for ExTensor.

Tests basic arithmetic operations with operator overloading.
"""

from shared.core import ExTensor, zeros, ones, full


fn main() raises:
    """Test basic arithmetic operations."""
    print("ExTensor Arithmetic Test")
    print("=" * 50)

    # Create test tensors
    var shape = List[Int]()
    var a = ones(shape, DType.float32)
    var b = full(shape, 2.0, DType.float32)

    print("\nCreated tensors:")
    print("  a = ones((2, 3)) - all values are 1.0")
    print("  b = full((2, 3), 2.0) - all values are 2.0")

    # Test addition
    print("\nTesting addition: c = a + b")
    var c = a + b
    print("  Expected: all 3.0")
    print("  Result numel:", c.numel())
    print("  Result shape:", c.shape()[0], "x", c.shape()[1])
    print("  First element:", c._get_float64(0))

    # Test subtraction
    print("\nTesting subtraction: d = b - a")
    var d = b - a
    print("  Expected: all 1.0")
    print("  First element:", d._get_float64(0))

    # Test multiplication
    print("\nTesting multiplication: e = a * b")
    var e = a * b
    print("  Expected: all 2.0")
    print("  First element:", e._get_float64(0))

    # Test chained operations
    print("\nTesting chained: f = (a + b) * b")
    var f = (a + b) * b
    print("  Expected: all 6.0  [(1 + 2) * 2 = 6]")
    print("  First element:", f._get_float64(0))

    # Test zeros
    print("\nTesting zeros addition: g = a + zeros")
    var z = zeros(shape, DType.float32)
    var g = a + z
    print("  Expected: all 1.0")
    print("  First element:", g._get_float64(0))

    print("\n" + "=" * 50)
    print("All tests completed!")
    print("\nNote: Broadcasting not yet implemented")
    print("  - Only same-shape operations work")
    print("  - Scalar broadcasting: TODO")
