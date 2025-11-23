"""Test double (float64) data type support for sgd_momentum_update_inplace.

This test verifies that the sgd_momentum_update_inplace function now supports
both float32 and float64 dtypes.
"""

from shared.core.extensor import ExTensor, zeros_like
from shared.training.optimizers.sgd import sgd_momentum_update_inplace
from collections import List


fn test_sgd_momentum_float32() raises:
    """Test sgd_momentum_update_inplace with float32."""
    print("Testing sgd_momentum_update_inplace with float32...")

    # Create float32 tensors
    var shape = List[Int]()
    var param = ExTensor(shape, DType.float32)
    var grad = ExTensor(shape, DType.float32)
    var velocity = zeros_like(param)

    # Set initial values: param = [1.0, 2.0, 3.0]
    param._set_float64(0, 1.0)
    param._set_float64(1, 2.0)
    param._set_float64(2, 3.0)

    # Set gradients: grad = [0.1, 0.2, 0.3]
    grad._set_float64(0, 0.1)
    grad._set_float64(1, 0.2)
    grad._set_float64(2, 0.3)

    # Apply SGD with momentum
    sgd_momentum_update_inplace(param, grad, velocity, lr=0.1, momentum=0.9)

    # Verify param was updated
    var p0 = param._get_float64(0)
    var p1 = param._get_float64(1)
    var p2 = param._get_float64(2)

    print("  Updated params (float32): [", p0, ",", p1, ",", p2, "]")

    # velocity = 0.9 * 0 - 0.1 * grad = -0.1 * [0.1, 0.2, 0.3] = [-0.01, -0.02, -0.03]
    # param = param + velocity = [1, 2, 3] + [-0.01, -0.02, -0.03] = [0.99, 1.98, 2.97]
    if abs(p0 - 0.99) > 0.001:
        raise Error("Float32 param[0] should be ~0.99, got " + String(p0))
    if abs(p1 - 1.98) > 0.001:
        raise Error("Float32 param[1] should be ~1.98, got " + String(p1))
    if abs(p2 - 2.97) > 0.001:
        raise Error("Float32 param[2] should be ~2.97, got " + String(p2))

    print("  ✓ Float32 test passed")


fn test_sgd_momentum_float64() raises:
    """Test sgd_momentum_update_inplace with float64."""
    print("Testing sgd_momentum_update_inplace with float64...")

    # Create float64 tensors
    var shape = List[Int]()
    var param = ExTensor(shape, DType.float64)
    var grad = ExTensor(shape, DType.float64)
    var velocity = zeros_like(param)

    # Set initial values: param = [1.0, 2.0, 3.0]
    param._set_float64(0, 1.0)
    param._set_float64(1, 2.0)
    param._set_float64(2, 3.0)

    # Set gradients: grad = [0.1, 0.2, 0.3]
    grad._set_float64(0, 0.1)
    grad._set_float64(1, 0.2)
    grad._set_float64(2, 0.3)

    # Apply SGD with momentum
    sgd_momentum_update_inplace(param, grad, velocity, lr=0.1, momentum=0.9)

    # Verify param was updated
    var p0 = param._get_float64(0)
    var p1 = param._get_float64(1)
    var p2 = param._get_float64(2)

    print("  Updated params (float64): [", p0, ",", p1, ",", p2, "]")

    # velocity = 0.9 * 0 - 0.1 * grad = -0.1 * [0.1, 0.2, 0.3] = [-0.01, -0.02, -0.03]
    # param = param + velocity = [1, 2, 3] + [-0.01, -0.02, -0.03] = [0.99, 1.98, 2.97]
    if abs(p0 - 0.99) > 0.001:
        raise Error("Float64 param[0] should be ~0.99, got " + String(p0))
    if abs(p1 - 1.98) > 0.001:
        raise Error("Float64 param[1] should be ~1.98, got " + String(p1))
    if abs(p2 - 2.97) > 0.001:
        raise Error("Float64 param[2] should be ~2.97, got " + String(p2))

    print("  ✓ Float64 test passed")


fn test_sgd_momentum_multi_step_float64() raises:
    """Test multiple SGD momentum updates with float64."""
    print("Testing multi-step SGD momentum with float64...")

    # Create float64 tensors
    var shape = List[Int]()
    var param = ExTensor(shape, DType.float64)
    var grad = ExTensor(shape, DType.float64)
    var velocity = zeros_like(param)

    # Set initial values: param = [10.0, 20.0]
    param._set_float64(0, 10.0)
    param._set_float64(1, 20.0)

    # Set gradients: grad = [1.0, 2.0]
    grad._set_float64(0, 1.0)
    grad._set_float64(1, 2.0)

    # Apply SGD multiple times
    for i in range(5):
        sgd_momentum_update_inplace(param, grad, velocity, lr=0.1, momentum=0.9)

    # Verify params decreased
    var p0 = param._get_float64(0)
    var p1 = param._get_float64(1)

    print("  Final params after 5 steps (float64): [", p0, ",", p1, "]")

    # Params should be less than initial values
    if p0 >= 10.0:
        raise Error("Float64 param[0] should decrease from 10.0")
    if p1 >= 20.0:
        raise Error("Float64 param[1] should decrease from 20.0")

    print("  ✓ Multi-step float64 test passed")


fn main() raises:
    """Run all double support tests."""
    print("=" * 60)
    print("Double (float64) Data Type Support Tests")
    print("=" * 60)

    test_sgd_momentum_float32()
    test_sgd_momentum_float64()
    test_sgd_momentum_multi_step_float64()

    print("=" * 60)
    print("All double support tests passed! ✓")
    print("=" * 60)
