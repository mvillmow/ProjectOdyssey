"""Diagnostic script to understand dropout gradient issue."""

from shared.core.extensor import ExTensor, zeros, ones, ones_like
from shared.core.dropout import dropout, dropout_backward
from shared.testing import check_gradient


fn main() raises:
    """Diagnose dropout gradient checking failure."""

    # Same setup as test
    var shape = List[Int]()
    shape.append(5)
    var x = zeros(shape, DType.float32)

    # Set non-uniform values
    x._data.bitcast[Float32]()[0] = -0.5
    x._data.bitcast[Float32]()[1] = 0.0
    x._data.bitcast[Float32]()[2] = 0.2
    x._data.bitcast[Float32]()[3] = 0.5
    x._data.bitcast[Float32]()[4] = 1.0

    print("Input values:")
    for i in range(x.numel()):
        print(
            "  x[" + String(i) + "] = " + String(x._data.bitcast[Float32]()[i])
        )

    # Forward pass to create mask
    var (output, mask) = dropout(x, p=0.3, training=True, seed=42)

    print("\nMask values (seed=42):")
    for i in range(mask.numel()):
        print(
            "  mask["
            + String(i)
            + "] = "
            + String(mask._data.bitcast[Float32]()[i])
        )

    print("\nOutput values (x * mask / (1-p)):")
    var scale = 1.0 / (1.0 - 0.3)
    for i in range(output.numel()):
        var expected = (
            x._data.bitcast[Float32]()[i]
            * mask._data.bitcast[Float32]()[i]
            * Float32(scale)
        )
        var actual = output._data.bitcast[Float32]()[i]
        print(
            "  output["
            + String(i)
            + "] = "
            + String(actual)
            + " (expected: "
            + String(expected)
            + ")"
        )

    # Test backward
    var grad_out = ones_like(output)
    var grad_input = dropout_backward(grad_out, mask, p=0.3)

    print("\nGradient values (grad_out * mask / (1-p)):")
    for i in range(grad_input.numel()):
        var expected = (
            grad_out._data.bitcast[Float32]()[i]
            * mask._data.bitcast[Float32]()[i]
            * Float32(scale)
        )
        var actual = grad_input._data.bitcast[Float32]()[i]
        print(
            "  grad["
            + String(i)
            + "] = "
            + String(actual)
            + " (expected: "
            + String(expected)
            + ")"
        )

    # Test with another mask to verify consistency
    var (output2, mask2) = dropout(x, p=0.3, training=True, seed=42)

    print("\nMask values (second call with seed=42):")
    for i in range(mask2.numel()):
        var same = (
            mask._data.bitcast[Float32]()[i]
            == mask2._data.bitcast[Float32]()[i]
        )
        print(
            "  mask2["
            + String(i)
            + "] = "
            + String(mask2._data.bitcast[Float32]()[i])
            + " (same as first: "
            + String(same)
            + ")"
        )

    # Now test numerical gradient manually
    print("\nManual numerical gradient check:")
    var epsilon = 1e-5

    for i in range(x.numel()):
        # Forward perturbation
        var x_plus = zeros(shape, DType.float32)
        for j in range(x.numel()):
            x_plus._data.bitcast[Float32]()[j] = x._data.bitcast[Float32]()[j]
        x_plus._data.bitcast[Float32]()[i] += Float32(epsilon)

        var (out_plus, _) = dropout(x_plus, p=0.3, training=True, seed=42)
        var loss_plus: Float32 = 0.0
        for j in range(out_plus.numel()):
            loss_plus += (
                out_plus._data.bitcast[Float32]()[j]
                * grad_out._data.bitcast[Float32]()[j]
            )

        # Backward perturbation
        var x_minus = zeros(shape, DType.float32)
        for j in range(x.numel()):
            x_minus._data.bitcast[Float32]()[j] = x._data.bitcast[Float32]()[j]
        x_minus._data.bitcast[Float32]()[i] -= Float32(epsilon)

        var (out_minus, _) = dropout(x_minus, p=0.3, training=True, seed=42)
        var loss_minus: Float32 = 0.0
        for j in range(out_minus.numel()):
            loss_minus += (
                out_minus._data.bitcast[Float32]()[j]
                * grad_out._data.bitcast[Float32]()[j]
            )

        # Central difference
        var numerical_grad = (loss_plus - loss_minus) / (2.0 * Float32(epsilon))
        var analytical_grad = grad_input._data.bitcast[Float32]()[i]

        var diff = abs(numerical_grad - analytical_grad)
        var rel_error = diff / (abs(numerical_grad) + 1e-8)

        print("  Index " + String(i) + ":")
        print("    Numerical: " + String(numerical_grad))
        print("    Analytical: " + String(analytical_grad))
        print("    Difference: " + String(diff))
        print("    Relative error: " + String(rel_error))
