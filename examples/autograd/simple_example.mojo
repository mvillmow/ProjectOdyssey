"""Simple autograd example demonstrating basic usage.

This example shows how to:
1. Create Variables with gradient tracking
2. Enable gradient tape recording
3. Perform operations (will be recorded when fully implemented)
4. Compute gradients via backward pass
5. Update parameters using an optimizer

Note:
    This is a minimal example showing the API design. Full autograd functionality
    (automatic operation recording and backward pass) is still being implemented.
    Currently, this demonstrates the interface and manual gradient computation.
"""

from shared.autograd import Variable, GradientTape, SGD
from shared.core.extensor import ExTensor, zeros, ones
from shared.core.arithmetic import add, multiply, subtract
from shared.core.reduction import sum as tensor_sum, mean
from shared.core.loss import mean_squared_error, mean_squared_error_backward
from shared.core.reduction import mean_backward


fn simple_linear_regression() raises:
    """Train a simple linear model using manual gradients.

    Model: y = w * x + b

    This example demonstrates the training loop structure that will work
    automatically once autograd operation recording is fully implemented.
    """
    print("=== Simple Linear Regression with Autograd ===\n")

    # Hyperparameters
    var learning_rate: Float64 = 0.01
    var num_epochs: Int = 10

    # Create gradient tape (needed for Variable creation)
    var tape = GradientTape()
    tape.enable()

    # Create parameters
    var w_data = zeros(List[Int](), DType.float32)
    w_data._set_float64(0, 0.5)  # Initialize w = 0.5
    var w = Variable(w_data, requires_grad=True, tape=tape)

    var b_data = zeros(List[Int](), DType.float32)
    b_data._set_float64(0, 0.0)  # Initialize b = 0.0
    var b = Variable(b_data, requires_grad=True, tape=tape)

    print("Initial parameters:")
    print("  w =", w.data._get_float64(0))
    print("  b =", b.data._get_float64(0))
    print()

    # Create simple training data: y = 2*x + 1
    # X = [1, 2, 3, 4, 5]
    # Y = [3, 5, 7, 9, 11]
    var X = ExTensor(List[Int](), DType.float32)
    for i in range(5):
        X._set_float64(i, Float64(i + 1))

    var Y = ExTensor(List[Int](), DType.float32)
    for i in range(5):
        Y._set_float64(i, Float64(2 * (i + 1) + 1))

    # Create optimizer
    var optimizer = SGD(learning_rate)

    print("Training for", num_epochs, "epochs...")
    print()

    # Training loop
    for epoch in range(num_epochs):
        # Forward pass: predictions = w * X + b
        var w_expanded = ExTensor(List[Int](), DType.float32)
        for i in range(5):
            w_expanded._set_float64(i, w.data._get_float64(0))

        var b_expanded = ExTensor(List[Int](), DType.float32)
        for i in range(5):
            b_expanded._set_float64(i, b.data._get_float64(0))

        var wx = multiply(w_expanded, X)
        var predictions = add(wx, b_expanded)

        # Compute loss: MSE = mean((predictions - targets)^2)
        var squared_errors = mean_squared_error(predictions, Y)
        var loss = mean(squared_errors, axis=0, keepdims=False)

        # Backward pass (manual gradients for now)
        # TODO(#2725): Replace with tape.backward() when fully implemented

        # Gradient of loss (scalar): ∂loss/∂loss = 1
        var grad_loss = ones(loss.shape(), loss.dtype())

        # Gradient of mean: ∂loss/∂squared_errors
        var grad_squared_errors = mean_backward(
            grad_loss, squared_errors, axis=-1
        )

        # Gradient of MSE: ∂squared_errors/∂predictions
        var grad_predictions = mean_squared_error_backward(
            grad_squared_errors, predictions, Y
        )

        # Gradient of w: ∂predictions/∂w = X (since predictions = w*X + b)
        # So: ∂loss/∂w = sum(∂loss/∂predictions * X)
        var grad_w_expanded = multiply(grad_predictions, X)
        var grad_w_sum = tensor_sum(grad_w_expanded, axis=0, keepdims=False)

        # Gradient of b: ∂predictions/∂b = 1 (since predictions = w*X + b)
        # So: ∂loss/∂b = sum(∂loss/∂predictions)
        var grad_b_sum = tensor_sum(grad_predictions, axis=0, keepdims=False)

        # Store gradients in tape
        tape.registry.set_grad(w.id, grad_w_sum)
        tape.registry.set_grad(b.id, grad_b_sum)

        # Update parameters using optimizer
        var params: List[Variable] = []
        params.append(w.copy())
        params.append(b.copy())
        optimizer.step(params, tape)

        # Reset gradients
        optimizer.zero_grad(tape)

        # Print progress
        var loss_value = loss._get_float64(0)
        var w_value = w.data._get_float64(0)
        var b_value = b.data._get_float64(0)

        print("Epoch", epoch + 1, "/", num_epochs)
        print("  Loss:", loss_value)
        print("  w:", w_value)
        print("  b:", b_value)
        print()

    print("Training complete!")
    print()
    print("Final parameters (target: w=2.0, b=1.0):")
    print("  w =", w.data._get_float64(0))
    print("  b =", b.data._get_float64(0))
    print()


fn main() raises:
    """Run autograd examples."""
    simple_linear_regression()
