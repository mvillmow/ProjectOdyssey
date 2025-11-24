"""Example: Mojo Patterns - Trait-Based Design

This example demonstrates creating reusable interfaces with traits.

Usage:
    pixi run mojo run examples/mojo-patterns/trait_example.mojo

See documentation: docs/core/mojo-patterns.md
"""

from shared.core.types import Tensor


trait Module:
    """Base trait for neural network modules."""

    fn forward(mut self, input: Tensor) -> Tensor:
        """Forward pass."""
        ...

    fn parameters(mut self) -> List[Tensor]:
        """Get trainable parameters."""
        ...


trait Optimizer:
    """Base trait for optimizers."""

    fn step(self, inout parameters: List[Tensor]):
        """Update parameters."""
        ...

    fn zero_grad(self, inout parameters: List[Tensor]):
        """Zero gradients."""
        for i in range(len(parameters)):
            parameters[i].grad = Tensor.zeros_like(parameters[i])


struct Linear(Module):
    """Linear layer implementing Module trait."""
    var weight: Tensor
    var bias: Tensor

    fn __init__(mut self, input_size: Int, output_size: Int):
        self.weight = Tensor.randn(output_size, input_size)
        self.bias = Tensor.zeros(output_size, DType.float32)

    fn forward(mut self, input: Tensor) -> Tensor:
        return input @ self.weight.T + self.bias

    fn parameters(mut self) -> List[Tensor]:
        return [self.weight, self.bias]


struct Adam(Optimizer):
    """Adam optimizer implementing Optimizer trait."""
    var lr: Float64
    var beta1: Float64
    var beta2: Float64

    fn __init__(mut self, lr: Float64 = 0.001, beta1: Float64 = 0.9, beta2: Float64 = 0.999):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2

    fn step(self, inout parameters: List[Tensor]):
        # Adam update logic
        for i in range(len(parameters)):
            # Update with momentum and adaptive learning rate
            parameters[i] -= self.lr * parameters[i].grad


fn main() raises:
    """Demonstrate trait-based design."""
    var layer = Linear(784, 128)
    var optimizer = Adam(lr=0.001)

    var input = Tensor.randn(32, 784)
    var output = layer.forward(input)

    print("Layer output shape:", output.shape())
    print("Trait-based design example complete!")
