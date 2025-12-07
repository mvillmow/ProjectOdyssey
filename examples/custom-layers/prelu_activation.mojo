"""Example: Custom Layers - Parametric ReLU

This example implements a custom activation layer with learnable parameters.

Usage:
    pixi run mojo run examples/custom-layers/prelu_activation.mojo

See documentation: docs/advanced/custom-layers.md
"""

from shared.core import Module, Tensor


struct PReLU(Module):
    """Parametric ReLU activation.

    Formula: PReLU(x) = max(0, x) + α * min(0, x).
   """
    var alpha: Tensor  # Learnable slope for negative values

    fn __init__(out self, num_features: Int = 1, init_value: Float64 = 0.25):
        """Initialize PReLU.

        Args:
            num_features: Number of parameters (1 for shared, or per-channel).
            init_value: Initial value for alpha.
        """
        self.alpha = Tensor.ones(num_features, DType.float32) * init_value

    fn forward(mut self, input: Tensor) -> Tensor:
        """Forward pass.

        Returns:
            PReLU(input) = max(0, input) + α * min(0, input).
       """
        var positive = max(input, 0.0)
        var negative = min(input, 0.0)

        # Broadcast alpha if needed
        if self.alpha.size() == 1:
            return positive + self.alpha[0] * negative
        else:
            # Per-channel alpha
            return positive + self.alpha.reshape(1, -1, 1, 1) * negative

    fn parameters(mut self) -> List[Tensor]:
        """Return alpha for optimization."""
        return [self.alpha]


fn main() raises:
    """Demonstrate PReLU activation."""

    # Create PReLU layer
    var prelu = PReLU(num_features=1, init_value=0.25)

    # Test with sample input
    var input = Tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
    print("Input:", input)

    var output = prelu.forward(input)
    print("Output:", output)
    print("Expected: [-0.5, -0.25, 0.0, 1.0, 2.0]")

    # Show parameters
    var params = prelu.parameters()
    print("Alpha parameter:", params[0])

    print("\nPReLU example complete!")
