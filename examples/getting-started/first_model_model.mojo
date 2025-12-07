"""Example: First Model - Model Definition

Defines the DigitClassifier model architecture.

Usage:
    from model import DigitClassifier

See documentation: docs/getting-started/first_model.md

FIXME: This example does not compile. Issues:
1. API Mismatch: Imports Layer, Sequential, ReLU, Softmax classes that do
   NOT exist in shared.core. The actual library provides pure functional
   operations: relu(), softmax() functions, not classes.

2. Type issue: Imports Tensor from shared.core.types but actual export is
   ExTensor from shared.core.extensor

3. Syntax error: Uses incorrect `mut self` parameter syntax. Mojo requires
   either `mut self: Self` or just `self` for methods. The parameters on
   lines 20, 37, 41 are invalid.

4. Missing import: Uses List type without importing it

This example represents a high-level OOP design that doesn't match the
pure functional implementation. It needs complete redesign or the library
needs to implement these classes first.
"""

# FIXME: These imports don't exist in shared library
# from shared.core import Layer, Sequential, ReLU, Softmax
# from shared.core.types import Tensor


struct DigitClassifier:
    """Simple 3-layer neural network for digit classification."""

    var model: Sequential

    fn __init__(mut self):
        """Create a 3-layer network: 784 -> 128 -> 64 -> 10."""

        # Input: 784 pixels (28x28 flattened)
        # Hidden layer 1: 128 neurons with ReLU activation
        # Hidden layer 2: 64 neurons with ReLU activation
        # Output: 10 classes (digits 0-9) with Softmax

        self.model = Sequential(
            [
                Layer("linear", input_size=784, output_size=128),
                ReLU(),
                Layer("linear", input_size=128, output_size=64),
                ReLU(),
                Layer("linear", input_size=64, output_size=10),
                Softmax(),
            ]
        )

    fn forward(mut self, input: Tensor) -> Tensor:
        """Forward pass through the network."""
        return self.model.forward(input)

    fn parameters(mut self) -> List[Tensor]:
        """Get all trainable parameters."""
        return self.model.parameters()
