"""Example: Mojo Patterns - Ownership and Borrowing

This example demonstrates Mojo's ownership system for memory safety.

Usage:
    pixi run mojo run examples/mojo-patterns/ownership_example.mojo

See documentation: docs/core/mojo-patterns.md
"""

from shared.core.types import Tensor


# Borrowed: read-only access (no ownership transfer)
fn compute_loss(predictions: Tensor, targets: Tensor) -> Float64:
    """Compute loss without taking ownership."""
    var diff = predictions - targets
    return (diff * diff).mean()


# Owned: take ownership (move semantics)
fn consume_tensor(var tensor: Tensor) -> Float64:
    """Take ownership and consume tensor."""
    var result = tensor.sum()
    # tensor is destroyed here
    return result


# Inout: mutable reference (modify in place)
fn update_weights(mut weights: Tensor, gradients: Tensor, lr: Float64):
    """Update weights in place."""
    weights -= lr * gradients  # Modifies original


fn main() raises:
    """Demonstrate ownership patterns."""

    # Example 1: Borrowed parameters (read-only)
    var pred = Tensor.randn(10, 10)
    var target = Tensor.randn(10, 10)
    var loss = compute_loss(pred, target)  # No ownership transfer
    print("Loss (borrowed):", loss)

    # Example 2: Owned parameter (transfer ownership)
    var temp_tensor = Tensor.randn(5, 5)
    var sum_value = consume_tensor(temp_tensor)  # temp_tensor is consumed
    print("Sum (owned):", sum_value)
    # Cannot use temp_tensor here - it was consumed!

    # Example 3: Inout parameter (mutable reference)
    var weights = Tensor.randn(10, 10)
    var grads = Tensor.randn(10, 10)
    print("Weights before update:", weights[0, 0])
    update_weights(weights, grads, 0.01)  # Modifies weights in place
    print("Weights after update:", weights[0, 0])

    print("\nOwnership example complete!")
