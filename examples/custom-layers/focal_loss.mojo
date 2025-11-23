"""Example: Custom Layers - Focal Loss

This example implements a custom loss function for imbalanced datasets.

Usage:
    pixi run mojo run examples/custom-layers/focal_loss.mojo

See documentation: docs/advanced/custom-layers.md
"""

from shared.core.types import Tensor


struct FocalLoss:
    """Focal loss for addressing class imbalance.

    FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)

    Reference:
        Lin et al., "Focal Loss for Dense Object Detection", ICCV 2017
    """
    var alpha: Float64  # Weighting factor
    var gamma: Float64  # Focusing parameter

    fn __init__(out self, alpha: Float64 = 0.25, gamma: Float64 = 2.0):
        self.alpha = alpha
        self.gamma = gamma

    fn __call__(self, borrowed predictions: Tensor, borrowed targets: Tensor) -> Tensor:
        """Compute focal loss.

        Args:
            predictions: Model predictions (after softmax), shape [batch, num_classes].
            targets: Ground truth labels, shape [batch].

        Returns:
            Scalar loss value.
        """
        var batch_size = predictions.shape[0]
        var num_classes = predictions.shape[1]

        # Get predicted probabilities for true class
        var p_t = Tensor.zeros(batch_size, DType.float32)
        for i in range(batch_size):
            var true_class = int(targets[i])
            p_t[i] = predictions[i, true_class]

        # Focal loss: -α * (1 - p_t)^γ * log(p_t)
        var alpha_t = self.alpha
        var focal_weight = (1.0 - p_t) ** self.gamma
        var ce_loss = -log(p_t + 1e-7)  # Add epsilon for stability

        var loss = alpha_t * focal_weight * ce_loss
        return loss.mean()


fn main() raises:
    """Demonstrate focal loss."""

    # Create focal loss
    var loss_fn = FocalLoss(alpha=0.25, gamma=2.0)

    # Test with sample predictions and targets
    var predictions = Tensor([
        [0.1, 0.2, 0.7],  # Correct class: 2
        [0.8, 0.1, 0.1],  # Correct class: 0
        [0.3, 0.3, 0.4],  # Correct class: 1
    ])
    var targets = Tensor([2.0, 0.0, 1.0])

    print("Predictions shape:", predictions.shape)
    print("Targets shape:", targets.shape)

    var loss = loss_fn(predictions, targets)
    print("Focal loss:", loss)

    print("\nFocal loss example complete!")
