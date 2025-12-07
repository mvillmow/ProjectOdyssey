"""
Training Library

The training library provides reusable training infrastructure including optimizers,
schedulers, metrics, callbacks, and training loops for ML Odyssey paper implementations.

All components are implemented in Mojo for maximum performance.

FIXME: Placeholder import tests in tests/shared/test_imports.mojo require:
- test_training_imports (line 80+)
- test_training_optimizers_imports (line 95+)
- test_training_schedulers_imports (line 110+)
- test_training_metrics_imports (line 125+)
- test_training_callbacks_imports (line 140+)
- test_training_loops_imports (line 155+)
All tests marked as "(placeholder)" and require uncommented imports as Issue #49 progresses.
See Issue #49 for details
"""

from python import PythonObject
from shared.core.extensor import ExTensor
from shared.core.traits import Model, Loss, Optimizer

# Package version
from ..version import VERSION

# ============================================================================
# Exports - Training Components
# ============================================================================

# Export model utilities for weight persistence (Issue #2294)
from .model_utils import (
    save_model_weights,
    load_model_weights,
    get_model_parameter_names,
    validate_shapes,
)

# Export base interfaces and utilities
from .base import (
    Callback,
    CallbackSignal,
    CONTINUE,
    STOP,
    TrainingState,
    LRScheduler,
    has_nan_or_inf,
    compute_gradient_norm,
    is_valid_loss,
    clip_gradients,
)

# Export scheduler implementations
from .schedulers import StepLR, CosineAnnealingLR, WarmupLR, ReduceLROnPlateau

# Export callback implementations
# NOTE: Callbacks must be imported directly from submodules due to Mojo limitations:
#   from shared.training.callbacks import EarlyStopping
# NOT from shared.training import EarlyStopping
from .callbacks import EarlyStopping, ModelCheckpoint, LoggingCallback

# ============================================================================
# Core Training Components (Issue #1939)
# ============================================================================

# SGD Optimizer
struct SGD(Optimizer, Movable):
    """Stochastic Gradient Descent optimizer.

    Implements the Optimizer trait for use with generic TrainingLoop.
    """
    var learning_rate: Float32

    fn __init__(out self, learning_rate: Float32):
        """Initialize SGD optimizer.

        Args:
            learning_rate: Learning rate for gradient descent.
        """
        self.learning_rate = learning_rate

    fn step(mut self, params: List[ExTensor]) raises:
        """Update parameters using gradients.

        Implements: param = param - learning_rate * grad

        Args:
            params: List of parameter tensors to update

        Note:
            This is a stub implementation for Issue #2397.
            Full gradient-based updates will be implemented later.
        """
        # TODO: Implement actual parameter updates when gradient system is ready
        pass

    fn zero_grad(mut self) raises:
        """Reset optimizer state.

        Note:
            SGD has no internal state, so this is a no-op.
        """
        pass


# MSELoss - Mean Squared Error Loss
struct MSELoss(Loss, Movable):
    """Mean Squared Error loss function for regression.

    Implements the Loss trait for use with generic TrainingLoop.
    """
    var reduction: String

    fn __init__(out self, reduction: String = "mean"):
        """Initialize MSE loss.

        Args:
            reduction: Type of reduction ('mean' or 'sum').
        """
        self.reduction = reduction

    fn compute(self, pred: ExTensor, target: ExTensor) raises -> ExTensor:
        """Compute MSE loss between predictions and targets.

        Implements the Loss trait interface.

        Args:
            pred: Model predictions
            target: Ground truth targets

        Returns:
            Scalar loss value as ExTensor

        Note:
            This is a stub implementation for Issue #2397.
            Full MSE computation will be implemented later.
        """
        # TODO: Implement actual MSE computation
        # For now, return a dummy scalar tensor
        from shared.core.extensor import zeros
        return zeros(List[Int](), DType.float32)

    fn forward(self, output: ExTensor, target: ExTensor) raises -> Float32:
        """Compute MSE loss (legacy interface for backward compatibility).

        Args:
            output: Model outputs.
            target: Target values.

        Returns:
            Loss value.
        """
        return Float32(0.0)

    fn backward(self, grad_output: ExTensor) raises -> ExTensor:
        """Compute gradient of loss.

        Args:
            grad_output: Upstream gradient.

        Returns:
            Gradient tensor.
        """
        # Placeholder implementation - returns gradient same shape as input
        return grad_output


# TrainingLoop - Main training orchestrator (Generic with Trait Bounds)
struct TrainingLoop[M: Model & Movable, L: Loss & Movable, O: Optimizer & Movable]:
    """Orchestrates training with forward/backward/optimize cycle.

    Generic training loop with compile-time type safety via trait bounds.
    Converts from PythonObject to pure Mojo types for zero-cost abstraction.

    Type Parameters:
        M: Model type (must implement Model trait)
        L: Loss type (must implement Loss trait)
        O: Optimizer type (must implement Optimizer trait)

    Example:
        ```mojo
        ar model = SimpleMLP(...)
        var optimizer = SGD(learning_rate=0.01)
        var loss_fn = MSELoss()
        var loop = TrainingLoop[SimpleMLP, MSELoss, SGD](model, optimizer, loss_fn)
        ```
    """
    var model: M
    var optimizer: O
    var loss_fn: L

    fn __init__(out self, var model: M, var optimizer: O, var loss_fn: L):
        """Initialize training loop with generic components.

        Args:
            model: Model implementing Model trait
            optimizer: Optimizer implementing Optimizer trait
            loss_fn: Loss function implementing Loss trait
        """
        self.model = model^
        self.optimizer = optimizer^
        self.loss_fn = loss_fn^

    fn step(mut self, inputs: ExTensor, targets: ExTensor) raises -> ExTensor:
        """Perform single training step.

        Implements the training loop cycle using trait methods:
        1. Forward pass: outputs = model.forward(inputs)  [Model trait]
        2. Loss computation: loss = loss_fn.compute(outputs, targets)  [Loss trait]
        3. Backward pass: compute gradients (TODO: when gradient system ready)
        4. Optimizer step: optimizer.step(params)  [Optimizer trait]
        5. Return loss value

        Args:
            inputs: Input ExTensor
            targets: Target ExTensor

        Returns:
            Scalar loss value as ExTensor

        Note:
            Uses trait methods for type-safe dispatch.
            Backward pass stub until gradient system is implemented.
        """
        # Forward pass via Model trait
        var outputs = self.forward(inputs)

        # Compute loss via Loss trait
        var loss_value = self.compute_loss(outputs, targets)

        # TODO: Backward pass when gradient system is ready
        # TODO: Optimizer step when gradient system is ready

        return loss_value^

    fn forward(mut self, inputs: ExTensor) raises -> ExTensor:
        """Execute forward pass via Model trait.

        Args:
            inputs: Input ExTensor

        Returns:
            Model output ExTensor
        """
        # Call model.forward() via Model trait
        return self.model.forward(inputs)

    fn compute_loss(self, outputs: ExTensor, targets: ExTensor) raises -> ExTensor:
        """Compute loss via Loss trait.

        Args:
            outputs: Model outputs
            targets: Ground truth targets

        Returns:
            Scalar loss value as ExTensor
        """
        # Call loss_fn.compute() via Loss trait
        return self.loss_fn.compute(outputs, targets)

    fn run_epoch(mut self, data_loader: PythonObject) raises -> Float32:
        """Run single epoch over dataset.

        Iterates through all batches in data loader and performs training
        step on each batch, returning the average loss for the epoch.

        Args:
            data_loader: Data loader providing batches (PythonObject for now)

        Returns:
            Average loss for the epoch

        Note:
            data_loader remains PythonObject until Track 4 implements
            Mojo data loading infrastructure.
        """
        var total_loss = Float64(0.0)
        var num_batches = Int(0)

        # TODO: Iterate through batches when Python integration is complete
        # The data_loader is currently a PythonObject, but step() requires ExTensor.
        # This will be implemented once the data loading infrastructure is ready.
        # For now, return 0.0 as a placeholder.
        _ = data_loader  # Suppress unused variable warning

        # Return average loss
        if num_batches > 0:
            return Float32(total_loss / Float64(num_batches))
        else:
            return Float32(0.0)


# Export validation loop
from .loops.validation_loop import ValidationLoop

# Export evaluation module (Issue #2352)
from .evaluation import (
    EvaluationResult,
    evaluate_model,
    evaluate_model_simple,
    evaluate_topk,
)

# Export mixed precision training utilities
from .mixed_precision import (
    GradientScaler,
    convert_to_fp32_master,
    update_model_from_master,
    check_gradients_finite,
    clip_gradients_by_norm,
    clip_gradients_by_value,
)

# Export precision configuration
from .precision_config import PrecisionConfig, PrecisionMode

# Export CrossEntropyLoss wrapper (wraps core.loss.cross_entropy)
struct CrossEntropyLoss(Loss, Movable):
    """Cross entropy loss function for classification.

    Implements the Loss trait for use with generic TrainingLoop.
    """
    var reduction: String

    fn __init__(out self, reduction: String = "mean"):
        """Initialize cross entropy loss.

        Args:
            reduction: Type of reduction ('mean' or 'sum').
        """
        self.reduction = reduction

    fn compute(self, pred: ExTensor, target: ExTensor) raises -> ExTensor:
        """Compute cross entropy loss between predictions and targets.

        Args:
            pred: Model predictions (logits)
            target: Ground truth targets (class indices or one-hot)

        Returns:
            Scalar loss value as ExTensor
        """
        from shared.core.loss import cross_entropy
        return cross_entropy(pred, target)

# ============================================================================
# Public API
# ============================================================================
