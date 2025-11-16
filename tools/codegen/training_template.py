#!/usr/bin/env python3

"""
Tool: codegen/training_template.py
Purpose: Generate training loop boilerplate code

Language: Python
Justification:
  - Template processing with string substitution
  - String manipulation for code generation
  - No performance requirements (one-time generation)
  - No ML/AI computation involved

Reference: ADR-001
Last Review: 2025-11-16
"""

from typing import Dict, List
import argparse
import sys


def generate_training_loop(
    optimizer: str = "SGD",
    loss_fn: str = "CrossEntropy",
    metrics: List[str] = None
) -> str:
    """
    Generate a basic training loop template.

    Args:
        optimizer: Optimizer name (SGD, Adam, etc.)
        loss_fn: Loss function name
        metrics: List of metric names to track

    Returns:
        Generated Mojo training loop code
    """
    if metrics is None:
        metrics = ["loss"]

    metrics_str = ", ".join(metrics)

    template = f"""fn train_epoch(
    inout model: Model,
    borrowed train_data: DataLoader,
    inout optimizer: {optimizer},
    borrowed loss_fn: {loss_fn}
) -> TrainingMetrics:
    '''Train model for one epoch.'''
    var total_loss: Float64 = 0.0
    var num_batches: Int = 0

    # Training loop
    for batch in train_data:
        # Forward pass
        let outputs = model.forward(batch.inputs)

        # Compute loss
        let loss = loss_fn.compute(outputs, batch.targets)
        total_loss += loss

        # Backward pass
        let gradients = model.backward(loss)

        # Update weights
        optimizer.step(model, gradients)

        num_batches += 1

    # Compute metrics
    let avg_loss = total_loss / num_batches

    return TrainingMetrics(
        loss=avg_loss,
        # TODO: Add other metrics: {metrics_str}
    )


fn train(
    inout model: Model,
    borrowed train_data: DataLoader,
    borrowed val_data: DataLoader,
    num_epochs: Int,
    learning_rate: Float64
) -> List[TrainingMetrics]:
    '''Main training function.'''
    var optimizer = {optimizer}(learning_rate)
    var loss_fn = {loss_fn}()
    var history = List[TrainingMetrics]()

    for epoch in range(num_epochs):
        print("Epoch", epoch + 1, "/", num_epochs)

        # Train
        let train_metrics = train_epoch(model, train_data, optimizer, loss_fn)
        print("  Train loss:", train_metrics.loss)

        # Validate
        let val_metrics = validate(model, val_data, loss_fn)
        print("  Val loss:", val_metrics.loss)

        history.append(train_metrics)

    return history


fn validate(
    borrowed model: Model,
    borrowed val_data: DataLoader,
    borrowed loss_fn: {loss_fn}
) -> TrainingMetrics:
    '''Validate model on validation set.'''
    var total_loss: Float64 = 0.0
    var num_batches: Int = 0

    for batch in val_data:
        let outputs = model.forward(batch.inputs)
        let loss = loss_fn.compute(outputs, batch.targets)
        total_loss += loss
        num_batches += 1

    let avg_loss = total_loss / num_batches
    return TrainingMetrics(loss=avg_loss)
"""
    return template


def main() -> int:
    """Main CLI interface."""
    parser = argparse.ArgumentParser(
        description="Generate training loop boilerplate code"
    )

    parser.add_argument(
        "--optimizer",
        default="SGD",
        help="Optimizer name (default: SGD)"
    )
    parser.add_argument(
        "--loss",
        default="CrossEntropy",
        help="Loss function name (default: CrossEntropy)"
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=["loss"],
        help="Metrics to track (default: loss)"
    )

    args = parser.parse_args()

    try:
        code = generate_training_loop(
            optimizer=args.optimizer,
            loss_fn=args.loss,
            metrics=args.metrics
        )
        print(code)
        return 0

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
