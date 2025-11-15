# Visualization Guide

Visualizing training metrics, model architecture, and results in ML Odyssey.

## Overview

ML Odyssey provides visualization utilities for understanding model behavior, debugging, and presenting results.
This guide covers plotting training curves, confusion matrices, feature maps, and more.

## Training Visualization

### Training Curves

Plot loss and accuracy over time:

```mojo
from shared.utils.visualization import plot_training_curves

fn visualize_training(history: TrainingHistory):
    """Plot training and validation curves."""

    plot_training_curves(
        train_losses=history.train_losses,
        val_losses=history.val_losses,
        train_metrics=history.train_accuracies,
        val_metrics=history.val_accuracies,
        metric_name="Accuracy",
        save_path="results/training_curves.png"
    )
```

Output:

![Training Curves Example](Training curves showing loss decreasing and accuracy increasing over epochs)

### Custom Metrics

Plot multiple metrics:

```mojo
from shared.utils.visualization import plot_metrics

fn plot_all_metrics(history: TrainingHistory):
    """Plot all tracked metrics."""

    var metrics = {
        "Loss": (history.train_losses, history.val_losses),
        "Accuracy": (history.train_accs, history.val_accs),
        "F1 Score": (history.train_f1, history.val_f1),
    }

    for name, (train, val) in metrics.items():
        plot_metrics(
            train_values=train,
            val_values=val,
            title=name + " Over Time",
            ylabel=name,
            save_path=f"results/{name.lower()}.png"
        )
```

### Learning Rate Schedule

Visualize learning rate changes:

```mojo
from shared.utils.visualization import plot_lr_schedule

fn visualize_lr_schedule():
    """Plot learning rate schedule."""
    var scheduler = CosineAnnealingLR(T_max=100, eta_min=1e-6)

    var lrs = []
    for epoch in range(100):
        lrs.append(scheduler.get_lr())
        scheduler.step()

    plot_lr_schedule(
        learning_rates=lrs,
        save_path="results/lr_schedule.png"
    )
```

## Model Visualization

### Architecture Diagram

Visualize model structure:

```mojo
from shared.utils.visualization import plot_model

fn visualize_architecture():
    """Create model architecture diagram."""
    var model = LeNet5()

    plot_model(
        model,
        input_shape=[1, 28, 28],
        save_path="results/architecture.png",
        show_shapes=True,
        show_layer_names=True
    )
```

### Parameter Distribution

Visualize weight distributions:

```mojo
from shared.utils.visualization import plot_weight_distribution

fn visualize_weights(model: LeNet5):
    """Plot distribution of model weights."""

    for i, param in enumerate(model.parameters()):
        plot_weight_distribution(
            weights=param,
            title=f"Layer {i} Weight Distribution",
            save_path=f"results/weights_layer_{i}.png"
        )
```

### Gradient Flow

Visualize gradient magnitudes:

```mojo
from shared.utils.visualization import plot_gradient_flow

fn visualize_gradients(model: LeNet5):
    """Plot gradient flow through network."""

    var layer_names = []
    var avg_grads = []

    for name, param in model.named_parameters():
        if param.grad is not None:
            layer_names.append(name)
            avg_grads.append(param.grad.abs().mean())

    plot_gradient_flow(
        layer_names=layer_names,
        gradients=avg_grads,
        save_path="results/gradient_flow.png"
    )
```

## Evaluation Visualization

### Confusion Matrix

Visualize classification results:

```mojo
from shared.utils.visualization import plot_confusion_matrix

fn visualize_predictions(model: LeNet5, test_loader: BatchLoader):
    """Plot confusion matrix for test set."""

    # Get predictions
    var all_preds = []
    var all_targets = []

    for batch in test_loader:
        var inputs, targets = batch
        var outputs = model.forward(inputs)
        var preds = outputs.argmax(dim=1)

        all_preds.extend(preds)
        all_targets.extend(targets)

    # Compute confusion matrix
    var cm = confusion_matrix(all_targets, all_preds)

    # Plot
    plot_confusion_matrix(
        cm=cm,
        class_names=["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"],
        title="MNIST Classification Results",
        save_path="results/confusion_matrix.png",
        normalize=True  # Show percentages
    )
```

### ROC Curve

For binary classification:

```mojo
from shared.utils.visualization import plot_roc_curve

fn visualize_roc(predictions: Tensor, targets: Tensor):
    """Plot ROC curve."""

    var fpr, tpr, thresholds = roc_curve(targets, predictions)
    var auc_score = auc(fpr, tpr)

    plot_roc_curve(
        fpr=fpr,
        tpr=tpr,
        auc_score=auc_score,
        title=f"ROC Curve (AUC = {auc_score:.3f})",
        save_path="results/roc_curve.png"
    )
```

### Precision-Recall Curve

```mojo
from shared.utils.visualization import plot_pr_curve

fn visualize_pr_curve(predictions: Tensor, targets: Tensor):
    """Plot precision-recall curve."""

    var precision, recall, thresholds = precision_recall_curve(targets, predictions)
    var ap_score = average_precision(precision, recall)

    plot_pr_curve(
        precision=precision,
        recall=recall,
        ap_score=ap_score,
        save_path="results/pr_curve.png"
    )
```

## Data Visualization

### Dataset Examples

Visualize training examples:

```mojo
from shared.utils.visualization import plot_image_grid

fn visualize_dataset(dataset: TensorDataset, num_samples: Int = 25):
    """Display grid of dataset samples."""

    var images = []
    var labels = []

    for i in range(num_samples):
        var img, label = dataset[i]
        images.append(img)
        labels.append(label)

    plot_image_grid(
        images=images,
        labels=labels,
        grid_size=(5, 5),
        title="MNIST Training Examples",
        save_path="results/dataset_samples.png"
    )
```

### Data Augmentation

Visualize augmentation effects:

```mojo
from shared.utils.visualization import plot_augmentation

fn visualize_augmentation(image: Tensor):
    """Show effect of data augmentation."""

    var transforms = [
        ("Original", identity),
        ("Random Crop", RandomCrop(28, padding=4)),
        ("Horizontal Flip", RandomHorizontalFlip()),
        ("Rotation", RandomRotation(15)),
        ("Color Jitter", ColorJitter(0.2, 0.2, 0.2)),
    ]

    var augmented_images = []
    var titles = []

    for name, transform in transforms:
        augmented_images.append(transform(image))
        titles.append(name)

    plot_augmentation(
        images=augmented_images,
        titles=titles,
        save_path="results/augmentation.png"
    )
```

## Feature Visualization

### Feature Maps

Visualize convolutional layer outputs:

```mojo
from shared.utils.visualization import plot_feature_maps

fn visualize_conv_features(model: LeNet5, input: Tensor):
    """Visualize convolutional layer activations."""

    # Get intermediate outputs
    var conv1_output = model.conv1.forward(input)

    plot_feature_maps(
        feature_maps=conv1_output[0],  # First image in batch
        num_features=6,  # Number of channels
        title="Conv1 Feature Maps",
        save_path="results/conv1_features.png"
    )
```

### Activation Heatmap

Visualize which parts of input activate neurons:

```mojo
from shared.utils.visualization import plot_activation_heatmap

fn visualize_activations(model: LeNet5, input: Tensor, layer_name: String):
    """Plot activation heatmap for specific layer."""

    # Get activations
    var activations = get_layer_activations(model, input, layer_name)

    plot_activation_heatmap(
        activations=activations,
        input_image=input,
        save_path=f"results/{layer_name}_heatmap.png"
    )
```

### t-SNE Visualization

Visualize high-dimensional embeddings:

```mojo
from shared.utils.visualization import plot_tsne

fn visualize_embeddings(model: LeNet5, data_loader: BatchLoader):
    """Visualize learned representations with t-SNE."""

    var embeddings = []
    var labels = []

    # Extract embeddings
    for batch in data_loader:
        var inputs, targets = batch
        var embedding = model.get_embedding(inputs)  # Before final layer
        embeddings.append(embedding)
        labels.append(targets)

    # Compute t-SNE
    var tsne_result = tsne(embeddings, n_components=2)

    # Plot
    plot_tsne(
        embeddings=tsne_result,
        labels=labels,
        class_names=["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"],
        save_path="results/tsne.png"
    )
```

## Interactive Visualization

### TensorBoard Integration

Log metrics for TensorBoard:

```mojo
from shared.utils import TensorBoardLogger

fn train_with_tensorboard():
    """Train with TensorBoard logging."""

    var logger = TensorBoardLogger(log_dir="runs/lenet5")

    for epoch in range(num_epochs):
        for i, batch in enumerate(train_loader):
            # Training step...
            var loss = train_step(model, batch)

            # Log scalar
            logger.add_scalar("Loss/train", loss, step=i)

            # Log images
            if i % 100 == 0:
                logger.add_images("Inputs", batch.images, step=i)

            # Log histogram
            logger.add_histogram("Weights/conv1", model.conv1.weight, step=i)

        # Log at epoch end
        logger.add_scalar("Accuracy/val", val_acc, step=epoch)

    logger.close()
```

View in TensorBoard:

```bash
pixi run tensorboard --logdir runs/
```

### Real-Time Plotting

Live updating plots during training:

```mojo
from shared.utils.visualization import LivePlot

fn train_with_live_plot():
    """Train with live updating plot."""

    var live_plot = LivePlot(
        metrics=["loss", "accuracy"],
        window_title="Training Progress"
    )

    for epoch in range(num_epochs):
        var train_loss, train_acc = train_epoch(model, train_loader)
        var val_loss, val_acc = validate(model, val_loader)

        # Update plot
        live_plot.update(
            epoch=epoch,
            train_loss=train_loss,
            train_acc=train_acc,
            val_loss=val_loss,
            val_acc=val_acc
        )

    live_plot.close()
```

## Styling and Customization

### Plot Style

Customize plot appearance:

```mojo
from shared.utils.visualization import set_plot_style

# Use predefined style
set_plot_style("seaborn")  # Options: "seaborn", "ggplot", "classic"

# Or custom style
set_plot_style({
    "figure.figsize": (10, 6),
    "font.size": 12,
    "axes.labelsize": 14,
    "axes.titlesize": 16,
    "legend.fontsize": 12,
    "lines.linewidth": 2,
})
```

### Color Schemes

Use consistent colors:

```mojo
from shared.utils.visualization import ColorScheme

var colors = ColorScheme.qualitative(num_colors=10)  # For classes
var gradient = ColorScheme.sequential("viridis", num_colors=64)  # Heatmaps
```

## Exporting Visualizations

### Multiple Formats

Save in different formats:

```mojo
fn save_visualization(figure: Figure, base_path: String):
    """Save figure in multiple formats."""
    figure.savefig(base_path + ".png", dpi=300)
    figure.savefig(base_path + ".pdf")
    figure.savefig(base_path + ".svg")
```

### High-Resolution

For publications:

```mojo
fn save_publication_figure(figure: Figure, path: String):
    """Save publication-quality figure."""
    figure.savefig(
        path,
        dpi=600,
        bbox_inches="tight",
        transparent=True,
        format="pdf"
    )
```

## Best Practices

### DO

- ✅ Visualize training progress regularly
- ✅ Save plots with descriptive names
- ✅ Use appropriate plot types for data
- ✅ Include titles, labels, and legends
- ✅ Normalize confusion matrices
- ✅ Show multiple metrics together

### DON'T

- ❌ Plot too many lines on one graph
- ❌ Use default colors for everything
- ❌ Skip axis labels
- ❌ Forget to close figures (memory leak)
- ❌ Use bitmap formats for publications
- ❌ Plot every single batch (too noisy)

## Next Steps

- **[Debugging](debugging.md)** - Debug models visually
- **[Performance Guide](performance.md)** - Profile with visualization
- **[Custom Layers](custom-layers.md)** - Visualize custom components
- **[Paper Implementation](../core/paper-implementation.md)** - Reproduce paper figures

## Related Documentation

- [Shared Library](../core/shared-library.md) - Visualization utilities
- [Testing Strategy](../core/testing-strategy.md) - Visual testing
- [First Model Tutorial](../getting-started/first_model.md) - Basic visualization
