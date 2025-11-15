# Python Integration Guide

Integrating ML Odyssey with Python libraries and tools.

## Overview

While ML Odyssey is built in Mojo for performance, it integrates seamlessly with the Python ecosystem. This guide
covers using Python libraries for visualization, data loading, and interoperability with existing ML tools.

## Mojo-Python Interoperability

### Calling Python from Mojo

Import and use Python libraries:

```mojo
from python import Python

fn use_numpy() raises:
    """Use NumPy from Mojo."""

    # Import Python module
    var np = Python.import_module("numpy")

    # Create NumPy array
    var arr = np.array([1, 2, 3, 4, 5])

    print("NumPy array:", arr)
    print("Mean:", arr.mean())
    print("Sum:", arr.sum())

fn use_matplotlib() raises:
    """Use Matplotlib from Mojo."""

    var plt = Python.import_module("matplotlib.pyplot")
    var np = Python.import_module("numpy")

    # Create data
    var x = np.linspace(0, 10, 100)
    var y = np.sin(x)

    # Plot
    plt.plot(x, y)
    plt.title("Sine Wave")
    plt.savefig("sine.png")
```

### Type Conversion

Convert between Mojo and Python types:

```mojo
from python import Python, PythonObject

fn convert_types() raises:
    """Convert between Mojo and Python types."""
    var np = Python.import_module("numpy")

    # Mojo Tensor to NumPy array
    var mojo_tensor = Tensor.randn(10, 10)
    var numpy_array = tensor_to_numpy(mojo_tensor)

    # NumPy array to Mojo Tensor
    var np_arr = np.random.randn(10, 10)
    var mojo_tens = numpy_to_tensor(np_arr)

fn tensor_to_numpy(borrowed tensor: Tensor) raises -> PythonObject:
    """Convert Mojo Tensor to NumPy array."""
    var np = Python.import_module("numpy")

    # Get pointer to data
    var data_ptr = tensor.data.unsafe_ptr()
    var shape = tensor.shape

    # Create NumPy array from pointer
    return np.array(data_ptr, dtype=np.float32).reshape(shape)

fn numpy_to_tensor(numpy_array: PythonObject) -> Tensor:
    """Convert NumPy array to Mojo Tensor."""

    # Get shape
    var shape = [int(numpy_array.shape[i]) for i in range(len(numpy_array.shape))]

    # Create Mojo tensor
    var tensor = Tensor.zeros(shape)

    # Copy data
    for i in range(tensor.size()):
        tensor.data[i] = float(numpy_array.flat[i])

    return tensor
```

## Visualization with Python

### Matplotlib Integration

Plot training curves:

```mojo
from python import Python

fn plot_with_matplotlib(
    train_losses: List[Float64],
    val_losses: List[Float64]
) raises:
    """Plot training curves with Matplotlib."""

    var plt = Python.import_module("matplotlib.pyplot")

    var epochs = range(len(train_losses))

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_losses, label="Train Loss")
    plt.plot(epochs, val_losses, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Progress")
    plt.legend()
    plt.grid(True)
    plt.savefig("training_curves.png", dpi=300)
    print("Plot saved to training_curves.png")
```

### Seaborn for Advanced Plots

Create publication-quality visualizations:

```mojo
fn plot_confusion_matrix_seaborn(
    cm: Tensor,
    class_names: List[String]
) raises:
    """Plot confusion matrix with Seaborn."""

    var sns = Python.import_module("seaborn")
    var plt = Python.import_module("matplotlib.pyplot")

    # Convert to NumPy
    var cm_numpy = tensor_to_numpy(cm)

    # Create heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm_numpy,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names
    )
    plt.title("Confusion Matrix")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.savefig("confusion_matrix.png", dpi=300)
```

## Data Loading with Python

### Using PyTorch DataLoaders

Load data with PyTorch:

```mojo
from python import Python

struct TorchDataLoader:
    """Wrapper for PyTorch DataLoader."""
    var loader: PythonObject
    var iterator: PythonObject

    fn __init__(inout self, dataset_name: String, batch_size: Int) raises:
        var torch = Python.import_module("torch")
        var torchvision = Python.import_module("torchvision")

        # Load dataset
        var transform = torchvision.transforms.ToTensor()

        var dataset = torchvision.datasets.MNIST(
            root="data/",
            train=True,
            download=True,
            transform=transform
        )

        # Create DataLoader
        self.loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True
        )

        self.iterator = iter(self.loader)

    fn next(inout self) raises -> (Tensor, Tensor):
        """Get next batch."""

        try:
            var batch = next(self.iterator)
        except:
            # Restart iterator
            self.iterator = iter(self.loader)
            var batch = next(self.iterator)

        var images = numpy_to_tensor(batch[0].numpy())
        var labels = numpy_to_tensor(batch[1].numpy())

        return (images, labels)
```

Usage:

```mojo
fn train_with_torch_data() raises:
    """Train using PyTorch DataLoader."""

    var data_loader = TorchDataLoader("MNIST", batch_size=32)

    for epoch in range(10):
        for i in range(1000):  # 1000 batches per epoch
            var images, labels = data_loader.next()

            # Train with Mojo model
            var outputs = model.forward(images)
            var loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
```

## Experiment Tracking

### Weights & Biases Integration

Track experiments with W&B:

```mojo
from python import Python

struct WandBLogger:
    """Weights & Biases logger."""
    var wandb: PythonObject
    var run: PythonObject

    fn __init__(inout self, project: String, name: String, config: Dict) raises:
        self.wandb = Python.import_module("wandb")

        # Initialize run
        self.run = self.wandb.init(
            project=project,
            name=name,
            config=config
        )

    fn log(self, metrics: Dict[String, Float64], step: Int):
        """Log metrics."""
        self.wandb.log(metrics, step=step)

    fn log_image(self, key: String, image: Tensor):
        """Log image."""
        var img = tensor_to_numpy(image)
        self.wandb.log({key: self.wandb.Image(img)})

    fn finish(self):
        """Finish run."""
        self.run.finish()
```

Usage:

```mojo
fn train_with_wandb() raises:
    """Train with W&B logging."""

    var logger = WandBLogger(
        project="ml-odyssey",
        name="lenet5-mnist",
        config={
            "learning_rate": 0.01,
            "batch_size": 32,
            "epochs": 100,
        }
    )

    for epoch in range(100):
        var train_loss = train_epoch(model, train_loader)
        var val_loss, val_acc = validate(model, val_loader)

        logger.log({
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_accuracy": val_acc,
        }, step=epoch)

    logger.finish()
```

### TensorBoard Integration

Log to TensorBoard:

```mojo
from python import Python

struct TensorBoardLogger:
    """TensorBoard logger."""
    var writer: PythonObject

    fn __init__(inout self, log_dir: String) raises:
        var torch = Python.import_module("torch")
        self.writer = torch.utils.tensorboard.SummaryWriter(log_dir)

    fn add_scalar(self, tag: String, value: Float64, step: Int):
        """Log scalar value."""
        self.writer.add_scalar(tag, value, step)

    fn add_image(self, tag: String, image: Tensor, step: Int):
        """Log image."""
        var img = tensor_to_numpy(image)
        self.writer.add_image(tag, img, step)

    fn add_histogram(self, tag: String, values: Tensor, step: Int):
        """Log histogram."""
        var vals = tensor_to_numpy(values)
        self.writer.add_histogram(tag, vals, step)

    fn close(self):
        """Close writer."""
        self.writer.close()
```

## Model Interoperability

### Export to ONNX

Export Mojo model to ONNX format:

```mojo
fn export_to_onnx(
    model: Model,
    input_shape: List[Int],
    output_path: String
) raises:
    """Export model to ONNX format."""

    var torch = Python.import_module("torch")
    var onnx = Python.import_module("onnx")

    # Convert model to PyTorch
    var torch_model = mojo_to_torch(model)

    # Create dummy input
    var dummy_input = torch.randn(input_shape)

    # Export to ONNX
    torch.onnx.export(
        torch_model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"]
    )

    print("Model exported to", output_path)
```

### Load PyTorch Models

Import pretrained PyTorch models:

```mojo
fn load_pytorch_weights(
    inout mojo_model: Model,
    pytorch_path: String
) raises:
    """Load PyTorch weights into Mojo model."""

    var torch = Python.import_module("torch")

    # Load PyTorch checkpoint
    var checkpoint = torch.load(pytorch_path)
    var state_dict = checkpoint["state_dict"]

    # Copy weights
    for name, param in mojo_model.named_parameters():
        if name in state_dict:
            var torch_param = state_dict[name]
            param.copy_from(numpy_to_tensor(torch_param.cpu().numpy()))
            print("Loaded", name)
```

## Scientific Computing

### SciPy Integration

Use SciPy for optimization:

```mojo
fn optimize_with_scipy(
    objective_fn: fn(Tensor) -> Float64,
    initial_params: Tensor
) raises -> Tensor:
    """Optimize using SciPy."""

    var scipy = Python.import_module("scipy.optimize")

    # Wrapper for SciPy
    def py_objective(x):
        var mojo_x = numpy_to_tensor(x)
        return float(objective_fn(mojo_x))

    # Optimize
    var result = scipy.minimize(
        py_objective,
        tensor_to_numpy(initial_params),
        method="L-BFGS-B"
    )

    return numpy_to_tensor(result.x)
```

### Scikit-learn Integration

Use sklearn for preprocessing:

```mojo
fn preprocess_with_sklearn(data: Tensor) raises -> Tensor:
    """Preprocess data using scikit-learn."""

    var sklearn = Python.import_module("sklearn.preprocessing")

    # Convert to NumPy
    var numpy_data = tensor_to_numpy(data)

    # StandardScaler
    var scaler = sklearn.StandardScaler()
    var scaled = scaler.fit_transform(numpy_data)

    # Convert back
    return numpy_to_tensor(scaled)
```

## Best Practices

### DO

- ✅ Use Python for ecosystem integration
- ✅ Convert data at boundaries (not in hot loops)
- ✅ Cache Python module imports
- ✅ Use Python for visualization
- ✅ Leverage existing data loaders

### DON'T

- ❌ Call Python in performance-critical loops
- ❌ Pass data back and forth unnecessarily
- ❌ Reimplement standard utilities (use Python)
- ❌ Import modules repeatedly

## Performance Considerations

### Minimize Conversions

```mojo
# Bad: Convert every iteration
for batch in loader:
    var mojo_batch = python_to_mojo(batch)  # Slow!
    train_step(model, mojo_batch)

# Good: Convert once
var all_data = python_to_mojo(full_dataset)  # Convert once
for batch in batch_split(all_data):
    train_step(model, batch)  # Pure Mojo
```

### Batch Operations

```mojo
# Bad: Many small Python calls
for i in range(1000):
    var result = python_func(data[i])  # 1000 Python calls

# Good: Single batched call
var results = python_func(data)  # 1 Python call
```

## Next Steps

- **[Visualization](visualization.md)** - Python plotting libraries
- **[Debugging](debugging.md)** - Debug with Python tools
- **[Performance Guide](performance.md)** - Optimize Python integration
- **[Testing Strategy](../core/testing-strategy.md)** - Test integration code

## Related Documentation

- [Mojo Python Interop](https://docs.modular.com/mojo/manual/python/) - Official Mojo docs
- [Shared Library](../core/shared-library.md) - Integration utilities
- [First Model Tutorial](../getting-started/first_model.md) - Basic usage
