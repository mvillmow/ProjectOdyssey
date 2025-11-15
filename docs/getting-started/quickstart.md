# Quickstart Guide

Get up and running with ML Odyssey in 5 minutes.

## Prerequisites

- Linux or macOS (WSL2 on Windows)
- Python 3.10 or later
- Git

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/mvillmow/ml-odyssey.git
cd ml-odyssey
```

### 2. Install Pixi

ML Odyssey uses [Pixi](https://pixi.sh) for environment management:

```bash
curl -fsSL https://pixi.sh/install.sh | bash
```

### 3. Set Up Environment

```bash
pixi install
```

This installs all dependencies including Mojo, Python packages, and development tools.

## Verify Installation

Test that everything is working:

```bash
# Test Mojo installation
pixi run mojo --version

# Run tests
pixi run pytest tests/

# Format code
pixi run mojo format shared/
```

## Your First Model

Let's create a simple neural network using the shared library.

See `examples/getting-started/quickstart_example.mojo`](

Key concepts demonstrated:

```mojo
# Create a simple network
var model = Sequential([
    Layer("linear", input_size=784, output_size=128),
    Layer("relu"),
    Layer("linear", input_size=128, output_size=10),
])

# Train the model
var optimizer = SGD(learning_rate=0.01)
var trainer = Trainer(model, optimizer)
trainer.train(train_loader, epochs=10)
```

Full example: `examples/getting-started/quickstart_example.mojo`

Run the example:

```bash
pixi run mojo run examples/getting-started/quickstart_example.mojo
```

## Next Steps

Now that you have ML Odyssey running:

- **[Installation Guide](installation.md)** - Detailed setup instructions
- **[First Model Tutorial](first_model.md)** - Complete walkthrough of building a model
- **[Project Structure](../core/project-structure.md)** - Understanding the codebase
- **[Mojo Patterns](../core/mojo-patterns.md)** - Learn Mojo-specific ML patterns

## Common Issues

### Mojo Not Found

If `mojo` command is not found:

```bash
# Ensure Pixi environment is activated
pixi shell

# Or use pixi run prefix
pixi run mojo --version
```

### Import Errors

If you see import errors:

```bash
# Reinstall dependencies
pixi install --force-reinstall

# Verify shared library is accessible
ls -la shared/
```

### Performance Issues

For optimal performance:

- Use release builds: `pixi run mojo build --release`
- Enable SIMD optimizations in your code
- See [Performance Guide](../advanced/performance.md)

## Getting Help

- [Documentation](../index.md) - Complete documentation
- [GitHub Issues](https://github.com/mvillmow/ml-odyssey/issues) - Report bugs or ask questions
- [Contributing](https://github.com/mvillmow/ml-odyssey/blob/main/CONTRIBUTING.md) - Contribution guidelines
