# Quick Start

Get up and running with ML Odyssey in 5 minutes.

## Prerequisites

Before starting, ensure you have completed the [Installation Guide](installation.md).

## Verify Installation

First, verify that Mojo and ML Odyssey are properly installed:

```bash

# Check Mojo version
pixi run mojo --version

# Verify you're in the repository
cd ml-odyssey
ls shared/

```text

Expected output:

```text

mojo 25.1.0 (...)
core  data  training  utils

```text

## Run Your First Example

Let's create and run a simple neural network layer:

```bash

# Create example file
cat > hello_ml.mojo << 'EOF'
from shared.core import Layer, ReLU
from shared.utils import create_tensor

fn main() raises:
    print("=== ML Odyssey Quick Start ===\n")

    # Create a simple linear layer
    var layer = Layer("linear", input_size=10, output_size=5)
    var activation = ReLU()

    # Create sample input (batch of 3 examples, 10 features each)
    var input_data = create_tensor(3, 10, fill_value=0.5)

    # Forward pass
    var output = layer.forward(input_data)
    var activated = activation.forward(output)

    print("Input shape:  ", input_data.shape())
    print("Output shape: ", activated.shape())
    print("\nSuccess! Your ML Odyssey environment is ready.")

main()
EOF

# Run the example
pixi run mojo run hello_ml.mojo

```text

Expected output:

```text

=== ML Odyssey Quick Start ===

Input shape:  (3, 10)
Output shape: (3, 5)

Success! Your ML Odyssey environment is ready.

```text

## What Just Happened

You created a simple neural network layer and ran a forward pass:

1. **Linear Layer** - Transforms 10-dimensional input to 5-dimensional output
2. **ReLU Activation** - Applies non-linearity to the output
3. **Batch Processing** - Processed 3 examples simultaneously

This demonstrates the core building blocks you'll use to build complete models.

## Next Steps

### Build Your First Model

Ready for a complete example? Follow the [First Model Tutorial](first_model.md) to:

- Build a handwritten digit classifier (MNIST)
- Train a 3-layer neural network
- Achieve ~95% accuracy in 30 minutes

### Explore the Repository

- **[Repository Structure](repository-structure.md)** - Navigate the codebase
- **[Shared Library Guide](../core/shared-library.md)** - Available components
- **[Mojo Patterns](../core/mojo-patterns.md)** - Mojo-specific ML patterns

### Try Examples

```bash

# Explore example implementations
ls examples/
cd examples/getting-started/

```text

## Troubleshooting

### "Module not found" Error

```bash

# Ensure you're running from the repository root
cd /path/to/ml-odyssey
pixi run mojo run hello_ml.mojo

```text

### Mojo Version Issues

```bash

# Update Mojo to latest version
pixi update
pixi run mojo --version

```text

### Still Having Issues

- Check [Installation Guide](installation.md) for detailed setup
- Open an issue on GitHub

## Summary

You've successfully:

- ✅ Verified ML Odyssey installation
- ✅ Created and ran a neural network layer
- ✅ Processed data with activation functions

**Ready to build real models?** → [First Model Tutorial](first_model.md)
