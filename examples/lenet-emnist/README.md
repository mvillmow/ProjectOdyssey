# LeNet-5 on EMNIST Example

A simple implementation of LeNet-5 convolutional neural network for EMNIST character recognition, following KISS principles.

## Overview

This example demonstrates how to use ML Odyssey's shared library to build, train, and run inference with the classic
LeNet-5 architecture on the EMNIST dataset.

**Architecture**: LeNet-5 (LeCun et al., 1998)

**Dataset**: EMNIST Balanced (47 classes: digits 0-9, uppercase A-Z, and select lowercase letters)

**Status**: 🚧 **Skeleton Implementation** - Demonstrates structure while waiting for stable Mojo file I/O and autograd

## Quick Start

### 1. Download Dataset

```bash
python scripts/download_emnist.py --split balanced
```

This downloads the EMNIST Balanced split (131,600 training samples, 18,800 test samples, 47 classes) to
`datasets/emnist/`.

### 2. Train Model (When Available)

```bash
mojo run examples/lenet-emnist/train.mojo --epochs 10 --batch-size 32 --lr 0.01
```

### 3. Run Inference (When Available)

```bash
# Evaluate on test set
mojo run examples/lenet-emnist/inference.mojo --weights lenet5_emnist.weights

# Infer single image
mojo run examples/lenet-emnist/inference.mojo --weights lenet5_emnist.weights --image path/to/image.png
```

## Dataset Information

### EMNIST Dataset

The EMNIST dataset is an extension of MNIST that includes handwritten letters in addition to digits.

- **Source**: NIST Special Database 19
- **Format**: 28×28 grayscale images
- **File Format**: IDX (same as MNIST) and MATLAB

### Available Splits

| Split        | Samples | Classes | Description                         |
|--------------|---------|---------|-------------------------------------|
| **Balanced** | 131,600 | 47      | Recommended - balanced distribution |
| ByClass      | 814,255 | 62      | Unbalanced, all characters          |
| ByMerge      | 814,255 | 47      | Unbalanced, merged similar chars    |
| Digits       | 280,000 | 10      | Digits only (0-9)                   |
| Letters      | 145,600 | 26      | Uppercase letters only (A-Z)        |
| MNIST        | 70,000  | 10      | Original MNIST for comparison       |

**Default**: This example uses the **Balanced** split (47 classes).

### EMNIST Balanced Classes

- **Digits (10)**: 0-9
- **Uppercase Letters (26)**: A-Z
- **Lowercase Letters (11)**: a, b, d, e, f, g, h, n, q, r, t

Note: Only select lowercase letters are included to reduce confusion with similar-looking uppercase letters.

## Model Architecture

### LeNet-5

The classic convolutional neural network architecture introduced by LeCun et al. (1998).

```text
Input (28×28×1)
    ↓
Conv2D (6 filters, 5×5) + ReLU
    ↓
MaxPool (2×2, stride=2)
    ↓
Conv2D (16 filters, 5×5) + ReLU
    ↓
MaxPool (2×2, stride=2)
    ↓
Flatten (16×4×4 = 256)
    ↓
Linear (256 → 120) + ReLU
    ↓
Linear (120 → 84) + ReLU
    ↓
Linear (84 → 47)
    ↓
Output (47 classes)
```

### Parameters

- **Input Shape**: (batch, 1, 28, 28)
- **Output Shape**: (batch, 47)
- **Total Parameters**: ~61,706
  - Conv1: 6×(1×5×5+1) = 156
  - Conv2: 16×(6×5×5+1) = 2,416
  - FC1: 256×120+120 = 30,840
  - FC2: 120×84+84 = 10,164
  - FC3: 84×47+47 = 3,995

## File Structure

```text
examples/lenet-emnist/
├── README.md           # This file
├── model.mojo          # LeNet-5 model definition
├── train.mojo          # Training script
└── inference.mojo      # Inference script
```

## Usage Details

### Training Options

```bash
mojo run examples/lenet-emnist/train.mojo \
    --epochs 10 \
    --batch-size 32 \
    --lr 0.01 \
    --data-dir datasets/emnist
```

**Arguments**:

- `--epochs`: Number of training epochs (default: 10)
- `--batch-size`: Mini-batch size (default: 32)
- `--lr`: Learning rate for SGD (default: 0.01)
- `--data-dir`: Path to EMNIST dataset directory (default: `datasets/emnist`)

### Inference Options

```bash
# Test set evaluation
mojo run examples/lenet-emnist/inference.mojo \
    --weights lenet5_emnist.weights \
    --data-dir datasets/emnist

# Single image inference
mojo run examples/lenet-emnist/inference.mojo \
    --weights lenet5_emnist.weights \
    --image path/to/image.png
```

**Arguments**:

- `--weights`: Path to saved model weights (default: `lenet5_emnist.weights`)
- `--data-dir`: Path to EMNIST dataset for test set evaluation
- `--image`: Path to single image for inference (switches to single-image mode)

## Implementation Status

### ✅ Completed

- [x] Dataset download script with IDX format support
- [x] LeNet-5 model architecture definition
- [x] Training script structure
- [x] Inference script structure
- [x] Comprehensive documentation

### 🚧 Pending (Waiting for Mojo Stabilization)

- [ ] IDX file loading (waiting for stable Mojo file I/O)
- [ ] Image loading (waiting for stable Mojo image I/O)
- [ ] Backward pass implementation (waiting for autograd)
- [ ] Weight serialization/deserialization (waiting for stable Mojo file I/O)
- [ ] Complete training loop
- [ ] Complete inference pipeline

### Current Limitations

This is a **skeleton implementation** that demonstrates the structure and KISS principles. The following features
are marked as TODO pending Mojo stdlib stabilization:

1. **File I/O**: IDX file reading, image loading, weight saving/loading
2. **Autograd**: Automatic differentiation for backward pass
3. **Optimizers**: Parameter updates with SGD/Adam

The implementation shows the **intended structure** and will be completed as Mojo's stdlib matures.

## Expected Performance

Based on the reference implementation and similar experiments:

- **Training Time**: ~2-3 hours on CPU for 20 epochs (batch_size=256)
- **Expected Accuracy**: 93-95% on EMNIST Balanced after 10 epochs
- **Peak Accuracy**: 98%+ on EMNIST Digits (easier task)

## Design Principles (KISS)

This example follows **Keep It Simple, Stupid** principles:

1. **Minimal Dependencies**: Uses only ML Odyssey shared library
2. **Functional Design**: Model uses functional ops (conv2d, linear, relu) from shared/core
3. **Clear Structure**: Separate files for model, training, and inference
4. **Simple Interfaces**: Command-line arguments for configuration
5. **No Over-Engineering**: Direct implementation without unnecessary abstractions

## References

### Papers

1. **LeNet-5 (Original)**:
   LeCun, Y., Bottou, L., Bengio, Y., & Haffner, P. (1998).
   Gradient-based learning applied to document recognition.
   *Proceedings of the IEEE*, 86(11), 2278-2324.
   [DOI: 10.1109/5.726791](https://doi.org/10.1109/5.726791)

2. **EMNIST Dataset**:
   Cohen, G., Afshar, S., Tapson, J., & van Schaik, A. (2017).
   EMNIST: an extension of MNIST to handwritten letters.
   *arXiv preprint arXiv:1702.05373v1*.
   [arXiv:1702.05373](https://arxiv.org/abs/1702.05373)

### Datasets

- **EMNIST Official Page**: <https://www.nist.gov/itl/products-and-services/emnist-dataset>
- **Download**: <http://www.itl.nist.gov/iaui/vip/cs_links/EMNIST/gzip.zip>

### Reference Implementations

- **LeNet from Scratch (NumPy)**: <https://github.com/mattwang44/LeNet-from-Scratch>
  - Pure NumPy implementation without deep learning frameworks
  - Demonstrates manual backpropagation
  - Achieves 98.6% accuracy on MNIST

### Related Resources

- **Original MNIST**: <http://yann.lecun.com/exdb/mnist/>
- **LeNet-5 Architecture**: <http://yann.lecun.com/exdb/lenet/>

## Contributing

This example is part of ML Odyssey. Contributions welcome!

1. Complete pending TODOs when Mojo stdlib stabilizes
2. Optimize performance (SIMD, parallelization)
3. Add data augmentation
4. Implement learning rate scheduling
5. Add visualization tools

See [CONTRIBUTING.md](../../CONTRIBUTING.md) for guidelines.

## License

This example is part of ML Odyssey and follows the project's license.

See [LICENSE](../../LICENSE) for details.

## Acknowledgments

- **Dataset**: NIST for creating and releasing the EMNIST dataset
- **Architecture**: Yann LeCun et al. for the LeNet-5 architecture
- **Reference Implementation**: Matt Wang for the clear NumPy reference implementation
- **ML Odyssey**: The shared library providing functional neural network operations
