# DenseNet-121 on CIFAR-10 Example

A complete implementation of DenseNet-121 for CIFAR-10 image classification, demonstrating dense connectivity and efficient feature reuse.

## Overview

This example shows how to build, train, and run inference with the DenseNet-121 architecture using ML Odyssey's shared library.

**Architecture**: DenseNet-121 (Huang et al., 2016) - Densely Connected Convolutional Networks

**Dataset**: CIFAR-10 (10 classes of RGB images: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck)

**Status**: 🚧 **In Development** - Implementation in progress

## Quick Start

### 1. Download Dataset

```bash
python examples/densenet121-cifar10/download_cifar10.py
```

This downloads CIFAR-10 (50,000 training + 10,000 test samples) to `datasets/cifar10/`.

### 2. Train Model

```bash
mojo run examples/densenet121-cifar10/train.mojo --epochs 200 --batch-size 64 --lr 0.01
```

### 3. Run Inference

```bash
mojo run examples/densenet121-cifar10/inference.mojo --weights-dir densenet121_weights
```

## Key Innovation: Dense Connectivity

DenseNet introduced **dense connectivity** - a pattern where each layer receives feature maps from ALL previous layers and passes its own to ALL subsequent layers.

### Dense Connectivity Pattern

Traditional CNNs have sequential connections:

```text
Layer 1 → Layer 2 → Layer 3 → Layer 4 → ...
```

DenseNet has dense connections:

```text
Layer 1 ──┬─→ Layer 2 ──┬─→ Layer 3 ──┬─→ Layer 4
          │             │             │
          ├─────────────┼─────────────┼─→ ...
          │             │             │
          └─────────────┴─────────────┴─→ ...
```

Each layer receives inputs from ALL previous layers in the dense block!

### Dense Block Structure

Within a dense block of L layers:

```text
Input (x_0)
    ↓
Layer 1: BN → ReLU → Conv3×3 → output (x_1)
    ↓
Layer 2: concat([x_0, x_1]) → BN → ReLU → Conv3×3 → output (x_2)
    ↓
Layer 3: concat([x_0, x_1, x_2]) → BN → ReLU → Conv3×3 → output (x_3)
    ↓
...
    ↓
Layer L: concat([x_0, x_1, ..., x_{L-1}]) → BN → ReLU → Conv3×3 → output (x_L)
    ↓
Output: concat([x_0, x_1, x_2, ..., x_L])
```

**Key Insights**:

- **Feature reuse**: All layers can access features from all previous layers
- **Short paths**: Gradients flow directly from loss to all layers
- **Compact**: Each layer adds only k feature maps (growth rate k)
- **Efficient**: Fewer parameters than ResNet despite being deeper

### Growth Rate

DenseNet uses a **growth rate** (k) - the number of feature maps each layer produces:

- If input has `c` channels and growth rate is `k`
- Layer 1 receives `c` channels, outputs `k` channels
- Layer 2 receives `c + k` channels, outputs `k` channels
- Layer 3 receives `c + 2k` channels, outputs `k` channels
- Layer L receives `c + (L-1)k` channels, outputs `k` channels

Typical growth rate: k = 32 for DenseNet-121

### Bottleneck Layers

To reduce computation, DenseNet uses **bottleneck layers** (1×1 convolutions before 3×3):

```text
Input (many channels)
    ↓
BN → ReLU → Conv1×1 (4k outputs) → BN → ReLU → Conv3×3 (k outputs)
```

This reduces the number of input channels to the expensive 3×3 convolution.

### Transition Layers

Between dense blocks, **transition layers** reduce spatial dimensions and feature map count:

```text
Input (c channels, H×W)
    ↓
BN → Conv1×1 (c/2 outputs) → AvgPool2×2
    ↓
Output (c/2 channels, H/2×W/2)
```

Compression factor: θ = 0.5 (reduce channels by half)

## Model Architecture

### DenseNet-121 (Adapted for 32×32 Input)

The classic DenseNet-121 adapted for CIFAR-10's smaller images.

```text
Input (32×32×3)
    ↓
Initial Block:
    Conv2D(2k, 3×3, stride=1, pad=1) → BN → ReLU
    ↓ (32×32×64, where k=32)
Dense Block 1 (6 layers, growth rate k=32):
    Layer 1: [64] → BN → ReLU → Conv1×1(4k) → BN → ReLU → Conv3×3(k)
    Layer 2: [64+32] → ...
    Layer 3: [64+64] → ...
    Layer 4: [64+96] → ...
    Layer 5: [64+128] → ...
    Layer 6: [64+160] → ...
    ↓ (32×32×256 = 64 + 6×32)
Transition 1:
    BN → Conv1×1(128) → AvgPool2×2
    ↓ (16×16×128)
Dense Block 2 (12 layers, growth rate k=32):
    Layer 1: [128] → BN → ReLU → Conv1×1(4k) → BN → ReLU → Conv3×3(k)
    ...
    Layer 12: [128+352] → ...
    ↓ (16×16×512 = 128 + 12×32)
Transition 2:
    BN → Conv1×1(256) → AvgPool2×2
    ↓ (8×8×256)
Dense Block 3 (24 layers, growth rate k=32):
    Layer 1: [256] → ...
    ...
    Layer 24: [256+736] → ...
    ↓ (8×8×1024 = 256 + 24×32)
Transition 3:
    BN → Conv1×1(512) → AvgPool2×2
    ↓ (4×4×512)
Dense Block 4 (16 layers, growth rate k=32):
    Layer 1: [512] → ...
    ...
    Layer 16: [512+480] → ...
    ↓ (4×4×1024 = 512 + 16×32)
Global Average Pool (4×4 → 1×1)
    ↓ (1024)
Linear(1024 → 10)
    ↓
Output (10 classes)
```

### Adaptations for CIFAR-10

Compared to the original DenseNet-121 for ImageNet (224×224):

1. **Smaller initial conv**: 3×3 instead of 7×7 (input is already small)
2. **No initial pooling**: Skip max pooling after first conv
3. **Same dense blocks**: Keep 4 dense blocks with [6, 12, 24, 16] layers
4. **Same growth rate**: k = 32
5. **Smaller FC layer**: 1024 → 10 instead of 1024 → 1000

### Parameters

- **Input Shape**: (batch, 3, 32, 32)
- **Output Shape**: (batch, 10)
- **Total Layers**: 121 (1 + 6 + 12 + 24 + 16 + 3 + 1 = 63 conv layers × 2 - 1)
- **Total Trainable Parameters**: ~7M
  - Initial conv: ~2K
  - Dense Block 1: ~300K
  - Dense Block 2: ~800K
  - Dense Block 3: ~2.5M
  - Dense Block 4: ~3M
  - FC layer: ~10K
- **Memory**: ~28MB for float32 weights

### Architecture Details

Each **Dense Layer** (within a dense block) consists of:

1. **Bottleneck**: BN → ReLU → Conv1×1 (4k filters)
2. **Convolution**: BN → ReLU → Conv3×3 (k filters)

Number of parameters per dense layer:
- Bottleneck: (input_channels) × (4k) × 1 × 1
- Convolution: (4k) × k × 3 × 3

Each **Transition Layer** (between dense blocks) consists of:

1. **Compression**: BN → Conv1×1 (θ × input_channels)
2. **Downsampling**: AvgPool 2×2 (stride=2)

Where θ = 0.5 (compression factor)

**Total Connections**:

In a dense block with L layers:
- Layer 1: receives 1 input (c channels)
- Layer 2: receives 2 inputs (c + k channels)
- Layer L: receives L inputs (c + (L-1)k channels)
- Total: L(L+1)/2 connections (quadratic!)

For DenseNet-121: 6 + 12 + 24 + 16 = 58 layers in dense blocks
- Total connections: (6×7)/2 + (12×13)/2 + (24×25)/2 + (16×17)/2 = 549 connections!

## Dataset Information

### CIFAR-10 Dataset

60,000 32×32 color images in 10 classes (6,000 per class).

- **Source**: Canadian Institute for Advanced Research
- **Training**: 50,000 images (5 batches of 10,000)
- **Test**: 10,000 images (1 batch)
- **Format**: Binary batches (converted to IDX for Mojo)

### Classes

| Index | Class      | Description               |
|-------|------------|---------------------------|
| 0     | airplane   | Various types of aircraft |
| 1     | automobile | Cars and trucks           |
| 2     | bird       | Various bird species      |
| 3     | cat        | Domestic cats             |
| 4     | deer       | Deer in various poses     |
| 5     | dog        | Domestic dogs             |
| 6     | frog       | Frogs and toads           |
| 7     | horse      | Horses                    |
| 8     | ship       | Boats and ships           |
| 9     | truck      | Large trucks              |

## File Structure

```text
examples/densenet121-cifar10/
├── README.md              # This file
├── model.mojo             # DenseNet-121 model with dense connectivity
├── train.mojo             # Training with manual backward passes
├── inference.mojo         # Inference with weight loading
├── data_loader.mojo       # CIFAR-10 binary format loading (symlink to resnet18)
├── weights.mojo           # Hex-based weight serialization
└── download_cifar10.py    # Python script to download dataset (symlink to resnet18)
```

## Implementation Status

### ✅ Planned

- [ ] Model architecture (4 dense blocks with 6, 12, 24, 16 layers)
- [ ] Forward pass through all 121 layers
- [ ] Dense connectivity implementation (concatenation)
- [ ] Transition layers (compression + pooling)
- [ ] Batch normalization integration
- [ ] Weight save/load functionality
- [ ] CIFAR-10 data loading (reuse from ResNet-18)
- [ ] Inference script
- [ ] Training script structure
- [ ] Comprehensive documentation

### 🔮 Future Enhancements

- [ ] Data augmentation integration
- [ ] Memory-efficient implementation (checkpointing)
- [ ] SIMD optimization for concatenation
- [ ] Learning rate schedules (cosine annealing)

## Expected Performance

Based on reference implementations and similar experiments:

- **Training Time**: ~40-50 hours on CPU for 200 epochs (batch_size=64)
- **Expected Accuracy**: 94-95% on CIFAR-10 after 200 epochs
- **Peak Accuracy**: 95-96% with data augmentation
- **Memory Usage**: ~28MB for model weights, ~500MB peak during training

### Comparison with Other Architectures

| Model      | Parameters | Layers | CIFAR-10 Accuracy | Training Time | Key Feature                |
|------------|------------|--------|-------------------|---------------|----------------------------|
| LeNet-5    | 61K        | 7      | 70-75%            | 2-3 hours     | Early CNN                  |
| AlexNet    | 2.3M       | 8      | 80-85%            | 8-12 hours    | Large kernels, dropout     |
| VGG-16     | 15M        | 16     | 91-93%            | 30-40 hours   | Very deep                  |
| ResNet-18  | 11M        | 18     | 93-94%            | 40-50 hours   | Skip connections           |
| GoogLeNet  | 6.8M       | 22     | 92-94%            | 35-45 hours   | Inception modules          |
| MobileNetV1| 4.2M       | 28     | 90-92%            | 25-35 hours   | Depthwise separable        |
| DenseNet   | 7M         | 121    | 94-95%            | 40-50 hours   | Dense connectivity         |

**Why DenseNet-121 Achieves High Accuracy**:

1. **Feature reuse**: All layers can access all previous features
2. **Short gradient paths**: Direct connections from loss to all layers
3. **Parameter efficiency**: Despite 121 layers, only 7M parameters
4. **No vanishing gradients**: Dense connections ensure gradient flow

## Advanced Features

### Concatenation Mathematics

For layer ℓ in a dense block:

```text
Input: x_ℓ = [x_0, x_1, ..., x_{ℓ-1}]
Channels: c_ℓ = c_0 + (ℓ-1) × k

where:
  c_0 = initial channels
  k = growth rate
  ℓ = layer index
```

Forward pass:
```text
x_ℓ = H_ℓ([x_0, x_1, ..., x_{ℓ-1}])
```

Where H_ℓ is the composite function: BN → ReLU → Conv1×1 → BN → ReLU → Conv3×3

### Backward Pass

Concatenation backward:
```text
grad_input = [grad_x_0, grad_x_1, ..., grad_x_{ℓ-1}]
```

Each layer receives gradients from ALL subsequent layers!

### Memory Efficiency

DenseNet's memory consumption:
- **Forward**: Must store all intermediate feature maps for concatenation
- **Backward**: Must store all activations for gradient computation
- **Peak memory**: Proportional to L² (quadratic in depth)

Memory optimization techniques:
1. **Shared memory buffers**: Reuse memory for concatenation
2. **Checkpointing**: Recompute activations during backward pass
3. **Mixed precision**: Use float16 for activations, float32 for weights

### Batch Normalization

Applied after every convolution (bottleneck and 3×3):

```text
x_norm = (x - mean) / sqrt(var + eps)
y = gamma * x_norm + beta
```

### Learning Rate Scheduling

Step decay schedule (same as other models):

- **Schedule**: Decay by 10× at epochs 50, 75
- **Formula**:
  - Epochs 0-49: lr = 0.01
  - Epochs 50-74: lr = 0.001
  - Epochs 75+: lr = 0.0001

### Weight Initialization

**He initialization** for all convolutions:
- Formula: `weights ~ N(0, sqrt(2 / fan_in))`
- Critical for training very deep networks

## Design Principles (KISS)

This example follows **Keep It Simple, Stupid** principles:

1. **Minimal Dependencies**: Uses only ML Odyssey shared library
2. **Functional Design**: Uses functional ops from shared/core
3. **Clear Structure**: Separate files for model, training, and inference
4. **Simple Interfaces**: Command-line arguments for configuration
5. **No Over-Engineering**: Direct implementation without unnecessary abstractions
6. **Pattern Reuse**: Follows the same structure as previous examples

## Usage Details

### Training Options

```bash
mojo run examples/densenet121-cifar10/train.mojo \
    --epochs 200 \
    --batch-size 64 \
    --lr 0.01 \
    --momentum 0.9 \
    --data-dir datasets/cifar10 \
    --weights-dir densenet121_weights
```

**Arguments**:

- `--epochs`: Number of training epochs (default: 200)
- `--batch-size`: Mini-batch size (default: 64, smaller due to memory)
- `--lr`: Initial learning rate for SGD (default: 0.01)
- `--momentum`: Momentum factor for SGD (default: 0.9)
- `--data-dir`: Path to CIFAR-10 dataset directory (default: `datasets/cifar10`)
- `--weights-dir`: Directory to save model weights (default: `densenet121_weights`)

### Inference Options

```bash
mojo run examples/densenet121-cifar10/inference.mojo \
    --weights-dir densenet121_weights \
    --data-dir datasets/cifar10
```

**Arguments**:

- `--weights-dir`: Directory containing saved model weights (default: `densenet121_weights`)
- `--data-dir`: Path to CIFAR-10 dataset for test set evaluation (default: `datasets/cifar10`)

## References

### Papers

1. **DenseNet (Original)**:
   Huang, G., Liu, Z., Van Der Maaten, L., & Weinberger, K. Q. (2017).
   Densely connected convolutional networks.
   *CVPR 2017*.
   [arXiv Paper](https://arxiv.org/abs/1608.06993)

2. **Batch Normalization**:
   Ioffe, S., & Szegedy, C. (2015).
   Batch normalization: Accelerating deep network training by reducing internal covariate shift.
   *ICML 2015*.
   [Paper](https://arxiv.org/abs/1502.03167)

3. **CIFAR-10 Dataset**:
   Krizhevsky, A., & Hinton, G. (2009).
   Learning multiple layers of features from tiny images.
   *Technical report, University of Toronto*.
   [Tech Report](https://www.cs.toronto.edu/~kriz/learning-features-2009-TR.pdf)

### Datasets

- **CIFAR-10 Official Page**: <https://www.cs.toronto.edu/~kriz/cifar.html>
- **Download**: <https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz>

### Reference Implementations

- **DenseNet PyTorch**: <https://github.com/pytorch/vision/blob/main/torchvision/models/densenet.py>
  - Official PyTorch implementation
  - Demonstrates dense connectivity patterns

- **DenseNet TensorFlow**: <https://github.com/liuzhuang13/DenseNet>
  - Original implementation by authors

### Related Resources

- **Papers with Code - DenseNet**: <https://paperswithcode.com/method/densenet>
- **DenseNet Explained**: <https://towardsdatascience.com/understanding-and-visualizing-densenets-7f688092391a>
- **Dense Connectivity**: <https://d2l.ai/chapter_convolutional-modern/densenet.html>

## Contributing

This example is part of ML Odyssey. Contributions welcome!

**Priority Tasks**:

1. **Implement dense layer** with concatenation
2. **Complete model architecture** (4 dense blocks)
3. **Implement training script** with backward pass
4. **Memory optimization** (checkpointing for backward pass)
5. **Optimize concatenation** with SIMD vectorization

See [CONTRIBUTING.md](../../CONTRIBUTING.md) for guidelines.

## License

This example is part of ML Odyssey and follows the project's license.

See [LICENSE](../../LICENSE) for details.

## Acknowledgments

- **Dataset**: CIFAR (Canadian Institute for Advanced Research) for creating and releasing the CIFAR-10 dataset
- **Architecture**: Gao Huang et al. for the DenseNet architecture and dense connectivity
- **Inspiration**: Previous examples (ResNet-18, GoogLeNet, VGG-16, MobileNetV1) for patterns
- **ML Odyssey**: The shared library providing functional neural network operations

## Next Steps

1. Implement dense layer with feature concatenation
2. Build complete DenseNet-121 model with 4 dense blocks
3. Implement transition layers (compression + pooling)
4. Create training script with dense block backward pass
5. Test on CIFAR-10 dataset
6. Compare with other architectures (accuracy, efficiency, training time)
7. Experiment with memory optimization techniques
