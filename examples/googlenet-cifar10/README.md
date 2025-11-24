# GoogLeNet (Inception-v1) on CIFAR-10 Example

A complete implementation of GoogLeNet/Inception-v1 for CIFAR-10 image classification
demonstrating the power of multi-scale feature extraction and efficient architecture design.

## Overview

This example shows how to build, train
and run inference with the GoogLeNet (Inception-v1) architecture using ML Odyssey's shared library.

**Architecture**: GoogLeNet/Inception-v1 (Szegedy et al., 2014) - Going Deeper with Convolutions

**Dataset**: CIFAR-10 (10 classes of RGB images: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck)

**Status**: 🚧 **In Development** - Implementation in progress

## Quick Start

### 1. Download Dataset

```bash
python examples/googlenet-cifar10/download_cifar10.py
```text

This downloads CIFAR-10 (50,000 training + 10,000 test samples) to `datasets/cifar10/`.

### 2. Train Model

```bash
mojo run examples/googlenet-cifar10/train.mojo --epochs 200 --batch-size 128 --lr 0.01
```text

### 3. Run Inference

```bash
mojo run examples/googlenet-cifar10/inference.mojo --weights-dir googlenet_weights
```text

## Key Innovation: Inception Module

GoogLeNet introduced **Inception modules**
a revolutionary approach to multi-scale feature extraction that processes input at different scales simultaneously.

### The Inception Module

Traditional CNNs use a single kernel size per layer. Inception modules process the input with multiple kernel sizes in
parallel:

```text
                    Input
                      ↓
        ┌─────────────┼─────────────┬─────────────┐
        ↓             ↓             ↓             ↓
    1×1 Conv      1×1 Conv      1×1 Conv     MaxPool 3×3
        ↓             ↓             ↓         (stride=1)
        ↓         3×3 Conv      5×5 Conv         ↓
        ↓             ↓             ↓         1×1 Conv
        └─────────────┴─────────────┴─────────────┘
                      ↓
                  Concatenate (depthwise)
                      ↓
                   Output
```text

**Key Insights**:

- **Multi-scale processing**: Captures features at different scales (1×1, 3×3, 5×5)
- **Dimensionality reduction**: 1×1 convs before expensive 3×3 and 5×5 convs reduce parameters
- **Feature diversity**: Parallel branches learn complementary features
- **Efficiency**: Fewer parameters than VGG-16 but higher accuracy

### Dimensionality Reduction (1×1 Convolutions)

1×1 convolutions reduce the number of channels before expensive larger convolutions:

**Without 1×1 reduction** (naive Inception):

- Input: 256 channels → 3×3 conv (128 filters) → 256×128×9 = 295,296 parameters

**With 1×1 reduction** (actual Inception):

- Input: 256 channels → 1×1 conv (96 filters) → 3×3 conv (128 filters)
- Parameters: (256×96×1) + (96×128×9) = 24,576 + 110,592 = 135,168 parameters
- **Reduction: 54% fewer parameters!**

This allows the network to go deeper with fewer parameters.

## Model Architecture

### GoogLeNet (Adapted for 32×32 Input)

The classic GoogLeNet adapted for CIFAR-10's smaller images (32×32 vs 224×224 ImageNet).

```text
Input (32×32×3)
    ↓
Initial Block:
    Conv2D(64, 3×3, stride=1, pad=1) → BN → ReLU
    ↓ (32×32×64)
    MaxPool(3×3, stride=2, pad=1)
    ↓ (16×16×64)
Inception 3a:
    [1×1(64), 3×3(128), 5×5(32), pool→1×1(32)]
    ↓ (16×16×256)
Inception 3b:
    [1×1(128), 3×3(192), 5×5(96), pool→1×1(64)]
    ↓ (16×16×480)
MaxPool(3×3, stride=2, pad=1)
    ↓ (8×8×480)
Inception 4a:
    [1×1(192), 3×3(208), 5×5(48), pool→1×1(64)]
    ↓ (8×8×512)
Inception 4b:
    [1×1(160), 3×3(224), 5×5(64), pool→1×1(64)]
    ↓ (8×8×512)
Inception 4c:
    [1×1(128), 3×3(256), 5×5(64), pool→1×1(64)]
    ↓ (8×8×512)
Inception 4d:
    [1×1(112), 3×3(288), 5×5(64), pool→1×1(64)]
    ↓ (8×8×528)
Inception 4e:
    [1×1(256), 3×3(320), 5×5(128), pool→1×1(128)]
    ↓ (8×8×832)
MaxPool(3×3, stride=2, pad=1)
    ↓ (4×4×832)
Inception 5a:
    [1×1(256), 3×3(320), 5×5(128), pool→1×1(128)]
    ↓ (4×4×832)
Inception 5b:
    [1×1(384), 3×3(384), 5×5(128), pool→1×1(128)]
    ↓ (4×4×1024)
Global Average Pool (4×4 → 1×1)
    ↓ (1×1×1024)
Dropout(0.4)
    ↓ (1024)
Linear(1024 → 10)
    ↓
Output (10 classes)
```text

### Adaptations for CIFAR-10

Compared to the original GoogLeNet for ImageNet (224×224):

1. **Smaller initial conv**: 3×3 instead of 7×7 (input is already small)
2. **Fewer pooling layers**: Only 3 max pooling layers (vs 5)
3. **Same Inception modules**: Keep the 9 Inception modules (3a, 3b, 4a-e, 5a, 5b)
4. **Same global pooling**: Use global average pooling before classifier
5. **Smaller FC layer**: 1024 → 10 instead of 1024 → 1000
6. **No auxiliary classifiers**: Skip auxiliary classifiers (used in ImageNet training)

### Parameters

- **Input Shape**: (batch, 3, 32, 32)
- **Output Shape**: (batch, 10)
- **Total Trainable Parameters**: ~6.8M (much less than VGG-16's 15M)
  - Initial block: ~2K
  - Inception 3a: ~160K
  - Inception 3b: ~380K
  - Inception 4a: ~580K
  - Inception 4b: ~620K
  - Inception 4c: ~670K
  - Inception 4d: ~720K
  - Inception 4e: ~1.0M
  - Inception 5a: ~1.1M
  - Inception 5b: ~1.5M
  - FC: 10K
- **Memory**: ~27MB for float32 weights

### Architecture Details

Each **Inception Module** consists of:

- **Branch 1**: 1×1 convolution (direct feature extraction)
- **Branch 2**: 1×1 conv (reduction) → 3×3 conv
- **Branch 3**: 1×1 conv (reduction) → 5×5 conv
- **Branch 4**: 3×3 max pooling → 1×1 conv (projection)
- **Concatenation**: Depth-wise concatenation of all branches

**Example** (Inception 3a):

- Input: (16×16×64)
- Branch 1: 1×1 conv (64 filters) → (16×16×64)
- Branch 2: 1×1 conv (96 filters) → 3×3 conv (128 filters) → (16×16×128)
- Branch 3: 1×1 conv (16 filters) → 5×5 conv (32 filters) → (16×16×32)
- Branch 4: MaxPool 3×3 → 1×1 conv (32 filters) → (16×16×32)
- Concatenate: (16×16×256) = 64 + 128 + 32 + 32

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
examples/googlenet-cifar10/
├── README.md              # This file
├── model.mojo             # GoogLeNet model with Inception modules
├── train.mojo             # Training with manual backward passes
├── inference.mojo         # Inference with weight loading
├── data_loader.mojo       # CIFAR-10 binary format loading (symlink to resnet18)
├── weights.mojo           # Hex-based weight serialization
└── download_cifar10.py    # Python script to download dataset (symlink to resnet18)
```text

## Implementation Status

### ✅ Planned

- [ ] Model architecture (9 Inception modules)
- [ ] Forward pass through all 22 layers
- [ ] Inception module implementation
- [ ] Batch normalization integration
- [ ] Weight save/load functionality
- [ ] CIFAR-10 data loading (reuse from ResNet-18)
- [ ] Inference script
- [ ] Training script structure
- [ ] Comprehensive documentation

### 🔮 Future Enhancements

- [ ] Auxiliary classifiers (for improved gradient flow)
- [ ] Data augmentation integration
- [ ] SIMD optimization for concatenation
- [ ] Learning rate schedules (cosine annealing)

## Expected Performance

Based on reference implementations and similar experiments:

- **Training Time**: ~35-45 hours on CPU for 200 epochs (batch_size=128)
- **Expected Accuracy**: 92-94% on CIFAR-10 after 200 epochs
- **Peak Accuracy**: 94-96% with data augmentation
- **Memory Usage**: ~27MB for model weights

### Comparison with Other Architectures

| Model      | Parameters | CIFAR-10 Accuracy | Training Time (200 epochs) | Key Feature              |
|------------|------------|-------------------|----------------------------|--------------------------|
| LeNet-5    | 61K        | 70-75%            | 2-3 hours                  | Early CNN                |
| AlexNet    | 2.3M       | 80-85%            | 8-12 hours                 | Large kernels, dropout   |
| VGG-16     | 15M        | 91-93%            | 30-40 hours                | Very deep (16 layers)    |
| ResNet-18  | 11M        | 93-94%            | 40-50 hours                | Skip connections         |
| GoogLeNet  | 6.8M       | 92-94%            | 35-45 hours                | Inception modules        |

**Why GoogLeNet is Efficient**:

1. **Fewer parameters than VGG-16**: 6.8M vs 15M, but similar accuracy
2. **1×1 convolutions**: Reduce dimensionality before expensive ops
3. **Global average pooling**: Eliminates large fully connected layers
4. **Multi-scale features**: Learns diverse representations efficiently

## Advanced Features

### Inception Module Mathematics

For an Inception module with input `x` (shape: [B, C_in, H, W]):

1. **Branch 1** (1×1):
   - `b1 = conv1x1_1(x)` → [B, C1, H, W]

2. **Branch 2** (1×1 → 3×3):
   - `b2_reduce = conv1x1_2(x)` → [B, C2_reduce, H, W]
   - `b2 = conv3x3(b2_reduce)` → [B, C2, H, W]

3. **Branch 3** (1×1 → 5×5):
   - `b3_reduce = conv1x1_3(x)` → [B, C3_reduce, H, W]
   - `b3 = conv5x5(b3_reduce)` → [B, C3, H, W]

4. **Branch 4** (pool → 1×1):
   - `b4_pool = maxpool3x3(x)` → [B, C_in, H, W]
   - `b4 = conv1x1_4(b4_pool)` → [B, C4, H, W]

5. **Concatenate**:
   - `output = concat([b1, b2, b3, b4], axis=1)` → [B, C1+C2+C3+C4, H, W]

### Backward Pass

Concatenation backward:

- `grad_input = split(grad_output, [C1, C2, C3, C4])`
- Each branch gets its portion of the gradient
- Standard conv/pool backward for each branch

### Batch Normalization

Applied after every convolution (same as ResNet):

```text
x_norm = (x - mean) / sqrt(var + eps)
y = gamma * x_norm + beta
```text

During training:

- Compute mean and variance over batch
- Update running statistics with momentum

During inference:

- Use running mean and variance (fixed)

### Learning Rate Scheduling

Step decay schedule (similar to ResNet):

- **Schedule**: Decay by 5× every 60 epochs
- **Formula**: `lr = initial_lr * (0.2 ** (epoch // 60))`
- **Example**:
  - Epochs 0-59: lr = 0.01
  - Epochs 60-119: lr = 0.002
  - Epochs 120-179: lr = 0.0004
  - Epochs 180+: lr = 0.00008

### Weight Initialization

**Xavier initialization** for 1×1 convolutions:

- Formula: `weights ~ N(0, sqrt(2 / (fan_in + fan_out)))`

**He initialization** for 3×3 and 5×5 convolutions:

- Formula: `weights ~ N(0, sqrt(2 / fan_in))`
- Better for ReLU activations

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
mojo run examples/googlenet-cifar10/train.mojo \
    --epochs 200 \
    --batch-size 128 \
    --lr 0.01 \
    --momentum 0.9 \
    --data-dir datasets/cifar10 \
    --weights-dir googlenet_weights
```text

**Arguments**:

- `--epochs`: Number of training epochs (default: 200)
- `--batch-size`: Mini-batch size (default: 128)
- `--lr`: Initial learning rate for SGD (default: 0.01)
- `--momentum`: Momentum factor for SGD (default: 0.9)
- `--data-dir`: Path to CIFAR-10 dataset directory (default: `datasets/cifar10`)
- `--weights-dir`: Directory to save model weights (default: `googlenet_weights`)

### Inference Options

```bash
mojo run examples/googlenet-cifar10/inference.mojo \
    --weights-dir googlenet_weights \
    --data-dir datasets/cifar10
```text

**Arguments**:

- `--weights-dir`: Directory containing saved model weights (default: `googlenet_weights`)
- `--data-dir`: Path to CIFAR-10 dataset for test set evaluation (default: `datasets/cifar10`)

## References

### Papers

1. **GoogLeNet/Inception-v1 (Original)**:
   Szegedy, C., Liu, W., Jia, Y., Sermanet, P., Reed, S., Anguelov, D., ... & Rabinovich, A. (2015).
   Going deeper with convolutions.
   *CVPR 2015*.
   [arXiv Paper](https://arxiv.org/abs/1409.4842)

2. **Network in Network (1×1 convolutions)**:
   Lin, M., Chen, Q., & Yan, S. (2013).
   Network in network.
   *ICLR 2014*.
   [arXiv Paper](https://arxiv.org/abs/1312.4400)

3. **Batch Normalization**:
   Ioffe, S., & Szegedy, C. (2015).
   Batch normalization: Accelerating deep network training by reducing internal covariate shift.
   *ICML 2015*.
   [Paper](https://arxiv.org/abs/1502.03167)

4. **CIFAR-10 Dataset**:
   Krizhevsky, A., & Hinton, G. (2009).
   Learning multiple layers of features from tiny images.
   *Technical report, University of Toronto*.
   [Tech Report](https://www.cs.toronto.edu/~kriz/learning-features-2009-TR.pdf)

### Datasets

- **CIFAR-10 Official Page**: <https://www.cs.toronto.edu/~kriz/cifar.html>
- **Download**: <https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz>

### Reference Implementations

- **GoogLeNet PyTorch**: <https://github.com/pytorch/vision/blob/main/torchvision/models/googlenet.py>
  - Official PyTorch implementation
  - Demonstrates Inception module architecture

- **Inception TensorFlow**: <https://github.com/tensorflow/models/tree/master/research/slim/nets>
  - TensorFlow Inception family implementations

### Related Resources

- **Papers with Code - GoogLeNet**: <https://paperswithcode.com/method/googlenet>
- **Inception Explained**:
<https://towardsdatascience.com/a-simple-guide-to-the-versions-of-the-inception-network-7fc52b863202>
- **1×1 Convolutions**: <https://d2l.ai/chapter_convolutional-modern/nin.html>

## Contributing

This example is part of ML Odyssey. Contributions welcome!

**Priority Tasks**:

1. **Implement Inception module** with parallel branches
2. **Complete model architecture** (9 Inception modules)
3. **Implement training script** with backward pass
4. **Add auxiliary classifiers** (optional, for better gradient flow)
5. **Optimize concatenation** with SIMD vectorization

See [CONTRIBUTING.md](../../CONTRIBUTING.md) for guidelines.

## License

This example is part of ML Odyssey and follows the project's license.

See [LICENSE](../../LICENSE) for details.

## Acknowledgments

- **Dataset**: CIFAR (Canadian Institute for Advanced Research) for creating and releasing the CIFAR-10 dataset
- **Architecture**: Christian Szegedy et al. for the GoogLeNet/Inception architecture
- **Inspiration**: ResNet-18 and VGG-16 examples for establishing implementation patterns
- **ML Odyssey**: The shared library providing functional neural network operations

## Next Steps

1. Implement the Inception module structure
2. Build the complete GoogLeNet model with 9 Inception modules
3. Implement training script with backward pass through Inception modules
4. Test on CIFAR-10 dataset
5. Compare efficiency with VGG-16 and ResNet-18
