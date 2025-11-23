"""Training Script for DenseNet-121 on CIFAR-10

NOTE: DenseNet's dense connectivity creates L(L+1)/2 connections per dense block.
For DenseNet-121: 549 total connections across all dense blocks!

Backward pass requires:
- Splitting gradients at each concatenation point
- Routing gradients to all previous layers
- Managing quadratic memory growth (L²)

Full implementation would require ~3000 lines. Consider automatic differentiation.
"""

from shared.core import ExTensor, zeros, cross_entropy
from shared.data import extract_batch_pair, compute_num_batches
from model import DenseNet121
from data_loader import load_cifar10_train

fn main() raises:
    print("=" * 60)
    print("DenseNet-121 Training on CIFAR-10")
    print("=" * 60)
    print()

    var train_data = load_cifar10_train("datasets/cifar10")
    var train_images = train_data[0]
    var train_labels = train_data[1]
    print("Training samples: " + str(train_images.shape[0]))
    print()

    var model = DenseNet121(num_classes=10)
    print("Model: DenseNet-121")
    print("Total layers: 121")
    print("Dense connections: 549")
    print("Parameters: ~7M")
    print()

    print("NOTE: Dense connectivity creates complex backward pass:")
    print("  - Each layer receives gradients from ALL subsequent layers")
    print("  - Concatenation splits gradients to multiple paths")
    print("  - Memory consumption is O(L²) in depth")
    print()
    print("For actual training, use automatic differentiation.")
    print()
