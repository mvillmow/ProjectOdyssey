"""Inference Script for DenseNet-121 on CIFAR-10"""

from shared.core import ExTensor, zeros
from shared.data import extract_batch_pair, compute_num_batches
from shared.data.datasets import load_cifar10_test
from model import DenseNet121

alias CLASS_NAMES = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

fn main() raises:
    print("=" * 60)
    print("DenseNet-121 Inference on CIFAR-10")
    print("=" * 60)
    print()

    var test_data = load_cifar10_test("datasets/cifar10")
    var test_images = test_data[0]
    var test_labels = test_data[1]
    print("Test samples: " + str(test_images.shape()[0]))
    print()

    var model = DenseNet121(num_classes=10)
    print("Model: DenseNet-121 (121 layers, dense connectivity)")
    print("Parameters: ~7M")
    print()

    print("Expected accuracy: 94-95% on CIFAR-10")
    print("Key feature: Dense connections ensure gradient flow")
    print()
