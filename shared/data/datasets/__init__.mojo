"""Dataset implementations and utilities.

Provides high-level dataset interfaces for common ML datasets including CIFAR-10.

Modules:
    `cifar10`: CIFAR-10 dataset wrapper for image classification

Classes:
    `CIFAR10Dataset`: High-level interface for CIFAR-10 data access

Example:
    from shared.data.datasets import CIFAR10Dataset

    # Create dataset and load data
    var dataset = CIFAR10Dataset("/path/to/cifar10/data")

    # Access training data
    var train_images, train_labels = dataset.get_train_data()

    # Access test data
    var test_images, test_labels = dataset.get_test_data()

    # Get individual samples
    var image, label = dataset.__getitem__(0)

    # Query dataset properties
    print(dataset.num_train_samples())  # 50000
    print(dataset.num_test_samples())   # 10000
    print(dataset.num_classes())        # 10
    print(dataset.get_class_name(0))    # "airplane"
"""

from .cifar10 import CIFAR10Dataset, CIFAR10_CLASSES
