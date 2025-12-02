"""CIFAR-10 Dataset Loader

Provides functions to load CIFAR-10 dataset from IDX file format (converted from binary batches).

CIFAR-10 Structure:
    - Images: 32x32 RGB (3 channels)
    - Labels: 10 classes (0-9)
    - Training: 50,000 images (5 batches of 10,000)
    - Test: 10,000 images (1 batch)

References:
    - CIFAR-10 Dataset: https://www.cs.toronto.edu/~kriz/cifar.html
    - IDX Format: http://yann.lecun.com/exdb/mnist/
"""

from shared.core import ExTensor, zeros
from shared.core.normalization import normalize_rgb as normalize_rgb_shared
from .idx_loader import load_idx_labels, load_idx_images_rgb


fn normalize_images_rgb(mut images: ExTensor) raises -> ExTensor:
    """Normalize uint8 RGB images to float32 with ImageNet normalization.

    Args:
        images: Input images as uint8 ExTensor of shape (N, 3, H, W)

    Returns:
        Normalized images as float32 ExTensor

    Note:
        Applies ImageNet normalization:
        - mean=[0.485, 0.456, 0.406] for RGB channels
        - std=[0.229, 0.224, 0.225] for RGB channels
        - Converts pixel values from [0, 255] to normalized float

        Uses shared library implementation.
    """
    # ImageNet normalization parameters (R, G, B)
    var mean = (Float32(0.485), Float32(0.456), Float32(0.406))
    var std = (Float32(0.229), Float32(0.224), Float32(0.225))

    # Use shared library normalize_rgb function
    return normalize_rgb_shared(images, mean, std)


fn _copy_batch_data(
    mut dest_images: ExTensor,
    mut dest_labels: ExTensor,
    src_images: ExTensor,
    src_labels: ExTensor,
    offset: Int
) raises:
    """Helper function to copy batch data into full training tensor.

    Args:
        dest_images: Destination image tensor
        dest_labels: Destination label tensor
        src_images: Source batch images (10,000 images)
        src_labels: Source batch labels (10,000 labels)
        offset: Starting index in destination tensor
    """
    var batch_size = src_images.shape()[0]
    var pixels_per_image = 3 * 32 * 32  # RGB channels * height * width

    # Copy images
    var dest_img_data = dest_images._data.bitcast[Float32]()
    var src_img_data = src_images._data.bitcast[Float32]()
    for i in range(batch_size):
        for j in range(pixels_per_image):
            dest_img_data[offset * pixels_per_image + i * pixels_per_image + j] = src_img_data[i * pixels_per_image + j]

    # Copy labels
    var dest_label_data = dest_labels._data
    var src_label_data = src_labels._data
    for i in range(batch_size):
        dest_label_data[offset + i] = src_label_data[i]


fn load_cifar10_batch(batch_dir: String, batch_name: String) raises -> Tuple[ExTensor, ExTensor]:
    """Load a single CIFAR-10 batch (images and labels).

    Args:
        batch_dir: Directory containing CIFAR-10 IDX files
        batch_name: Batch name without extension (e.g., "train_batch_1", "test_batch")

    Returns:
        Tuple of (images, labels):
        - images: ExTensor of shape (N, 3, 32, 32) normalized float32
        - labels: ExTensor of shape (N,) uint8

    Raises:
        Error: If batch files cannot be read
    """
    var images_path = batch_dir + "/" + batch_name + "_images.idx"
    var labels_path = batch_dir + "/" + batch_name + "_labels.idx"

    # Load raw images and labels
    var images_raw = load_idx_images_rgb(images_path)
    var labels = load_idx_labels(labels_path)

    # Normalize images
    var images_normalized = normalize_images_rgb(images_raw)

    return (images_normalized^, labels^)
