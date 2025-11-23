"""Data Loading Utilities for CIFAR-10

Provides functions to load CIFAR-10 dataset from IDX file format (converted from binary batches).

IDX File Format:
    Labels: [magic(4B)][count(4B)][label_data...]
    Images: [magic(4B)][count(4B)][channels(4B)][rows(4B)][cols(4B)][pixel_data...]

All integers are big-endian (network byte order).

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
from memory import UnsafePointer


fn read_uint32_be(data: UnsafePointer[UInt8], offset: Int) -> Int:
    """Read 32-bit unsigned integer in big-endian format.

    Args:
        data: Pointer to byte array
        offset: Byte offset to read from

    Returns:
        Integer value in host byte order
    """
    var b0 = Int(data[offset])
    var b1 = Int(data[offset + 1])
    var b2 = Int(data[offset + 2])
    var b3 = Int(data[offset + 3])

    return (b0 << 24) | (b1 << 16) | (b2 << 8) | b3


fn load_idx_labels(filepath: String) raises -> ExTensor:
    """Load labels from IDX file format.

    Args:
        filepath: Path to IDX labels file

    Returns:
        ExTensor of shape (num_samples,) with uint8 label values

    Raises:
        Error: If file format is invalid or cannot be read
    """
    # Read entire file
    var content: String
    with open(filepath, "r") as f:
        content = f.read()

    # Convert string to bytes (this is a workaround - ideally we'd read binary)
    var file_size = len(content)
    if file_size < 8:
        raise Error("IDX file too small")

    # Parse header (treating String as bytes - this is a simplification)
    var data_bytes = content._as_ptr()

    var magic = read_uint32_be(data_bytes, 0)
    if magic != 2049:  # Label file magic number
        raise Error("Invalid IDX label file magic number: " + str(magic))

    var num_items = read_uint32_be(data_bytes, 4)

    if file_size < 8 + num_items:
        raise Error("IDX file size mismatch")

    # Create output tensor
    var shape = List[Int]()
    shape[0] = num_items
    var labels = zeros(shape, DType.uint8)

    # Copy label data
    var labels_data = labels._data
    for i in range(num_items):
        labels_data[i] = data_bytes[8 + i]

    return labels^


fn load_idx_images_rgb(filepath: String) raises -> ExTensor:
    """Load RGB images from IDX file format.

    Args:
        filepath: Path to IDX RGB images file

    Returns:
        ExTensor of shape (num_samples, channels, rows, cols) with uint8 pixel values

    Raises:
        Error: If file format is invalid or cannot be read

    Note:
        For CIFAR-10: (N, 3, 32, 32) where 3 channels are RGB
    """
    # Read entire file
    var content: String
    with open(filepath, "r") as f:
        content = f.read()

    var file_size = len(content)
    if file_size < 20:  # Header is 20 bytes for RGB images (magic + count + channels + rows + cols)
        raise Error("IDX file too small")

    var data_bytes = content._as_ptr()

    # Parse header
    var magic = read_uint32_be(data_bytes, 0)
    if magic != 2052:  # RGB image file magic number (custom extension)
        raise Error("Invalid IDX RGB image file magic number: " + str(magic))

    var num_images = read_uint32_be(data_bytes, 4)
    var num_channels = read_uint32_be(data_bytes, 8)
    var num_rows = read_uint32_be(data_bytes, 12)
    var num_cols = read_uint32_be(data_bytes, 16)

    if num_channels != 3:
        raise Error("Expected 3 RGB channels, got: " + str(num_channels))

    var expected_size = 20 + (num_images * num_channels * num_rows * num_cols)
    if file_size < expected_size:
        raise Error("IDX file size mismatch")

    # Create output tensor (num_images, channels, rows, cols) for CNN input
    var shape = List[Int]()
    shape[0] = num_images
    shape[1] = num_channels  # RGB channels
    shape[2] = num_rows
    shape[3] = num_cols
    var images = zeros(shape, DType.uint8)

    # Copy image data
    var images_data = images._data
    var total_pixels = num_images * num_channels * num_rows * num_cols
    for i in range(total_pixels):
        images_data[i] = data_bytes[20 + i]

    return images^


fn normalize_images_rgb(inout images: ExTensor) raises -> ExTensor:
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

        Now uses shared library implementation.
    """
    # ImageNet normalization parameters (R, G, B)
    var mean = (Float32(0.485), Float32(0.456), Float32(0.406))
    var std = (Float32(0.229), Float32(0.224), Float32(0.225))

    # Use shared library normalize_rgb function
    return normalize_rgb_shared(images, mean, std)


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


fn load_cifar10_train(data_dir: String) raises -> Tuple[ExTensor, ExTensor]:
    """Load all CIFAR-10 training data (all 5 batches).

    Args:
        data_dir: Directory containing CIFAR-10 IDX files

    Returns:
        Tuple of (images, labels):
        - images: ExTensor of shape (50000, 3, 32, 32) normalized float32
        - labels: ExTensor of shape (50000,) uint8

    Raises:
        Error: If batch files cannot be read

    Note:
        Concatenates all 5 training batches (10,000 images each) into a single tensor.
    """
    # Load first batch to get shape information
    var batch1 = load_cifar10_batch(data_dir, "train_batch_1")
    var batch1_images = batch1[0]
    var batch1_labels = batch1[1]

    # Create full training tensors (50,000 samples)
    var train_shape = List[Int]()
    train_shape[0] = 50000  # 5 batches * 10,000 images
    train_shape[1] = 3      # RGB channels
    train_shape[2] = 32     # height
    train_shape[3] = 32     # width
    var all_images = zeros(train_shape, DType.float32)

    var label_shape = List[Int]()
    label_shape[0] = 50000
    var all_labels = zeros(label_shape, DType.uint8)

    # Copy first batch
    var offset = 0
    _copy_batch_data(all_images, all_labels, batch1_images, batch1_labels, offset)

    # Load and copy remaining 4 batches
    for batch_idx in range(2, 6):  # batches 2-5
        var batch_name = "train_batch_" + str(batch_idx)
        var batch = load_cifar10_batch(data_dir, batch_name)
        offset += 10000
        _copy_batch_data(all_images, all_labels, batch[0], batch[1], offset)

    return (all_images^, all_labels^)


fn load_cifar10_test(data_dir: String) raises -> Tuple[ExTensor, ExTensor]:
    """Load CIFAR-10 test data.

    Args:
        data_dir: Directory containing CIFAR-10 IDX files

    Returns:
        Tuple of (images, labels):
        - images: ExTensor of shape (10000, 3, 32, 32) normalized float32
        - labels: ExTensor of shape (10000,) uint8

    Raises:
        Error: If test batch file cannot be read
    """
    return load_cifar10_batch(data_dir, "test_batch")


fn _copy_batch_data(
    inout dest_images: ExTensor,
    inout dest_labels: ExTensor,
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
    var batch_size = src_images.shape[0]
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
