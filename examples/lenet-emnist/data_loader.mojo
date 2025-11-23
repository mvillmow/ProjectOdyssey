"""Data Loading Utilities for EMNIST

Provides functions to load EMNIST dataset from IDX file format.

IDX File Format:
    Labels: [magic(4B)][count(4B)][label_data...]
    Images: [magic(4B)][count(4B)][rows(4B)][cols(4B)][pixel_data...]

All integers are big-endian (network byte order).

References:
    - IDX Format: http://yann.lecun.com/exdb/mnist/
    - EMNIST Dataset: https://www.nist.gov/itl/products-and-services/emnist-dataset
"""

from shared.core import ExTensor, zeros
from collections import List
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
    var data_bytes = content.unsafe_ptr()

    var magic = read_uint32_be(data_bytes, 0)
    if magic != 2049:  # Label file magic number
        raise Error("Invalid IDX label file magic number: " + String(magic))

    var num_items = read_uint32_be(data_bytes, 4)

    if file_size < 8 + num_items:
        raise Error("IDX file size mismatch")

    # Create output tensor
    var shape = List[Int]()
    shape.append(num_items)
    var labels = zeros(shape, DType.uint8)

    # Copy label data
    var labels_data = labels._data
    for i in range(num_items):
        labels_data[i] = data_bytes[8 + i]

    return labels^


fn load_idx_images(filepath: String) raises -> ExTensor:
    """Load images from IDX file format.

    Args:
        filepath: Path to IDX images file

    Returns:
        ExTensor of shape (num_samples, rows, cols) with uint8 pixel values

    Raises:
        Error: If file format is invalid or cannot be read
    """
    # Read entire file
    var content: String
    with open(filepath, "r") as f:
        content = f.read()

    var file_size = len(content)
    if file_size < 16:
        raise Error("IDX file too small")

    var data_bytes = content.unsafe_ptr()

    # Parse header
    var magic = read_uint32_be(data_bytes, 0)
    if magic != 2051:  # Image file magic number
        raise Error("Invalid IDX image file magic number: " + String(magic))

    var num_images = read_uint32_be(data_bytes, 4)
    var num_rows = read_uint32_be(data_bytes, 8)
    var num_cols = read_uint32_be(data_bytes, 12)

    var expected_size = 16 + (num_images * num_rows * num_cols)
    if file_size < expected_size:
        raise Error("IDX file size mismatch")

    # Create output tensor (num_images, 1, rows, cols) for CNN input
    var shape = List[Int]()
    shape.append(num_images)
    shape.append(1)  # Single channel (grayscale)
    shape.append(num_rows)
    shape.append(num_cols)
    var images = zeros(shape, DType.uint8)

    # Copy image data
    var images_data = images._data
    var total_pixels = num_images * num_rows * num_cols
    for i in range(total_pixels):
        images_data[i] = data_bytes[16 + i]

    return images^


fn normalize_images(mut images: ExTensor) raises -> ExTensor:
    """Normalize uint8 images to float32 in range [0, 1].

    Args:
        images: Input images as uint8 ExTensor

    Returns:
        Normalized images as float32 ExTensor

    Note:
        Converts pixel values from [0, 255] to [0.0, 1.0]
    """
    var shape = images.shape
    var normalized = zeros(shape, DType.float32)

    var num_elements = images.numel()
    var src_data = images._data
    var dst_data = normalized._data.bitcast[Float32]()

    for i in range(num_elements):
        dst_data[i] = Float32(src_data[i]) / 255.0

    return normalized^


fn one_hot_encode(labels: ExTensor, num_classes: Int) raises -> ExTensor:
    """Convert integer labels to one-hot encoded float32 tensor.

    Args:
        labels: Integer labels as uint8 ExTensor, shape (num_samples,)
        num_classes: Total number of classes

    Returns:
        One-hot encoded labels as float32 ExTensor, shape (num_samples, num_classes)

    Example:
        labels = [0, 2, 1]  # 3 samples, 3 classes
        one_hot = one_hot_encode(labels, 3)
        # Result shape: (3, 3)
        # [[1.0, 0.0, 0.0],
        #  [0.0, 0.0, 1.0],
        #  [0.0, 1.0, 0.0]]
    """
    var num_samples = labels.shape[0]

    # Create output tensor (num_samples, num_classes)
    var shape = List[Int]()
    shape.append(num_samples)
    shape.append(num_classes)
    var one_hot = zeros(shape, DType.float32)

    # Fill one-hot encoding
    var labels_data = labels._data
    var one_hot_data = one_hot._data.bitcast[Float32]()

    for i in range(num_samples):
        var label_idx = Int(labels_data[i])
        if label_idx < 0 or label_idx >= num_classes:
            raise Error("Label index out of range: " + String(label_idx))

        # Set the corresponding class to 1.0
        var offset = i * num_classes + label_idx
        one_hot_data[offset] = 1.0

    return one_hot^
