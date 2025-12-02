"""IDX File Format Loader

Provides functions to load data from IDX file format (used by MNIST, EMNIST, and similar datasets).

IDX File Format:
    Labels: [magic(4B)][count(4B)][label_data...]
    Images: [magic(4B)][count(4B)][rows(4B)][cols(4B)][pixel_data...]
    Images RGB: [magic(4B)][count(4B)][channels(4B)][rows(4B)][cols(4B)][pixel_data...]

All integers are big-endian (network byte order).

Magic Numbers:
    - 2049: Label files
    - 2051: Grayscale image files (1 channel)
    - 2052: RGB image files (3 channels, custom extension)

References:
    - IDX Format: http://yann.lecun.com/exdb/mnist/
"""

from shared.core import ExTensor, zeros
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
    """Load grayscale images from IDX file format.

    Args:
        filepath: Path to IDX grayscale images file

    Returns:
        ExTensor of shape (num_samples, 1, rows, cols) with uint8 pixel values

    Raises:
        Error: If file format is invalid or cannot be read

    Note:
        For single-channel (grayscale) images. Shape includes channel dimension for
        consistency with multi-channel loaders.
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


fn load_idx_images_rgb(filepath: String) raises -> ExTensor:
    """Load RGB images from IDX file format.

    Args:
        filepath: Path to IDX RGB images file

    Returns:
        ExTensor of shape (num_samples, channels, rows, cols) with uint8 pixel values

    Raises:
        Error: If file format is invalid or cannot be read

    Note:
        For CIFAR-10 and similar: (N, 3, 32, 32) where 3 channels are RGB
    """
    # Read entire file
    var content: String
    with open(filepath, "r") as f:
        content = f.read()

    var file_size = len(content)
    if file_size < 20:  # Header is 20 bytes for RGB images (magic + count + channels + rows + cols)
        raise Error("IDX file too small")

    var data_bytes = content.unsafe_ptr()

    # Parse header
    var magic = read_uint32_be(data_bytes, 0)
    if magic != 2052:  # RGB image file magic number (custom extension)
        raise Error("Invalid IDX RGB image file magic number: " + String(magic))

    var num_images = read_uint32_be(data_bytes, 4)
    var num_channels = read_uint32_be(data_bytes, 8)
    var num_rows = read_uint32_be(data_bytes, 12)
    var num_cols = read_uint32_be(data_bytes, 16)

    if num_channels != 3:
        raise Error("Expected 3 RGB channels, got: " + String(num_channels))

    var expected_size = 20 + (num_images * num_channels * num_rows * num_cols)
    if file_size < expected_size:
        raise Error("IDX file size mismatch")

    # Create output tensor (num_images, channels, rows, cols) for CNN input
    var shape = List[Int]()
    shape.append(num_images)
    shape.append(num_channels)  # RGB channels
    shape.append(num_rows)
    shape.append(num_cols)
    var images = zeros(shape, DType.uint8)

    # Copy image data
    var images_data = images._data
    var total_pixels = num_images * num_channels * num_rows * num_cols
    for i in range(total_pixels):
        images_data[i] = data_bytes[20 + i]

    return images^
