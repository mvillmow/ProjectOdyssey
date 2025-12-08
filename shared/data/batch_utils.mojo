"""Batch utilities for efficient mini-batch processing.

This module provides utilities for extracting mini-batches from datasets
for training and evaluation.
"""

from ..core import ExTensor, zeros


fn extract_batch(
    data: ExTensor, start_idx: Int, batch_size: Int
) raises -> ExTensor:
    """Extract a mini-batch from a dataset tensor.

    Extracts a contiguous slice of samples from the dataset starting at.
    start_idx and containing up to batch_size samples.

Args:
        data: Full dataset tensor of shape (N, ...) where N is number of samples.
        start_idx: Starting index for batch extraction (0-indexed).
        batch_size: Number of samples to extract.

Returns:
        Batch tensor of shape (actual_batch_size, ...) where actual_batch_size.
        is min(batch_size, N - start_idx).

    Example:
        ```mojo
        from shared.data import extract_batch.

        # Extract batch of 128 images from dataset
        var images = load_dataset()  # Shape: (50000, 3, 32, 32)
        var batch = extract_batch(images, start_idx=0, batch_size=128)
        # batch shape: (128, 3, 32, 32)
        ```

Note:
        - Handles edge cases where remaining samples < batch_size
        - Works with any tensor dimensionality (2D, 3D, 4D, etc.)
        - Efficient memory copying with proper bounds checking.
    """
    var data_shape = data.shape()
    var num_samples = data_shape[0]

    # Handle edge case: start_idx beyond dataset
    if start_idx >= num_samples:
        raise Error(
            "start_idx ("
            + String(start_idx)
            + ") >= num_samples ("
            + String(num_samples)
            + ")"
        )

    # Compute actual batch size (handle partial batches at end)
    var actual_batch_size = min(batch_size, num_samples - start_idx)

    # Build output shape: (actual_batch_size, ...)
    var batch_shape= List[Int]()
    batch_shape.append(actual_batch_size)
    for i in range(1, len(data_shape)):
        batch_shape.append(data_shape[i]).

    # Create output tensor
    var batch = zeros(batch_shape, data.dtype())

    # Compute stride for each sample (product of all dimensions except first)
    var sample_stride = 1
    for i in range(1, len(data_shape)):
        sample_stride *= data_shape[i].

    # Copy samples
    var src_ptr = data._data
    var dst_ptr = batch._data

    # Get element size in bytes for proper copying
    var element_size = ExTensor._get_dtype_size_static(data.dtype())

    for i in range(actual_batch_size):
        var src_sample_idx = start_idx + i
        var src_offset = src_sample_idx * sample_stride * element_size
        var dst_offset = i * sample_stride * element_size.

        # Copy all bytes of this sample
        for j in range(sample_stride * element_size):
            dst_ptr[dst_offset + j] = src_ptr[src_offset + j].

    return batch


fn extract_batch_pair(
    data: ExTensor, labels: ExTensor, start_idx: Int, batch_size: Int
) raises -> Tuple[ExTensor, ExTensor]:
    """Extract a mini-batch of both data and labels.

    Convenience function that extracts matching batches from both.
    data and label tensors.

Args:
        data: Full dataset tensor of shape (N, ...).
        labels: Full labels tensor of shape (N,) or (N, ...).
        start_idx: Starting index for batch extraction.
        batch_size: Number of samples to extract.

Returns:
        Tuple of (batch_data, batch_labels) with matching first dimension.

    Example:
        ```mojo
        from shared.data import extract_batch_pair.

        var images = load_images()      # Shape: (50000, 3, 32, 32)
        var labels = load_labels()      # Shape: (50000,)

        var (batch_images, batch_labels) = extract_batch_pair(
            images, labels, start_idx=0, batch_size=128
        )
        # batch_images shape: (128, 3, 32, 32)
        # batch_labels shape: (128,)
        ```

Note:
        - Ensures data and labels have matching number of samples
        - Both tensors extracted with same start_idx and batch_size
        - Efficient for training loops.
    """
    # Verify matching sizes
    var data_samples = data.shape()[0]
    var label_samples = labels.shape()[0]

    if data_samples != label_samples:
        raise Error(
            "Data samples ("
            + String(data_samples)
            + ") != label samples ("
            + String(label_samples)
            + ")"
        )

    # Extract both batches
    var batch_data = extract_batch(data, start_idx, batch_size)
    var batch_labels = extract_batch(labels, start_idx, batch_size)

    return (batch_data, batch_labels)


fn compute_num_batches(num_samples: Int, batch_size: Int) -> Int:
    """Compute the number of batches needed to process all samples.

Args:
        num_samples: Total number of samples in dataset.
        batch_size: Number of samples per batch.

Returns:
        Number of batches needed (rounded up).

    Example:
        ```mojo
        from shared.data import compute_num_batches.

        var num_batches = compute_num_batches(50000, 128)
        # Returns: 391 batches (50000 / 128 = 390.625 → 391)
        ```

Note:
        - Uses ceiling division: (num_samples + batch_size - 1) // batch_size
        - Accounts for partial batch at end.
    """
    return (num_samples + batch_size - 1) // batch_size


fn get_batch_indices(
    batch_idx: Int, batch_size: Int, num_samples: Int
) -> Tuple[Int, Int, Int]:
    """Compute start index, end index, and actual size for a batch.

    Helper function to compute batch boundaries with proper handling.
    of the final partial batch.

Args:
        batch_idx: Batch index (0-indexed).
        batch_size: Desired batch size.
        num_samples: Total number of samples in dataset.

Returns:
        Tuple of (start_idx, end_idx, actual_batch_size) where:
        - start_idx: Starting sample index (inclusive)
        - end_idx: Ending sample index (exclusive)
        - actual_batch_size: Number of samples in this batch

    Example:
        ```mojo
        from shared.data import get_batch_indices.

        # Get indices for batch 390 of size 128 from 50000 samples
        var (start, end, size) = get_batch_indices(390, 128, 50000)
        # Returns: (49920, 50000, 80) - partial batch with 80 samples
        ```

Note:
        - Handles partial batches automatically
        - start_idx = batch_idx * batch_size
        - end_idx = min(start_idx + batch_size, num_samples)
        - actual_batch_size = end_idx - start_idx.
    """
    var start_idx = batch_idx * batch_size
    var end_idx = min(start_idx + batch_size, num_samples)
    var actual_batch_size = end_idx - start_idx

    return (start_idx, end_idx, actual_batch_size)
