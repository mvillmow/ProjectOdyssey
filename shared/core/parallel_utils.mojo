"""Parallel processing utilities for batch operations.

This module provides adaptive parallel execution based on batch size,
avoiding thread overhead for small batches while enabling parallelism
for large batches.
"""

from algorithm import parallelize

# Minimum batch size to warrant parallelization
alias PARALLEL_BATCH_THRESHOLD: Int = 4

# Default worker count (0 = system decides)
alias DEFAULT_NUM_WORKERS: Int = 0


fn should_parallelize(batch_size: Int, threshold: Int = PARALLEL_BATCH_THRESHOLD) -> Bool:
    """Determine if batch size warrants parallel execution.

    Args:
        batch_size: Number of batch elements
        threshold: Minimum batch size for parallelization

    Returns:
        True if parallelization is beneficial
    """
    return batch_size >= threshold


fn parallel_for_batch[
    func: fn(Int) capturing -> None
](batch_size: Int, num_workers: Int = DEFAULT_NUM_WORKERS):
    """Execute function across batch indices in parallel.

    Args:
        func: Function to execute for each batch index
        batch_size: Number of batch elements
        num_workers: Number of worker threads (0 = auto)
    """
    if num_workers > 0:
        parallelize[func](batch_size, num_workers)
    else:
        parallelize[func](batch_size)
