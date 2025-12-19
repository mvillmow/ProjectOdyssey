"""Test fixtures for ExTensor testing.

Provides common tensor creation utilities for tests, including
random tensors, sequential tensors, and special value tensors.

These fixtures wrap the comprehensive infrastructure in shared.testing
with convenient test-specific APIs.
"""

from shared.core.extensor import ExTensor, zeros, ones
from shared.testing.data_generators import random_tensor as shared_random_tensor
from shared.testing.data_generators import random_uniform


fn random_tensor(shape: List[Int], dtype: DType = DType.float32) raises -> ExTensor:
    """Create a tensor with random values from uniform distribution [0, 1).

    Args:
        shape: Shape of the output tensor as a list of dimensions.
        dtype: Data type of tensor elements (default: float32).

    Returns:
        ExTensor with random values uniformly distributed in [0, 1).

    Example:
        ```mojo
        var weights = random_tensor([10, 5], DType.float32)
        # Creates 10x5 tensor with random values in [0, 1)
        ```

    Raises:
        Error: If operation fails.
    """
    return shared_random_tensor(shape, dtype)


fn sequential_tensor(shape: List[Int], dtype: DType = DType.float32) raises -> ExTensor:
    """Create tensor with sequential values 0, 1, 2, 3, ...

    Tensor is filled with sequential values in row-major order, then reshaped
    to the requested shape.

    Args:
        shape: Shape of the output tensor as a list of dimensions.
        dtype: Data type of tensor elements (default: float32).

    Returns:
        ExTensor with values 0, 1, 2, ... in flattened order.

    Example:
        ```mojo
        var tensor = sequential_tensor([2, 3], DType.float32)
        # Returns tensor [[0, 1, 2], [3, 4, 5]]
        ```

    Raises:
        Error: If operation fails.
    """
    var tensor = zeros(shape, dtype)

    # Calculate total number of elements
    var numel = 1
    for dim in shape:
        numel *= dim

    # Fill with sequential values
    for i in range(numel):
        tensor._set_float64(i, Float64(i))

    return tensor


fn nan_tensor(shape: List[Int]) raises -> ExTensor:
    """Create tensor filled with NaN values.

    Args:
        shape: Shape of the output tensor as a list of dimensions.

    Returns:
        ExTensor with all elements set to NaN.

    Example:
        ```mojo
        var tensor = nan_tensor([3, 3])
        # Returns 3x3 tensor with all NaN values
        ```

    Note:
        Creates float32 tensors with NaN values. NaN is represented as
        0x7fc00000 in float32 bit representation.

    Raises:
        Error: If operation fails.
    """
    var tensor = zeros(shape, DType.float32)

    # Calculate total number of elements
    var numel = 1
    for dim in shape:
        numel *= dim

    # Fill with NaN values
    # NaN in float32 is typically 0x7fc00000 in bits
    var nan_val = 0.0 / 0.0  # IEEE 754 NaN
    for i in range(numel):
        tensor._set_float64(i, nan_val)

    return tensor


fn inf_tensor(shape: List[Int]) raises -> ExTensor:
    """Create tensor filled with infinity values.

    Args:
        shape: Shape of the output tensor as a list of dimensions.

    Returns:
        ExTensor with all elements set to positive infinity.

    Example:
        ```mojo
        var tensor = inf_tensor([3, 3])
        # Returns 3x3 tensor with all infinity values
        ```

    Note:
        Creates float32 tensors with positive infinity values.
        Positive infinity is represented as 0x7f800000 in float32 bit representation.

    Raises:
        Error: If operation fails.
    """
    var tensor = zeros(shape, DType.float32)

    # Calculate total number of elements
    var numel = 1
    for dim in shape:
        numel *= dim

    # Fill with positive infinity
    var inf_val = 1.0 / 0.0  # IEEE 754 positive infinity
    for i in range(numel):
        tensor._set_float64(i, inf_val)

    return tensor


fn ones_like(tensor: ExTensor) raises -> ExTensor:
    """Create tensor of ones matching input shape and dtype.

    Args:
        tensor: Template tensor to match shape and dtype from.

    Returns:
        ExTensor of ones with same shape and dtype as input.

    Example:
        ```mojo
        var t1 = random_tensor([3, 4], DType.float32)
        var t2 = ones_like(t1)
        # t2 has shape [3, 4] and dtype float32, all values are 1.0
        ```

    Raises:
        Error: If operation fails.
    """
    return ones(tensor.shape(), tensor.dtype())


fn zeros_like(tensor: ExTensor) raises -> ExTensor:
    """Create tensor of zeros matching input shape and dtype.

    Args:
        tensor: Template tensor to match shape and dtype from.

    Returns:
        ExTensor of zeros with same shape and dtype as input.

    Example:
        ```mojo
        var t1 = random_tensor([3, 4], DType.float32)
        var t2 = zeros_like(t1)
        # t2 has shape [3, 4] and dtype float32, all values are 0.0
        ```

    Raises:
        Error: If operation fails.
    """
    return zeros(tensor.shape(), tensor.dtype())
