"""Test fixtures for ExTensor testing.

Provides common tensor creation utilities for tests, including
random tensors, sequential tensors, and special value tensors.
"""

from collections import List
from shared.core.extensor import (
    ExTensor,
    zeros,
    ones,
    full,
    nan_tensor,
    inf_tensor,
)
from random import random_float64, seed as set_seed


fn random_tensor(
    shape: List[Int], dtype: DType, seed_value: Int = 42
) raises -> ExTensor:
    """Create tensor with seeded random values for reproducibility.

    Args:
        shape: Shape of the tensor to create
        dtype: Data type of the tensor
        seed_value: Random seed for reproducibility (default: 42)

    Returns:
        ExTensor filled with random values in [0.0, 1.0)

    Examples:
        ```
        var shape = List[Int]()
        shape.append(3)
        shape.append(4)
        var t = random_tensor(shape, DType.float32, seed=42)
        ```
    """
    set_seed(seed_value)
    var tensor = ExTensor(shape, dtype)
    var numel = tensor.numel()

    if dtype == DType.float32:
        var ptr = tensor._data.bitcast[Scalar[DType.float32]]()
        for i in range(numel):
            ptr[i] = Scalar[DType.float32](random_float64())
    elif dtype == DType.float64:
        var ptr = tensor._data.bitcast[Scalar[DType.float64]]()
        for i in range(numel):
            ptr[i] = Scalar[DType.float64](random_float64())
    elif dtype == DType.float16:
        var ptr = tensor._data.bitcast[Scalar[DType.float16]]()
        for i in range(numel):
            ptr[i] = Scalar[DType.float16](random_float64())
    elif dtype == DType.int32:
        var ptr = tensor._data.bitcast[Scalar[DType.int32]]()
        for i in range(numel):
            ptr[i] = Scalar[DType.int32](Int(random_float64() * 100))
    elif dtype == DType.int64:
        var ptr = tensor._data.bitcast[Scalar[DType.int64]]()
        for i in range(numel):
            ptr[i] = Scalar[DType.int64](Int(random_float64() * 100))
    else:
        raise Error("random_tensor: unsupported dtype")

    return tensor^


fn sequential_tensor(shape: List[Int], dtype: DType) raises -> ExTensor:
    """Create tensor with sequential values [0, 1, 2, ...].

    Args:
        shape: Shape of the tensor to create
        dtype: Data type of the tensor

    Returns:
        ExTensor filled with values 0, 1, 2, ... up to numel-1

    Examples:
        ```
        var shape = List[Int]()
        shape.append(3)
        shape.append(4)
        var t = sequential_tensor(shape, DType.float32)
        # Result: [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]]
        ```
    """
    var tensor = ExTensor(shape, dtype)
    var numel = tensor.numel()

    if dtype == DType.float32:
        var ptr = tensor._data.bitcast[Scalar[DType.float32]]()
        for i in range(numel):
            ptr[i] = Scalar[DType.float32](Float32(i))
    elif dtype == DType.float64:
        var ptr = tensor._data.bitcast[Scalar[DType.float64]]()
        for i in range(numel):
            ptr[i] = Scalar[DType.float64](Float64(i))
    elif dtype == DType.float16:
        var ptr = tensor._data.bitcast[Scalar[DType.float16]]()
        for i in range(numel):
            ptr[i] = Scalar[DType.float16](Float32(i))
    elif dtype == DType.int32:
        var ptr = tensor._data.bitcast[Scalar[DType.int32]]()
        for i in range(numel):
            ptr[i] = Scalar[DType.int32](i)
    elif dtype == DType.int64:
        var ptr = tensor._data.bitcast[Scalar[DType.int64]]()
        for i in range(numel):
            ptr[i] = Scalar[DType.int64](i)
    else:
        raise Error("sequential_tensor: unsupported dtype")

    return tensor^


fn nan_tensor_fixture(shape: List[Int]) raises -> ExTensor:
    """Create tensor filled with NaN values (float32).

    Args:
        shape: Shape of the tensor to create

    Returns:
        ExTensor filled with NaN values

    Examples:
        ```
        var shape = List[Int]()
        shape.append(3)
        shape.append(4)
        var t = nan_tensor_fixture(shape)
        ```
    """
    return nan_tensor(shape, DType.float32)


fn inf_tensor_fixture(shape: List[Int]) raises -> ExTensor:
    """Create tensor filled with positive infinity values (float32).

    Args:
        shape: Shape of the tensor to create

    Returns:
        ExTensor filled with infinity values

    Examples:
        ```
        var shape = List[Int]()
        shape.append(3)
        shape.append(4)
        var t = inf_tensor_fixture(shape)
        ```
    """
    return inf_tensor(shape, DType.float32)


fn ones_like_fixture(tensor: ExTensor) raises -> ExTensor:
    """Create ones tensor with same shape and dtype as input.

    Args:
        tensor: Template tensor for shape and dtype

    Returns:
        ExTensor filled with ones, matching input shape and dtype

    Examples:
        ```
        var shape = List[Int]()
        shape.append(3)
        shape.append(4)
        var t1 = zeros(shape, DType.float32)
        var t2 = ones_like_fixture(t1)  # Shape (3, 4), all ones
        ```
    """
    return ones(tensor.shape(), tensor.dtype())


fn zeros_like_fixture(tensor: ExTensor) raises -> ExTensor:
    """Create zeros tensor with same shape and dtype as input.

    Args:
        tensor: Template tensor for shape and dtype

    Returns:
        ExTensor filled with zeros, matching input shape and dtype

    Examples:
        ```
        var shape = List[Int]()
        shape.append(3)
        shape.append(4)
        var t1 = ones(shape, DType.float32)
        var t2 = zeros_like_fixture(t1)  # Shape (3, 4), all zeros
        ```
    """
    return zeros(tensor.shape(), tensor.dtype())
