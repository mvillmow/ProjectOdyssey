"""
Testing Utilities - Data Generators

Purpose: Generate synthetic test data for ML implementations
Language: Mojo (required for performance-critical data generation, SIMD optimization)
"""

from random import random_float64
from tensor import Tensor, TensorShape
from memory import memset_zero


struct TensorGenerator(Copyable, Movable):
    """Generate test tensors with various patterns and distributions."""

    fn __init__(mut self):
        """Initialize tensor generator."""
        pass

    fn generate_random(
        self,
        shape: TensorShape,
        min_val: Float64 = 0.0,
        max_val: Float64 = 1.0
    ) -> Tensor[DType.float32]:
        """
        Generate random tensor with uniform distribution.

        Args:
            shape: Tensor shape
            min_val: Minimum value
            max_val: Maximum value

        Returns:
            Random tensor
        """
        # Calculate total size
        var size: Int = 1
        for i in range(shape.rank()):
            size *= shape[i]

        # Create tensor
        var tensor = Tensor[DType.float32](shape)

        # Fill with random values
        # TODO: Use proper random number generation
        # For now, this is a placeholder implementation
        for i in range(size):
            let val = random_float64(0, 1)
            let scaled = min_val + val * (max_val - min_val)
            tensor[i] = scaled.cast[DType.float32]()

        return tensor

    fn generate_zeros(self, shape: TensorShape) -> Tensor[DType.float32]:
        """
        Generate tensor filled with zeros.

        Args:
            shape: Tensor shape

        Returns:
            Zero tensor
        """
        var tensor = Tensor[DType.float32](shape)
        # TODO: Use memset_zero or similar
        return tensor

    fn generate_ones(self, shape: TensorShape) -> Tensor[DType.float32]:
        """
        Generate tensor filled with ones.

        Args:
            shape: Tensor shape

        Returns:
            Tensor filled with 1.0
        """
        var tensor = Tensor[DType.float32](shape)

        # Fill with ones
        var size: Int = 1
        for i in range(shape.rank()):
            size *= shape[i]

        for i in range(size):
            tensor[i] = 1.0

        return tensor

    fn generate_batch(
        self,
        batch_size: Int,
        *dims: Int,
        min_val: Float64 = 0.0,
        max_val: Float64 = 1.0
    ) -> Tensor[DType.float32]:
        """
        Generate a batch of random data.

        Args:
            batch_size: Batch size (first dimension)
            dims: Remaining dimensions
            min_val: Minimum value
            max_val: Maximum value

        Returns:
            Random batch tensor
        """
        # Build shape with batch_size as first dimension
        # TODO: Implement proper variadic handling
        # For now, simplified version for common cases

        # Create 2D batch (batch_size, feature_dim)
        if dims.__len__() == 1:
            let shape = TensorShape(batch_size, dims[0])
            return self.generate_random(shape, min_val, max_val)

        # Create 4D batch (batch_size, channels, height, width)
        if dims.__len__() == 3:
            let shape = TensorShape(batch_size, dims[0], dims[1], dims[2])
            return self.generate_random(shape, min_val, max_val)

        # Fallback: 1D batch
        let shape = TensorShape(batch_size)
        return self.generate_random(shape, min_val, max_val)


fn create_test_tensor(
    *dims: Int,
    fill_value: Float64 = 0.0
) -> Tensor[DType.float32]:
    """
    Convenience function to create a test tensor with specific value.

    Args:
        dims: Tensor dimensions
        fill_value: Value to fill tensor with

    Returns:
        Filled tensor
    """
    # TODO: Implement variadic dimension handling
    # Simplified version for common cases

    var tensor: Tensor[DType.float32]

    if dims.__len__() == 1:
        tensor = Tensor[DType.float32](TensorShape(dims[0]))
    elif dims.__len__() == 2:
        tensor = Tensor[DType.float32](TensorShape(dims[0], dims[1]))
    elif dims.__len__() == 3:
        tensor = Tensor[DType.float32](TensorShape(dims[0], dims[1], dims[2]))
    elif dims.__len__() == 4:
        tensor = Tensor[DType.float32](TensorShape(dims[0], dims[1], dims[2], dims[3]))
    else:
        # Fallback: 1D with size 10
        tensor = Tensor[DType.float32](TensorShape(10))

    # Fill with value
    let size = tensor.num_elements()
    for i in range(size):
        tensor[i] = fill_value.cast[DType.float32]()

    return tensor
