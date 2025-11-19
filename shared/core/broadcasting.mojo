"""Broadcasting utilities for ExTensor.

Implements NumPy-style broadcasting rules for tensor operations.
"""

from collections.vector import DynamicVector


fn broadcast_shapes(
    shape1: DynamicVector[Int], shape2: DynamicVector[Int]
) raises -> DynamicVector[Int]:
    """Compute the broadcast shape of two tensor shapes.

    Args:
        shape1: First tensor shape
        shape2: Second tensor shape

    Returns:
        The broadcast result shape

    Raises:
        Error if shapes are not broadcast-compatible

    Broadcasting rules:
        1. Compare shapes element-wise from right to left
        2. Dimensions are compatible if they are equal or one is 1
        3. Missing dimensions are treated as 1
        4. Output shape is element-wise maximum of input shapes

    Examples:
        broadcast_shapes([3, 4, 5], [4, 5]) -> [3, 4, 5]
        broadcast_shapes([3, 1, 5], [3, 4, 5]) -> [3, 4, 5]
        broadcast_shapes([3, 4], [5, 4]) -> Error (incompatible)
    """
    let ndim1 = len(shape1)
    let ndim2 = len(shape2)
    let max_ndim = max(ndim1, ndim2)

    var result_shape = DynamicVector[Int]()

    # Process dimensions from right to left
    for i in range(max_ndim):
        # Get dimension from each shape (1 if dimension doesn't exist)
        let dim1_idx = ndim1 - 1 - i
        let dim2_idx = ndim2 - 1 - i

        let dim1 = shape1[dim1_idx] if dim1_idx >= 0 else 1
        let dim2 = shape2[dim2_idx] if dim2_idx >= 0 else 1

        # Check compatibility
        if dim1 != dim2 and dim1 != 1 and dim2 != 1:
            raise Error(
                "Shapes are not broadcast-compatible: dimension " + str(i) + " has sizes "
                + str(dim1) + " and " + str(dim2)
            )

        # Result dimension is the maximum
        let result_dim = max(dim1, dim2)
        result_shape.push_back(result_dim)

    # Reverse to get correct order (we built it backwards)
    var final_shape = DynamicVector[Int]()
    for i in range(len(result_shape) - 1, -1, -1):
        final_shape.push_back(result_shape[i])

    return final_shape


fn are_shapes_broadcastable(
    shape1: DynamicVector[Int], shape2: DynamicVector[Int]
) -> Bool:
    """Check if two shapes are broadcast-compatible.

    Args:
        shape1: First tensor shape
        shape2: Second tensor shape

    Returns:
        True if shapes are broadcast-compatible, False otherwise

    Examples:
        are_shapes_broadcastable([3, 4, 5], [4, 5]) -> True
        are_shapes_broadcastable([3, 4], [5, 4]) -> False
    """
    let ndim1 = len(shape1)
    let ndim2 = len(shape2)
    let max_ndim = max(ndim1, ndim2)

    for i in range(max_ndim):
        let dim1_idx = ndim1 - 1 - i
        let dim2_idx = ndim2 - 1 - i

        let dim1 = shape1[dim1_idx] if dim1_idx >= 0 else 1
        let dim2 = shape2[dim2_idx] if dim2_idx >= 0 else 1

        if dim1 != dim2 and dim1 != 1 and dim2 != 1:
            return False

    return True


fn compute_broadcast_strides(
    original_shape: DynamicVector[Int],
    broadcast_shape: DynamicVector[Int],
) -> DynamicVector[Int]:
    """Compute strides for broadcasting a tensor to a new shape.

    Args:
        original_shape: The original tensor shape
        broadcast_shape: The target broadcast shape

    Returns:
        Strides for the broadcast tensor (0 for broadcasted dimensions)

    Note:
        Dimensions that are 1 in the original shape get stride 0 in the broadcast.
        This allows efficient broadcasting without materializing extra copies.

    Examples:
        original_shape = [3, 1, 5]
        broadcast_shape = [3, 4, 5]
        result = [stride_for_3, 0, stride_for_5]  # Middle dimension has stride 0
    """
    let ndim_orig = len(original_shape)
    let ndim_broad = len(broadcast_shape)

    var broadcast_strides = DynamicVector[Int]()

    # Calculate original row-major strides
    var orig_strides = DynamicVector[Int]()
    var stride = 1
    for i in range(ndim_orig - 1, -1, -1):
        orig_strides.push_back(0)  # Preallocate
    for i in range(ndim_orig - 1, -1, -1):
        orig_strides[i] = stride
        stride *= original_shape[i]

    # Compute broadcast strides
    for i in range(ndim_broad):
        let orig_idx = i - (ndim_broad - ndim_orig)

        if orig_idx < 0:
            # Dimension doesn't exist in original -> stride 0
            broadcast_strides.push_back(0)
        elif original_shape[orig_idx] == 1 and broadcast_shape[i] > 1:
            # Dimension is 1 and being broadcast -> stride 0
            broadcast_strides.push_back(0)
        else:
            # Normal dimension -> use original stride
            broadcast_strides.push_back(orig_strides[orig_idx])

    return broadcast_strides


struct BroadcastIterator:
    """Iterator for efficiently iterating over broadcast tensor elements.

    This allows element-wise operations to work efficiently with broadcasting
    without materializing the full broadcast tensor.
    """

    var shape: DynamicVector[Int]
    var strides1: DynamicVector[Int]
    var strides2: DynamicVector[Int]
    var size: Int
    var position: Int

    fn __init__(
        inout self,
        shape: DynamicVector[Int],
        strides1: DynamicVector[Int],
        strides2: DynamicVector[Int],
    ):
        """Initialize broadcast iterator.

        Args:
            shape: The broadcast output shape
            strides1: Broadcast strides for first tensor
            strides2: Broadcast strides for second tensor
        """
        self.shape = shape
        self.strides1 = strides1
        self.strides2 = strides2
        self.position = 0

        # Calculate total size
        self.size = 1
        for i in range(len(shape)):
            self.size *= shape[i]

    fn __iter__(self) -> Self:
        """Return iterator."""
        return self

    fn __next__(inout self) raises -> (Int, Int):
        """Get next pair of indices for the two tensors.

        Returns:
            Tuple of (index1, index2) for accessing elements

        Raises:
            Error when iteration is complete
        """
        if self.position >= self.size:
            raise Error("Iterator exhausted")

        # Compute multi-dimensional index from flat position
        var remaining = self.position
        var idx1 = 0
        var idx2 = 0

        for i in range(len(self.shape)):
            let dim_size = self.shape[i]
            let coord = remaining // self.size  # TODO: Fix this calculation
            remaining = remaining % dim_size

            idx1 += coord * self.strides1[i]
            idx2 += coord * self.strides2[i]

        self.position += 1
        return (idx1, idx2)

    fn has_next(self) -> Bool:
        """Check if more elements remain."""
        return self.position < self.size
