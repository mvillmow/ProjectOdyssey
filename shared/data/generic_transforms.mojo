"""Generic data transformation utilities.

This module provides domain-agnostic transformations that work across
modalities (images, tensors, arrays). Includes composition patterns,
utility transforms, batch processing, and type conversions.

Key Features:
- Identity transform (passthrough)
- Lambda transforms (inline functions)
- Conditional transforms (predicate-based application)
- Clamp transforms (value limiting)
- Debug transforms (inspection/logging)
- Sequential composition (chaining transforms)
- Batch transforms (apply to lists)
- Type conversions (Float32, Int32)

Example:
    >>> # Create preprocessing pipeline
    >>> fn scale(x: Float32) -> Float32:
    ...     return x / 255.0
    >>>
    >>> var pipeline = SequentialTransform()
    >>> pipeline.append(LambdaTransform(scale))
    >>> pipeline.append(ClampTransform(0.0, 1.0))
    >>>
    >>> var result = pipeline(data)
"""

from tensor import Tensor
from shared.data.transforms import Transform


# ============================================================================
# Identity Transform
# ============================================================================


@value
struct IdentityTransform(Transform):
    """Identity transform - returns input unchanged.

    Useful as a placeholder or for conditional pipelines where
    no transformation should be applied under certain conditions.

    Time Complexity: O(1) - just returns reference to input.
    Space Complexity: O(1) - no allocation.

    Example:
        >>> var identity = IdentityTransform()
        >>> var result = identity(data)  # result == data
    """

    fn __call__(self, data: Tensor) raises -> Tensor:
        """Apply identity transform (passthrough).

        Args:
            data: Input tensor.

        Returns:
            Input tensor unchanged.
        """
        return data


# ============================================================================
# Lambda Transform
# ============================================================================


@value
struct LambdaTransform(Transform):
    """Apply a function element-wise to tensor values.

    Provides flexible inline transformations without defining
    a full transform struct. The function is applied to each
    element independently.

    Time Complexity: O(n) where n is number of elements.
    Space Complexity: O(n) for output tensor.

    Example:
        >>> fn double(x: Float32) -> Float32:
        ...     return x * 2.0
        >>>
        >>> var transform = LambdaTransform(double)
        >>> var result = transform(data)
    """

    var func: fn (Float32) -> Float32

    fn __init__(out self, func: fn (Float32) -> Float32):
        """Create lambda transform.

        Args:
            func: Function to apply element-wise.
        """
        self.func = func

    fn __call__(self, data: Tensor) raises -> Tensor:
        """Apply function to each element.

        Args:
            data: Input tensor.

        Returns:
            Transformed tensor with function applied to each element.
        """
        var result_values = List[Float32](capacity=data.num_elements())

        for i in range(data.num_elements()):
            var value = data[i]
            var transformed = self.func(value)
            result_values.append(transformed)

        return Tensor(result_values^)


# ============================================================================
# Conditional Transform
# ============================================================================


@value
struct ConditionalTransform(Transform):
    """Apply transform only if predicate is true.

    Evaluates a predicate function on the input tensor. If true,
    applies the transform. If false, returns input unchanged.

    Time Complexity: O(p + t) where p is predicate cost, t is transform cost.
    Space Complexity: O(n) if transform applied, O(1) otherwise.

    Example:
        >>> fn is_large(tensor: Tensor) -> Bool:
        ...     return tensor.num_elements() > 100
        >>>
        >>> var transform = ConditionalTransform(is_large, augment)
        >>> var result = transform(data)  # Only augments large tensors
    """

    var predicate: fn (Tensor) -> Bool
    var transform: Transform

    fn __init__(
        out self,
        predicate: fn (Tensor) -> Bool,
        owned transform: Transform,
    ):
        """Create conditional transform.

        Args:
            predicate: Function to evaluate on tensor.
            transform: Transform to apply if predicate is true.
        """
        self.predicate = predicate
        self.transform = transform^

    fn __call__(self, data: Tensor) raises -> Tensor:
        """Apply transform if predicate is true.

        Args:
            data: Input tensor.

        Returns:
            Transformed tensor if predicate true, otherwise original.
        """
        if self.predicate(data):
            return self.transform(data)
        else:
            return data


# ============================================================================
# Clamp Transform
# ============================================================================


@value
struct ClampTransform(Transform):
    """Clamp tensor values to specified range [min_val, max_val].

    Limits all values to be within the specified range. Values below
    min_val are set to min_val, values above max_val are set to max_val.

    Time Complexity: O(n) where n is number of elements.
    Space Complexity: O(n) for output tensor.

    Example:
        >>> var clamp = ClampTransform(0.0, 1.0)
        >>> var result = clamp(data)  # All values in [0, 1]
    """

    var min_val: Float32
    var max_val: Float32

    fn __init__(out self, min_val: Float32, max_val: Float32):
        """Create clamp transform.

        Args:
            min_val: Minimum allowed value.
            max_val: Maximum allowed value.

        Raises:
            Error if min_val > max_val.
        """
        if min_val > max_val:
            raise Error("min_val must be <= max_val")

        self.min_val = min_val
        self.max_val = max_val

    fn __call__(self, data: Tensor) raises -> Tensor:
        """Clamp all values to [min_val, max_val].

        Args:
            data: Input tensor.

        Returns:
            Tensor with all values clamped to range.
        """
        var result_values = List[Float32](capacity=data.num_elements())

        for i in range(data.num_elements()):
            var value = data[i]

            # Clamp to range
            if value < self.min_val:
                result_values.append(self.min_val)
            elif value > self.max_val:
                result_values.append(self.max_val)
            else:
                result_values.append(value)

        return Tensor(result_values^)


# ============================================================================
# Debug Transform
# ============================================================================


@value
struct DebugTransform(Transform):
    """Debug transform for logging/inspection.

    Prints tensor information (shape, statistics) for debugging
    purposes, then returns the tensor unchanged. Useful for
    inspecting intermediate results in transform pipelines.

    Time Complexity: O(n) for statistics computation.
    Space Complexity: O(1) - no allocation.

    Example:
        >>> var debug = DebugTransform("layer1_output")
        >>> var result = debug(data)  # Prints info, returns data
    """

    var name: String

    fn __init__(out self, name: String):
        """Create debug transform.

        Args:
            name: Name to display in debug output.
        """
        self.name = name

    fn __call__(self, data: Tensor) raises -> Tensor:
        """Print tensor info and return unchanged.

        Args:
            data: Input tensor.

        Returns:
            Input tensor unchanged.
        """
        print("[DEBUG: " + self.name + "]")
        print("  Elements:", data.num_elements())

        # Compute basic statistics if tensor is non-empty
        if data.num_elements() > 0:
            var min_val = data[0]
            var max_val = data[0]
            var sum_val: Float32 = 0.0

            for i in range(data.num_elements()):
                var val = data[i]
                if val < min_val:
                    min_val = val
                if val > max_val:
                    max_val = val
                sum_val += val

            var mean_val = sum_val / Float32(data.num_elements())

            print("  Min:", min_val)
            print("  Max:", max_val)
            print("  Mean:", mean_val)

        return data


# ============================================================================
# Sequential Transform
# ============================================================================


@value
struct SequentialTransform(Transform):
    """Apply transforms sequentially in order.

    Chains multiple transforms together, applying them in sequence.
    The output of each transform becomes the input to the next.

    Time Complexity: O(sum of all transform costs).
    Space Complexity: O(n) for intermediate results.

    Example:
        >>> var transforms = List[Transform]()
        >>> transforms.append(normalize)
        >>> transforms.append(clamp)
        >>>
        >>> var pipeline = SequentialTransform(transforms^)
        >>> var result = pipeline(data)
    """

    var transforms: List[Transform]

    fn __init__(out self, owned transforms: List[Transform]):
        """Create sequential composition.

        Args:
            transforms: List of transforms to apply in order.
        """
        self.transforms = transforms^

    fn __call__(self, data: Tensor) raises -> Tensor:
        """Apply all transforms sequentially.

        Args:
            data: Input tensor.

        Returns:
            Tensor after all transforms applied.
        """
        var result = data

        # Apply each transform in sequence
        for i in range(len(self.transforms)):
            result = self.transforms[i](result)

        return result


# ============================================================================
# Batch Transform
# ============================================================================


@value
struct BatchTransform:
    """Apply transform to a batch of tensors.

    Applies the same transform to each tensor in a list,
    useful for batch processing in data pipelines.

    Time Complexity: O(b * t) where b is batch size, t is transform cost.
    Space Complexity: O(b * n) for output batch.

    Example:
        >>> var batch = List[Tensor]()
        >>> # ... fill batch ...
        >>>
        >>> var transform = BatchTransform(normalize)
        >>> var results = transform(batch)
    """

    var transform: Transform

    fn __init__(out self, owned transform: Transform):
        """Create batch transform.

        Args:
            transform: Transform to apply to each tensor in batch.
        """
        self.transform = transform^

    fn __call__(self, batch: List[Tensor]) raises -> List[Tensor]:
        """Apply transform to each tensor in batch.

        Args:
            batch: List of input tensors.

        Returns:
            List of transformed tensors (same order as input).
        """
        var results = List[Tensor](capacity=len(batch))

        for i in range(len(batch)):
            var transformed = self.transform(batch[i])
            results.append(transformed)

        return results


# ============================================================================
# Type Conversion Transforms
# ============================================================================


@value
struct ToFloat32(Transform):
    """Convert tensor to Float32 dtype.

    Converts all elements to Float32. If already Float32,
    returns a copy. Preserves values exactly for compatible types.

    Time Complexity: O(n) where n is number of elements.
    Space Complexity: O(n) for output tensor.

    Example:
        >>> var converter = ToFloat32()
        >>> var result = converter(int_tensor)
    """

    fn __call__(self, data: Tensor) raises -> Tensor:
        """Convert to Float32.

        Args:
            data: Input tensor.

        Returns:
            Tensor with all values as Float32.
        """
        # Tensor is already Float32 in current implementation
        # Just create a copy with Float32 values
        var result_values = List[Float32](capacity=data.num_elements())

        for i in range(data.num_elements()):
            result_values.append(Float32(data[i]))

        return Tensor(result_values^)


@value
struct ToInt32(Transform):
    """Convert tensor to Int32 dtype (truncation).

    Converts all elements to Int32 by truncating decimal places.
    Positive values round toward zero: 2.9 -> 2.
    Negative values round toward zero: -2.9 -> -2.

    Time Complexity: O(n) where n is number of elements.
    Space Complexity: O(n) for output tensor.

    Example:
        >>> var converter = ToInt32()
        >>> var result = converter(float_tensor)  # Truncates decimals
    """

    fn __call__(self, data: Tensor) raises -> Tensor:
        """Convert to Int32 (truncate).

        Args:
            data: Input tensor.

        Returns:
            Tensor with all values truncated to Int32.

        Note:
            Truncates toward zero: 2.9 -> 2, -2.9 -> -2.
        """
        var result_values = List[Float32](capacity=data.num_elements())

        for i in range(data.num_elements()):
            var value = data[i]
            # Truncate to int and convert back to float for storage
            var int_value = int(value)
            result_values.append(Float32(int_value))

        return Tensor(result_values^)


# ============================================================================
# Helper Functions
# ============================================================================


fn apply_to_tensor(
    data: Tensor, func: fn (Float32) -> Float32
) raises -> Tensor:
    """Apply function element-wise to tensor.

    Helper function for creating ad-hoc transforms without
    defining a transform struct.

    Args:
        data: Input tensor.
        func: Function to apply to each element.

    Returns:
        Transformed tensor.

    Example:
        >>> fn square(x: Float32) -> Float32:
        ...     return x * x
        >>>
        >>> var result = apply_to_tensor(data, square)
    """
    var transform = LambdaTransform(func)
    return transform(data)


fn compose_transforms(owned transforms: List[Transform]) raises -> SequentialTransform:
    """Create sequential composition of transforms.

    Convenience function for building transform pipelines.

    Args:
        transforms: List of transforms to compose.

    Returns:
        SequentialTransform that applies all transforms in order.

    Example:
        >>> var pipeline = compose_transforms(List(norm, clamp, debug))
        >>> var result = pipeline(data)
    """
    return SequentialTransform(transforms^)
