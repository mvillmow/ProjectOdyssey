"""Data transformation and augmentation utilities.

This module provides transformations for preprocessing and augmenting data.
"""

from tensor import Tensor
from math import sqrt, floor, ceil
from random import random_si64


# ============================================================================
# Transform Trait
# ============================================================================


trait Transform:
    """Base interface for all transforms.

    Transforms modify data in-place or return transformed copies.
    """

    fn __call__(self, data: Tensor) raises -> Tensor:
        """Apply the transform to data.

        Args:
            data: Input tensor.

        Returns:
            Transformed tensor.

        Raises:
            Error if transform cannot be applied.
        """
        ...


# ============================================================================
# Compose Transform
# ============================================================================


@value
struct Compose(Transform):
    """Compose multiple transforms sequentially.

    Applies transforms in order, passing output of each to the next.
    """

    var transforms: List[Transform]

    fn __init__(out self, owned transforms: List[Transform]):
        """Create composition of transforms.

        Args:
            transforms: List of transforms to apply in order.
        """
        self.transforms = transforms^

    fn __call__(self, data: Tensor) raises -> Tensor:
        """Apply all transforms sequentially.

        Args:
            data: Input tensor.

        Returns:
            Transformed tensor after all transforms.

        Raises:
            Error if any transform cannot be applied.
        """
        var result = data
        for t in self.transforms:
            result = t[](result)
        return result

    fn __len__(self) -> Int:
        """Return number of transforms."""
        return len(self.transforms)

    fn append(inoutself, transform: Transform):
        """Add a transform to the pipeline.

        Args:
            transform: Transform to add.
        """
        self.transforms.append(transform)


# ============================================================================
# Tensor Transforms
# ============================================================================


@value
struct ToTensor(Transform):
    """Convert data to tensor format.

    Ensures data is in tensor format with appropriate dtype.
    """

    fn __call__(self, data: Tensor) raises -> Tensor:
        """Convert to tensor.

        Args:
            data: Input data.

        Returns:
            Data as tensor.

        Raises:
            Error if conversion fails.
        """
        # Already a tensor, just return
        return data


@value
struct Normalize(Transform):
    """Normalize tensor with mean and standard deviation.

    Applies: (x - mean) / std
    """

    var mean: Float64
    var std: Float64

    fn __init__(out self, mean: Float64 = 0.0, std: Float64 = 1.0):
        """Create normalize transform.

        Args:
            mean: Mean to subtract.
            std: Standard deviation to divide by.
        """
        self.mean = mean
        self.std = std

    fn __call__(self, data: Tensor) raises -> Tensor:
        """Normalize tensor by subtracting mean and dividing by std.

        Applies the formula: (data - mean) / std to all elements.

        Args:
            data: Input tensor.

        Returns:
            Normalized tensor.

        Raises:
            Error if std is zero.
        """
        if self.std == 0.0:
            raise Error("Cannot normalize with std=0")

        # Create a list to hold normalized values
        var normalized = List[Float32](capacity=data.num_elements())

        # Normalize each element: (x - mean) / std
        for i in range(data.num_elements()):
            var value = Float64(data[i])
            var norm_value = (value - self.mean) / self.std
            normalized.append(Float32(norm_value))

        # Create tensor from normalized values
        return Tensor(normalized^)


@value
struct Reshape(Transform):
    """Reshape tensor to target shape.

    Changes tensor dimensions while preserving total elements.
    """

    var target_shape: List[Int]

    fn __init__(out self, owned target_shape: List[Int]):
        """Create reshape transform.

        Args:
            target_shape: Target shape for tensor.
        """
        self.target_shape = target_shape^

    fn __call__(self, data: Tensor) raises -> Tensor:
        """Reshape tensor to target shape.

        Validates that the total number of elements remains the same.

        Args:
            data: Input tensor.

        Returns:
            Reshaped tensor.

        Raises:
            Error if target shape has different number of elements.
        """
        # Calculate total elements in target shape
        var target_elements = 1
        for dim in self.target_shape:
            target_elements *= dim[]

        # Validate element count matches
        if target_elements != data.num_elements():
            raise Error(
                "Cannot reshape tensor with "
                + str(data.num_elements())
                + " elements to shape with "
                + str(target_elements)
                + " elements"
            )

        # Copy all values (reshape is just a view change, data stays the same)
        var values = List[Float32](capacity=data.num_elements())
        for i in range(data.num_elements()):
            values.append(Float32(data[i]))

        # TODO: Properly set shape metadata on returned tensor
        # For now, return flattened tensor (Mojo's Tensor API limitation)
        return Tensor(values^)


# ============================================================================
# Image Transforms
# ============================================================================


@value
struct Resize(Transform):
    """Resize image to target size.

    Resizes spatial dimensions of image tensors.
    """

    var size: Tuple[Int, Int]
    var interpolation: String

    fn __init__(
        out self, size: Tuple[Int, Int], interpolation: String = "bilinear"
    ):
        """Create resize transform.

        Args:
            size: Target (height, width).
            interpolation: Interpolation method.
        """
        self.size = size
        self.interpolation = interpolation

    fn __call__(self, data: Tensor) raises -> Tensor:
        """Resize image tensor using nearest-neighbor sampling.

        This is a simplified implementation for 1D tensors.
        Proper 2D image resizing requires bilinear/bicubic interpolation.

        Args:
            data: Input image tensor.

        Returns:
            Resized image tensor.

        Raises:
            Error if operation fails.
        """
        var old_size = data.num_elements()
        var new_size = self.size[0] * self.size[1]

        # Simplified nearest-neighbor resize for 1D tensors
        var resized = List[Float32](capacity=new_size)

        for i in range(new_size):
            # Map new index to old index using nearest-neighbor
            var old_idx = int((float(i) / float(new_size)) * float(old_size))
            if old_idx >= old_size:
                old_idx = old_size - 1

            resized.append(Float32(data[old_idx]))

        # TODO: Implement proper 2D image resizing with interpolation
        # This requires:
        # 1. Understanding tensor layout (H, W, C) vs (C, H, W)
        # 2. Bilinear or bicubic interpolation for quality
        # 3. Handling edge cases and aspect ratio
        return Tensor(resized^)


@value
struct CenterCrop(Transform):
    """Crop the center of an image.

    Extracts a center crop of specified size.
    """

    var size: Tuple[Int, Int]

    fn __init__(out self, size: Tuple[Int, Int]):
        """Create center crop transform.

        Args:
            size: Target (height, width) of crop.
        """
        self.size = size

    fn __call__(self, data: Tensor) raises -> Tensor:
        """Center crop image to target size.

        For 1D tensors, crops the center portion of specified size.
        For multi-dimensional image tensors, proper 2D cropping is needed.

        Args:
            data: Input image tensor.

        Returns:
            Cropped image tensor.

        Raises:
            Error if crop size exceeds tensor size.
        """
        var num_elements = data.num_elements()
        var crop_size = self.size[0] * self.size[1]  # Total elements to keep

        if crop_size > num_elements:
            raise Error("Crop size exceeds tensor size")

        # For 1D tensor, crop center portion
        var offset = (num_elements - crop_size) // 2
        var cropped = List[Float32](capacity=crop_size)

        for i in range(offset, offset + crop_size):
            cropped.append(Float32(data[i]))

        # TODO: Implement proper 2D center cropping for image tensors
        # This requires understanding tensor layout (H, W, C) and
        # extracting the center rectangle
        return Tensor(cropped^)


@value
struct RandomCrop(Transform):
    """Random crop from an image.

    Extracts a random crop of specified size.
    """

    var size: Tuple[Int, Int]
    var padding: Optional[Int]

    fn __init__(out self, size: Tuple[Int, Int], padding: Optional[Int] = None):
        """Create random crop transform.

        Args:
            size: Target (height, width) of crop.
            padding: Optional padding before cropping.
        """
        self.size = size
        self.padding = padding

    fn __call__(self, data: Tensor) raises -> Tensor:
        """Random crop image to target size.

        For 1D tensors, crops a random portion of specified size.
        For multi-dimensional image tensors, proper 2D random cropping is needed.

        Args:
            data: Input image tensor.

        Returns:
            Randomly cropped image tensor.

        Raises:
            Error if crop size exceeds tensor size.
        """
        var num_elements = data.num_elements()
        var crop_size = self.size[0] * self.size[1]  # Total elements to keep

        if crop_size > num_elements:
            raise Error("Crop size exceeds tensor size")

        # For 1D tensor, crop random portion
        var max_offset = num_elements - crop_size
        var offset = int(random_si64(0, max_offset + 1))

        var cropped = List[Float32](capacity=crop_size)
        for i in range(offset, offset + crop_size):
            cropped.append(Float32(data[i]))

        # TODO: Implement proper 2D random cropping for image tensors
        # This requires understanding tensor layout (H, W, C) and
        # extracting a random rectangle with optional padding
        return Tensor(cropped^)


@value
struct RandomHorizontalFlip(Transform):
    """Randomly flip image horizontally.

    Flips with specified probability.
    """

    var p: Float64

    fn __init__(out self, p: Float64 = 0.5):
        """Create random horizontal flip transform.

        Args:
            p: Probability of flipping.
        """
        self.p = p

    fn __call__(self, data: Tensor) raises -> Tensor:
        """Randomly flip image horizontally with probability p.

        For 1D tensors, reverses the order of elements with probability p.
        For multi-dimensional tensors, this is a simplified implementation.

        Args:
            data: Input image tensor.

        Returns:
            Possibly flipped image tensor.

        Raises:
            Error if operation fails.
        """
        # Generate random number in [0, 1)
        var rand_val = float(random_si64(0, 1000000)) / 1000000.0

        # Don't flip if random value >= probability
        if rand_val >= self.p:
            return data

        # Flip the tensor by reversing element order
        var flipped = List[Float32](capacity=data.num_elements())
        for i in range(data.num_elements() - 1, -1, -1):
            flipped.append(Float32(data[i]))

        # TODO: For proper image flipping, need to reverse only width dimension
        # This simplified implementation reverses all elements
        return Tensor(flipped^)


@value
struct RandomRotation(Transform):
    """Randomly rotate image.

    Rotates within specified degree range.
    """

    var degrees: Tuple[Float64, Float64]
    var fill_value: Float64

    fn __init__(
        out self, degrees: Tuple[Float64, Float64], fill_value: Float64 = 0.0
    ):
        """Create random rotation transform.

        Args:
            degrees: Range of rotation degrees (min, max).
            fill_value: Value to fill empty pixels after rotation.
        """
        self.degrees = degrees
        self.fill_value = fill_value

    fn __call__(self, data: Tensor) raises -> Tensor:
        """Randomly rotate image within specified degree range.

        This is a placeholder implementation that returns the original tensor.
        Proper rotation requires affine transformations and interpolation.

        Args:
            data: Input image tensor.

        Returns:
            Image tensor (currently unrotated - TODO).

        Raises:
            Error if operation fails.
        """
        # Generate random rotation angle in degrees range
        var angle_range = self.degrees[1] - self.degrees[0]
        var rand_val = float(random_si64(0, 1000000)) / 1000000.0
        var angle = self.degrees[0] + (rand_val * angle_range)

        # TODO: Implement proper image rotation
        # This requires:
        # 1. Convert angle to radians
        # 2. Create rotation matrix [cos(θ), -sin(θ); sin(θ), cos(θ)]
        # 3. Apply affine transformation to each pixel coordinate
        # 4. Use interpolation to sample rotated pixels
        # 5. Fill empty regions with fill_value
        #
        # For now, return original tensor unchanged
        # This allows tests to run even though rotation isn't fully implemented
        return data
