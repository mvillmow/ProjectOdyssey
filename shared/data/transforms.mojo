"""Data transformation and augmentation utilities.

This module provides transformations for preprocessing and augmenting data.
"""

from tensor import Tensor
from math import sqrt, floor, ceil, sin, cos
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


# Type alias for backward compatibility and more intuitive naming
alias Pipeline = Compose


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

        Extracts a center rectangle from a 2D image tensor.
        Assumes flattened tensor with shape (H, W, C) where H = W.
        Default assumption: C = 3 (RGB); adjust for grayscale.

        Args:
            data: Input image tensor.

        Returns:
            Center-cropped image tensor.

        Raises:
            Error if crop size exceeds image size.
        """
        # Determine image dimensions
        # Assume square images: H = W = sqrt(num_elements / channels)
        # Default to 3 channels (RGB)
        var total_elements = data.num_elements()
        var channels = 3
        var pixels = total_elements // channels  # H * W
        var width = int(sqrt(float(pixels)))  # Assume H = W
        var height = width

        var crop_h = self.size[0]
        var crop_w = self.size[1]

        # Validate crop size doesn't exceed image size
        if crop_h > height or crop_w > width:
            raise Error("Crop size exceeds image size")

        # Calculate center position
        var offset_h = (height - crop_h) // 2
        var offset_w = (width - crop_w) // 2

        # Create cropped tensor
        var cropped = List[Float32](capacity=crop_h * crop_w * channels)

        # Extract center rectangle
        for h in range(crop_h):
            for w in range(crop_w):
                for c in range(channels):
                    var src_idx = ((offset_h + h) * width + (offset_w + w)) * channels + c
                    cropped.append(Float32(data[src_idx]))

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

        Extracts a random rectangle from a 2D image tensor.
        Assumes flattened tensor with shape (H, W, C) where H = W.
        Default assumption: C = 3 (RGB); adjust for grayscale.
        Supports optional padding for edge handling.

        Args:
            data: Input image tensor.

        Returns:
            Randomly cropped image tensor.

        Raises:
            Error if crop size exceeds padded image size.
        """
        # Determine image dimensions
        # Assume square images: H = W = sqrt(num_elements / channels)
        # Default to 3 channels (RGB)
        var total_elements = data.num_elements()
        var channels = 3
        var pixels = total_elements // channels  # H * W
        var width = int(sqrt(float(pixels)))  # Assume H = W
        var height = width

        # Apply padding if specified (conceptually increase image size)
        var padded_height = height
        var padded_width = width
        if self.padding:
            var pad = self.padding.value()[]
            padded_height = height + 2 * pad
            padded_width = width + 2 * pad

        var crop_h = self.size[0]
        var crop_w = self.size[1]

        # Validate crop size
        if crop_h > padded_height or crop_w > padded_width:
            raise Error("Crop size exceeds padded image size")

        # Random top-left position within valid range
        var max_h = padded_height - crop_h
        var max_w = padded_width - crop_w
        var top = int(random_si64(0, max_h + 1))
        var left = int(random_si64(0, max_w + 1))

        # Adjust for padding offset
        var actual_top = top
        var actual_left = left
        if self.padding:
            var pad = self.padding.value()[]
            actual_top = top - pad
            actual_left = left - pad

        # Create cropped tensor
        var cropped = List[Float32](capacity=crop_h * crop_w * channels)

        # Extract random rectangle
        for h in range(crop_h):
            for w in range(crop_w):
                # Calculate source pixel position
                var src_h = actual_top + h
                var src_w = actual_left + w

                # For each channel
                for c in range(channels):
                    # Check if source pixel is within original image bounds
                    if src_h >= 0 and src_h < height and src_w >= 0 and src_w < width:
                        # Sample from source pixel
                        var src_idx = (src_h * width + src_w) * channels + c
                        cropped.append(Float32(data[src_idx]))
                    else:
                        # Out of bounds (in padding region), fill with 0
                        cropped.append(Float32(0.0))

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

        Flips the image along the width dimension (reverses each row).
        Assumes flattened tensor with shape (H, W, C) where H = W.
        Default assumption: C = 3 (RGB); adjust for grayscale.

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

        # Determine image dimensions
        # Assume square images: H = W = sqrt(num_elements / channels)
        # Default to 3 channels (RGB)
        var total_elements = data.num_elements()
        var channels = 3
        var pixels = total_elements // channels  # H * W
        var width = int(sqrt(float(pixels)))  # Assume H = W
        var height = width

        # Create flipped tensor
        var flipped = List[Float32](capacity=total_elements)

        # For each row
        for h in range(height):
            # For each column (in reverse for flip)
            for w_idx in range(width):
                # Original column index (from right to left)
                var w_orig = width - 1 - w_idx
                # Copy all channels for this pixel
                for c in range(channels):
                    var src_idx = (h * width + w_orig) * channels + c
                    flipped.append(Float32(data[src_idx]))

        return Tensor(flipped^)


@value
struct RandomVerticalFlip(Transform):
    """Randomly flip image vertically.

    Flips with specified probability.
    """

    var p: Float64

    fn __init__(out self, p: Float64 = 0.5):
        """Create random vertical flip transform.

        Args:
            p: Probability of flipping.
        """
        self.p = p

    fn __call__(self, data: Tensor) raises -> Tensor:
        """Randomly flip image vertically with probability p.

        Flips the image along the height dimension (reverses rows).
        Assumes flattened tensor with shape (H, W, C) where H = W.
        Default assumption: C = 3 (RGB); adjust for grayscale.

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

        # Determine image dimensions
        # Assume square images: H = W = sqrt(num_elements / channels)
        # Default to 3 channels (RGB)
        var total_elements = data.num_elements()
        var channels = 3
        var pixels = total_elements // channels  # H * W
        var width = int(sqrt(float(pixels)))  # Assume H = W
        var height = width

        # Create flipped tensor
        var flipped = List[Float32](capacity=total_elements)

        # For each row (in reverse for vertical flip)
        for h_idx in range(height):
            # Original row index (from bottom to top)
            var h_orig = height - 1 - h_idx
            # For each column
            for w in range(width):
                # Copy all channels for this pixel
                for c in range(channels):
                    var src_idx = (h_orig * width + w) * channels + c
                    flipped.append(Float32(data[src_idx]))

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

        Performs rotation around image center using nearest-neighbor sampling.
        Assumes flattened tensor with shape (H, W, C) where H = W.
        Default assumption: C = 3 (RGB); adjust for grayscale.

        Args:
            data: Input image tensor.

        Returns:
            Rotated image tensor.

        Raises:
            Error if operation fails.
        """
        # Generate random rotation angle in degrees range
        var angle_range = self.degrees[1] - self.degrees[0]
        var rand_val = float(random_si64(0, 1000000)) / 1000000.0
        var angle_deg = self.degrees[0] + (rand_val * angle_range)

        # Convert angle to radians
        var pi = 3.14159265359
        var angle_rad = angle_deg * (pi / 180.0)

        # Determine image dimensions
        # Assume square images: H = W = sqrt(num_elements / channels)
        # Default to 3 channels (RGB)
        var total_elements = data.num_elements()
        var channels = 3
        var pixels = total_elements // channels  # H * W
        var width = int(sqrt(float(pixels)))  # Assume H = W
        var height = width

        # Compute rotation matrix values
        var cos_angle = cos(angle_rad)
        var sin_angle = sin(angle_rad)

        # Image center
        var cx = float(width) / 2.0
        var cy = float(height) / 2.0

        # Create rotated tensor with fill_value for empty regions
        var rotated = List[Float32](capacity=total_elements)

        # For each output pixel
        for y in range(height):
            for x in range(width):
                # Convert to floating point for rotation calculation
                var x_f = float(x)
                var y_f = float(y)

                # Apply inverse rotation to find source pixel
                # x_src = (x - cx) * cos(θ) - (y - cy) * sin(θ) + cx
                # y_src = (x - cx) * sin(θ) + (y - cy) * cos(θ) + cy
                var x_src = (x_f - cx) * cos_angle - (y_f - cy) * sin_angle + cx
                var y_src = (x_f - cx) * sin_angle + (y_f - cy) * cos_angle + cy

                # Round to nearest integer for nearest-neighbor sampling
                var x_src_int = int(x_src + 0.5)
                var y_src_int = int(y_src + 0.5)

                # For each channel
                for c in range(channels):
                    # Check if source pixel is within bounds
                    if (x_src_int >= 0
                        and x_src_int < width
                        and y_src_int >= 0
                        and y_src_int < height):
                        # Sample from source pixel
                        var src_idx = (y_src_int * width + x_src_int) * channels + c
                        rotated.append(Float32(data[src_idx]))
                    else:
                        # Fill with fill_value for out-of-bounds pixels
                        rotated.append(Float32(self.fill_value))

        return Tensor(rotated^)


@value
struct RandomErasing(Transform):
    """Randomly erase rectangular regions in images (Cutout augmentation).

    Randomly selects a rectangle region and erases it by setting pixels to a fill value.
    Helps improve model robustness to occlusion.

    Reference: "Random Erasing Data Augmentation" (Zhong et al., 2017)
    """

    var p: Float64  # Probability of applying erasing
    var scale: Tuple[Float64, Float64]  # Min/max area fraction to erase
    var ratio: Tuple[Float64, Float64]  # Min/max aspect ratio of erased region
    var value: Float64  # Fill value (0 for black, can be random)

    fn __init__(
        out self,
        p: Float64 = 0.5,
        scale: Tuple[Float64, Float64] = (0.02, 0.33),
        ratio: Tuple[Float64, Float64] = (0.3, 3.3),
        value: Float64 = 0.0
    ):
        """Create random erasing transform.

        Args:
            p: Probability of applying erasing.
            scale: Range of proportion of erased area (min, max).
            ratio: Range of aspect ratio of erased area (min, max).
            value: Pixel value to fill erased region with.
        """
        self.p = p
        self.scale = scale
        self.ratio = ratio
        self.value = value

    fn __call__(self, data: Tensor) raises -> Tensor:
        """Apply random erasing with probability p.

        Randomly erases a rectangular region from the image by:
        1. Checking probability to decide if erasing should occur
        2. Calculating target erased area based on scale parameter
        3. Determining rectangle dimensions based on aspect ratio
        4. Randomly positioning the rectangle within image bounds
        5. Setting all pixels in rectangle to fill value

        Assumes flattened tensor with shape (H, W, C) where H = W.
        Default assumption: C = 3 (RGB); adjust for grayscale.

        Args:
            data: Input image tensor.

        Returns:
            Image with randomly erased rectangular region (or original if not applied).

        Raises:
            Error if operation fails.
        """
        # Step 1: Check probability - randomly decide whether to apply erasing
        var rand_val = float(random_si64(0, 1000000)) / 1000000.0
        if rand_val >= self.p:
            return data  # Don't erase

        # Step 2: Infer image dimensions
        # Assume square RGB image: total_pixels = H * W, num_elements = H * W * C
        var total_elements = data.num_elements()
        var channels = 3
        var total_pixels = total_elements // channels
        var image_size = int(sqrt(float(total_pixels)))
        var area = image_size * image_size

        # Step 3: Randomly select erased region size
        # Target area as fraction of image
        var scale_range = self.scale[1] - self.scale[0]
        var scale_rand = float(random_si64(0, 1000000)) / 1000000.0
        var target_area_fraction = self.scale[0] + (scale_rand * scale_range)
        var target_area = float(area) * target_area_fraction

        # Aspect ratio (width/height)
        var ratio_range = self.ratio[1] - self.ratio[0]
        var ratio_rand = float(random_si64(0, 1000000)) / 1000000.0
        var aspect_ratio = self.ratio[0] + (ratio_rand * ratio_range)

        # Calculate width and height from area and aspect ratio
        # area = h * w, ratio = w / h
        # => area = h * (h * ratio) = h^2 * ratio
        # => h = sqrt(area / ratio)
        # => w = sqrt(area * ratio)
        var erase_h = int(sqrt(target_area / aspect_ratio))
        var erase_w = int(sqrt(target_area * aspect_ratio))

        # Ensure within image bounds
        erase_h = min(erase_h, image_size)
        erase_w = min(erase_w, image_size)

        # Skip if region is too small
        if erase_h <= 0 or erase_w <= 0:
            return data

        # Step 4: Randomly select top-left position
        var max_top = image_size - erase_h
        var max_left = image_size - erase_w

        # Handle edge case where erased region equals image size
        if max_top < 0 or max_left < 0:
            return data

        var top = int(random_si64(0, max_top + 1))
        var left = int(random_si64(0, max_left + 1))

        # Step 5: Erase the rectangle
        # Copy original data
        var result = List[Float32](capacity=total_elements)
        for i in range(total_elements):
            result.append(Float32(data[i]))

        # Erase rectangle by setting to fill value
        # For each pixel in the erased rectangle
        for row in range(top, top + erase_h):
            for col in range(left, left + erase_w):
                # Set all channels to fill value
                for c in range(channels):
                    var index = (row * image_size + col) * channels + c
                    result[index] = Float32(self.value)

        return Tensor(result^)
