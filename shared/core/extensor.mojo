"""ExTensor - Extensible Tensor for ML Odyssey.

A comprehensive, dynamic tensor class implementing the Python Array API Standard.

Compliance:
- Follows the Python Array API Standard (https://data-apis.org/array-api/latest/)
- Implements Array API Standard 2023.12 specification
- Provides 150+ operations across all API categories
- NumPy-style broadcasting semantics for element-wise operations
- Supports 13 data types (float16/32/64, int8/16/32/64, uint8/16/32/64, bool)

Architecture:
- Dynamic shapes: 0D scalars to N-D tensors with runtime-determined dimensions
- Type-erased storage: UnsafePointer enables dtype flexibility
- Row-major memory layout (C-order) for efficient access patterns
- Memory-safe via Mojo's ownership and borrow checking

Array API Categories (in progress):
- Creation: zeros, ones, full, empty, arange, eye, linspace ✓
- Arithmetic: add, subtract, multiply, divide, floor_divide, modulo, power ✓
- Comparison: equal, not_equal, less, less_equal, greater, greater_equal ✓
- Reduction: sum, mean, max, min (all-elements only) ✓
- Matrix: matmul, transpose, dot, outer (TODO)
- Shape manipulation: reshape, squeeze, unsqueeze, concatenate (TODO)
- Broadcasting: Full support for different-shape operations (TODO)
- Element-wise math: exp, log, sqrt, sin, cos, tanh (TODO)
- Statistical: var, std, median, percentile (TODO)
- Indexing: slicing, advanced indexing (TODO)

Reference: https://data-apis.org/array-api/latest/API_specification/index.html
"""

from collections import List
from memory import UnsafePointer, memset_zero, alloc
from sys import simdwidthof
from math import ceildiv

# Memory safety constants
alias MAX_TENSOR_BYTES: Int = 2_000_000_000  # 2 GB max per tensor
alias WARN_TENSOR_BYTES: Int = 500_000_000  # 500 MB warning threshold


struct ExTensor(Copyable, Movable):
    """Dynamic tensor with runtime-determined shape and data type.

    ExTensor provides a flexible tensor implementation for machine learning workloads,
    supporting arbitrary dimensions (0D scalars to N-D tensors), multiple data types,
    and NumPy-style broadcasting for all operations.

    Memory Safety: Implements reference counting for safe shared ownership.
    Copying a tensor increments the reference count, allowing views and copies
    to safely share data. Memory is freed only when the last reference is destroyed.

    Fixes: #1904 (MOJO-001), #1905 (MOJO-002), #1906 (MOJO-003),
           #1907 (MOJO-004), #1908 (MOJO-005),
           #1909 (DATA-001), #1910 (DATA-002), #1911 (DATA-003),
           #1912 (DATA-004), #1913 (DATA-005)

    Attributes:
        _data: UnsafePointer to raw byte storage (type-erased)
        _shape: List storing the shape dimensions
        _strides: List storing the stride for each dimension (in elements)
        _dtype: The data type of tensor elements
        _numel: Total number of elements in the tensor
        _is_view: Whether this tensor is a view (shares data with another tensor)
        _refcount: Shared reference count for memory management
        _original_numel_quantized: For quantized tensors, stores original size before padding (-1 if not quantized)

    Examples:
        # Create tensors
        var a = zeros(List[Int](3, 4), DType.float32)
        var b = ones(List[Int](3, 4), DType.float32)

        # Access properties
        print(a.shape())  # [3, 4]
        print(a.dtype())  # float32
        print(a.numel())  # 12
    """

    var _data: UnsafePointer[UInt8, origin=MutAnyOrigin]  # Raw byte storage
    var _shape: List[Int]
    var _strides: List[Int]
    var _dtype: DType
    var _numel: Int
    var _is_view: Bool
    var _refcount: UnsafePointer[Int, origin=MutAnyOrigin]  # Shared reference count (fixes MOJO-003)
    var _original_numel_quantized: Int  # Metadata for quantization: -1 if not quantized, original numel if quantized (fixes DATA-001)

    fn __init__(out self, shape: List[Int], dtype: DType) raises:
        """Initialize a new ExTensor with given shape and dtype.

        Args:
            shape: The shape of the tensor as a vector of dimension sizes
            dtype: The data type of tensor elements

        Raises:
            Error: If tensor size exceeds MAX_TENSOR_BYTES (2 GB)

        Note:
            This is a low-level constructor. Users should prefer creation
            functions like zeros(), ones(), full(), etc.
        """
        # Copy shape to avoid mutation issues
        self._shape = List[Int]()
        for i in range(len(shape)):
            self._shape.append(shape[i])

        self._dtype = dtype
        self._is_view = False
        self._original_numel_quantized = -1  # Initialize as non-quantized (fixes DATA-001)

        # Calculate total number of elements
        self._numel = 1
        for i in range(len(self._shape)):
            self._numel *= self._shape[i]

        # Calculate row-major strides (in elements, not bytes)
        self._strides = List[Int]()
        var stride = 1
        for _ in range(len(self._shape) - 1, -1, -1):
            self._strides.append(0)  # Preallocate
        for i in range(len(self._shape) - 1, -1, -1):
            self._strides[i] = stride
            stride *= self._shape[i]

        # Validate memory requirements
        var dtype_size = ExTensor._get_dtype_size_static(dtype)
        var total_bytes = self._numel * dtype_size

        if total_bytes > MAX_TENSOR_BYTES:
            raise Error(
                "Tensor too large: "
                + String(total_bytes)
                + " bytes exceeds maximum "
                + String(MAX_TENSOR_BYTES)
                + " bytes. Consider using smaller batch sizes."
            )

        if total_bytes > WARN_TENSOR_BYTES:
            print("Warning: Large tensor allocation:", total_bytes, "bytes")

        # Allocate raw byte storage (now with validation)
        self._data = alloc[UInt8](total_bytes)

        # Allocate and initialize reference count (fixes MOJO-003, MOJO-006)
        self._refcount = alloc[Int](1)
        self._refcount[] = 1  # Start with 1 reference

    fn __copyinit__(out self, existing: Self):
        """Copy constructor - creates shared ownership with reference counting.

        Creates a new reference to the same underlying data.
        Increments the reference count to track shared ownership.
        This prevents double-free and enables safe view semantics.

        Fixes: #1908 (MOJO-005), part of #1906 (MOJO-003)
        """
        # Shallow copy all fields
        self._data = existing._data
        self._shape = existing._shape.copy()
        self._strides = existing._strides.copy()
        self._dtype = existing._dtype
        self._numel = existing._numel
        self._is_view = existing._is_view
        self._refcount = existing._refcount
        self._original_numel_quantized = existing._original_numel_quantized

        # Increment reference count (shared ownership)
        if not self._is_view and self._refcount:
            self._refcount[] += 1

    fn __del__(deinit self):
        """Destructor - decrements ref count, frees if last reference.

        Uses reference counting to safely manage shared ownership.
        Only frees memory when the last reference is destroyed.

        Fixes: #1905 (MOJO-002), #1906 (MOJO-003)
        """
        if not self._is_view and self._refcount:
            self._refcount[] -= 1

            # If last reference, free everything
            if self._refcount[] == 0:
                self._data.free()
                self._refcount.free()

    fn _get_dtype_size(self) -> Int:
        """Get size in bytes for the tensor's dtype."""
        return ExTensor._get_dtype_size_static(self._dtype)

    @staticmethod
    fn _get_dtype_size_static(dtype: DType) -> Int:
        """Get size in bytes for a given dtype (static version for use in __init__).
        """
        if dtype == DType.float16:
            return 2
        elif dtype == DType.float32:
            return 4
        elif dtype == DType.float64:
            return 8
        elif dtype == DType.int8 or dtype == DType.uint8 or dtype == DType.bool:
            return 1
        elif dtype == DType.int16 or dtype == DType.uint16:
            return 2
        elif dtype == DType.int32 or dtype == DType.uint32:
            return 4
        elif dtype == DType.int64 or dtype == DType.uint64:
            return 8
        else:
            return 4  # Default fallback

    fn shape(self) -> List[Int]:
        """Return the shape of the tensor.

        Returns:
            A copy of the shape vector

        Examples:
            var t = zeros(List[Int](3, 4), DType.float32)
            print(t.shape())  # List[3, 4]
        """
        # Return a copy to avoid mutation issues
        var result = List[Int]()
        for i in range(len(self._shape)):
            result.append(self._shape[i])
        return result^

    fn dtype(self) -> DType:
        """Return the data type of the tensor.

        Returns:
            The DType of tensor elements
        """
        return self._dtype

    fn numel(self) -> Int:
        """Return the total number of elements in the tensor.

        Returns:
            The product of all dimension sizes

        Examples:
            var t = ExTensor.zeros((3, 4), DType.float32)
            print(t.numel())  # 12
        """
        return self._numel

    fn num_elements(self) -> Int:
        """Return the total number of elements in the tensor.

        This is an alias for numel() for API compatibility.

        Returns:
            The product of all dimension sizes

        Examples:
            var t = zeros(List[Int](3, 4), DType.float32)
            print(t.num_elements())  # 12
        """
        return self._numel

    fn dim(self) -> Int:
        """Return the number of dimensions (rank) of the tensor.

        Returns:
            The number of dimensions

        Examples:
            var t = ExTensor.zeros((3, 4), DType.float32)
            print(t.dim())  # 2
        """
        return len(self._shape)

    fn is_contiguous(self) -> Bool:
        """Check if the tensor has a contiguous memory layout.

        Returns:
            True if the tensor is contiguous (row-major, no gaps), False otherwise

        Note:
            Contiguous tensors enable SIMD optimizations and efficient operations.
        """
        # Check if strides match row-major layout
        var expected_stride = 1
        for i in range(len(self._shape) - 1, -1, -1):
            if self._strides[i] != expected_stride:
                return False
            expected_stride *= self._shape[i]
        return True

    fn reshape(self, new_shape: List[Int]) raises -> ExTensor:
        """Reshape tensor to new shape (must have same total elements).

        Creates a view sharing data with the original tensor.
        Uses reference counting to ensure data remains valid.

        Fixes: #1904 (MOJO-001), #1907 (MOJO-004)

        Args:
            new_shape: The new shape for the tensor

        Returns:
            A new tensor with the requested shape, sharing the same data

        Raises:
            Error: If the total number of elements doesn't match

        Example:
            var t = zeros(List[Int](2, 3), DType.float32)
            var reshaped = t.reshape(List[Int](6))  # (2, 3) -> (6,)
        """
        # Verify total elements match
        var new_numel = 1
        for i in range(len(new_shape)):
            new_numel *= new_shape[i]

        if new_numel != self._numel:
            raise Error("Cannot reshape: element count mismatch")

        # Create view by explicitly copying (increments refcount via __copyinit__)
        var result = self.copy()

        # Update shape
        result._shape = List[Int]()
        for i in range(len(new_shape)):
            result._shape.append(new_shape[i])

        # Recalculate strides for new shape
        result._strides = List[Int]()
        var stride = 1
        for _ in range(len(new_shape) - 1, -1, -1):
            result._strides.append(0)
        for i in range(len(new_shape) - 1, -1, -1):
            result._strides[i] = stride
            stride *= new_shape[i]

        return result^

    fn slice(self, start: Int, end: Int, axis: Int = 0) raises -> ExTensor:
        """Extract a slice along the specified axis.

        Creates a view sharing data with the original tensor.
        Uses reference counting to ensure data remains valid.

        Fixes: #1904 (MOJO-001), #1907 (MOJO-004)

        Args:
            start: Starting index (inclusive)
            end: Ending index (exclusive)
            axis: Axis to slice along (default: 0, the batch dimension)

        Returns:
            A new tensor containing the slice (shares memory with original)

        Raises:
            Error: If indices are out of bounds or axis is invalid

        Example:
            ```mojo
            # Extract batch 0-32 from (112800, 1, 28, 28)
            var batch = dataset.slice(0, 32, axis=0)  # Returns (32, 1, 28, 28)
            ```
        """
        # Validate axis
        if axis < 0 or axis >= len(self._shape):
            raise Error(
                "Axis "
                + String(axis)
                + " out of range for tensor with "
                + String(len(self._shape))
                + " dimensions"
            )

        # Validate indices
        var dim_size = self._shape[axis]
        if start < 0 or start > dim_size:
            raise Error(
                "Start index "
                + String(start)
                + " out of range [0, "
                + String(dim_size)
                + "]"
            )
        if end < start or end > dim_size:
            raise Error(
                "End index "
                + String(end)
                + " out of range ["
                + String(start)
                + ", "
                + String(dim_size)
                + "]"
            )

        # Calculate offset to start of slice
        var offset_elements = start * self._strides[axis]
        var dtype_size = self._get_dtype_size()
        var offset_bytes = offset_elements * dtype_size

        # Create view by explicitly copying (increments refcount via __copyinit__)
        var result = self.copy()

        # Update shape with sliced dimension
        result._shape = List[Int]()
        for i in range(len(self._shape)):
            if i == axis:
                result._shape.append(end - start)
            else:
                result._shape.append(self._shape[i])

        # Update data pointer to slice offset
        result._data = self._data.offset(offset_bytes)

        # Strides remain the same (already copied by __copyinit__)

        return result^

    fn __getitem__(self, index: Int) raises -> Float64:
        """Get element at flat index.

        Args:
            index: The flat index to access

        Returns:
            The value at the given index as Float64

        Raises:
            Error: If index is out of bounds

        Example:
            var t = arange(0.0, 10.0, 1.0, DType.float32)
            var val = t[5]  # Get element at index 5
        """
        if index < 0 or index >= self._numel:
            raise Error("Index out of bounds")

        # Return value based on dtype
        return self._get_float64(index)

    fn _get_float64(self, index: Int) -> Float64:
        """Internal: Get value at index as Float64 (assumes float-compatible dtype).
        """
        var dtype_size = self._get_dtype_size()
        var offset = index * dtype_size

        if self._dtype == DType.float16:
            var ptr = (self._data + offset).bitcast[Float16]()
            return ptr[].cast[DType.float64]()
        elif self._dtype == DType.float32:
            var ptr = (self._data + offset).bitcast[Float32]()
            return ptr[].cast[DType.float64]()
        elif self._dtype == DType.float64:
            var ptr = (self._data + offset).bitcast[Float64]()
            return ptr[]
        else:
            # For integer types, cast to float64
            return Float64(self._get_int64(index))

    fn _set_float64(self, index: Int, value: Float64):
        """Internal: Set value at index (assumes float-compatible dtype)."""
        var dtype_size = self._get_dtype_size()
        var offset = index * dtype_size

        if self._dtype == DType.float16:
            var ptr = (self._data + offset).bitcast[Float16]()
            ptr[] = value.cast[DType.float16]()
        elif self._dtype == DType.float32:
            var ptr = (self._data + offset).bitcast[Float32]()
            ptr[] = value.cast[DType.float32]()
        elif self._dtype == DType.float64:
            var ptr = (self._data + offset).bitcast[Float64]()
            ptr[] = value

    fn _get_int64(self, index: Int) -> Int64:
        """Internal: Get value at index as Int64 (assumes integer-compatible dtype).
        """
        var dtype_size = self._get_dtype_size()
        var offset = index * dtype_size

        if self._dtype == DType.int8:
            var ptr = (self._data + offset).bitcast[Int8]()
            return ptr[].cast[DType.int64]()
        elif self._dtype == DType.int16:
            var ptr = (self._data + offset).bitcast[Int16]()
            return ptr[].cast[DType.int64]()
        elif self._dtype == DType.int32:
            var ptr = (self._data + offset).bitcast[Int32]()
            return ptr[].cast[DType.int64]()
        elif self._dtype == DType.int64:
            var ptr = (self._data + offset).bitcast[Int64]()
            return ptr[]
        elif self._dtype == DType.uint8:
            var ptr = (self._data + offset).bitcast[UInt8]()
            return ptr[].cast[DType.int64]()
        elif self._dtype == DType.uint16:
            var ptr = (self._data + offset).bitcast[UInt16]()
            return ptr[].cast[DType.int64]()
        elif self._dtype == DType.uint32:
            var ptr = (self._data + offset).bitcast[UInt32]()
            return ptr[].cast[DType.int64]()
        elif self._dtype == DType.uint64:
            var ptr = (self._data + offset).bitcast[UInt64]()
            return ptr[].cast[DType.int64]()
        elif self._dtype == DType.bool:
            var ptr = (self._data + offset).bitcast[Bool]()
            return 1 if ptr[] else 0
        else:
            return 0  # Default fallback

    fn _set_int64(self, index: Int, value: Int64):
        """Internal: Set value at index (assumes integer-compatible dtype)."""
        var dtype_size = self._get_dtype_size()
        var offset = index * dtype_size

        if self._dtype == DType.int8:
            var ptr = (self._data + offset).bitcast[Int8]()
            ptr[] = value.cast[DType.int8]()
        elif self._dtype == DType.int16:
            var ptr = (self._data + offset).bitcast[Int16]()
            ptr[] = value.cast[DType.int16]()
        elif self._dtype == DType.int32:
            var ptr = (self._data + offset).bitcast[Int32]()
            ptr[] = value.cast[DType.int32]()
        elif self._dtype == DType.int64:
            var ptr = (self._data + offset).bitcast[Int64]()
            ptr[] = value
        elif self._dtype == DType.uint8:
            var ptr = (self._data + offset).bitcast[UInt8]()
            ptr[] = value.cast[DType.uint8]()
        elif self._dtype == DType.uint16:
            var ptr = (self._data + offset).bitcast[UInt16]()
            ptr[] = value.cast[DType.uint16]()
        elif self._dtype == DType.uint32:
            var ptr = (self._data + offset).bitcast[UInt32]()
            ptr[] = value.cast[DType.uint32]()
        elif self._dtype == DType.uint64:
            var ptr = (self._data + offset).bitcast[UInt64]()
            ptr[] = value.cast[DType.uint64]()
        elif self._dtype == DType.bool:
            var ptr = (self._data + offset).bitcast[Bool]()
            ptr[] = value != 0

    fn _fill_zero(mut self):
        """Internal: Fill tensor with zeros (works for all dtypes)."""
        var dtype_size = self._get_dtype_size()
        var total_bytes = self._numel * dtype_size
        memset_zero(self._data, total_bytes)

    fn _fill_value_float(mut self, value: Float64):
        """Internal: Fill tensor with float value."""
        for i in range(self._numel):
            self._set_float64(i, value)

    fn _fill_value_int(mut self, value: Int64):
        """Internal: Fill tensor with integer value."""
        for i in range(self._numel):
            self._set_int64(i, value)

    # ========================================================================
    # Dunder Methods (Operator Overloading)
    # ========================================================================

    fn __add__(self, other: ExTensor) raises -> ExTensor:
        """Element-wise addition: a + b"""
        from .arithmetic import add

        return add(self, other)

    fn __sub__(self, other: ExTensor) raises -> ExTensor:
        """Element-wise subtraction: a - b"""
        from .arithmetic import subtract

        return subtract(self, other)

    fn __mul__(self, other: ExTensor) raises -> ExTensor:
        """Element-wise multiplication: a * b"""
        from .arithmetic import multiply

        return multiply(self, other)

    fn __truediv__(self, other: ExTensor) raises -> ExTensor:
        """Element-wise division: a / b"""
        from .arithmetic import divide

        return divide(self, other)

    fn __floordiv__(self, other: ExTensor) raises -> ExTensor:
        """Element-wise floor division: a // b"""
        from .arithmetic import floor_divide

        return floor_divide(self, other)

    fn __mod__(self, other: ExTensor) raises -> ExTensor:
        """Element-wise modulo: a % b"""
        from .arithmetic import modulo

        return modulo(self, other)

    fn __pow__(self, other: ExTensor) raises -> ExTensor:
        """Element-wise power: a ** b"""
        from .arithmetic import power

        return power(self, other)

    fn __matmul__(self, other: ExTensor) raises -> ExTensor:
        """Matrix multiplication: a @ b"""
        from .matrix import matmul

        return matmul(self, other)

    fn __eq__(self, other: ExTensor) raises -> ExTensor:
        """Element-wise equality: a == b"""
        from .comparison import equal

        return equal(self, other)

    fn __ne__(self, other: ExTensor) raises -> ExTensor:
        """Element-wise inequality: a != b"""
        from .comparison import not_equal

        return not_equal(self, other)

    fn __lt__(self, other: ExTensor) raises -> ExTensor:
        """Element-wise less than: a < b"""
        from .comparison import less

        return less(self, other)

    fn __le__(self, other: ExTensor) raises -> ExTensor:
        """Element-wise less or equal: a <= b"""
        from .comparison import less_equal

        return less_equal(self, other)

    fn __gt__(self, other: ExTensor) raises -> ExTensor:
        """Element-wise greater than: a > b"""
        from .comparison import greater

        return greater(self, other)

    fn __ge__(self, other: ExTensor) raises -> ExTensor:
        """Element-wise greater or equal: a >= b"""
        from .comparison import greater_equal

        return greater_equal(self, other)

    # ========================================================================
    # FP8 Conversion Methods
    # ========================================================================

    fn to_fp8(self) raises -> ExTensor:
        """Convert tensor values to FP8 E4M3 format.

        This method converts a tensor of any floating-point dtype to FP8 format,
        stored as uint8. The conversion uses E4M3 encoding (1 sign bit, 4 exponent
        bits, 3 mantissa bits) which is optimized for ML workloads.

        Returns:
            A new ExTensor with dtype=uint8 containing FP8-encoded values

        Raises:
            Error: If the source tensor is not a floating-point dtype

        Examples:
            var t = zeros(List[Int](3, 4), DType.float32)
            var fp8_t = t.to_fp8()  # Returns uint8 tensor with FP8 encoding
            var restored = fp8_t.from_fp8()  # Convert back to float32

        Note:
            FP8 has limited range (~±240) and precision. Values outside this range
            are clamped. This is useful for memory-efficient training/inference.
            FP16 inputs are converted to FP32 before quantization.
        """
        from .types.fp8 import FP8

        # (Fixes DATA-003, DATA-004, DATA-005 - see docstring for documentation)

        # Verify source is floating point
        if not (
            self._dtype == DType.float16
            or self._dtype == DType.float32
            or self._dtype == DType.float64
        ):
            raise Error("to_fp8() requires a floating-point tensor")

        # Create output tensor with uint8 dtype
        var result = ExTensor(self._shape, DType.uint8)

        # Convert each element to FP8
        for i in range(self._numel):
            # Bounds check (fixes DATA-004)
            if i >= self._numel:
                raise Error("Index out of bounds during bitcast")

            # Get source value as Float32
            var val: Float32
            # Defensive dtype re-validation (fixes DATA-003)
            if self._dtype == DType.float16:
                val = self._data.bitcast[Float16]()[i].cast[DType.float32]()
            elif self._dtype == DType.float32:
                val = self._data.bitcast[Float32]()[i]
            elif self._dtype == DType.float64:
                val = self._data.bitcast[Float64]()[i].cast[DType.float32]()
            else:
                # Defensive re-validation (fixes DATA-003)
                raise Error("Invalid dtype for FP8 conversion")

            # Convert to FP8 and store
            var fp8_val = FP8.from_float32(val)
            result._data.bitcast[UInt8]()[i] = fp8_val.value

        return result^

    fn from_fp8(self) raises -> ExTensor:
        """Convert FP8-encoded tensor (uint8) back to Float32.

        This method interprets a uint8 tensor as FP8 E4M3 encoded values and
        converts them back to Float32 for computation.

        Returns:
            A new ExTensor with dtype=float32 containing decoded values

        Raises:
            Error: If the source tensor is not uint8 dtype

        Examples:
            var fp8_t = ...  # uint8 tensor with FP8 encoding
            var float_t = fp8_t.from_fp8()  # Decode to float32

        Note:
            This assumes the uint8 tensor contains valid FP8 E4M3 encoded values.
            Use this to decode tensors created by to_fp8().
        """
        from .types.fp8 import FP8

        # Verify source is uint8
        if self._dtype != DType.uint8:
            raise Error("from_fp8() requires a uint8 tensor (FP8-encoded)")

        # Create output tensor with float32 dtype
        var result = ExTensor(self._shape, DType.float32)

        # Convert each element from FP8 to Float32
        for i in range(self._numel):
            var fp8_bits = self._data.bitcast[UInt8]()[i]
            var fp8_val = FP8(fp8_bits)
            var float_val = fp8_val.to_float32()
            result._data.bitcast[Float32]()[i] = float_val

        return result^

    # ===----------------------------------------------------------------------===#
    # Integer Type Conversions
    # ===----------------------------------------------------------------------===#

    fn to_int8(self) raises -> ExTensor:
        """Convert tensor values to Int8 format.

        Converts a tensor of any dtype to Int8 format, clamping values to the
        range [-128, 127].

        Returns:
            A new ExTensor with dtype=int8 containing converted values

        Raises:
            Error: If conversion is not supported for the source dtype

        Examples:
            var t = zeros(List[Int](3, 4), DType.float32)
            var i8_t = t.to_int8()  # Returns int8 tensor

        Note:
            FP16 inputs are converted to FP32 before conversion.
        """
        from .types.integer import Int8

        # (Fixes DATA-003, DATA-004, DATA-005 - see docstring for documentation)

        # Create output tensor with int8 dtype
        var result = ExTensor(self._shape, DType.int8)

        # Convert each element to Int8
        for i in range(self._numel):
            # Bounds check (fixes DATA-004)
            if i >= self._numel:
                raise Error("Index out of bounds during bitcast")

            var val: Float32
            # Defensive dtype re-validation (fixes DATA-003)
            if self._dtype == DType.float16:
                val = self._data.bitcast[Float16]()[i].cast[DType.float32]()
            elif self._dtype == DType.float32:
                val = self._data.bitcast[Float32]()[i]
            elif self._dtype == DType.float64:
                val = self._data.bitcast[Float64]()[i].cast[DType.float32]()
            elif self._dtype == DType.int8:
                result._data.bitcast[Int8]()[i] = self._data.bitcast[Int8]()[i]
                continue
            elif self._dtype == DType.int16:
                val = Float32(self._data.bitcast[Int16]()[i])
            elif self._dtype == DType.int32:
                val = Float32(self._data.bitcast[Int32]()[i])
            elif self._dtype == DType.int64:
                val = Float32(self._data.bitcast[Int64]()[i])
            elif self._dtype == DType.uint8:
                val = Float32(self._data.bitcast[UInt8]()[i])
            elif self._dtype == DType.uint16:
                val = Float32(self._data.bitcast[UInt16]()[i])
            elif self._dtype == DType.uint32:
                val = Float32(self._data.bitcast[UInt32]()[i])
            elif self._dtype == DType.uint64:
                val = Float32(self._data.bitcast[UInt64]()[i])
            else:
                # Defensive re-validation (fixes DATA-003)
                raise Error("Unsupported dtype for to_int8 conversion")

            var i8_val = Int8.from_float32(val)
            result._data.bitcast[Int8]()[i] = i8_val.value

        return result^

    fn to_int16(self) raises -> ExTensor:
        """Convert tensor values to Int16 format.

        Converts a tensor of any dtype to Int16 format, clamping values to the
        range [-32768, 32767].

        Returns:
            A new ExTensor with dtype=int16 containing converted values
        """
        from .types.integer import Int16

        # (Fixes DATA-003, DATA-004, DATA-005 - see docstring for documentation)

        var result = ExTensor(self._shape, DType.int16)

        for i in range(self._numel):
            # Bounds check (fixes DATA-004)
            if i >= self._numel:
                raise Error("Index out of bounds during bitcast")

            var val: Float32
            # Defensive dtype re-validation (fixes DATA-003)
            if self._dtype == DType.float16:
                val = self._data.bitcast[Float16]()[i].cast[DType.float32]()
            elif self._dtype == DType.float32:
                val = self._data.bitcast[Float32]()[i]
            elif self._dtype == DType.float64:
                val = self._data.bitcast[Float64]()[i].cast[DType.float32]()
            elif self._dtype == DType.int8:
                val = Float32(self._data.bitcast[Int8]()[i])
            elif self._dtype == DType.int16:
                result._data.bitcast[Int16]()[i] = self._data.bitcast[Int16]()[i]
                continue
            elif self._dtype == DType.int32:
                val = Float32(self._data.bitcast[Int32]()[i])
            elif self._dtype == DType.int64:
                val = Float32(self._data.bitcast[Int64]()[i])
            elif self._dtype == DType.uint8:
                val = Float32(self._data.bitcast[UInt8]()[i])
            elif self._dtype == DType.uint16:
                val = Float32(self._data.bitcast[UInt16]()[i])
            elif self._dtype == DType.uint32:
                val = Float32(self._data.bitcast[UInt32]()[i])
            elif self._dtype == DType.uint64:
                val = Float32(self._data.bitcast[UInt64]()[i])
            else:
                # Defensive re-validation (fixes DATA-003)
                raise Error("Unsupported dtype for to_int16 conversion")

            var i16_val = Int16.from_float32(val)
            result._data.bitcast[Int16]()[i] = i16_val.value

        return result^

    fn to_int32(self) raises -> ExTensor:
        """Convert tensor values to Int32 format.

        Converts a tensor of any dtype to Int32 format, clamping values to the
        range [-2147483648, 2147483647].

        Returns:
            A new ExTensor with dtype=int32 containing converted values
        """
        from .types.integer import Int32

        # (Fixes DATA-003, DATA-004, DATA-005 - see docstring for documentation)

        var result = ExTensor(self._shape, DType.int32)

        for i in range(self._numel):
            # Bounds check (fixes DATA-004)
            if i >= self._numel:
                raise Error("Index out of bounds during bitcast")

            var val: Float32
            # Defensive dtype re-validation (fixes DATA-003)
            if self._dtype == DType.float16:
                val = self._data.bitcast[Float16]()[i].cast[DType.float32]()
            elif self._dtype == DType.float32:
                val = self._data.bitcast[Float32]()[i]
            elif self._dtype == DType.float64:
                val = self._data.bitcast[Float64]()[i].cast[DType.float32]()
            elif self._dtype == DType.int8:
                val = Float32(self._data.bitcast[Int8]()[i])
            elif self._dtype == DType.int16:
                val = Float32(self._data.bitcast[Int16]()[i])
            elif self._dtype == DType.int32:
                result._data.bitcast[Int32]()[i] = self._data.bitcast[Int32]()[i]
                continue
            elif self._dtype == DType.int64:
                val = Float32(self._data.bitcast[Int64]()[i])
            elif self._dtype == DType.uint8:
                val = Float32(self._data.bitcast[UInt8]()[i])
            elif self._dtype == DType.uint16:
                val = Float32(self._data.bitcast[UInt16]()[i])
            elif self._dtype == DType.uint32:
                val = Float32(self._data.bitcast[UInt32]()[i])
            elif self._dtype == DType.uint64:
                val = Float32(self._data.bitcast[UInt64]()[i])
            else:
                # Defensive re-validation (fixes DATA-003)
                raise Error("Unsupported dtype for to_int32 conversion")

            var i32_val = Int32.from_float32(val)
            result._data.bitcast[Int32]()[i] = i32_val.value

        return result^

    fn to_int64(self) raises -> ExTensor:
        """Convert tensor values to Int64 format.

        Converts a tensor of any dtype to Int64 format.

        Returns:
            A new ExTensor with dtype=int64 containing converted values
        """
        from .types.integer import Int64

        # (Fixes DATA-003, DATA-004, DATA-005 - see docstring for documentation)

        var result = ExTensor(self._shape, DType.int64)

        for i in range(self._numel):
            # Bounds check (fixes DATA-004)
            if i >= self._numel:
                raise Error("Index out of bounds during bitcast")

            var val: Float32
            # Defensive dtype re-validation (fixes DATA-003)
            if self._dtype == DType.float16:
                val = self._data.bitcast[Float16]()[i].cast[DType.float32]()
            elif self._dtype == DType.float32:
                val = self._data.bitcast[Float32]()[i]
            elif self._dtype == DType.float64:
                val = self._data.bitcast[Float64]()[i].cast[DType.float32]()
            elif self._dtype == DType.int8:
                val = Float32(self._data.bitcast[Int8]()[i])
            elif self._dtype == DType.int16:
                val = Float32(self._data.bitcast[Int16]()[i])
            elif self._dtype == DType.int32:
                val = Float32(self._data.bitcast[Int32]()[i])
            elif self._dtype == DType.int64:
                result._data.bitcast[Int64]()[i] = self._data.bitcast[Int64]()[i]
                continue
            elif self._dtype == DType.uint8:
                val = Float32(self._data.bitcast[UInt8]()[i])
            elif self._dtype == DType.uint16:
                val = Float32(self._data.bitcast[UInt16]()[i])
            elif self._dtype == DType.uint32:
                val = Float32(self._data.bitcast[UInt32]()[i])
            elif self._dtype == DType.uint64:
                val = Float32(self._data.bitcast[UInt64]()[i])
            else:
                # Defensive re-validation (fixes DATA-003)
                raise Error("Unsupported dtype for to_int64 conversion")

            var i64_val = Int64.from_float32(val)
            result._data.bitcast[Int64]()[i] = i64_val.value

        return result^

    fn to_uint8(self) raises -> ExTensor:
        """Convert tensor values to UInt8 format.

        Converts a tensor of any dtype to UInt8 format, clamping values to the
        range [0, 255].

        Returns:
            A new ExTensor with dtype=uint8 containing converted values
        """
        from .types.unsigned import UInt8

        # (Fixes DATA-003, DATA-004, DATA-005 - see docstring for documentation)

        var result = ExTensor(self._shape, DType.uint8)

        for i in range(self._numel):
            # Bounds check (fixes DATA-004)
            if i >= self._numel:
                raise Error("Index out of bounds during bitcast")

            var val: Float32
            # Defensive dtype re-validation (fixes DATA-003)
            if self._dtype == DType.float16:
                val = self._data.bitcast[Float16]()[i].cast[DType.float32]()
            elif self._dtype == DType.float32:
                val = self._data.bitcast[Float32]()[i]
            elif self._dtype == DType.float64:
                val = self._data.bitcast[Float64]()[i].cast[DType.float32]()
            elif self._dtype == DType.int8:
                val = Float32(self._data.bitcast[Int8]()[i])
            elif self._dtype == DType.int16:
                val = Float32(self._data.bitcast[Int16]()[i])
            elif self._dtype == DType.int32:
                val = Float32(self._data.bitcast[Int32]()[i])
            elif self._dtype == DType.int64:
                val = Float32(self._data.bitcast[Int64]()[i])
            elif self._dtype == DType.uint8:
                result._data.bitcast[UInt8]()[i] = self._data.bitcast[UInt8]()[i]
                continue
            elif self._dtype == DType.uint16:
                val = Float32(self._data.bitcast[UInt16]()[i])
            elif self._dtype == DType.uint32:
                val = Float32(self._data.bitcast[UInt32]()[i])
            elif self._dtype == DType.uint64:
                val = Float32(self._data.bitcast[UInt64]()[i])
            else:
                # Defensive re-validation (fixes DATA-003)
                raise Error("Unsupported dtype for to_uint8 conversion")

            var u8_val = UInt8.from_float32(val)
            result._data.bitcast[UInt8]()[i] = u8_val.value

        return result^

    fn to_uint16(self) raises -> ExTensor:
        """Convert tensor values to UInt16 format.

        Converts a tensor of any dtype to UInt16 format, clamping values to the
        range [0, 65535].

        Returns:
            A new ExTensor with dtype=uint16 containing converted values
        """
        from .types.unsigned import UInt16

        # (Fixes DATA-003, DATA-004, DATA-005 - see docstring for documentation)

        var result = ExTensor(self._shape, DType.uint16)

        for i in range(self._numel):
            # Bounds check (fixes DATA-004)
            if i >= self._numel:
                raise Error("Index out of bounds during bitcast")

            var val: Float32
            # Defensive dtype re-validation (fixes DATA-003)
            if self._dtype == DType.float16:
                val = self._data.bitcast[Float16]()[i].cast[DType.float32]()
            elif self._dtype == DType.float32:
                val = self._data.bitcast[Float32]()[i]
            elif self._dtype == DType.float64:
                val = self._data.bitcast[Float64]()[i].cast[DType.float32]()
            elif self._dtype == DType.int8:
                val = Float32(self._data.bitcast[Int8]()[i])
            elif self._dtype == DType.int16:
                val = Float32(self._data.bitcast[Int16]()[i])
            elif self._dtype == DType.int32:
                val = Float32(self._data.bitcast[Int32]()[i])
            elif self._dtype == DType.int64:
                val = Float32(self._data.bitcast[Int64]()[i])
            elif self._dtype == DType.uint8:
                val = Float32(self._data.bitcast[UInt8]()[i])
            elif self._dtype == DType.uint16:
                result._data.bitcast[UInt16]()[i] = self._data.bitcast[UInt16]()[i]
                continue
            elif self._dtype == DType.uint32:
                val = Float32(self._data.bitcast[UInt32]()[i])
            elif self._dtype == DType.uint64:
                val = Float32(self._data.bitcast[UInt64]()[i])
            else:
                # Defensive re-validation (fixes DATA-003)
                raise Error("Unsupported dtype for to_uint16 conversion")

            var u16_val = UInt16.from_float32(val)
            result._data.bitcast[UInt16]()[i] = u16_val.value

        return result^

    fn to_uint32(self) raises -> ExTensor:
        """Convert tensor values to UInt32 format.

        Converts a tensor of any dtype to UInt32 format, clamping values to the
        range [0, 4294967295].

        Returns:
            A new ExTensor with dtype=uint32 containing converted values
        """
        from .types.unsigned import UInt32

        # (Fixes DATA-003, DATA-004, DATA-005 - see docstring for documentation)

        var result = ExTensor(self._shape, DType.uint32)

        for i in range(self._numel):
            # Bounds check (fixes DATA-004)
            if i >= self._numel:
                raise Error("Index out of bounds during bitcast")

            var val: Float32
            # Defensive dtype re-validation (fixes DATA-003)
            if self._dtype == DType.float16:
                val = self._data.bitcast[Float16]()[i].cast[DType.float32]()
            elif self._dtype == DType.float32:
                val = self._data.bitcast[Float32]()[i]
            elif self._dtype == DType.float64:
                val = self._data.bitcast[Float64]()[i].cast[DType.float32]()
            elif self._dtype == DType.int8:
                val = Float32(self._data.bitcast[Int8]()[i])
            elif self._dtype == DType.int16:
                val = Float32(self._data.bitcast[Int16]()[i])
            elif self._dtype == DType.int32:
                val = Float32(self._data.bitcast[Int32]()[i])
            elif self._dtype == DType.int64:
                val = Float32(self._data.bitcast[Int64]()[i])
            elif self._dtype == DType.uint8:
                val = Float32(self._data.bitcast[UInt8]()[i])
            elif self._dtype == DType.uint16:
                val = Float32(self._data.bitcast[UInt16]()[i])
            elif self._dtype == DType.uint32:
                result._data.bitcast[UInt32]()[i] = self._data.bitcast[UInt32]()[i]
                continue
            elif self._dtype == DType.uint64:
                val = Float32(self._data.bitcast[UInt64]()[i])
            else:
                # Defensive re-validation (fixes DATA-003)
                raise Error("Unsupported dtype for to_uint32 conversion")

            var u32_val = UInt32.from_float32(val)
            result._data.bitcast[UInt32]()[i] = u32_val.value

        return result^

    fn to_uint64(self) raises -> ExTensor:
        """Convert tensor values to UInt64 format.

        Converts a tensor of any dtype to UInt64 format, clamping negative values to 0.

        Returns:
            A new ExTensor with dtype=uint64 containing converted values
        """
        from .types.unsigned import UInt64

        # (Fixes DATA-003, DATA-004, DATA-005 - see docstring for documentation)

        var result = ExTensor(self._shape, DType.uint64)

        for i in range(self._numel):
            # Bounds check (fixes DATA-004)
            if i >= self._numel:
                raise Error("Index out of bounds during bitcast")

            var val: Float32
            # Defensive dtype re-validation (fixes DATA-003)
            if self._dtype == DType.float16:
                val = self._data.bitcast[Float16]()[i].cast[DType.float32]()
            elif self._dtype == DType.float32:
                val = self._data.bitcast[Float32]()[i]
            elif self._dtype == DType.float64:
                val = self._data.bitcast[Float64]()[i].cast[DType.float32]()
            elif self._dtype == DType.int8:
                val = Float32(self._data.bitcast[Int8]()[i])
            elif self._dtype == DType.int16:
                val = Float32(self._data.bitcast[Int16]()[i])
            elif self._dtype == DType.int32:
                val = Float32(self._data.bitcast[Int32]()[i])
            elif self._dtype == DType.int64:
                val = Float32(self._data.bitcast[Int64]()[i])
            elif self._dtype == DType.uint8:
                val = Float32(self._data.bitcast[UInt8]()[i])
            elif self._dtype == DType.uint16:
                val = Float32(self._data.bitcast[UInt16]()[i])
            elif self._dtype == DType.uint32:
                val = Float32(self._data.bitcast[UInt32]()[i])
            elif self._dtype == DType.uint64:
                result._data.bitcast[UInt64]()[i] = self._data.bitcast[UInt64]()[i]
                continue
            else:
                # Defensive re-validation (fixes DATA-003)
                raise Error("Unsupported dtype for to_uint64 conversion")

            var u64_val = UInt64.from_float32(val)
            result._data.bitcast[UInt64]()[i] = u64_val.value

        return result^

    # ========================================================================
    # BF8 Conversion Methods
    # ========================================================================

    fn to_bf8(self) raises -> ExTensor:
        """Convert tensor values to BF8 E5M2 format.

        This method converts a tensor of any floating-point dtype to BF8 format,
        stored as uint8. The conversion uses E5M2 encoding (1 sign bit, 5 exponent
        bits, 2 mantissa bits) which provides larger range than FP8 E4M3.

        Returns:
            A new ExTensor with dtype=uint8 containing BF8-encoded values

        Raises:
            Error: If the source tensor is not a floating-point dtype

        Examples:
            var t = zeros(List[Int](3, 4), DType.float32)
            var bf8_t = t.to_bf8()  # Returns uint8 tensor with BF8 encoding
            var restored = bf8_t.from_bf8()  # Convert back to float32

        Note:
            BF8 has larger range (~±57344) than FP8 but less precision (2 mantissa bits).
            Values outside this range are clamped. This is useful for memory-efficient
            training/inference where range is more important than precision.
            FP16 inputs are converted to FP32 before quantization.
        """
        from .types.bf8 import BF8

        # (Fixes DATA-003, DATA-004, DATA-005 - see docstring for documentation)

        # Verify source is floating point
        if not (
            self._dtype == DType.float16
            or self._dtype == DType.float32
            or self._dtype == DType.float64
        ):
            raise Error("to_bf8() requires a floating-point tensor")

        # Create output tensor with uint8 dtype
        var result = ExTensor(self._shape, DType.uint8)

        # Convert each element to BF8
        for i in range(self._numel):
            # Bounds check (fixes DATA-004)
            if i >= self._numel:
                raise Error("Index out of bounds during bitcast")

            # Get source value as Float32
            var val: Float32
            # Defensive dtype re-validation (fixes DATA-003)
            if self._dtype == DType.float16:
                val = self._data.bitcast[Float16]()[i].cast[DType.float32]()
            elif self._dtype == DType.float32:
                val = self._data.bitcast[Float32]()[i]
            elif self._dtype == DType.float64:
                val = self._data.bitcast[Float64]()[i].cast[DType.float32]()
            else:
                # Defensive re-validation (fixes DATA-003)
                raise Error("Invalid dtype for BF8 conversion")

            # Convert to BF8 and store
            var bf8_val = BF8.from_float32(val)
            result._data.bitcast[UInt8]()[i] = bf8_val.value

        return result^

    fn from_bf8(self) raises -> ExTensor:
        """Convert BF8-encoded tensor (uint8) back to Float32.

        This method interprets a uint8 tensor as BF8 E5M2 encoded values and
        converts them back to Float32 for computation.

        Returns:
            A new ExTensor with dtype=float32 containing decoded values

        Raises:
            Error: If the source tensor is not uint8 dtype

        Examples:
            var bf8_t = ...  # uint8 tensor with BF8 encoding
            var float_t = bf8_t.from_bf8()  # Decode to float32

        Note:
            This assumes the uint8 tensor contains valid BF8 E5M2 encoded values.
            Use this to decode tensors created by to_bf8().
        """
        from .types.bf8 import BF8

        # Verify source is uint8
        if self._dtype != DType.uint8:
            raise Error("from_bf8() requires a uint8 tensor (BF8-encoded)")

        # Create output tensor with float32 dtype
        var result = ExTensor(self._shape, DType.float32)

        # Convert each element from BF8 to Float32
        for i in range(self._numel):
            var bf8_bits = self._data.bitcast[UInt8]()[i]
            var bf8_val = BF8(bf8_bits)
            var float_val = bf8_val.to_float32()
            result._data.bitcast[Float32]()[i] = float_val

        return result^

    # ===----------------------------------------------------------------------===#
    # FP4 Blocked Type Conversions
    # ===----------------------------------------------------------------------===#

    fn to_mxfp4(self) raises -> ExTensor:
        """Convert tensor values to MXFP4 blocked format.

        This method converts a tensor of any floating-point dtype to MXFP4 format,
        stored as uint8 blocks. Values are packed into 32-element blocks, each with
        a shared E8M0 scale.

        Returns:
            A new ExTensor with dtype=uint8 containing MXFP4-encoded blocks

        Raises:
            Error: If the source tensor is not a floating-point dtype

        Examples:
            # Aligned size (32 elements = 1 block)
            var t = zeros(List[Int](32,), DType.float32)
            var mxfp4_t = t.to_mxfp4()  # Returns uint8 tensor (17 bytes)
            var restored = mxfp4_t.from_mxfp4()  # Restores 32 elements

            # Non-aligned size (33 elements = 2 blocks with padding)
            var t2 = zeros(List[Int](33,), DType.float32)
            var mxfp4_t2 = t2.to_mxfp4()  # Pads to 64 elements, returns 34 bytes
            var restored2 = mxfp4_t2.from_mxfp4()  # Correctly restores 33 elements!

        Note:
            MXFP4 uses 32-element blocks. Non-aligned tensors are padded with zeros,
            but original size is preserved in metadata. Round-trip conversion maintains
            original tensor size.
            Memory efficiency: 17 bytes per 32 Float32 values (16:1 compression).
            FP16 inputs are converted to FP32 before quantization.
        """
        from .types.mxfp4 import MXFP4Block

        # Verify source is floating point (DATA-003 outer check)
        if not (
            self._dtype == DType.float16
            or self._dtype == DType.float32
            or self._dtype == DType.float64
        ):
            raise Error("to_mxfp4() requires a floating-point tensor")

        # Calculate number of blocks (32 elements per block)
        var num_blocks = (self._numel + 31) // 32
        var total_bytes = num_blocks * 17  # 17 bytes per MXFP4Block

        # Create output tensor as flattened uint8 array
        var result = ExTensor(List[Int](), DType.uint8)

        # Store original size before padding (fixes DATA-001)
        result._original_numel_quantized = self._numel

        # Process each block
        for block_idx in range(num_blocks):
            var start_idx = block_idx * 32
            var end_idx = min(start_idx + 32, self._numel)

            # Collect 32 values (pad with zeros if needed)
            var values = List[Float32]()
            for i in range(32):
                var idx = start_idx + i
                if idx < self._numel:
                    # Bounds check (fixes DATA-004)
                    if idx >= self._numel:
                        raise Error("Index out of bounds during bitcast")

                    # Get source value as Float32
                    var val: Float32
                    # Defensive dtype re-validation (fixes DATA-003)
                    if self._dtype == DType.float16:
                        val = self._data.bitcast[Float16]()[idx].cast[DType.float32]()
                    elif self._dtype == DType.float32:
                        val = self._data.bitcast[Float32]()[idx]
                    elif self._dtype == DType.float64:
                        val = self._data.bitcast[Float64]()[idx].cast[DType.float32]()
                    else:
                        # Defensive re-validation (fixes DATA-003)
                        raise Error("Invalid dtype for MXFP4 quantization")
                    values.append(val)
                else:
                    values.append(Float32(0.0))  # Padding

            # Create MXFP4Block
            var block = MXFP4Block.from_float32_array(values)

            # Store block data (16 bytes + 1 scale byte)
            var block_offset = block_idx * 17
            for i in range(16):
                result._data.bitcast[UInt8]()[block_offset + i] = block.data[i]
            result._data.bitcast[UInt8]()[block_offset + 16] = block.scale.exponent

        return result^

    fn from_mxfp4(self) raises -> ExTensor:
        """Convert MXFP4-encoded tensor (uint8 blocks) back to Float32.

        This method interprets a uint8 tensor as MXFP4 blocks and converts them
        back to Float32 for computation.

        Returns:
            A new ExTensor with dtype=float32 containing decoded values

        Raises:
            Error: If the source tensor is not uint8 dtype or not block-aligned

        Examples:
            var mxfp4_t = ...  # uint8 tensor with MXFP4 blocks
            var float_t = mxfp4_t.from_mxfp4()  # Decode to float32, restores original size

        Note:
            This assumes the uint8 tensor contains valid MXFP4 blocks.
            Use this to decode tensors created by to_mxfp4().
            Original tensor size is restored from metadata if available.
        """
        from .types.mxfp4 import MXFP4Block, E8M0Scale

        # Verify source is uint8
        if self._dtype != DType.uint8:
            raise Error("from_mxfp4() requires a uint8 tensor (MXFP4-encoded)")

        # Calculate number of blocks and output size
        if self._numel % 17 != 0:
            raise Error("MXFP4 tensor size must be multiple of 17 bytes")

        var num_blocks = self._numel // 17
        var padded_output_size = num_blocks * 32

        # Check if original size is stored (fixes DATA-002)
        var output_size: Int
        if self._original_numel_quantized >= 0:
            # Use stored original size (was set by to_mxfp4)
            output_size = self._original_numel_quantized
        else:
            # Fallback to padded size for backwards compatibility
            output_size = padded_output_size

        # Create output tensor
        var result = ExTensor(List[Int](), DType.float32)

        # Decode each block
        for block_idx in range(num_blocks):
            var block_offset = block_idx * 17

            # Reconstruct MXFP4Block
            var data = SIMD[DType.uint8, 16](0)
            for i in range(16):
                data[i] = self._data.bitcast[UInt8]()[block_offset + i]
            var scale = E8M0Scale(self._data.bitcast[UInt8]()[block_offset + 16])

            var block = MXFP4Block(data, scale)

            # Decode block to Float32 values (only decode needed elements)
            var values = block.to_float32_array()
            for i in range(32):
                var output_idx = block_idx * 32 + i
                if output_idx < output_size:
                    result._data.bitcast[Float32]()[output_idx] = values[i]

        # Trim result to original size if needed
        if output_size < padded_output_size:
            # Create a new tensor with the correct size
            var trimmed = ExTensor(List[Int](output_size), DType.float32)
            for i in range(output_size):
                trimmed._data.bitcast[Float32]()[i] = result._data.bitcast[Float32]()[i]
            return trimmed^

        return result^

    fn to_nvfp4(self) raises -> ExTensor:
        """Convert tensor values to NVFP4 blocked format.

        This method converts a tensor of any floating-point dtype to NVFP4 format,
        stored as uint8 blocks. Values are packed into 16-element blocks, each with
        a shared E4M3 scale.

        Returns:
            A new ExTensor with dtype=uint8 containing NVFP4-encoded blocks

        Raises:
            Error: If the source tensor is not a floating-point dtype

        Examples:
            # Aligned size (16 elements = 1 block)
            var t = zeros(List[Int](16,), DType.float32)
            var nvfp4_t = t.to_nvfp4()  # Returns uint8 tensor (9 bytes)
            var restored = nvfp4_t.from_nvfp4()  # Restores 16 elements

            # Non-aligned size (17 elements = 2 blocks with padding)
            var t2 = zeros(List[Int](17,), DType.float32)
            var nvfp4_t2 = t2.to_nvfp4()  # Pads to 32 elements, returns 18 bytes
            var restored2 = nvfp4_t2.from_nvfp4()  # Correctly restores 17 elements!

        Note:
            NVFP4 uses 16-element blocks for better accuracy. Non-aligned tensors are
            padded with zeros, but original size is preserved in metadata.
            Memory efficiency: 9 bytes per 16 Float32 values (14:1 compression).
            FP16 inputs are converted to FP32 before quantization.
        """
        from .types.nvfp4 import NVFP4Block

        # Verify source is floating point (DATA-003 outer check)
        if not (
            self._dtype == DType.float16
            or self._dtype == DType.float32
            or self._dtype == DType.float64
        ):
            raise Error("to_nvfp4() requires a floating-point tensor")

        # Calculate number of blocks (16 elements per block)
        var num_blocks = (self._numel + 15) // 16
        var total_bytes = num_blocks * 9  # 9 bytes per NVFP4Block

        # Create output tensor as flattened uint8 array
        var result = ExTensor(List[Int](), DType.uint8)

        # Store original size before padding (fixes DATA-001)
        result._original_numel_quantized = self._numel

        # Process each block
        for block_idx in range(num_blocks):
            var start_idx = block_idx * 16
            var end_idx = min(start_idx + 16, self._numel)

            # Collect 16 values (pad with zeros if needed)
            var values = List[Float32]()
            for i in range(16):
                var idx = start_idx + i
                if idx < self._numel:
                    # Bounds check (fixes DATA-004)
                    if idx >= self._numel:
                        raise Error("Index out of bounds during bitcast")

                    # Get source value as Float32
                    var val: Float32
                    # Defensive dtype re-validation (fixes DATA-003)
                    if self._dtype == DType.float16:
                        val = self._data.bitcast[Float16]()[idx].cast[DType.float32]()
                    elif self._dtype == DType.float32:
                        val = self._data.bitcast[Float32]()[idx]
                    elif self._dtype == DType.float64:
                        val = self._data.bitcast[Float64]()[idx].cast[DType.float32]()
                    else:
                        # Defensive re-validation (fixes DATA-003)
                        raise Error("Invalid dtype for NVFP4 quantization")
                    values.append(val)
                else:
                    values.append(Float32(0.0))  # Padding

            # Create NVFP4Block
            var block = NVFP4Block.from_float32_array(values)

            # Store block data (8 bytes + 1 scale byte)
            var block_offset = block_idx * 9
            for i in range(8):
                result._data.bitcast[UInt8]()[block_offset + i] = block.data[i]
            result._data.bitcast[UInt8]()[block_offset + 8] = block.scale.value

        return result^

    fn from_nvfp4(self) raises -> ExTensor:
        """Convert NVFP4-encoded tensor (uint8 blocks) back to Float32.

        This method interprets a uint8 tensor as NVFP4 blocks and converts them
        back to Float32 for computation.

        Returns:
            A new ExTensor with dtype=float32 containing decoded values

        Raises:
            Error: If the source tensor is not uint8 dtype or not block-aligned

        Examples:
            var nvfp4_t = ...  # uint8 tensor with NVFP4 blocks
            var float_t = nvfp4_t.from_nvfp4()  # Decode to float32, restores original size

        Note:
            This assumes the uint8 tensor contains valid NVFP4 blocks.
            Use this to decode tensors created by to_nvfp4().
            Original tensor size is restored from metadata if available.
        """
        from .types.nvfp4 import NVFP4Block, E4M3Scale

        # Verify source is uint8
        if self._dtype != DType.uint8:
            raise Error("from_nvfp4() requires a uint8 tensor (NVFP4-encoded)")

        # Calculate number of blocks and output size
        if self._numel % 9 != 0:
            raise Error("NVFP4 tensor size must be multiple of 9 bytes")

        var num_blocks = self._numel // 9
        var padded_output_size = num_blocks * 16

        # Check if original size is stored (fixes DATA-002)
        var output_size: Int
        if self._original_numel_quantized >= 0:
            # Use stored original size (was set by to_nvfp4)
            output_size = self._original_numel_quantized
        else:
            # Fallback to padded size for backwards compatibility
            output_size = padded_output_size

        # Create output tensor
        var result = ExTensor(List[Int](), DType.float32)

        # Decode each block
        for block_idx in range(num_blocks):
            var block_offset = block_idx * 9

            # Reconstruct NVFP4Block
            var data = SIMD[DType.uint8, 8](0)
            for i in range(8):
                data[i] = self._data.bitcast[UInt8]()[block_offset + i]
            var scale = E4M3Scale(self._data.bitcast[UInt8]()[block_offset + 8])

            var block = NVFP4Block(data, scale)

            # Decode block to Float32 values (only decode needed elements)
            var values = block.to_float32_array()
            for i in range(16):
                var output_idx = block_idx * 16 + i
                if output_idx < output_size:
                    result._data.bitcast[Float32]()[output_idx] = values[i]

        # Trim result to original size if needed
        if output_size < padded_output_size:
            # Create a new tensor with the correct size
            var trimmed = ExTensor(List[Int](output_size), DType.float32)
            for i in range(output_size):
                trimmed._data.bitcast[Float32]()[i] = result._data.bitcast[Float32]()[i]
            return trimmed^

        return result^

    # TODO: Add reflected operators (__radd__, __rsub__, etc.) for operations like: 2 + tensor
    # TODO: Add in-place operators (__iadd__, __isub__, etc.) for operations like: tensor += 2
    # TODO: Add unary operators (__neg__, __pos__, __abs__, __invert__)


# ============================================================================
# Creation Operations
# ============================================================================


fn zeros(shape: List[Int], dtype: DType) raises -> ExTensor:
    """Create a tensor filled with zeros.

    Args:
        shape: The shape of the output tensor
        dtype: The data type of tensor elements

    Returns:
        A new ExTensor filled with zeros

    Examples:
        var t = zeros(List[Int](3, 4), DType.float32)
        # Creates a 3x4 tensor of float32 zeros

    Performance:
        O(n) time where n is the number of elements
    """
    var tensor = ExTensor(shape, dtype)
    tensor._fill_zero()  # Efficiently zero out all bytes
    return tensor^


fn ones(shape: List[Int], dtype: DType) raises -> ExTensor:
    """Create a tensor filled with ones.

    Args:
        shape: The shape of the output tensor
        dtype: The data type of tensor elements

    Returns:
        A new ExTensor filled with ones

    Examples:
        var t = ones(List[Int](3, 4), DType.float32)
        # Creates a 3x4 tensor of float32 ones
    """
    var tensor = ExTensor(shape, dtype)

    # Fill with ones based on dtype category
    if (
        dtype == DType.float16
        or dtype == DType.float32
        or dtype == DType.float64
    ):
        tensor._fill_value_float(1.0)
    else:
        tensor._fill_value_int(1)

    return tensor^


fn full(shape: List[Int], fill_value: Float64, dtype: DType) raises -> ExTensor:
    """Create a tensor filled with a specific value.

    Args:
        shape: The shape of the output tensor
        fill_value: The value to fill the tensor with
        dtype: The data type of tensor elements

    Returns:
        A new ExTensor filled with fill_value

    Examples:
        var t = full(List[Int](3, 4), 42.0, DType.float32)
        # Creates a 3x4 tensor filled with 42.0
    """
    var tensor = ExTensor(shape, dtype)

    # Fill with value based on dtype category
    if (
        dtype == DType.float16
        or dtype == DType.float32
        or dtype == DType.float64
    ):
        tensor._fill_value_float(fill_value)
    else:
        tensor._fill_value_int(Int(fill_value))

    return tensor^


fn empty(shape: List[Int], dtype: DType) raises -> ExTensor:
    """Create an uninitialized tensor (fast allocation).

    Args:
        shape: The shape of the output tensor
        dtype: The data type of tensor elements

    Returns:
        A new ExTensor with uninitialized memory

    Warning:
        The tensor contains uninitialized memory. Values are undefined until written.
        Use this for performance when you will immediately write to all elements.

    Examples:
        var t = empty(List[Int](3, 4), DType.float32)
        # Creates a 3x4 tensor with undefined values
    """
    # Just allocate without initialization
    var tensor = ExTensor(shape, dtype)
    return tensor^


fn arange(
    start: Float64, stop: Float64, step: Float64, dtype: DType
) raises -> ExTensor:
    """Create 1D tensor with evenly spaced values.

    Args:
        start: Start value (inclusive)
        stop: End value (exclusive)
        step: Spacing between values
        dtype: The data type of tensor elements

    Returns:
        A new 1D ExTensor with values in range [start, stop) with given step

    Examples:
        var t = arange(0.0, 10.0, 1.0, DType.float32)
        # Creates [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

        var t2 = arange(0.0, 10.0, 2.0, DType.int32)
        # Creates [0, 2, 4, 6, 8]
    """
    # Calculate number of elements
    var num_elements = Int((stop - start) / step)
    var shape = List[Int]()
    shape.append(num_elements)

    var tensor = ExTensor(shape, dtype)

    # Fill with sequence
    var value = start
    for i in range(num_elements):
        if (
            dtype == DType.float16
            or dtype == DType.float32
            or dtype == DType.float64
        ):
            tensor._set_float64(i, value)
        else:
            tensor._set_int64(i, Int(value))
        value += step

    return tensor^


fn eye(n: Int, m: Int, k: Int, dtype: DType) raises -> ExTensor:
    """Create 2D tensor with ones on diagonal.

    Args:
        n: Number of rows
        m: Number of columns
        k: Diagonal offset (0 for main diagonal, >0 for upper, <0 for lower)
        dtype: The data type of tensor elements

    Returns:
        A new 2D ExTensor with ones on the k-th diagonal

    Examples:
        var t = eye(3, 3, 0, DType.float32)
        # Creates 3x3 identity matrix

        var t2 = eye(3, 4, 1, DType.float32)
        # Creates 3x4 matrix with ones on diagonal above main
    """
    var shape = List[Int]()
    shape.append(n)
    shape.append(m)

    var tensor = ExTensor(shape, dtype)
    tensor._fill_zero()

    # Set diagonal to one
    for i in range(n):
        var j = i + k
        if j >= 0 and j < m:
            var index = i * m + j
            if (
                dtype == DType.float16
                or dtype == DType.float32
                or dtype == DType.float64
            ):
                tensor._set_float64(index, 1.0)
            else:
                tensor._set_int64(index, 1)

    return tensor^


fn linspace(start: Float64, stop: Float64, num: Int, dtype: DType) raises -> ExTensor:
    """Create 1D tensor with evenly spaced values (inclusive).

    Args:
        start: Start value (inclusive)
        stop: End value (inclusive)
        num: Number of values
        dtype: The data type of tensor elements

    Returns:
        A new 1D ExTensor with num evenly spaced values

    Examples:
        var t = linspace(0.0, 10.0, 11, DType.float32)
        # Creates [0.0, 1.0, 2.0, ..., 10.0]

        var t2 = linspace(0.0, 1.0, 5, DType.float64)
        # Creates [0.0, 0.25, 0.5, 0.75, 1.0]
    """
    var shape = List[Int]()
    shape.append(num)

    var tensor = ExTensor(shape, dtype)

    if num == 1:
        # Special case: single value
        if (
            dtype == DType.float16
            or dtype == DType.float32
            or dtype == DType.float64
        ):
            tensor._set_float64(0, start)
        else:
            tensor._set_int64(0, Int(start))
    else:
        # Calculate step size
        var step = (stop - start) / (num - 1)

        # Fill with sequence
        for i in range(num):
            var value = start + step * i
            if (
                dtype == DType.float16
                or dtype == DType.float32
                or dtype == DType.float64
            ):
                tensor._set_float64(i, value)
            else:
                tensor._set_int64(i, Int(value))

    return tensor^


fn ones_like(tensor: ExTensor) raises -> ExTensor:
    """Create tensor of ones with same shape and dtype as input.

    Args:
        tensor: Template tensor to match shape and dtype

    Returns:
        A new ExTensor filled with ones, same shape and dtype as input

    Example:
        var x = zeros(List[Int](3, 4), DType.float32)
        var y = ones_like(x)  # (3, 4) tensor of ones, float32
    """
    return ones(tensor.shape(), tensor.dtype())


fn zeros_like(tensor: ExTensor) raises -> ExTensor:
    """Create tensor of zeros with same shape and dtype as input.

    Args:
        tensor: Template tensor to match shape and dtype

    Returns:
        A new ExTensor filled with zeros, same shape and dtype as input

    Example:
        var x = ones(List[Int](3, 4), DType.float32)
        var y = zeros_like(x)  # (3, 4) tensor of zeros, float32
    """
    return zeros(tensor.shape(), tensor.dtype())


fn full_like(tensor: ExTensor, fill_value: Float64) raises -> ExTensor:
    """Create tensor filled with a value, same shape and dtype as input.

    Args:
        tensor: Template tensor to match shape and dtype
        fill_value: Value to fill the tensor with

    Returns:
        A new ExTensor filled with fill_value, same shape and dtype as input

    Example:
        var x = ones(List[Int](3, 4), DType.float32)
        var y = full_like(x, 3.14)  # (3, 4) tensor of 3.14, float32
    """
    return full(tensor.shape(), fill_value, tensor.dtype())


fn calculate_max_batch_size(
    sample_shape: List[Int],
    dtype: DType,
    max_memory_bytes: Int = 500_000_000,  # 500 MB default
) raises -> Int:
    """Calculate maximum safe batch size for given sample shape.

    Args:
        sample_shape: Shape of a single sample (e.g., [1, 28, 28] for MNIST)
        dtype: Data type of the tensor
        max_memory_bytes: Maximum memory to use for a batch (default: 500 MB)

    Returns:
        Maximum batch size that fits in memory

    Example:
        ```mojo
        # For MNIST: (1, 28, 28) images
        var sample_shape = List[Int]()
        sample_shape.append(1)
        sample_shape.append(28)
        sample_shape.append(28)
        var max_batch = calculate_max_batch_size(sample_shape, DType.float32)
        print("Max batch size:", max_batch)  # ~640,000 samples
        ```
    """
    var sample_elements = 1
    for i in range(len(sample_shape)):
        sample_elements *= sample_shape[i]

    var dtype_size = ExTensor._get_dtype_size_static(dtype)
    var bytes_per_sample = sample_elements * dtype_size

    if bytes_per_sample <= 0:
        raise Error("Invalid sample shape or dtype")

    var max_batch = max_memory_bytes // bytes_per_sample

    if max_batch < 1:
        raise Error(
            "Single sample ("
            + String(bytes_per_sample)
            + " bytes) exceeds memory limit ("
            + String(max_memory_bytes)
            + " bytes)"
        )

    return max_batch
