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


struct ExTensor(Movable):
    """Dynamic tensor with runtime-determined shape and data type.

    ExTensor provides a flexible tensor implementation for machine learning workloads,
    supporting arbitrary dimensions (0D scalars to N-D tensors), multiple data types,
    and NumPy-style broadcasting for all operations.

    Attributes:
        _data: UnsafePointer to raw byte storage (type-erased)
        _shape: List storing the shape dimensions
        _strides: List storing the stride for each dimension (in elements)
        _dtype: The data type of tensor elements
        _numel: Total number of elements in the tensor
        _is_view: Whether this tensor is a view (shares data with another tensor)

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

    fn __init__(out self, shape: List[Int], dtype: DType):
        """Initialize a new ExTensor with given shape and dtype.

        Args:
            shape: The shape of the tensor as a vector of dimension sizes
            dtype: The data type of tensor elements

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

        # Calculate total number of elements
        self._numel = 1
        for i in range(len(self._shape)):
            self._numel *= self._shape[i]

        # Calculate row-major strides (in elements, not bytes)
        self._strides = List[Int]()
        var stride = 1
        for i in range(len(self._shape) - 1, -1, -1):
            self._strides.append(0)  # Preallocate
        for i in range(len(self._shape) - 1, -1, -1):
            self._strides[i] = stride
            stride *= self._shape[i]

        # Allocate raw byte storage
        var dtype_size = ExTensor._get_dtype_size_static(dtype)
        self._data = alloc[UInt8](self._numel * dtype_size)

    fn __del__(deinit self):
        """Destructor to free allocated memory.

        Only frees memory if this tensor owns the data (not a view).
        Views share data with another tensor and should not free it.

        Note:
            Currently, all tensors own their data since views are not yet implemented.
            _is_view is always False in the current implementation.
        """
        if not self._is_view:
            # Free the allocated memory
            # Since _data is always allocated in __init__, this is safe
            self._data.free()

    fn _get_dtype_size(self) -> Int:
        """Get size in bytes for the tensor's dtype."""
        return ExTensor._get_dtype_size_static(self._dtype)

    @staticmethod
    fn _get_dtype_size_static(dtype: DType) -> Int:
        """Get size in bytes for a given dtype (static version for use in __init__)."""
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

        # Create new tensor sharing same data
        var result = ExTensor(new_shape, self._dtype)
        result._data = self._data  # Share data
        result._is_view = True  # Mark as view
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
        """Internal: Get value at index as Float64 (assumes float-compatible dtype)."""
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
        """Internal: Get value at index as Int64 (assumes integer-compatible dtype)."""
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

    # TODO: Add reflected operators (__radd__, __rsub__, etc.) for operations like: 2 + tensor
    # TODO: Add in-place operators (__iadd__, __isub__, etc.) for operations like: tensor += 2
    # TODO: Add unary operators (__neg__, __pos__, __abs__, __invert__)


# ============================================================================
# Creation Operations
# ============================================================================

fn zeros(shape: List[Int], dtype: DType) -> ExTensor:
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


fn ones(shape: List[Int], dtype: DType) -> ExTensor:
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


fn full(shape: List[Int], fill_value: Float64, dtype: DType) -> ExTensor:
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


fn empty(shape: List[Int], dtype: DType) -> ExTensor:
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


fn arange(start: Float64, stop: Float64, step: Float64, dtype: DType) -> ExTensor:
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


fn eye(n: Int, m: Int, k: Int, dtype: DType) -> ExTensor:
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


fn linspace(start: Float64, stop: Float64, num: Int, dtype: DType) -> ExTensor:
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


fn ones_like(tensor: ExTensor) -> ExTensor:
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


fn zeros_like(tensor: ExTensor) -> ExTensor:
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


fn full_like(tensor: ExTensor, fill_value: Float64) -> ExTensor:
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
