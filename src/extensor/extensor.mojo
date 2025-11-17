"""ExTensor - Extensible Tensor for ML Odyssey.

A comprehensive, dynamic tensor class with arbitrary shapes, data types, and dimensions.
Implements 150+ operations from the Array API Standard 2024 with NumPy-style broadcasting.
"""

from memory import UnsafePointer, memset_zero
from sys import simdwidthof
from math import ceildiv


struct ExTensor:
    """Dynamic tensor with runtime-determined shape and data type.

    ExTensor provides a flexible tensor implementation for machine learning workloads,
    supporting arbitrary dimensions (0D scalars to N-D tensors), multiple data types,
    and NumPy-style broadcasting for all operations.

    Attributes:
        _data: UnsafePointer to raw byte storage (type-erased)
        _shape: DynamicVector storing the shape dimensions
        _strides: DynamicVector storing the stride for each dimension (in elements)
        _dtype: The data type of tensor elements
        _numel: Total number of elements in the tensor
        _is_view: Whether this tensor is a view (shares data with another tensor)

    Examples:
        # Create tensors
        var a = zeros(DynamicVector[Int](3, 4), DType.float32)
        var b = ones(DynamicVector[Int](3, 4), DType.float32)

        # Access properties
        print(a.shape())  # [3, 4]
        print(a.dtype())  # float32
        print(a.numel())  # 12
    """

    var _data: UnsafePointer[UInt8]  # Raw byte storage
    var _shape: DynamicVector[Int]
    var _strides: DynamicVector[Int]
    var _dtype: DType
    var _numel: Int
    var _is_view: Bool

    fn __init__(inout self, shape: DynamicVector[Int], dtype: DType):
        """Initialize a new ExTensor with given shape and dtype.

        Args:
            shape: The shape of the tensor as a vector of dimension sizes
            dtype: The data type of tensor elements

        Note:
            This is a low-level constructor. Users should prefer creation
            functions like zeros(), ones(), full(), etc.
        """
        self._shape = shape
        self._dtype = dtype
        self._is_view = False

        # Calculate total number of elements
        self._numel = 1
        for i in range(len(shape)):
            self._numel *= shape[i]

        # Calculate row-major strides (in elements, not bytes)
        self._strides = DynamicVector[Int]()
        var stride = 1
        for i in range(len(shape) - 1, -1, -1):
            self._strides.push_back(0)  # Preallocate
        for i in range(len(shape) - 1, -1, -1):
            self._strides[i] = stride
            stride *= shape[i]

        # Allocate raw byte storage
        let dtype_size = self._get_dtype_size()
        self._data = UnsafePointer[UInt8].alloc(self._numel * dtype_size)

    fn __del__(owned self):
        """Destructor to free allocated memory."""
        if not self._is_view:
            self._data.free()

    fn _get_dtype_size(self) -> Int:
        """Get size in bytes for the tensor's dtype."""
        if self._dtype == DType.float16:
            return 2
        elif self._dtype == DType.float32:
            return 4
        elif self._dtype == DType.float64:
            return 8
        elif self._dtype == DType.int8 or self._dtype == DType.uint8 or self._dtype == DType.bool:
            return 1
        elif self._dtype == DType.int16 or self._dtype == DType.uint16:
            return 2
        elif self._dtype == DType.int32 or self._dtype == DType.uint32:
            return 4
        elif self._dtype == DType.int64 or self._dtype == DType.uint64:
            return 8
        else:
            return 4  # Default fallback

    fn shape(self) -> DynamicVector[Int]:
        """Return the shape of the tensor.

        Returns:
            A vector containing the size of each dimension

        Examples:
            let t = ExTensor.zeros((3, 4), DType.float32)
            print(t.shape())  # DynamicVector[3, 4]
        """
        return self._shape

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
            let t = ExTensor.zeros((3, 4), DType.float32)
            print(t.numel())  # 12
        """
        return self._numel

    fn dim(self) -> Int:
        """Return the number of dimensions (rank) of the tensor.

        Returns:
            The number of dimensions

        Examples:
            let t = ExTensor.zeros((3, 4), DType.float32)
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

    fn _get_float64(self, index: Int) -> Float64:
        """Internal: Get value at index as Float64 (assumes float-compatible dtype)."""
        let dtype_size = self._get_dtype_size()
        let offset = index * dtype_size

        if self._dtype == DType.float16:
            let ptr = (self._data + offset).bitcast[Float16]()
            return ptr[].cast[DType.float64]()
        elif self._dtype == DType.float32:
            let ptr = (self._data + offset).bitcast[Float32]()
            return ptr[].cast[DType.float64]()
        elif self._dtype == DType.float64:
            let ptr = (self._data + offset).bitcast[Float64]()
            return ptr[]
        else:
            # For integer types, cast to float64
            return Float64(self._get_int64(index))

    fn _set_float64(self, index: Int, value: Float64):
        """Internal: Set value at index (assumes float-compatible dtype)."""
        let dtype_size = self._get_dtype_size()
        let offset = index * dtype_size

        if self._dtype == DType.float16:
            let ptr = (self._data + offset).bitcast[Float16]()
            ptr[] = value.cast[DType.float16]()
        elif self._dtype == DType.float32:
            let ptr = (self._data + offset).bitcast[Float32]()
            ptr[] = value.cast[DType.float32]()
        elif self._dtype == DType.float64:
            let ptr = (self._data + offset).bitcast[Float64]()
            ptr[] = value

    fn _get_int64(self, index: Int) -> Int64:
        """Internal: Get value at index as Int64 (assumes integer-compatible dtype)."""
        let dtype_size = self._get_dtype_size()
        let offset = index * dtype_size

        if self._dtype == DType.int8:
            let ptr = (self._data + offset).bitcast[Int8]()
            return ptr[].cast[DType.int64]()
        elif self._dtype == DType.int16:
            let ptr = (self._data + offset).bitcast[Int16]()
            return ptr[].cast[DType.int64]()
        elif self._dtype == DType.int32:
            let ptr = (self._data + offset).bitcast[Int32]()
            return ptr[].cast[DType.int64]()
        elif self._dtype == DType.int64:
            let ptr = (self._data + offset).bitcast[Int64]()
            return ptr[]
        elif self._dtype == DType.uint8:
            let ptr = (self._data + offset).bitcast[UInt8]()
            return ptr[].cast[DType.int64]()
        elif self._dtype == DType.uint16:
            let ptr = (self._data + offset).bitcast[UInt16]()
            return ptr[].cast[DType.int64]()
        elif self._dtype == DType.uint32:
            let ptr = (self._data + offset).bitcast[UInt32]()
            return ptr[].cast[DType.int64]()
        elif self._dtype == DType.uint64:
            let ptr = (self._data + offset).bitcast[UInt64]()
            return ptr[].cast[DType.int64]()
        elif self._dtype == DType.bool:
            let ptr = (self._data + offset).bitcast[Bool]()
            return 1 if ptr[] else 0
        else:
            return 0  # Default fallback

    fn _set_int64(self, index: Int, value: Int64):
        """Internal: Set value at index (assumes integer-compatible dtype)."""
        let dtype_size = self._get_dtype_size()
        let offset = index * dtype_size

        if self._dtype == DType.int8:
            let ptr = (self._data + offset).bitcast[Int8]()
            ptr[] = value.cast[DType.int8]()
        elif self._dtype == DType.int16:
            let ptr = (self._data + offset).bitcast[Int16]()
            ptr[] = value.cast[DType.int16]()
        elif self._dtype == DType.int32:
            let ptr = (self._data + offset).bitcast[Int32]()
            ptr[] = value.cast[DType.int32]()
        elif self._dtype == DType.int64:
            let ptr = (self._data + offset).bitcast[Int64]()
            ptr[] = value
        elif self._dtype == DType.uint8:
            let ptr = (self._data + offset).bitcast[UInt8]()
            ptr[] = value.cast[DType.uint8]()
        elif self._dtype == DType.uint16:
            let ptr = (self._data + offset).bitcast[UInt16]()
            ptr[] = value.cast[DType.uint16]()
        elif self._dtype == DType.uint32:
            let ptr = (self._data + offset).bitcast[UInt32]()
            ptr[] = value.cast[DType.uint32]()
        elif self._dtype == DType.uint64:
            let ptr = (self._data + offset).bitcast[UInt64]()
            ptr[] = value.cast[DType.uint64]()
        elif self._dtype == DType.bool:
            let ptr = (self._data + offset).bitcast[Bool]()
            ptr[] = value != 0

    fn _fill_zero(inout self):
        """Internal: Fill tensor with zeros (works for all dtypes)."""
        let dtype_size = self._get_dtype_size()
        let total_bytes = self._numel * dtype_size
        memset_zero(self._data, total_bytes)

    fn _fill_value_float(inout self, value: Float64):
        """Internal: Fill tensor with float value."""
        for i in range(self._numel):
            self._set_float64(i, value)

    fn _fill_value_int(inout self, value: Int64):
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

fn zeros(shape: DynamicVector[Int], dtype: DType) -> ExTensor:
    """Create a tensor filled with zeros.

    Args:
        shape: The shape of the output tensor
        dtype: The data type of tensor elements

    Returns:
        A new ExTensor filled with zeros

    Examples:
        var t = zeros(DynamicVector[Int](3, 4), DType.float32)
        # Creates a 3x4 tensor of float32 zeros

    Performance:
        O(n) time where n is the number of elements
    """
    var tensor = ExTensor(shape, dtype)
    tensor._fill_zero()  # Efficiently zero out all bytes
    return tensor^


fn ones(shape: DynamicVector[Int], dtype: DType) -> ExTensor:
    """Create a tensor filled with ones.

    Args:
        shape: The shape of the output tensor
        dtype: The data type of tensor elements

    Returns:
        A new ExTensor filled with ones

    Examples:
        var t = ones(DynamicVector[Int](3, 4), DType.float32)
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


fn full(shape: DynamicVector[Int], fill_value: Float64, dtype: DType) -> ExTensor:
    """Create a tensor filled with a specific value.

    Args:
        shape: The shape of the output tensor
        fill_value: The value to fill the tensor with
        dtype: The data type of tensor elements

    Returns:
        A new ExTensor filled with fill_value

    Examples:
        var t = full(DynamicVector[Int](3, 4), 42.0, DType.float32)
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
        tensor._fill_value_int(int(fill_value))

    return tensor^


fn empty(shape: DynamicVector[Int], dtype: DType) -> ExTensor:
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
        var t = empty(DynamicVector[Int](3, 4), DType.float32)
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
    let num_elements = int((stop - start) / step)
    var shape = DynamicVector[Int](1)
    shape[0] = num_elements

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
            tensor._set_int64(i, int(value))
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
    var shape = DynamicVector[Int](2)
    shape[0] = n
    shape[1] = m

    var tensor = ExTensor(shape, dtype)
    tensor._fill_zero()

    # Set diagonal to one
    for i in range(n):
        let j = i + k
        if j >= 0 and j < m:
            let index = i * m + j
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
    var shape = DynamicVector[Int](1)
    shape[0] = num

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
            tensor._set_int64(0, int(start))
    else:
        # Calculate step size
        let step = (stop - start) / (num - 1)

        # Fill with sequence
        for i in range(num):
            let value = start + step * i
            if (
                dtype == DType.float16
                or dtype == DType.float32
                or dtype == DType.float64
            ):
                tensor._set_float64(i, value)
            else:
                tensor._set_int64(i, int(value))

    return tensor^
