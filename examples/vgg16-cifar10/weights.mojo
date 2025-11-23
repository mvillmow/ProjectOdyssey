"""Weight Serialization for LeNet-5

Simple hex-based weight serialization format.

File Format (per weight file):
    Line 1: <name>
    Line 2: <dtype> <shape_dims...>
    Line 3+: <hex_encoded_weights>

Example:
    conv1_kernel
    float32 6 1 5 5
    3f800000 3f800000 ... (hex floats)

Each parameter is saved to its own file:
    - conv1_kernel.weights
    - conv1_bias.weights
    - etc.
"""

from shared.core import ExTensor, zeros
from memory import UnsafePointer
from pathlib import Path


fn bytes_to_hex(data: UnsafePointer[UInt8], num_bytes: Int) -> String:
    """Convert bytes to hexadecimal string.

    Args:
        data: Pointer to byte array
        num_bytes: Number of bytes to convert

    Returns:
        Hex string (e.g., "3f800000" for float 1.0)
    """
    var hex_chars = "0123456789abcdef"
    var result = String("")

    for i in range(num_bytes):
        var byte = Int(data[i])
        var high = (byte >> 4) & 0xF
        var low = byte & 0xF
        result += hex_chars[high]
        result += hex_chars[low]

    return result


fn hex_to_bytes(hex_str: String, output: UnsafePointer[UInt8]) raises:
    """Convert hexadecimal string to bytes.

    Args:
        hex_str: Hex string (e.g., "3f800000")
        output: Output buffer (must be pre-allocated)

    Raises:
        Error: If hex string is invalid
    """
    var length = len(hex_str)
    if length % 2 != 0:
        raise Error("Hex string must have even length")

    for i in range(0, length, 2):
        var high = _hex_char_to_int(hex_str[i])
        var low = _hex_char_to_int(hex_str[i + 1])
        output[i // 2] = UInt8((high << 4) | low)


fn _hex_char_to_int(c: String) raises -> Int:
    """Convert single hex character to integer.

    Args:
        c: Single character ('0'-'9', 'a'-'f', 'A'-'F')

    Returns:
        Integer value (0-15)

    Raises:
        Error: If character is not a valid hex digit
    """
    if c >= "0" and c <= "9":
        return ord(c) - ord("0")
    elif c >= "a" and c <= "f":
        return ord(c) - ord("a") + 10
    elif c >= "A" and c <= "F":
        return ord(c) - ord("A") + 10
    else:
        raise Error("Invalid hex character: " + c)


fn save_tensor(tensor: ExTensor, name: String, filepath: String) raises:
    """Save tensor to file in hex format.

    Args:
        tensor: Tensor to save
        name: Name of the parameter
        filepath: Output file path

    Format:
        Line 1: name
        Line 2: dtype shape_dim0 shape_dim1 ...
        Line 3: hex_data
    """
    var shape = tensor.shape
    var dtype = tensor.dtype()
    var numel = tensor.numel()

    # Build metadata line
    var dtype_str = str(dtype)
    var metadata = dtype_str + " "
    for i in range(len(shape)):
        metadata += str(shape[i])
        if i < len(shape) - 1:
            metadata += " "

    # Convert tensor data to hex
    var dtype_size = _get_dtype_size(dtype)
    var total_bytes = numel * dtype_size
    var hex_data = bytes_to_hex(tensor._data, total_bytes)

    # Write to file
    with open(filepath, "w") as f:
        _ = f.write(name + "\n")
        _ = f.write(metadata + "\n")
        _ = f.write(hex_data + "\n")


fn load_tensor(filepath: String) raises -> Tuple[String, ExTensor]:
    """Load tensor from file.

    Args:
        filepath: Input file path

    Returns:
        Tuple of (name, tensor)

    Raises:
        Error: If file format is invalid
    """
    # Read file
    var content: String
    with open(filepath, "r") as f:
        content = f.read()

    # Parse lines
    var lines = content.split("\n")
    if len(lines) < 3:
        raise Error("Invalid weight file format")

    var name = lines[0]
    var metadata = lines[1]
    var hex_data = lines[2]

    # Parse metadata
    var meta_parts = metadata.split(" ")
    if len(meta_parts) < 2:
        raise Error("Invalid metadata format")

    var dtype_str = meta_parts[0]
    var dtype = _parse_dtype(dtype_str)

    # Parse shape
    var shape = List[Int]()
    for i in range(1, len(meta_parts)):
        shape.append(int(meta_parts[i]))

    # Create tensor
    var tensor = zeros(shape, dtype)

    # Convert hex to bytes
    var dtype_size = _get_dtype_size(dtype)
    var total_bytes = tensor.numel() * dtype_size
    hex_to_bytes(hex_data, tensor._data)

    return (name, tensor^)


fn _get_dtype_size(dtype: DType) -> Int:
    """Get size in bytes for a dtype."""
    if dtype == DType.float16:
        return 2
    elif dtype == DType.float32:
        return 4
    elif dtype == DType.float64:
        return 8
    elif dtype == DType.int8 or dtype == DType.uint8:
        return 1
    elif dtype == DType.int16 or dtype == DType.uint16:
        return 2
    elif dtype == DType.int32 or dtype == DType.uint32:
        return 4
    elif dtype == DType.int64 or dtype == DType.uint64:
        return 8
    else:
        return 4


fn _parse_dtype(dtype_str: String) raises -> DType:
    """Parse dtype string to DType.

    Args:
        dtype_str: String representation (e.g., "float32")

    Returns:
        Corresponding DType

    Raises:
        Error: If dtype string is invalid
    """
    if dtype_str == "float16":
        return DType.float16
    elif dtype_str == "float32":
        return DType.float32
    elif dtype_str == "float64":
        return DType.float64
    elif dtype_str == "int8":
        return DType.int8
    elif dtype_str == "int16":
        return DType.int16
    elif dtype_str == "int32":
        return DType.int32
    elif dtype_str == "int64":
        return DType.int64
    elif dtype_str == "uint8":
        return DType.uint8
    elif dtype_str == "uint16":
        return DType.uint16
    elif dtype_str == "uint32":
        return DType.uint32
    elif dtype_str == "uint64":
        return DType.uint64
    else:
        raise Error("Unknown dtype: " + dtype_str)
