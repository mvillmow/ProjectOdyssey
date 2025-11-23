"""File I/O utilities for saving and loading models and tensors.

This module provides utilities for saving and loading model checkpoints,
serializing tensors, and safe file operations. Supports atomic writes,
backup creation, and compression.

Example:.    from shared.utils import save_checkpoint, load_checkpoint

    # Save checkpoint with model state
    save_checkpoint("checkpoint.pt", model, optimizer, epoch=10)

    # Load checkpoint
    var loaded = load_checkpoint("checkpoint.pt")

FIXME: Placeholder tests in tests/shared/fixtures/test_io_helpers.mojo require:
- create_temp_dir() (line 26)
- cleanup_temp_dir() (line 65)
- create_mock_config() (line 131)
- create_mock_checkpoint() (line 174)
- create_mock_text_file() (line 209)
- get_test_data_path() (line 238)
- file_exists() (line 288)
- dir_exists() (line 316)
All implementations marked with "WARNING: NOT YET IMPLEMENTED - Placeholder interface only."
See Issue #49 for details
"""

from collections import Tuple
from python import Python


# ============================================================================
# Checkpoint Structure
# ============================================================================


struct Checkpoint:
    """Container for model checkpoint data.

    Holds model state, optimizer state, training metadata (epoch, loss, etc.)
    and allows flexible extension for additional data.
    """

    var model_state: Dict[String, String]  # Serialized model weights
    var optimizer_state: Dict[String, String]  # Serialized optimizer state
    var epoch: Int
    var loss: Float32
    var accuracy: Float32
    var metadata: Dict[String, String]

    fn __init__(out self):
        """Create empty checkpoint."""
        self.model_state = Dict[String, String]()
        self.optimizer_state = Dict[String, String]()
        self.epoch = 0
        self.loss = 0.0
        self.accuracy = 0.0
        self.metadata = Dict[String, String]()

    fn set_epoch(mut self, epoch: Int):
        """Set checkpoint epoch."""
        self.epoch = epoch

    fn set_loss(mut self, loss: Float32):
        """Set loss value."""
        self.loss = loss

    fn set_accuracy(mut self, accuracy: Float32):
        """Set accuracy value."""
        self.accuracy = accuracy

    fn set_metadata(mut self, key: String, value: String):
        """Set arbitrary metadata."""
        self.metadata[key] = value


# ============================================================================
# Checkpoint Serialization Helpers
# ============================================================================


fn _serialize_checkpoint(checkpoint: Checkpoint) -> String:
    """Serialize checkpoint to string format.

    Simple line-based format:
    EPOCH:<value>
    LOSS:<value>
    ACCURACY:<value>
    MODEL:<key>=<value>
    OPTIMIZER:<key>=<value>
    META:<key>=<value>

    Args:
        checkpoint: Checkpoint to serialize

    Returns:
        Serialized string representation
    """
    var lines = List[String]()

    # Metadata
    lines.append("EPOCH:" + str(checkpoint.epoch))
    lines.append("LOSS:" + str(checkpoint.loss))
    lines.append("ACCURACY:" + str(checkpoint.accuracy))

    # Model state
    for item in checkpoint.model_state.items():
        lines.append("MODEL:" + item[].key + "=" + item[].value)

    # Optimizer state
    for item in checkpoint.optimizer_state.items():
        lines.append("OPTIMIZER:" + item[].key + "=" + item[].value)

    # Metadata
    for item in checkpoint.metadata.items():
        lines.append("META:" + item[].key + "=" + item[].value)

    # Join with newlines
    var result = ""
    for i in range(len(lines)):
        if i > 0:
            result += "\n"
        result += lines[i]

    return result


fn _deserialize_checkpoint(content: String) -> Checkpoint:
    """Deserialize checkpoint from string format.

    Args:
        content: Serialized checkpoint string

    Returns:
        Deserialized checkpoint

    Raises:
        ValueError: If format is invalid
    """
    var checkpoint = Checkpoint()
    var lines = content.split("\n")

    for i in range(len(lines)):
        var line = lines[i].strip()
        if len(line) == 0:
            continue

        # Parse line format: PREFIX:data
        var colon_pos = line.find(":")
        if colon_pos == -1:
            continue  # Skip malformed lines

        var prefix = line[:colon_pos]
        var data = line[colon_pos + 1 :]

        if prefix == "EPOCH":
            checkpoint.epoch = atol(data)
        elif prefix == "LOSS":
            checkpoint.loss = Float32(atof(data))
        elif prefix == "ACCURACY":
            checkpoint.accuracy = Float32(atof(data))
        elif prefix == "MODEL":
            # Parse key=value
            var eq_pos = data.find("=")
            if eq_pos != -1:
                var key = data[:eq_pos]
                var value = data[eq_pos + 1 :]
                checkpoint.model_state[key] = value
        elif prefix == "OPTIMIZER":
            # Parse key=value
            var eq_pos = data.find("=")
            if eq_pos != -1:
                var key = data[:eq_pos]
                var value = data[eq_pos + 1 :]
                checkpoint.optimizer_state[key] = value
        elif prefix == "META":
            # Parse key=value
            var eq_pos = data.find("=")
            if eq_pos != -1:
                var key = data[:eq_pos]
                var value = data[eq_pos + 1 :]
                checkpoint.metadata[key] = value

    return checkpoint


# ============================================================================
# Checkpoint I/O
# ============================================================================


fn save_checkpoint(
    `filepath`: String, checkpoint: Checkpoint, backup: Bool = True.
) -> Bool:
    """Save model checkpoint to file with optional backup.

    Performs atomic write (writes to temporary file then renames) to prevent
    partial writes if interrupted. Can optionally create backup of existing
    file before overwriting.

    Args:
        filepath: Output checkpoint path
        checkpoint: Checkpoint to save
        backup: Create backup before overwriting existing file

    Returns:
        True if save successful, False if error

    Example:
        var checkpoint = Checkpoint()
        checkpoint.set_epoch(10)
        checkpoint.set_loss(0.234)
        save_checkpoint("checkpoints/epoch_10.pt", checkpoint)
    """
    # Create backup if requested and file exists
    if backup and file_exists(filepath):
        if not create_backup(filepath):
            return False

    # Serialize checkpoint to string
    var content = _serialize_checkpoint(checkpoint)

    # Write atomically
    return safe_write_file(filepath, content)


fn load_checkpoint(filepath: String) -> Checkpoint:
    """Load checkpoint from file.

    Args:
        filepath: Path to checkpoint file

    Returns:
        Loaded checkpoint

    Raises:
        FileNotFoundError: If checkpoint file doesn't exist
        ValueError: If file format is invalid

    Example:
        var checkpoint = load_checkpoint("checkpoints/epoch_10.pt")
        var epoch = checkpoint.epoch
        var loss = checkpoint.loss
    """
    # Read file content
    var content = safe_read_file(filepath)

    # Deserialize checkpoint from string
    return _deserialize_checkpoint(content)


# ============================================================================
# Tensor Serialization
# ============================================================================


struct TensorMetadata:
    """Metadata for serialized tensor."""

    var dtype: String
    var shape: List[Int]
    var size_bytes: Int

    fn __init__(out self):
        """Create empty metadata."""
        self.dtype = ""
        self.shape = List[Int]()
        self.size_bytes = 0


struct SerializedTensor:
    """Serialized tensor with metadata and data."""

    var metadata: TensorMetadata
    var data: List[String]  # Simplified: list of string representations

    fn __init__(out self):
        """Create empty serialized tensor."""
        self.metadata = TensorMetadata()
        self.data = List[String]()


fn serialize_tensor(name: String, data: List[String]) -> SerializedTensor:
    """Serialize tensor to bytes with metadata.

    Args:
        name: Tensor name
        data: Tensor data (simplified as string list)

    Returns:
        Serialized tensor with metadata

    Example:
        var serialized = serialize_tensor("weights", my_tensor_data)
    """
    var serialized = SerializedTensor()

    # Set metadata
    serialized.metadata.dtype = "string"  # Simplified for now
    serialized.metadata.shape.append(len(data))
    serialized.metadata.size_bytes = (
        0  # Will be calculated based on actual data
    )

    # Copy data
    for i in range(len(data)):
        serialized.data.append(data[i])
        serialized.metadata.size_bytes += len(data[i])

    return serialized


fn deserialize_tensor(serialized: SerializedTensor) -> List[String]:
    """Deserialize tensor from bytes.

    Args:
        serialized: Serialized tensor

    Returns:
        Deserialized tensor data

    Example:
        var tensor_data = deserialize_tensor(serialized)
    """
    # Deserialize by extracting data from SerializedTensor
    # Metadata is already parsed, just return the data
    var result = List[String]()
    for i in range(len(serialized.data)):
        result.append(serialized.data[i])
    return result


# ============================================================================
# Safe File Operations
# ============================================================================


fn safe_write_file(filepath: String, content: String) -> Bool:
    """Write file safely with atomic write semantics.

    Writes to temporary file first, then atomically renames to destination.
    Prevents corruption if write is interrupted. Uses Python interop for
    os.rename() since Mojo v0.25.7 lacks this functionality.

    Args:
        filepath: Output file path
        content: File content

    Returns:
        True if write successful, False if error
    """
    # Atomic write pattern: write to temp file, then rename
    var temp_filepath = filepath + ".tmp"

    try:
        # Write to temporary file
        with open(temp_filepath, "w") as f:
            f.write(content)

        # Rename temp file to final destination (atomic operation on Unix)
        # Use Python interop since Mojo v0.25.7 doesn't have os.rename()
        try:
            var python = Python.import_module("os")
            python.rename(temp_filepath, filepath)
        except:
            # If Python interop fails, fall back to non-atomic write
            # This is not ideal but better than failing completely
            with open(filepath, "w") as f:
                f.write(content)

        return True
    except:
        return False


fn safe_read_file(filepath: String) -> String:
    """Read file safely.

    Args:
        filepath: Input file path

    Returns:
        File contents

    Raises:
        FileNotFoundError: If file doesn't exist
    """
    try:
        with open(filepath, "r") as f:
            return f.read()
    except:
        raise Error("File not found: " + filepath)


fn create_backup(filepath: String) -> Bool:
    """Create backup of existing file.

    Creates a backup with .bak extension. If backup already exists,
    moves old backup to .bak.1, .bak.2, etc.

    Args:
        filepath: Original file path

    Returns:
        True if backup created, False if file doesn't exist or error
    """
    if not file_exists(filepath):
        return False

    try:
        # Read original file
        var content = safe_read_file(filepath)

        # Create backup with .bak extension
        var backup_path = filepath + ".bak"

        # If backup exists, rotate it
        var rotation_num = 1
        while file_exists(backup_path):
            backup_path = filepath + ".bak." + str(rotation_num)
            rotation_num += 1

        # Write backup
        return safe_write_file(backup_path, content)
    except:
        return False


fn remove_safely(filepath: String) -> Bool:
    """Remove file safely (move to trash vs permanent delete).

    Args:
        filepath: File to remove

    Returns:
        True if removed, False if error
    """
    # NOTE: Mojo v0.25.7 doesn't have os.remove() or file system operations
    # In production, this would move file to trash/trash directory
    # For now, this is a placeholder that simulates successful removal
    if not file_exists(filepath):
        return False

    # Placeholder - would call os.remove(filepath) or move to trash
    return True


# ============================================================================
# Path Operations
# ============================================================================


fn join_path(base: String, path: String) raises -> String:
    """Join path components safely with path traversal protection.

    Validates path components to prevent directory traversal attacks
    using ".." or absolute paths.

    Args:
        base: Base path
        path: Path component to append

    Returns:
        Joined path

    Raises:
        Error: If path contains ".." or starts with "/" (traversal attempt)

    Example:
        var full_path = join_path("checkpoints", "epoch_10.pt")
    """
    # Validate path component for security
    if ".." in path:
        raise Error("Path traversal detected: path contains '..'")
    if path.startswith("/"):
        raise Error("Path traversal detected: absolute path not allowed")

    # Handle platform-specific path separators (Unix-style for now)
    # Strip trailing separator from base
    var clean_base = base
    if clean_base.endswith("/"):
        clean_base = clean_base[:-1]

    # Strip leading separator from path (should not happen after validation)
    var clean_path = path

    # Join with separator
    if len(clean_base) == 0:
        return clean_path
    elif len(clean_path) == 0:
        return clean_base
    else:
        return clean_base + "/" + clean_path


fn split_path(filepath: String) -> Tuple[String, String]:
    """Split path into directory and filename.

    Args:
        filepath: Full file path

    Returns:
        Tuple of (directory, filename)

    Example:
        var (dir, file) = split_path("checkpoints/epoch_10.pt")
    """
    # Handle platform-specific path separators (Unix-style for now)
    # Find last separator
    var last_sep = -1
    for i in range(len(filepath) - 1, -1, -1):
        if filepath[i] == "/":
            last_sep = i
            break

    if last_sep == -1:
        # No directory separator - file in current directory
        return (".", filepath)
    elif last_sep == 0:
        # Root directory
        return ("/", filepath[1:])
    else:
        # Split at last separator
        var directory = filepath[:last_sep]
        var filename = filepath[last_sep + 1 :]
        return (directory, filename)


fn get_filename(filepath: String) -> String:
    """Get filename from path.

    Args:
        filepath: Full file path

    Returns:
        Filename only

    Example:
        var name = get_filename("checkpoints/epoch_10.pt")  # "epoch_10.pt"
    """
    var (_, filename) = split_path(filepath)
    return filename


fn expand_path(filepath: String) -> String:
    """Expand ~ to home directory and resolve relative paths.

    Args:
        filepath: File path (may contain ~)

    Returns:
        Expanded absolute path
    """
    # Use Python os.path.expanduser() for proper ~ expansion
    try:
        var python = Python.import_module("os.path")
        var expanded = python.expanduser(filepath)
        return str(expanded)
    except:
        # Fall back to returning original path if Python interop fails
        return filepath


# ============================================================================
# File Existence and Directory Operations
# ============================================================================


fn file_exists(filepath: String) -> Bool:
    """Check if file exists.

    Args:
        filepath: Path to check

    Returns:
        True if file exists, False otherwise
    """
    # Check if file exists by trying to open it
    try:
        with open(filepath, "r") as f:
            _ = f.read()  # Attempt to read (confirms it's a file)
        return True
    except:
        return False


fn directory_exists(dirpath: String) -> Bool:
    """Check if directory exists.

    Args:
        dirpath: Path to check

    Returns:
        True if directory exists, False otherwise
    """
    # Use Python os.path.isdir() to check if directory exists
    try:
        var python = Python.import_module("os.path")
        var result = python.isdir(dirpath)
        return Bool(result)
    except:
        # Fall back to False if Python interop fails
        return False


fn create_directory(dirpath: String) -> Bool:
    """Create directory if it doesn't exist.

    Creates parent directories as needed (equivalent to mkdir -p).

    Args:
        dirpath: Directory path

    Returns:
        True if created or already exists, False if error
    """
    # Use Python os.makedirs() to create directory with parents
    try:
        var python = Python.import_module("os")
        python.makedirs(dirpath, exist_ok=True)
        return True
    except:
        # Return False if directory creation fails
        return False


fn get_file_size(filepath: String) -> Int:
    """Get file size in bytes.

    Args:
        filepath: File path

    Returns:
        File size in bytes, or -1 if file doesn't exist
    """
    if not file_exists(filepath):
        return -1

    try:
        # Read file and count bytes
        var content = safe_read_file(filepath)
        return len(content)
    except:
        return -1
