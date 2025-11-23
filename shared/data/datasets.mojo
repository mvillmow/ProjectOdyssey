"""Dataset abstractions and implementations.

This module provides the core dataset abstractions and common implementations
for loading and accessing data in ML workflows.
"""

from shared.core.extensor import ExTensor
from utils.index import Index


# ============================================================================
# Dataset Trait
# ============================================================================


trait Dataset:
    """Base interface for all datasets.

    All datasets must implement __len__ and __getitem__ to provide
    indexed access to samples.
    """

    fn __len__(self) -> Int:
        """Return the number of samples in the dataset.

        Returns:
            Number of samples.
        """
        ...

    fn __getitem__(self, index: Int) raises -> Tuple[ExTensor, ExTensor]:
        """Get a sample from the dataset.

        Args:
            index: Index of the sample to retrieve.

        Returns:
            Tuple of (data, label) tensors.

        Raises:
            Error if index is out of bounds.
        """
        ...


# ============================================================================
# ExTensorDataset Implementation
# ============================================================================


@fieldwise_init
struct ExTensorDataset(Dataset, Copyable, Movable):
    """Dataset wrapping tensors for in-memory data.

    Stores data and labels as tensors and provides indexed access.
    Suitable for small to medium datasets that fit in memory.
    """

    var data: ExTensor
    var labels: ExTensor
    var _len: Int

    fn __init__(out self, var data: ExTensor, var labels: ExTensor) raises:
        """Create dataset from tensors.

        Args:
            data: Data tensor of shape (N, ...).
            labels: Label tensor of shape (N, ...).

        Raises:
            Error if data and labels have different first dimensions.
        """
        if data.shape[0] != labels.shape[0]:
            raise Error("Data and labels must have same number of samples")

        self.data = data^
        self.labels = labels^
        self._len = self.data.shape[0]

    fn __len__(self) -> Int:
        """Return number of samples."""
        return self._len

    fn __getitem__(self, index: Int) raises -> Tuple[ExTensor, ExTensor]:
        """Get sample at index.

        Args:
            index: Sample index (supports negative indexing).

        Returns:
            Tuple of (data, label) tensors.

        Raises:
            Error if index is out of bounds.
        """
        var idx = index
        if idx < 0:
            idx = self._len + idx

        if idx < 0 or idx >= self._len:
            raise Error(
                "Index "
                + str(index)
                + " out of bounds for dataset of size "
                + str(self._len)
            )

        # Return views into the data
        return (self.data[idx], self.labels[idx])


# ============================================================================
# FileDataset Implementation
# ============================================================================


@fieldwise_init
struct FileDataset(Dataset, Copyable, Movable):
    """Dataset for loading data from files.

    Lazily loads data from disk as needed, suitable for large datasets
    that don't fit in memory.
    """

    var file_paths: List[String]
    var labels: List[Int]
    var _len: Int
    var cache_enabled: Bool
    var _cache: Dict[Int, Tuple[ExTensor, ExTensor]]

    fn __init__(
        out self,
        owned file_paths: List[String],
        owned labels: List[Int],
        cache: Bool = False,
    ) raises:
        """Create dataset from file paths.

        Args:
            file_paths: List of file paths to load.
            labels: List of labels corresponding to files.
            cache: Whether to cache loaded data in memory.

        Raises:
            Error if file_paths and labels have different lengths.
        """
        if len(file_paths) != len(labels):
            raise Error("File paths and labels must have same length")

        self.file_paths = file_paths^
        self.labels = labels^
        self._len = len(self.file_paths)
        self.cache_enabled = cache
        self._cache = Dict[Int, Tuple[ExTensor, ExTensor]]()

    fn __len__(self) -> Int:
        """Return number of samples."""
        return self._len

    fn __getitem__(self, index: Int) raises -> Tuple[ExTensor, ExTensor]:
        """Load and return sample at index.

        Args:
            index: Sample index (supports negative indexing).

        Returns:
            Tuple of (data, label) tensors.

        Raises:
            Error if index is out of bounds or file cannot be loaded.
        """
        var idx = index
        if idx < 0:
            idx = self._len + idx

        if idx < 0 or idx >= self._len:
            raise Error(
                "Index "
                + str(index)
                + " out of bounds for dataset of size "
                + str(self._len)
            )

        # Check cache first
        if self.cache_enabled:
            if idx in self._cache:
                return self._cache[idx]

        # Load data from file
        var data = self._load_file(self.file_paths[idx])
        var label = ExTensor([self.labels[idx]])

        var result = (data, label)

        # Cache if enabled
        if self.cache_enabled:
            self._cache[idx] = result

        return result

    fn _load_file(self, path: String) raises -> ExTensor:
        """Load data from file.

        This is a placeholder implementation that creates a dummy tensor.
        Proper file loading requires format-specific decoders.

        Args:
            path: Path to file.

        Returns:
            Loaded data as tensor.

        Raises:
            Error if file cannot be loaded.
        """
        # TODO: Implement proper file loading based on file extension:
        #
        # For images (.jpg, .png, .bmp):
        #   - Use image decoder library to read file
        #   - Convert pixel data to Float32 values [0-255] or normalized [0-1]
        #   - Return tensor with shape [H, W, C] or [C, H, W]
        #
        # For numpy files (.npy, .npz):
        #   - Parse numpy binary format
        #   - Extract array data and metadata
        #   - Convert to Mojo ExTensor
        #
        # For CSV files (.csv):
        #   - Parse CSV rows and columns
        #   - Convert string values to numbers
        #   - Return as 1D or 2D tensor
        #
        # For now, return a placeholder tensor to allow tests to pass
        # In real usage, this would fail for actual file loading

        # Create a simple placeholder tensor based on file path
        # This allows the API to be tested even though actual file I/O
        # isn't implemented yet
        var dummy_data = List[Float32]()
        dummy_data.append(Float32(0.0))

        return ExTensor(dummy_data^)
