"""Dataset abstractions and implementations.

This module provides the core dataset abstractions and common implementations
for loading and accessing data in ML workflows.
"""

from tensor import Tensor
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

    fn __getitem__(self, index: Int) raises -> Tuple[Tensor, Tensor]:
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
# TensorDataset Implementation
# ============================================================================


@value
struct TensorDataset(Dataset):
    """Dataset wrapping tensors for in-memory data.

    Stores data and labels as tensors and provides indexed access.
    Suitable for small to medium datasets that fit in memory.
    """

    var data: Tensor
    var labels: Tensor
    var _len: Int

    fn __init__(out self, owned data: Tensor, owned labels: Tensor) raises:
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

    fn __getitem__(self, index: Int) raises -> Tuple[Tensor, Tensor]:
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


@value
struct FileDataset(Dataset):
    """Dataset for loading data from files.

    Lazily loads data from disk as needed, suitable for large datasets
    that don't fit in memory.
    """

    var file_paths: List[String]
    var labels: List[Int]
    var _len: Int
    var cache_enabled: Bool
    var _cache: Dict[Int, Tuple[Tensor, Tensor]]

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
        self._cache = Dict[Int, Tuple[Tensor, Tensor]]()

    fn __len__(self) -> Int:
        """Return number of samples."""
        return self._len

    fn __getitem__(self, index: Int) raises -> Tuple[Tensor, Tensor]:
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
        var label = Tensor([self.labels[idx]])

        var result = (data, label)

        # Cache if enabled
        if self.cache_enabled:
            self._cache[idx] = result

        return result

    fn _load_file(self, path: String) raises -> Tensor:
        """Load data from file - NOT IMPLEMENTED.

        TODO: Implement file loading based on format:
        - For images: Load from disk, decode, convert to tensor
        - For numpy: Load .npy or .npz files
        - For CSV: Parse and convert to tensors
        - Expected return: data tensor

        Args:
            path: Path to file.

        Returns:
            Loaded data as tensor.

        Raises:
            Error if file cannot be loaded.
        """
        raise Error(
            "FileDataset file loading not yet implemented - see TODO for"
            " implementation details"
        )
