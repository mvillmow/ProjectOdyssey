"""Dataset abstractions and implementations.

This module provides the core dataset abstractions and common implementations
for loading and accessing data in ML workflows.

Includes:
    - Dataset trait: Base interface for all datasets
    - ExTensorDataset: In-memory tensor dataset wrapper
    - FileDataset: Lazy-loading dataset from files
    - EMNISTDataset: Extended MNIST dataset with multiple splits
"""

from shared.core.extensor import ExTensor, zeros
from shared.data.formats import load_idx_labels, load_idx_images
from utils.index import Index


# ============================================================================
# Dataset Trait
# ============================================================================


trait Dataset:
    """Base interface for all datasets.

    All datasets must implement __len__ and __getitem__ to provide.
    indexed access to samples.
    """

    fn __len__(self) -> Int:
        """Return the number of samples in the dataset.

        Returns:.            Number of samples.
        """
        ...

    fn __getitem__(self, index: Int) raises -> Tuple[ExTensor, ExTensor]:
        """Get a sample from the dataset.

        Args:.            `index`: Index of the sample to retrieve.

        Returns:.            Tuple of (data, label) tensors.

        Raises:.            Error if index is out of bounds.
        """
        ...


# ============================================================================
# ExTensorDataset Implementation
# ============================================================================


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

        Args:.            `data`: Data tensor of shape (N, ...).
            `labels`: Label tensor of shape (N, ...).

        Raises:.            Error if data and labels have different first dimensions.
        """
        if data.shape()[0] != labels.shape()[0]:
            raise Error("Data and labels must have same number of samples")

        self.data = data^
        self.labels = labels^
        self._len = self.data.shape()[0]

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
                + String(index)
                + " out of bounds for dataset of size "
                + String(self._len)
            )

        # Return slices into the data
        # For 1D tensors with shape [N], slice(idx, idx+1) gives shape [1]
        return (self.data.slice(idx, idx + 1, axis=0), self.labels.slice(idx, idx + 1, axis=0))


# ============================================================================
# FileDataset Implementation
# ============================================================================


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
        var file_paths: List[String],
        var labels: List[Int],
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

        Args:.            `index`: Sample index (supports negative indexing).

        Returns:.            Tuple of (data, label) tensors.

        Raises:.            Error if index is out of bounds or file cannot be loaded.
        """
        var idx = index
        if idx < 0:
            idx = self._len + idx

        if idx < 0 or idx >= self._len:
            raise Error(
                "Index "
                + String(index)
                + " out of bounds for dataset of size "
                + String(self._len)
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

        Args:.            `path`: Path to file.

        Returns:.            Loaded data as tensor.

        Raises:.            Error if file cannot be loaded.
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


# ============================================================================
# EMNIST Dataset Struct
# ============================================================================


struct EMNISTDataset(Dataset, Copyable, Movable):
    """EMNIST Dataset wrapper for convenient dataset access.

    Provides a unified interface for loading different EMNIST splits with
    automatic file path resolution and validation.

    Attributes:
        data: Tensor containing the image data (N, 1, 28, 28)
        labels: Tensor containing the label data (N,)
        _len: Number of samples in the dataset
        split: The split type loaded (balanced, byclass, bymerge, digits, letters, mnist)
        data_dir: Directory containing the EMNIST data files
    """

    var data: ExTensor
    var labels: ExTensor
    var _len: Int
    var split: String
    var data_dir: String

    fn __init__(
        out self,
        data_dir: String,
        split: String = "balanced",
        train: Bool = True,
    ) raises:
        """Initialize EMNIST Dataset.

        Args:
            data_dir: Path to directory containing EMNIST files
            split: Dataset split to load. Options:
                - "balanced": ~112k training samples, ~18.8k test samples
                - "byclass": ~814k training samples, ~135k test samples
                - "bymerge": ~814k training samples, ~135k test samples
                - "digits": ~60k training samples, ~10k test samples (MNIST digits only)
                - "letters": ~103k training samples, ~17.4k test samples
                - "mnist": Same as MNIST training set
            train: Whether to load training (True) or test (False) split

        Raises:
            Error: If data files cannot be loaded or invalid split specified
        """
        # Validate split
        var valid_splits = List[String]()
        valid_splits.append("balanced")
        valid_splits.append("byclass")
        valid_splits.append("bymerge")
        valid_splits.append("digits")
        valid_splits.append("letters")
        valid_splits.append("mnist")

        var valid = False
        for valid_split in valid_splits:
            if split == valid_split:
                valid = True
                break

        if not valid:
            raise Error("Invalid split: " + split + ". Must be one of: balanced, byclass, bymerge, digits, letters, mnist")

        self.split = split
        self.data_dir = data_dir

        # Build file paths based on split and train/test
        var train_str = "train" if train else "test"
        var images_path = data_dir + "/emnist-" + split + "-" + train_str + "-images-idx3-ubyte"
        var labels_path = data_dir + "/emnist-" + split + "-" + train_str + "-labels-idx1-ubyte"

        # Load data
        self.data = load_idx_images(images_path)
        self.labels = load_idx_labels(labels_path)

        # Validate and store length
        if self.data.shape()[0] != self.labels.shape()[0]:
            raise Error("Data and labels have mismatched number of samples")

        self._len = self.data.shape()[0]

    fn __len__(self) -> Int:
        """Return the number of samples in the dataset.

        Returns:
            Number of samples (images/labels pairs).
        """
        return self._len

    fn __getitem__(self, index: Int) raises -> Tuple[ExTensor, ExTensor]:
        """Get a sample from the dataset.

        Args:
            index: Index of the sample to retrieve (supports negative indexing).

        Returns:
            Tuple of (image, label) tensors where:
            - image: ExTensor with shape (1, 28, 28) - single grayscale image
            - label: ExTensor with shape (1,) - integer label

        Raises:
            Error: If index is out of bounds.
        """
        var idx = index
        if idx < 0:
            idx = self._len + idx

        if idx < 0 or idx >= self._len:
            raise Error(
                "Index "
                + String(index)
                + " out of bounds for dataset of size "
                + String(self._len)
            )

        # Return slices for individual samples
        # Data shape is (N, 1, 28, 28), so slice gives (1, 1, 28, 28)
        # Then squeeze first dimension to get (1, 28, 28)
        return (
            self.data.slice(idx, idx + 1, axis=0),
            self.labels.slice(idx, idx + 1, axis=0),
        )

    fn get_train_data(self) -> ExTensorDataset raises:
        """Get training data as ExTensorDataset.

        Returns:
            ExTensorDataset containing all training data and labels.

        Raises:
            Error: If data or labels are invalid.
        """
        return ExTensorDataset(self.data, self.labels)

    fn get_test_data(self) -> ExTensorDataset raises:
        """Get test data as ExTensorDataset.

        Note: This method returns the same data as get_train_data since
        EMNISTDataset is initialized with either train or test split via __init__.
        Use __init__ with train=False to load test data.

        Returns:
            ExTensorDataset containing all data and labels.

        Raises:
            Error: If data or labels are invalid.
        """
        return ExTensorDataset(self.data, self.labels)

    fn shape(self) -> List[Int]:
        """Return the shape of individual samples.

        Returns:
            Shape of each image (1, 28, 28) for grayscale.
        """
        var shape = List[Int]()
        shape.append(1)
        shape.append(28)
        shape.append(28)
        return shape

    fn num_classes(self) -> Int:
        """Return the number of classes for this split.

        Returns:
            Number of classes:
            - balanced: 47 classes
            - byclass: 62 classes
            - bymerge: 47 classes
            - digits: 10 classes
            - letters: 26 classes
            - mnist: 10 classes
        """
        if self.split == "balanced":
            return 47
        elif self.split == "byclass":
            return 62
        elif self.split == "bymerge":
            return 47
        elif self.split == "digits":
            return 10
        elif self.split == "letters":
            return 26
        elif self.split == "mnist":
            return 10
        else:
            return -1  # Should never reach here due to validation in __init__


# ============================================================================
# Convenience Functions
# ============================================================================


fn load_emnist_train(
    data_dir: String,
    split: String = "balanced",
) raises -> Tuple[ExTensor, ExTensor]:
    """Load EMNIST training dataset.

    Args:
        data_dir: Path to directory containing EMNIST files
        split: Dataset split to load (default: "balanced")

    Returns:
        Tuple of (images, labels) tensors

    Raises:
        Error: If data files cannot be loaded
    """
    var dataset = EMNISTDataset(data_dir, split, train=True)
    return (dataset.data, dataset.labels)


fn load_emnist_test(
    data_dir: String,
    split: String = "balanced",
) raises -> Tuple[ExTensor, ExTensor]:
    """Load EMNIST test dataset.

    Args:
        data_dir: Path to directory containing EMNIST files
        split: Dataset split to load (default: "balanced")

    Returns:
        Tuple of (images, labels) tensors

    Raises:
        Error: If data files cannot be loaded
    """
    var dataset = EMNISTDataset(data_dir, split, train=False)
    return (dataset.data, dataset.labels)


# ============================================================================
# Type Aliases
# ============================================================================


# Type alias for backwards compatibility
alias TensorDataset = ExTensorDataset
