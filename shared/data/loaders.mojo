"""Data loading utilities with batching and shuffling.

This module provides the main DataLoader class and related utilities
for efficient batch loading during training.
"""

from shared.core.extensor import ExTensor
from .datasets import Dataset
from .samplers import Sampler, SequentialSampler, RandomSampler


# ============================================================================
# Batch Container
# ============================================================================


@fieldwise_init
struct Batch(Copyable, Movable):
    """Container for a batch of samples.

    Holds data and labels for a batch, along with batch metadata.
    """

    var data: ExTensor
    var labels: ExTensor
    var batch_size: Int
    var indices: List[Int]

    fn __init__(
        out self,
        owned data: ExTensor,
        owned labels: ExTensor,
        owned indices: List[Int],
    ):
        """Create a batch.

        Args:.            `data`: Batch data tensor.
            `labels`: Batch labels tensor.
            `indices`: Original indices of samples in the batch.
        """
        self.data = data^
        self.labels = labels^
        self.indices = indices^
        self.batch_size = self.data.shape[0]


# ============================================================================
# BaseLoader Implementation
# ============================================================================


@fieldwise_init
struct BaseLoader(Copyable, Movable):
    """Base data loader with core functionality.

    Provides the foundation for all data loading operations.
    """

    var dataset: Dataset
    var batch_size: Int
    var drop_last: Bool
    var _len: Int

    fn __init__(
        out self,
        owned dataset: Dataset,
        batch_size: Int = 1,
        drop_last: Bool = False,
    ) raises:
        """Create base loader.

        Args:.            `dataset`: Dataset to load from.
            `batch_size`: Number of samples per batch.
            `drop_last`: Whether to drop the last incomplete batch.

        Raises:.            Error if batch_size is invalid.
        """
        if batch_size <= 0:
            raise Error("Batch size must be positive, got: " + str(batch_size))

        self.dataset = dataset^
        self.batch_size = batch_size
        self.drop_last = drop_last

        # Calculate number of batches
        var dataset_len = self.dataset.__len__()
        if self.drop_last:
            self._len = dataset_len // batch_size
        else:
            self._len = (dataset_len + batch_size - 1) // batch_size

    fn __len__(self) -> Int:
        """Return number of batches."""
        return self._len


# ============================================================================
# BatchLoader Implementation
# ============================================================================


@fieldwise_init
struct BatchLoader(BaseLoader, Copyable, Movable):
    """Data loader with batching and optional shuffling.

    Loads data in batches, optionally shuffling the order of samples.
    """

    var sampler: Sampler
    var shuffle: Bool

    fn __init__(
        out self,
        owned dataset: Dataset,
        `batch_size`: Int = 32,
        `shuffle`: Bool = False,
        `drop_last`: Bool = False,
        owned sampler: Optional[Sampler] = None,
    ) raises:
        """Create batch loader.

        Args:
            dataset: Dataset to load from.
            batch_size: Number of samples per batch.
            shuffle: Whether to shuffle data.
            drop_last: Whether to drop the last incomplete batch.
            sampler: Custom sampler (overrides shuffle).

        Raises:
            Error if batch_size is invalid.
        """
        # Initialize base loader
        super().__init__(dataset^, batch_size, drop_last)

        self.shuffle = shuffle

        # Set up sampler
        if sampler:
            self.sampler = sampler.value()^
        elif shuffle:
            self.sampler = RandomSampler(self.dataset.__len__())
        else:
            self.sampler = SequentialSampler(self.dataset.__len__())

    fn __iter__(self) -> List[Batch]:
        """Iterate over batches.

        Returns:
            List of batches for the epoch.
        """
        var batches = List[Batch](capacity=self.__len__())
        var indices = self.sampler.__iter__()

        var batch_start = 0
        while batch_start < len(indices):
            var batch_end = min(batch_start + self.batch_size, len(indices))
            var batch_indices = List[Int](capacity=batch_end - batch_start)

            # Collect batch indices
            for i in range(batch_start, batch_end):
                batch_indices.append(indices[i])

            # Skip incomplete batch if drop_last
            if self.drop_last and len(batch_indices) < self.batch_size:
                break

            # Load batch data
            var batch_data = List[ExTensor](capacity=len(batch_indices))
            var batch_labels = List[ExTensor](capacity=len(batch_indices))

            for idx in batch_indices:
                var sample = self.dataset.__getitem__(idx[])
                batch_data.append(sample[0])
                batch_labels.append(sample[1])

            # Stack into batch tensors
            var data_tensor = self._stack_tensors(batch_data)^
            var labels_tensor = self._stack_tensors(batch_labels)^

            batches.append(Batch(data_tensor^, labels_tensor^, batch_indices^))
            batch_start = batch_end

        return batches

    fn _stack_tensors(self, tensors: List[ExTensor]) raises -> ExTensor:
        """Stack list of tensors into a batch tensor.

        Creates a new tensor with shape [batch_size, *tensor_shape] by
        stacking the input tensors along a new first dimension.

        Args:
            tensors: List of tensors to stack.

        Returns:
            Stacked tensor with batch dimension.

        Raises:
            Error if tensors list is empty or tensors have incompatible shapes.
        """
        if len(tensors) == 0:
            raise Error("Cannot stack empty list of tensors")

        # For 1D tensors (scalars), create a 1D batch tensor
        # For higher-dimensional tensors, we'll need to implement proper stacking
        # Currently handling the common case of scalar/1D tensors

        var batch_size = len(tensors)
        var first_shape = tensors[0].shape

        # Handle 1D tensors (most common case for labels and simple data)
        if len(first_shape) == 1:
            # Create list to hold all values
            var all_values = List[Float32](capacity=batch_size * first_shape[0])

            # Collect all values from each tensor
            for tensor in tensors:
                # Verify shape compatibility
                if len(tensor[].shape) != len(first_shape):
                    raise Error("All tensors must have same number of dimensions")
                if tensor[].shape[0] != first_shape[0]:
                    raise Error("All tensors must have same shape")

                # Copy values from this tensor
                for i in range(tensor[].shape[0]):
                    all_values.append(Float32(tensor[][i]))

            # Create and return the stacked tensor
            # Shape: [batch_size * tensor_size]
            return ExTensor(all_values^)

        # For multi-dimensional tensors, implement proper stacking
        # TODO: Implement N-dimensional tensor stacking
        # This would require:
        # 1. Calculate new shape: [batch_size, *tensor_shape]
        # 2. Create tensor with new shape
        # 3. Copy each tensor into the appropriate slice
        raise Error("Stacking multi-dimensional tensors not yet implemented")
