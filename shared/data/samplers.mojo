"""Sampling strategies for data loading.

This module provides various sampling strategies for iterating through datasets.
"""

from random import random_si64, seed


# ============================================================================
# Sampler Trait
# ============================================================================


trait Sampler:
    """Base interface for all samplers.

    Samplers determine the order in which samples are accessed from a dataset.
    """

    fn __len__(self) -> Int:
        """Return the number of samples."""
        ...

    fn __iter__(mut self) -> List[Int]:
        """Return an iterator over sample indices.

        Returns:
            List of indices in the order they should be accessed.
        """
        ...


# ============================================================================
# SequentialSampler Implementation
# ============================================================================


struct SequentialSampler(Sampler, Copyable, Movable):
    """Samples elements sequentially in order.

    Always returns indices in the same order: 0, 1, 2, ..., n-1.
    """

    var data_source_len: Int
    var start_index: Int
    var end_index: Int

    fn __init__(
        out self,
        data_source_len: Int,
        start_index: Int = 0,
        end_index: Int = -1,
    ):
        """Create sequential sampler.

        Args:
            data_source_len: Length of the dataset.
            start_index: Starting index (inclusive).
            end_index: Ending index (exclusive), -1 for end of dataset.
        """
        self.data_source_len = data_source_len
        self.start_index = max(0, start_index)

        if end_index == -1:
            self.end_index = data_source_len
        else:
            self.end_index = min(data_source_len, end_index)

    fn __len__(self) -> Int:
        """Return number of samples."""
        return self.end_index - self.start_index

    fn __iter__(mut self) -> List[Int]:
        """Return sequential indices.

        Returns:.            List of indices from start to end.
        """
        var indices = List[Int](capacity=self.__len__())
        for i in range(self.start_index, self.end_index):
            indices.append(i)
        return indices^


# ============================================================================
# RandomSampler Implementation
# ============================================================================


struct RandomSampler(Sampler, Copyable, Movable):
    """Samples elements randomly without replacement.

    Generates a random permutation of indices for each iteration.
    """

    var data_source_len: Int
    var replacement: Bool
    var num_samples: Int
    var seed_value: Optional[Int]

    fn __init__(
        out self,
        data_source_len: Int,
        replacement: Bool = False,
        num_samples: Optional[Int] = None,
        seed_value: Optional[Int] = None,
    ):
        """Create random sampler.

        Args:.            `data_source_len`: Length of the dataset.
            `replacement`: Whether to sample with replacement.
            `num_samples`: Number of samples to draw (None = all).
            `seed_value`: Random seed for reproducibility.
        """
        self.data_source_len = data_source_len
        self.replacement = replacement
        self.seed_value = seed_value

        if num_samples:
            self.num_samples = num_samples.value()
        else:
            self.num_samples = data_source_len

    fn __len__(self) -> Int:
        """Return number of samples."""
        return self.num_samples

    fn __iter__(mut self) -> List[Int]:
        """Return random indices.

        Returns:
            List of randomly shuffled or sampled indices.
        """
        # Set seed if provided
        if self.seed_value:
            seed(self.seed_value.value())

        var indices = List[Int](capacity=self.num_samples)

        if self.replacement:
            # Sample with replacement
            for _ in range(self.num_samples):
                indices.append(Int(random_si64(0, self.data_source_len)))
        else:
            # Create shuffled indices
            var all_indices = List[Int](capacity=self.data_source_len)
            for i in range(self.data_source_len):
                all_indices.append(i)

            # Fisher-Yates shuffle
            for i in range(self.data_source_len - 1, 0, -1):
                var j = Int(random_si64(0, i + 1))
                var temp = all_indices[i]
                all_indices[i] = all_indices[j]
                all_indices[j] = temp

            # Take first num_samples
            for i in range(min(self.num_samples, self.data_source_len)):
                indices.append(all_indices[i])

        return indices^


# ============================================================================
# WeightedSampler Implementation
# ============================================================================


struct WeightedSampler(Sampler, Copyable, Movable):
    """Samples elements according to given weights.

    Each sample is drawn with probability proportional to its weight.
    """

    var weights: List[Float64]
    var num_samples: Int
    var replacement: Bool
    var seed_value: Optional[Int]

    fn __init__(
        out self,
        var weights: List[Float64],
        num_samples: Int,
        replacement: Bool = True,
        seed_value: Optional[Int] = None,
    ) raises:
        """Create weighted sampler.

        Args:
            weights: Weight for each sample.
            num_samples: Number of samples to draw.
            replacement: Whether to sample with replacement.
            seed_value: Random seed for reproducibility.

        Raises:
            Error if weights are invalid.
        """
        # Validate weights
        var total_weight = Float64(0)
        for i in range(len(weights)):
            if weights[i] < 0:
                raise Error("Weights must be non-negative")
            total_weight += weights[i]

        if total_weight == 0:
            raise Error("At least one weight must be positive")

        # Normalize weights
        self.weights = List[Float64](capacity=len(weights))
        for i in range(len(weights)):
            self.weights.append(weights[i] / total_weight)

        # Transfer ownership
        weights = List[Float64]()

        self.num_samples = num_samples
        self.replacement = replacement
        self.seed_value = seed_value

    fn __len__(self) -> Int:
        """Return number of samples."""
        return self.num_samples

    fn __iter__(mut self) -> List[Int]:
        """Return weighted random indices.

        Returns:.            List of indices sampled according to weights.
        """
        # Set seed if provided
        if self.seed_value:
            seed(self.seed_value.value())

        var indices = List[Int](capacity=self.num_samples)

        # Build cumulative weights for sampling
        var cumsum = List[Float64](capacity=len(self.weights))
        var total = Float64(0)
        for i in range(len(self.weights)):
            total += self.weights[i]
            cumsum.append(total)

        # Sample indices
        for _ in range(self.num_samples):
            var r = Float64(random_si64(0, 1000000)) / 1000000.0  # Random [0, 1)

            # Binary search for index
            var idx = 0
            for i in range(len(cumsum)):
                if r < cumsum[i]:
                    idx = i
                    break

            indices.append(idx)

            # If without replacement, set weight to 0 and renormalize
            if not self.replacement:
                self.weights[idx] = 0
                # Renormalize remaining weights
                total = Float64(0)
                for i in range(len(self.weights)):
                    total += self.weights[i]
                if total > 0:
                    for i in range(len(self.weights)):
                        self.weights[i] = self.weights[i] / total
                    # Rebuild cumsum
                    cumsum.clear()
                    total = Float64(0)
                    for i in range(len(self.weights)):
                        total += self.weights[i]
                        cumsum.append(total)

        return indices^
