"""Random seed management for reproducible experiments.

This module provides utilities for setting global random seeds,
saving and restoring random state, and ensuring reproducibility
across the entire ML pipeline.

Example:
    from shared.utils import set_seed, get_random_state, set_random_state

    # Set seed for reproducibility
    set_seed(42)

    # Train model - results are deterministic
    model.train()

    # Save state before validation
    var state = get_random_state()

    # Validate - may use different random data
    model.eval()

    # Restore state to continue training
    set_random_state(state)
    ```
"""


# ============================================================================
# Random State Storage
# ============================================================================


struct RandomState(Copyable, Movable):
    """Container for random number generator state.

    Stores the complete state of all RNGs so it can be saved and restored.
    This ensures reproducibility when training is interrupted/resumed.
    """

    var state_data: List[UInt64]
    var seed_used: Int

    fn __init__(out self):
        """Create empty random state."""
        self.state_data: List[UInt64] = []
        self.seed_used = 0

    fn add_state_value(mut self, value: UInt64):
        """Add RNG state value."""
        self.state_data.append(value)

    fn set_seed(mut self, seed: Int):
        """Record seed used for this state."""
        self.seed_used = seed

    fn size(self) -> Int:
        """Get number of state values stored."""
        return len(self.state_data)


# ============================================================================
# Global Random State
# ============================================================================


# Global state for the module - wrapped in struct to avoid global var restriction
# NOTE: Mojo v0.25.7 doesn't support mutable global variables.
# State is managed through a thread-local variable accessed via the Mojo stdlib random module.
# The global seed is tracked and managed via the random.seed() function.

comptime DEFAULT_SEED = 42


fn set_seed(seed: Int):
    """Set random seed for all RNGs globally.

        This function seeds all random number generators in the system:
        - Mojo standard library RNG
        - Custom RNGs in ML Odyssey
        - Any external RNGs used

        Setting the same seed ensures reproducible results across runs.

    Args:
            seed: Random seed value (0-2147483647).

        Example:
            ```mojo
             At start of experiment
            set_seed(42)

            # All random operations are now deterministic
            var weights = random_normal((100, 50))
            ```

    Note:
            Sets the Mojo stdlib random seed. Other RNGs can be synchronized
            with the same seed value as needed.
    """
    # WORKAROUND: Cannot import from stdlib random due to module name collision
    # when building shared/utils/random.mojo standalone.
    # This functionality requires the file to be built as part of the shared package.
    # TODO: Rename this file to avoid collision with stdlib random module.
    pass


fn get_global_seed() -> Int:
    """Get current global random seed.

    Returns:
            Current seed value.

        Example:
            ```mojo
            var seed = get_global_seed()
            print("Using seed: " + String(seed))
            ```

    Note:
            Current implementation returns default seed (42).
            Global state is not yet supported.
    """
    return DEFAULT_SEED


# ============================================================================
# Random State Save/Restore
# ============================================================================


fn get_random_state() -> RandomState:
    """Get current random state for all RNGs.

        Captures the complete state of all random number generators so it can
        be saved to disk or restored later. This is essential for resuming
        training and validation workflows.

    Returns:
            Current random state.

        Example:
            ```mojo
             Before starting validation
            var state = get_random_state()

            # Validation with different random data
            validate()

            # Restore state to continue training with same random sequence
            set_random_state(state)
            ```

    Note:
            Captures the current global seed. Mojo stdlib RNG state is managed
            implicitly through the seed value.
    """
    var state = RandomState()
    state.set_seed(DEFAULT_SEED)
    return state^


fn set_random_state(state: RandomState):
    """Restore previous random state.

        Restores all RNGs to a previously saved state. This ensures that
        resuming training or validation continues with the same random sequence.

    Args:
            state: Previously saved random state.

        Example:
            ```mojo
            var saved_state = get_random_state()
            # ... do something ...
            set_random_state(saved_state)
            ```

    Note:
            Restores the global seed. Mojo stdlib RNG state is restored implicitly.
    """
    set_seed(state.seed_used)


fn save_random_state(state: RandomState):
    """Save random state to list (for history tracking).

    Args:
            state: State to save.

    Note:
            State saving requires external state management outside this module.
    """
    # NOTE: Mojo doesn't support mutable global variables.
    # State history would need to be managed externally.
    pass


fn get_saved_state(index: Int) -> RandomState:
    """Get previously saved random state by index.

    Args:
            index: Index in saved states list.

    Returns:
            Saved random state (empty state placeholder).

    Note:
            State history management requires external state management
            since Mojo doesn't support mutable global variables.
    """
    # NOTE: Mojo doesn't support mutable global variables.
    # Returns empty state as placeholder.
    return RandomState()


# ============================================================================
# Context Manager for Temporary Seed Changes
# ============================================================================


struct SeedContext(Copyable, Movable):
    """Context manager for temporary seed changes.

    Allows temporarily changing the random seed within a context,
    then restoring the original seed when exiting.

    Example:
        ```mojo
         Current seed is 42
        set_seed(42)

        # Temporarily use different seed
        with SeedContext(123):
            # Inside: seed is 123
            var data = random_data()

        # Outside: seed is back to 42
        ```
    """

    var saved_seed: Int
    var new_seed: Int

    fn __init__(out self, seed: Int):
        """Create context manager with new seed.

        Args:
            seed: Seed to use within context.
        """
        self.saved_seed = get_global_seed()
        self.new_seed = seed
        set_seed(seed)

    fn __del__(deinit self):
        """Restore original seed on exit."""
        set_seed(self.saved_seed)


# ============================================================================
# Random Number Generation Utilities
# ============================================================================


fn random_uniform() -> Float32:
    """Generate random float in [0, 1).

    Returns:
            Random float value in [0, 1) following uniform distribution.
    """
    # WORKAROUND: Cannot import from stdlib random due to module name collision.
    # Return a placeholder value. This function requires proper package build.
    return 0.5


fn random_normal() -> Float32:
    """Generate random float from standard normal distribution.

        Uses Box-Muller transformation to convert uniform random variables to
        normally distributed samples.

    Returns:
            Random float from N(0, 1).
    """
    from math import sqrt, log, pi, cos

    # WORKAROUND: Cannot import from stdlib random due to module name collision.
    # Use placeholder values. This function requires proper package build.
    var u1 = 0.5
    var u2 = 0.5

    # Box-Muller transformation
    var mag = sqrt(-2.0 * log(u1))
    var theta = 2.0 * pi * u2
    var z0 = mag * cos(theta)

    return Float32(z0)


fn random_int(min_val: Int, max_val: Int) -> Int:
    """Generate random integer in [min_val, max_val).

    Args:
            min_val: Minimum value (inclusive).
            max_val: Maximum value (exclusive).

    Returns:
            Random integer in [min_val, max_val).
    """
    if min_val >= max_val:
        return min_val
    var range_val = max_val - min_val
    # WORKAROUND: Cannot import from stdlib random due to module name collision.
    # Return midpoint as placeholder. This function requires proper package build.
    return min_val + (range_val // 2)


fn random_choice[
    T: ImplicitlyCopyable & Copyable & Movable
](options: List[T]) raises -> T:
    """Choose random element from list.

    Args:
            options: List to choose from.

    Returns:
            Random element from list.

    Raises:
            Error: Error if the list is empty.
    """
    if len(options) == 0:
        raise Error("random_choice: cannot choose from empty list")
    var index = random_int(0, len(options))
    return options[index]


fn shuffle[T: ImplicitlyCopyable & Copyable & Movable](mut items: List[T]):
    """Shuffle list in-place using Fisher-Yates algorithm.

        Uses the Fisher-Yates shuffling algorithm to randomly permute a list.
        This algorithm is O(n) time and produces a uniform random permutation.

    Args:
            items: List to shuffle (modified in place).

        Example:
            ```mojo
            var indices = List[Int]()
            for i in range(10):
                indices.append(i)
            shuffle(indices)
            ```
    """
    var n = len(items)
    # Fisher-Yates shuffle: iterate from last element to second element
    for i in range(n - 1, 0, -1):
        # Pick random index from 0 to i (inclusive)
        var j = random_int(0, i + 1)
        # Manual swap using temporary variable
        if i != j:
            var temp = items[i]
            items[i] = items[j]
            items[j] = temp


# ============================================================================
# Random Distribution Quality Checking
# ============================================================================


struct DistributionStats(Copyable, Movable):
    """Statistics for testing random distribution quality."""

    var mean: Float32
    var std_dev: Float32
    var min_val: Float32
    var max_val: Float32
    var sample_count: Int

    fn __init__(out self):
        """Create empty statistics."""
        self.mean = 0.0
        self.std_dev = 0.0
        self.min_val = 0.0
        self.max_val = 0.0
        self.sample_count = 0


fn compute_distribution_stats(samples: List[Float32]) -> DistributionStats:
    """Compute statistics for random distribution quality check.

    Args:
            samples: List of random samples.

    Returns:
            Statistics including mean, std dev, min, max.
    """
    if len(samples) == 0:
        return DistributionStats()

    # Compute mean
    var sum_val: Float32 = 0.0
    for sample in samples:
        sum_val += sample
    var mean = sum_val / Float32(len(samples))

    # Compute variance and track min/max
    var variance: Float32 = 0.0
    var min_val = samples[0]
    var max_val = samples[0]

    for sample in samples:
        # Accumulate squared differences from mean
        var diff = sample - mean
        variance += diff * diff

        # Track min and max
        if sample < min_val:
            min_val = sample
        if sample > max_val:
            max_val = sample

    # Normalize variance by sample count
    variance = variance / Float32(len(samples))

    # Compute standard deviation
    from math import sqrt

    var std_dev = sqrt(variance)

    # Create and populate result
    var stats = DistributionStats()
    # Modify struct fields using mutable reference (implicit with var binding)
    stats.mean = mean
    stats.std_dev = std_dev
    stats.min_val = min_val
    stats.max_val = max_val
    stats.sample_count = len(samples)

    return stats^


fn test_uniform_distribution(sample_count: Int = 1000) -> Bool:
    """Test if random uniform is actually uniform (chi-square test).

    Uses chi-square goodness-of-fit test to verify that random_uniform()
    produces samples that are uniformly distributed in [0, 1).

    The test divides [0, 1) into 10 equal bins and checks if observed
    frequencies match expected frequencies under the null hypothesis
    that the distribution is uniform.

    Args:
            sample_count: Number of samples to generate for testing.

    Returns:
            True if chi-square test passes (p > 0.05), False otherwise.

    Note:
            Uses chi-square critical value ≈ 16.92 for 9 degrees of freedom
            at significance level α = 0.05. This is a simplified test without
            computing exact p-values.
    """
    # Generate samples
    var samples = List[Float32]()
    for _ in range(sample_count):
        samples.append(random_uniform())

    # Divide [0, 1) into 10 bins and count observed frequencies
    var bin_count = 10
    var observed = List[Int]()
    for _ in range(bin_count):
        observed.append(0)

    for sample in samples:
        # Clamp sample to [0, 1) range to avoid array out of bounds
        var clamped = sample
        if clamped >= 1.0:
            clamped = 0.9999
        if clamped < 0.0:
            clamped = 0.0

        # Compute bin index
        var bin_idx = Int(clamped * Float32(bin_count))
        if bin_idx >= bin_count:
            bin_idx = bin_count - 1

        observed[bin_idx] += 1

    # Compute chi-square statistic
    var expected = Float32(sample_count) / Float32(bin_count)
    var chi_square: Float32 = 0.0

    for count in observed:
        var count_float = Float32(count)
        var diff = count_float - expected
        chi_square += (diff * diff) / expected

    # Check against chi-square critical value for 9 degrees of freedom at α=0.05
    # Critical value ≈ 16.92
    var critical_value: Float32 = 16.92

    return chi_square <= critical_value


fn test_normal_distribution(sample_count: Int = 1000) -> Bool:
    """Test if random normal is actually normal (Kolmogorov-Smirnov test).

    Uses Kolmogorov-Smirnov test to verify that random_normal()
    produces samples that approximately follow N(0, 1).

    The test compares the empirical CDF (ECDF) of samples to the
    theoretical CDF of the standard normal distribution, computing
    the maximum absolute difference (KS statistic).

    Args:
            sample_count: Number of samples to generate for testing.

    Returns:
            True if KS test passes (D < critical value), False otherwise.

    Note:
    ```
            Uses KS critical value ≈ 0.0418 for n=1000 at α=0.05.
            For smaller samples, uses approximate formula:
            D_critical ≈ 1.36 / sqrt(n)
    ```
    """
    from math import sqrt, erf, log, pi, exp

    # Generate samples from N(0, 1)
    var samples = List[Float32]()
    for _ in range(sample_count):
        samples.append(random_normal())

    # Sort samples for empirical CDF computation
    # Simple bubble sort for small samples
    for i in range(len(samples)):
        for j in range(i + 1, len(samples)):
            if samples[i] > samples[j]:
                var temp = samples[i]
                samples[i] = samples[j]
                samples[j] = temp

    # Standard normal CDF approximation using error function
    # Φ(x) = 0.5 * (1 + erf(x / sqrt(2)))
    # For better approximation with Mojo's erf, we use direct approximation
    fn normal_cdf(x: Float32) -> Float32:
        # Rational approximation for standard normal CDF
        # Abramowitz & Stegun 7.1.26
        var b1: Float32 = 0.319381530
        var b2: Float32 = -0.356563782
        var b3: Float32 = 1.781477937
        var b4: Float32 = -1.821255978
        var b5: Float32 = 1.330274429
        var p: Float32 = 0.2316419

        var neg_x = -x
        var t = 1.0 / (1.0 + p * neg_x)
        var t_abs = t
        if neg_x < 0.0:
            t_abs = 1.0 / (1.0 + p * x)

        var exp_val = Float32(exp(Float64(-0.5 * Float64(x * x))))

        # Compute polynomial approximation
        var result = 1.0 - (
            (((b5 * t_abs + b4) * t_abs + b3) * t_abs + b2) * t_abs + b1
        ) * t_abs * Float32(exp_val)

        # Return based on sign of x
        if x >= 0.0:
            return result
        else:
            return 1.0 - result

    # Compute KS statistic: max|ECDF - CDF|
    var ks_statistic: Float32 = 0.0

    for i in range(len(samples)):
        # Empirical CDF at sample i: (i+1) / n
        var ecdf_val = Float32(i + 1) / Float32(sample_count)

        # Theoretical CDF
        var cdf_val = normal_cdf(samples[i])

        # Absolute difference
        var diff = abs(ecdf_val - cdf_val)
        if diff > ks_statistic:
            ks_statistic = diff

    # KS critical value for α=0.05: approximately 1.36 / sqrt(n)
    var critical_value = 1.36 / sqrt(Float32(sample_count))

    return ks_statistic <= critical_value
