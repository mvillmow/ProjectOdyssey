# Issue #432: [Cleanup] Data Utils - Refactoring and Finalization

## Objective

Refactor and finalize the data utilities module based on learnings from Test, Implementation, and Packaging phases. Address technical debt, improve error handling, and ensure production readiness.

## Deliverables

### Code Quality Improvements

- [ ] Improve error messages in FileDataset
- [ ] Standardize validation across all components
- [ ] Add missing edge case handling
- [ ] Simplify complex implementations
- [ ] Remove any code duplication

### Documentation Updates

- [ ] Add complexity analysis to all methods
- [ ] Document performance characteristics
- [ ] Create troubleshooting guide
- [ ] Add usage examples in docstrings

### Performance Verification

- [ ] Benchmark batching performance
- [ ] Verify shuffling efficiency
- [ ] Measure memory usage patterns
- [ ] Profile iterator overhead

## Success Criteria

- [ ] All error messages are clear and actionable
- [ ] Code coverage ≥ 90%
- [ ] No code duplication (DRY principle)
- [ ] All public APIs have comprehensive documentation
- [ ] Performance meets acceptable standards
- [ ] Production-ready code quality

## Refactoring Opportunities

### 1. FileDataset Error Messages

**Current State**: TODO comment in implementation

**Proposed Improvement**:

```mojo
fn _load_file(self, path: String) raises -> Tensor:
    """Load data from file.

    **IMPORTANT**: File loading not implemented in current Mojo ecosystem.

    For production use, preprocess files externally and use TensorDataset.
    See module documentation for recommended preprocessing workflow.

    Supported Workflow:
        1. Preprocess files using Python:
           ```python
           import numpy as np
           data = preprocess_images("path/to/images/")
           np.save("processed.npy", data)
           ```

        2. Load in Mojo:
           ```mojo
           var data = load_numpy("processed.npy")
           var dataset = TensorDataset(data, labels)
           ```

    Args:
        path: Path to file (format detection by extension).

    Returns:
        Never returns - always raises error.

    Raises:
        Error with format-specific guidance on preprocessing.
    """
    var ext = self._get_file_extension(path)

    var error_messages = Dict[String, String]()
    error_messages["jpg"] = self._image_error_message(path)
    error_messages["jpeg"] = self._image_error_message(path)
    error_messages["png"] = self._image_error_message(path)
    error_messages["bmp"] = self._image_error_message(path)
    error_messages["npy"] = self._numpy_error_message(path)
    error_messages["npz"] = self._numpy_error_message(path)
    error_messages["csv"] = self._csv_error_message(path)

    if ext in error_messages:
        raise Error(error_messages[ext])
    else:
        raise Error(self._unknown_format_error_message(path, ext))

fn _image_error_message(self, path: String) -> String:
    """Generate helpful error message for image files."""
    return (
        "Image loading not implemented.\n"
        "\n"
        "Recommended workflow:\n"
        "1. Preprocess in Python:\n"
        "   from PIL import Image\n"
        "   import numpy as np\n"
        "   img = np.array(Image.open('" + path + "'))\n"
        "   # ... preprocess (resize, normalize, etc.) ...\n"
        "   np.save('processed.npy', img)\n"
        "\n"
        "2. Load in Mojo:\n"
        "   var data = load_numpy('processed.npy')\n"
        "   var dataset = TensorDataset(data, labels)\n"
        "\n"
        "See shared/data/README.md for complete examples."
    )

fn _numpy_error_message(self, path: String) -> String:
    """Generate helpful error message for NumPy files."""
    return (
        "NumPy file loading not implemented.\n"
        "\n"
        "Recommended workflow:\n"
        "1. Load and convert in Python:\n"
        "   import numpy as np\n"
        "   data = np.load('" + path + "')\n"
        "   # Convert to format Mojo can load\n"
        "\n"
        "2. Or wait for Mojo NumPy parser (future work).\n"
        "\n"
        "See shared/data/README.md for alternatives."
    )

fn _csv_error_message(self, path: String) -> String:
    """Generate helpful error message for CSV files."""
    return (
        "CSV file loading not implemented.\n"
        "\n"
        "Recommended workflow:\n"
        "1. Parse in Python:\n"
        "   import pandas as pd\n"
        "   df = pd.read_csv('" + path + "')\n"
        "   data = df.values  # Convert to NumPy array\n"
        "   # Save as tensor or NumPy file\n"
        "\n"
        "See shared/data/README.md for examples."
    )

fn _unknown_format_error_message(self, path: String, ext: String) -> String:
    """Generate error message for unknown file format."""
    return (
        "Unsupported file format: '" + ext + "'\n"
        "File: " + path + "\n"
        "\n"
        "Supported (when implemented): .jpg, .png, .bmp, .npy, .csv\n"
        "Current recommendation: Use TensorDataset with preprocessed data.\n"
        "\n"
        "See shared/data/README.md for preprocessing workflows."
    )

fn _get_file_extension(self, path: String) -> String:
    """Extract lowercase file extension from path.

    Args:
        path: File path.

    Returns:
        Lowercase extension without dot, or empty string if no extension.

    Example:
        >>> self._get_file_extension("image.jpg")
        "jpg"
        >>> self._get_file_extension("data.tar.gz")
        "gz"
        >>> self._get_file_extension("noextension")
        ""
    """
    var parts = path.split(".")
    if len(parts) < 2:
        return ""
    return parts[len(parts) - 1].lower()
```

**Impact**: Users get clear, actionable guidance instead of cryptic errors

### 2. Validation Utilities

**Current State**: Ad-hoc validation in each component

**Proposed Consolidation**:

```mojo
# shared/data/validation.mojo

fn validate_positive_int(value: Int, param_name: String) raises:
    """Validate integer parameter is positive.

    Args:
        value: Value to validate.
        param_name: Parameter name for error messages.

    Raises:
        Error if value is not positive.

    Example:
        >>> validate_positive_int(32, "batch_size")  # OK
        >>> validate_positive_int(0, "batch_size")   # Raises error
    """
    if value <= 0:
        raise Error(
            param_name + " must be positive, got " + str(value)
        )

fn validate_probability(p: Float32, param_name: String) raises:
    """Validate probability is in [0, 1].

    Args:
        p: Probability to validate.
        param_name: Parameter name for error messages.

    Raises:
        Error if probability out of range.

    Example:
        >>> validate_probability(0.5, "p")      # OK
        >>> validate_probability(1.5, "p")      # Raises error
    """
    if p < 0.0 or p > 1.0:
        raise Error(
            param_name + " must be in [0.0, 1.0], got " + str(p)
        )

fn validate_index(index: Int, size: Int, container_name: String) raises:
    """Validate index is within bounds.

    Args:
        index: Index to validate.
        size: Size of container.
        container_name: Container name for error messages.

    Raises:
        Error if index out of bounds.

    Example:
        >>> validate_index(5, 10, "dataset")    # OK
        >>> validate_index(15, 10, "dataset")   # Raises error
        >>> validate_index(-1, 10, "dataset")   # Raises error
    """
    if index < 0 or index >= size:
        raise Error(
            "Index " + str(index) + " out of bounds for " +
            container_name + " of size " + str(size)
        )

fn validate_batch_size(batch_size: Int, dataset_size: Int) raises:
    """Validate batch size is reasonable.

    Args:
        batch_size: Requested batch size.
        dataset_size: Total dataset size.

    Raises:
        Error if batch size is invalid (≤0 or unreasonably large).

    Note:
        Batch size larger than dataset is allowed but warned about.
    """
    validate_positive_int(batch_size, "batch_size")

    # Warn if batch size larger than dataset
    if batch_size > dataset_size:
        print(
            "Warning: batch_size (" + str(batch_size) + ") > " +
            "dataset_size (" + str(dataset_size) + "). " +
            "Only one batch will be created."
        )
```

**Impact**: Consistent validation and error messages across all components

### 3. DataLoader Simplification

**Current State**: Iterator logic could be clearer

**Proposed Refactoring**:

```mojo
struct DataLoaderIterator:
    """Iterator over batches from DataLoader.

    Handles both shuffled and sequential iteration, correctly managing
    partial final batches and index bounds.
    """

    var loader: DataLoader
    var indices: List[Int]
    var current_index: Int

    fn __init__(out self, loader: DataLoader):
        """Initialize iterator with optional shuffling.

        Args:
            loader: DataLoader to iterate over.
        """
        self.loader = loader
        self.current_index = 0
        self.indices = self._create_indices()

    fn _create_indices(self) -> List[Int]:
        """Create index list (shuffled or sequential).

        Returns:
            List of indices to iterate over.
        """
        var size = len(self.loader.dataset)
        var indices = List[Int](capacity=size)

        # Create sequential indices
        for i in range(size):
            indices.append(i)

        # Shuffle if requested
        if self.loader.shuffle:
            self._shuffle_indices(indices, self.loader.seed)

        return indices

    fn _shuffle_indices(self, inout indices: List[Int], seed: Int):
        """Shuffle indices in-place using Fisher-Yates algorithm.

        Args:
            indices: Index list to shuffle in-place.
            seed: Random seed for reproducibility.
        """
        # Fisher-Yates shuffle with deterministic RNG
        var rng = RandomState(seed)
        for i in range(len(indices) - 1, 0, -1):
            var j = rng.random_int(0, i + 1)
            var temp = indices[i]
            indices[i] = indices[j]
            indices[j] = temp

    fn __next__(inout self) raises -> Batch:
        """Get next batch.

        Returns:
            Next batch of data.

        Raises:
            StopIteration when no more batches.
        """
        if self.current_index >= len(self.indices):
            raise StopIteration()

        # Determine batch end index
        var start = self.current_index
        var end = min(
            start + self.loader.batch_size,
            len(self.indices)
        )

        # Collect batch samples
        var batch_data = self._gather_batch(start, end)

        # Advance iterator
        self.current_index = end

        return batch_data

    fn _gather_batch(self, start: Int, end: Int) raises -> Batch:
        """Gather samples for batch.

        Args:
            start: Start index in self.indices.
            end: End index in self.indices (exclusive).

        Returns:
            Batch containing samples.
        """
        var batch_size = end - start
        var samples = List[Tensor](capacity=batch_size)

        for i in range(start, end):
            var dataset_index = self.indices[i]
            var sample = self.loader.dataset[dataset_index]
            samples.append(sample)

        return Batch(samples^)
```

**Impact**: Clearer logic, easier to maintain and test

### 4. Performance Optimization

**Batching Performance**:

```mojo
fn benchmark_batching():
    """Benchmark batching performance across dataset sizes."""
    print("Batching Performance Benchmark")
    print("=" * 50)

    var sizes = [100, 1000, 10000, 100000]
    var batch_size = 32

    for size in sizes:
        var data = create_test_data(size)
        var dataset = TensorDataset(data, data)
        var loader = DataLoader(dataset, batch_size=batch_size)

        var start = time.now()
        var batch_count = 0
        for batch in loader:
            batch_count += 1
        var elapsed = time.now() - start

        var throughput = Float64(size) / elapsed
        print(
            "Size: " + str(size).rjust(6) +
            " | Batches: " + str(batch_count).rjust(4) +
            " | Time: " + format_time(elapsed) +
            " | Throughput: " + format_throughput(throughput)
        )

fn benchmark_shuffling():
    """Benchmark shuffling overhead."""
    print("\nShuffling Performance Benchmark")
    print("=" * 50)

    var size = 10000
    var batch_size = 32
    var data = create_test_data(size)
    var dataset = TensorDataset(data, data)

    # Without shuffling
    var loader_no_shuffle = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    var start = time.now()
    for batch in loader_no_shuffle:
        pass
    var time_no_shuffle = time.now() - start

    # With shuffling
    var loader_shuffle = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    var start = time.now()
    for batch in loader_shuffle:
        pass
    var time_shuffle = time.now() - start

    print("No shuffle: " + format_time(time_no_shuffle))
    print("With shuffle: " + format_time(time_shuffle))
    print("Overhead: " + format_percent((time_shuffle - time_no_shuffle) / time_no_shuffle))
```

## Documentation Improvements

### 1. Complexity Analysis

Add to all methods:

```mojo
fn __getitem__(self, index: Int) raises -> Tensor:
    """Retrieve sample by index.

    Complexity:
        Time: O(1) for TensorDataset (direct array access)
        Space: O(1) - returns reference to existing tensor

    Args:
        index: Sample index (0 to len(dataset)-1).

    Returns:
        Tensor at specified index.

    Raises:
        Error if index out of bounds.

    Example:
        >>> var dataset = TensorDataset(data, labels)
        >>> var sample = dataset[0]
    """
    validate_index(index, len(self), "dataset")
    return self.data[index]
```

### 2. Performance Characteristics

Document in README:

```markdown
## Performance Characteristics

### TensorDataset

- **Memory**: Stores all data in RAM (size = dataset size)
- **Access Time**: O(1) constant time lookup
- **Iteration**: O(n) linear scan
- **Best For**: Datasets that fit in memory

### DataLoader

- **Batching**: O(n) where n = batch_size
- **Shuffling**: O(n) where n = dataset size (one-time cost)
- **Iteration**: O(dataset_size / batch_size) batches
- **Memory**: O(batch_size) for current batch

### Shuffling Overhead

- First epoch: ~5-10% overhead for shuffle
- Subsequent epochs: Same shuffle cost (deterministic)
- No overhead when shuffle=False

### Recommended Batch Sizes

| Dataset Size | Batch Size | Batches | Memory Impact |
|--------------|------------|---------|---------------|
| 100          | 32         | 4       | Minimal       |
| 1,000        | 32         | 32      | Low           |
| 10,000       | 64         | 157     | Moderate      |
| 100,000      | 128        | 782     | High          |

Memory impact = batch_size × sample_size × data_type_size
```

## Known Issues and TODOs

### High Priority

1. **FileDataset File Loading** (Issue #430):
   - Deferred to future work
   - Need Mojo ecosystem support
   - Documented workaround available

2. **Error Message Clarity**:
   - Improve FileDataset error messages ✅ (addressed in refactoring)
   - Add troubleshooting to README ⏳

### Medium Priority

1. **Memory Efficiency**:
   - TensorDataset loads all data
   - Consider streaming for large datasets (future)

2. **Sampler Strategies**:
   - Only sequential and random available
   - Add weighted, stratified samplers (future)

3. **Performance Optimization**:
   - Add multi-threading for data loading (future)
   - Prefetching for pipeline efficiency (future)

### Low Priority

1. **Advanced Features**:
   - Custom sampling strategies
   - Data streaming
   - GPU-accelerated transforms
   - Network data loading

## Testing and Validation

### 1. Edge Case Testing

**Add tests for**:
- Empty datasets
- Single-sample datasets
- Batch size larger than dataset
- Multiple epochs with shuffling
- Very large batch sizes
- Invalid indices (negative, too large)

### 2. Performance Testing

**Benchmark**:
- Batching speed for various sizes
- Shuffling overhead
- Memory usage patterns
- Iterator creation cost

### 3. Stress Testing

**Test limits**:
- Very large datasets (100k+ samples)
- Very large batch sizes (1000+)
- Many epochs (100+)
- Deep iterator nesting

## Timeline and Priorities

### High Priority (Must Complete)

1. ✅ Improve FileDataset error messages
2. ✅ Add validation utilities
3. ⏳ Create comprehensive README
4. ⏳ Add complexity analysis to docstrings
5. ⏳ Benchmark performance

### Medium Priority (Should Complete)

1. Simplify DataLoader iterator logic
2. Add performance characteristics to docs
3. Create troubleshooting guide
4. Test edge cases comprehensively
5. Profile memory usage

### Low Priority (Nice to Have)

1. Add advanced sampling strategies
2. Implement streaming support
3. Add multi-threading
4. Create visual diagrams
5. Benchmark against other frameworks

## References

### Related Issues

- Issue #428: [Plan] Data Utils
- Issue #429: [Test] Data Utils
- Issue #430: [Impl] Data Utils
- Issue #431: [Package] Data Utils
- Issue #432: [Cleanup] Data Utils (this issue)

### Implementation Files

- `/home/user/ml-odyssey/shared/data/datasets.mojo`
- `/home/user/ml-odyssey/shared/data/loaders.mojo`
- `/home/user/ml-odyssey/shared/data/samplers.mojo`

## Implementation Notes

### Cleanup Strategy

1. **Start with Error Messages**: Make failures helpful
2. **Add Validation**: Catch errors early
3. **Simplify Logic**: Reduce complexity where possible
4. **Document Performance**: Help users make informed decisions
5. **Test Thoroughly**: Ensure production readiness

### Quality Metrics

**Code Quality**:
- ⏳ All tests passing
- ⏳ Code coverage ≥ 90%
- ⏳ No code duplication
- ⏳ Clear error messages
- ⏳ Comprehensive docstrings

**Performance**:
- ⏳ Batching benchmarks established
- ⏳ Shuffling overhead measured
- ⏳ Memory usage profiled
- ⏳ Acceptable for typical use cases

**Documentation**:
- ⏳ README comprehensive
- ⏳ All methods documented
- ⏳ Performance characteristics clear
- ⏳ Troubleshooting guide complete

---

**Status**: Cleanup planning complete, refactoring tasks pending

**Last Updated**: 2025-11-19

**Prepared By**: Implementation Specialist (Level 3)
