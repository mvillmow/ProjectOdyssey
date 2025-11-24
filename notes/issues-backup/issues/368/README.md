# Issue #368: [Plan] Dataset Interface - Design and Documentation

## Objective

Define the base dataset interface that all dataset implementations must follow, establishing the contract for data access and enabling interchangeable dataset implementations with consistent usage patterns throughout the codebase.

## Deliverables

- Abstract base class or trait for datasets
- Required method signatures (`__len__`, `__getitem__`)
- Optional method specifications (transform, collate)
- Clear documentation and usage examples

## Success Criteria

- [ ] Interface is minimal and clear
- [ ] Required methods cover essential operations
- [ ] Documentation explains usage patterns
- [ ] Interface works with standard Python idioms

## Design Decisions

### 1. Python Sequence Protocol Compliance

**Decision**: Follow the Python sequence protocol by implementing `__len__` and `__getitem__` as required methods.

### Rationale

- Enables standard Python idioms (iteration, indexing, slicing)
- Makes datasets work with built-in Python functions (len(), list(), etc.)
- Familiar interface for Python developers
- Allows datasets to be used anywhere sequences are expected

### 2. Minimal Interface Design

**Decision**: Keep the interface minimal with only essential methods as required.

### Rationale

- Reduces implementation burden for custom datasets
- Easier to understand and use
- Follows YAGNI (You Aren't Gonna Need It) principle
- Optional methods can be added as needed without breaking compatibility

### 3. Abstract Base Class vs Trait

**Decision**: Use an abstract base class (or Mojo trait) to define the interface.

### Rationale

- Provides clear contract for implementers
- Enables type checking and validation
- Allows for default implementations of convenience methods
- Makes it easy to identify dataset types in the codebase

### 4. Required vs Optional Methods

**Decision**: Only `__len__` and `__getitem__` are required; transform and collate are optional.

### Rationale

- Core operations (length and indexing) are universally needed
- Transform and collate are dataset-specific features
- Allows simple datasets without unnecessary complexity
- Optional methods can be added through composition or inheritance

### 5. Support for Various Data Types

**Decision**: Interface should be data-agnostic, supporting images, text, tabular data, etc.

### Rationale

- Maximizes reusability across different ML tasks
- Prevents coupling to specific data formats
- Allows for future extensibility
- Follows Single Responsibility Principle (SRP)

### 6. Indexing and Iteration Support

**Decision**: Interface must support both indexed access and iteration.

### Rationale

- Indexed access enables random sampling and batching
- Iteration enables sequential processing
- Both patterns are common in ML workflows
- Python sequence protocol provides both automatically

## Architecture Overview

```text
Dataset Interface Hierarchy:

Dataset (Abstract Base)
├── __len__() -> int         [Required]
├── __getitem__(idx) -> T    [Required]
├── transform() -> Dataset   [Optional]
└── collate() -> Batch       [Optional]

Implementation Examples:
├── ImageDataset
├── TextDataset
└── TabularDataset
```text

## API Contract

### Required Methods

#### `__len__() -> int`

- Returns the total number of samples in the dataset
- Must be deterministic (same value across calls)
- Used for batching, progress tracking, and validation

#### `__getitem__(index: int) -> Sample`

- Returns a single sample at the specified index
- Index must be in range [0, len(self) - 1]
- Should raise IndexError for out-of-bounds access
- Return type depends on dataset implementation

### Optional Methods

#### `transform() -> Dataset`

- Applies transformations to dataset samples
- Returns a new dataset (doesn't modify in place)
- Allows for composable data augmentation

#### `collate() -> Batch`

- Combines multiple samples into a batch
- Handles padding, stacking, and other batch operations
- Used by dataloaders for efficient batch creation

## Usage Examples

### Basic Dataset Implementation

```mojo
struct SimpleDataset(Dataset):
    var data: List[Tensor]

    fn __len__(self) -> int:
        return len(self.data)

    fn __getitem__(self, index: int) -> Tensor:
        if index < 0 or index >= len(self):
            raise IndexError("Index out of range")
        return self.data[index]
```text

### Using Dataset with Standard Python Idioms

```python
# Create dataset
dataset = SimpleDataset(data)

# Get dataset size
size = len(dataset)

# Access samples by index
sample = dataset[0]

# Iterate over dataset
for sample in dataset:
    process(sample)

# Slice dataset (if supported)
subset = dataset[0:10]
```text

### Custom Dataset with Transform

```mojo
struct ImageDataset(Dataset):
    var paths: List[String]
    var transform_fn: Optional[Transform]

    fn __len__(self) -> int:
        return len(self.paths)

    fn __getitem__(self, index: int) -> Image:
        image = load_image(self.paths[index])
        if self.transform_fn:
            image = self.transform_fn.apply(image)
        return image

    fn transform(self, fn: Transform) -> ImageDataset:
        return ImageDataset(self.paths, fn)
```text

## Integration with Other Components

### Dataset Loader (Issues #373-376)

The dataset interface will be consumed by the dataset loader component, which:

- Creates dataset instances from configuration
- Applies splits (train/val/test)
- Handles data augmentation pipelines
- Manages dataset caching and preprocessing

### Data Pipeline (Issues #341-392)

The dataset interface is part of the broader data pipeline:

1. **Dataset Interface** (this component) - Defines how to access data
1. **Dataset Length** (#369-372) - Implements size tracking
1. **Dataset Getitem** (#377-380) - Implements indexed access
1. **Dataset Loader** (#373-376) - Creates and configures datasets

## References

### Source Plan

- [Dataset Interface Plan](notes/plan/02-shared-library/03-data-utils/01-base-dataset/01-dataset-interface/plan.md)
- [Base Dataset Parent Plan](notes/plan/02-shared-library/03-data-utils/01-base-dataset/plan.md)

### Related Issues

- Issue #369: [Test] Dataset Interface - Test Suite
- Issue #370: [Impl] Dataset Interface - Implementation
- Issue #371: [Package] Dataset Interface - Integration and Packaging
- Issue #372: [Cleanup] Dataset Interface - Refactoring and Finalization

### Related Components

- Issue #341-344: Base Dataset (parent component)
- Issue #345-348: Dataset Length
- Issue #349-352: Dataset Getitem
- Issue #373-376: Dataset Loader

### Comprehensive Documentation

- [Agent Hierarchy](agents/hierarchy.md)
- [5-Phase Workflow](notes/review/README.md)
- [Mojo Language Guidelines](.claude/agents/mojo-language-review-specialist.md)

## Implementation Notes

This section will be populated during the Test (#369), Implementation (#370), Packaging (#371), and Cleanup (#372) phases with:

- Discovered design patterns
- Implementation challenges and solutions
- Performance considerations
- Testing insights
- Integration issues and resolutions

## Next Steps

After this planning phase is complete:

1. **Test Phase (Issue #369)**: Write test suite following TDD principles
1. **Implementation Phase (Issue #370)**: Implement the dataset interface
1. **Packaging Phase (Issue #371)**: Integrate with build system and documentation
1. **Cleanup Phase (Issue #372)**: Refactor and finalize based on insights from parallel phases

All phases depend on this planning documentation being complete and approved.
