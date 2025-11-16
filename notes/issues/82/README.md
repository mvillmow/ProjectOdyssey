# Issue #82: [Plan] Directory Structure - Design and Documentation

## Objective

Design and document the comprehensive directory structure for `papers/` and `shared/` directories, establishing clear API contracts, integration patterns, and architectural guidelines for the ML Odyssey repository.

## Summary

This planning phase defines the complete directory structure and organization for:
- **Papers Directory**: Individual paper implementations with consistent structure
- **Shared Directory**: Reusable components across all paper implementations
- **API Contracts**: Clear interfaces between papers and shared code
- **Integration Patterns**: How components interact and depend on each other

## Deliverables

### 1. Directory Structure Specifications
- ✅ Papers directory template and organization
- ✅ Shared library component organization
- ✅ Clear separation of concerns
- ✅ Dependency flow documentation

### 2. API Contracts
- ✅ Interface definitions for shared components
- ✅ Data flow contracts between layers
- ✅ Type specifications for Mojo implementations
- ✅ Integration points documentation

### 3. Architectural Design Documents
- ✅ Component interaction patterns
- ✅ Dependency management strategy
- ✅ Performance considerations
- ✅ Migration and refactoring guidelines

## Success Criteria

- [x] Complete planning for papers/ and shared/ structure
- [x] Template structure designed for new papers
- [x] API contracts documented
- [x] Integration strategy defined
- [x] Clear separation between papers and shared code
- [x] Documentation complete in `/notes/issues/82/README.md`

## References

- [Shared Library README](/home/user/ml-odyssey/shared/README.md)
- [Papers Template README](/home/user/ml-odyssey/papers/_template/README.md)
- [Core Library Documentation](/home/user/ml-odyssey/shared/core/README.md)
- [Training Library Documentation](/home/user/ml-odyssey/shared/training/README.md)
- [Data Library Documentation](/home/user/ml-odyssey/shared/data/README.md)

## Implementation Notes

### Current State Analysis

The repository already has well-established structures for both `papers/` and `shared/` directories:

1. **Papers Directory** (`/papers/`)
   - Contains `_template/` for standardized paper implementations
   - Provides consistent structure for all paper implementations
   - Includes comprehensive documentation template

2. **Shared Directory** (`/shared/`)
   - `core/` - Fundamental ML/AI building blocks
   - `training/` - Training infrastructure and utilities
   - `data/` - Data loading and processing
   - `utils/` - Helper utilities and tools

3. **Supporting Directories** (from Issues #77-81)
   - `benchmarks/` - Performance testing
   - `docs/` - Project documentation
   - `agents/` - Claude agent configurations
   - `tools/` - Development utilities
   - `configs/` - Configuration management

---

## 1. Papers Directory Structure

### 1.1 Overall Organization

```text
papers/
├── README.md                    # Papers overview and index
├── _template/                   # Template for new papers (DO NOT MODIFY)
│   ├── README.md               # Template documentation
│   ├── src/                    # Mojo implementation
│   ├── scripts/                # Paper, dataset, reference downloads
│   ├── tests/                  # Test suite
│   ├── data/                   # Data management
│   ├── configs/                # Configuration files
│   ├── notebooks/              # Jupyter notebooks
│   └── examples/               # Usage examples
└── <paper-name>/               # Individual paper implementations
    ├── README.md               # Paper-specific documentation
    ├── paper.pdf               # Original paper (via git-lfs)
    ├── src/                    # Implementation code
    ├── scripts/                # Setup and download scripts
    ├── tests/                  # Paper-specific tests
    ├── data/                   # Paper-specific data
    ├── configs/                # Training configurations
    ├── notebooks/              # Experiments and analysis
    └── examples/               # Demonstration scripts
```

### 1.2 Paper Implementation Structure

Each paper follows this standardized structure:

#### Source Code (`src/`)
```text
src/
├── __init__.mojo               # Package exports
├── model.mojo                  # Main model implementation
├── layers/                     # Paper-specific layers
│   ├── __init__.mojo
│   └── custom_layer.mojo       # Novel layers from paper
├── loss.mojo                   # Paper-specific loss functions
├── metrics.mojo                # Paper-specific metrics
└── utils.mojo                  # Paper-specific utilities
```

#### Scripts (`scripts/`)
```text
scripts/
├── download_paper.mojo         # Download original paper PDF
├── download_dataset.mojo       # Download required datasets
├── download_reference.mojo     # Download reference implementation
└── setup.mojo                  # Complete setup script
```

#### Tests (`tests/`)
```text
tests/
├── __init__.mojo               # Test package initialization
├── test_model.mojo             # Model architecture tests
├── test_layers.mojo            # Custom layer tests
├── test_training.mojo          # Training pipeline tests
├── test_metrics.mojo           # Metric computation tests
└── test_integration.mojo       # End-to-end tests
```

#### Data Management (`data/`)
```text
data/
├── raw/                        # Original datasets (git-ignored)
├── processed/                  # Preprocessed data (git-ignored)
├── cache/                      # Cached computations (git-ignored)
└── .gitignore                  # Ignore data files
```

#### Configuration (`configs/`)
```text
configs/
├── base.yaml                   # Base configuration
├── train.yaml                  # Training configuration
├── eval.yaml                   # Evaluation configuration
└── hparams/                    # Hyperparameter sweeps
    ├── learning_rate.yaml
    └── batch_size.yaml
```

### 1.3 Paper Naming Conventions

Papers should be named using this convention:
- Format: `<year>-<short-name>` or just `<short-name>`
- Examples: `1998-lenet5`, `2012-alexnet`, `resnet`, `transformer`
- Use lowercase with hyphens (kebab-case)
- Keep names concise but recognizable

---

## 2. Shared Library Structure

### 2.1 Overall Organization

```text
shared/
├── README.md                   # Shared library overview
├── __init__.mojo               # Package root exports
├── core/                       # Fundamental components
│   ├── README.md
│   ├── __init__.mojo
│   ├── layers/                 # Neural network layers
│   ├── ops/                    # Mathematical operations
│   ├── types/                  # Data types and structures
│   └── utils/                  # Core utilities
├── training/                   # Training infrastructure
│   ├── README.md
│   ├── __init__.mojo
│   ├── optimizers/             # Optimization algorithms
│   ├── schedulers/             # Learning rate scheduling
│   ├── metrics/                # Evaluation metrics
│   ├── callbacks/              # Training callbacks
│   └── loops/                  # Training loop patterns
├── data/                       # Data processing
│   ├── README.md
│   ├── __init__.mojo
│   ├── datasets.mojo           # Dataset abstractions
│   ├── loaders.mojo            # Data loaders
│   ├── samplers.mojo           # Sampling strategies
│   └── transforms.mojo         # Data transformations
└── utils/                      # General utilities
    ├── README.md
    ├── __init__.mojo
    ├── logging.mojo            # Logging utilities
    ├── visualization.mojo      # Plotting and visualization
    ├── io.mojo                 # File I/O helpers
    └── profiling.mojo          # Performance profiling
```

### 2.2 Component Details

#### Core Components (`core/`)
```text
core/
├── layers/
│   ├── linear.mojo             # Fully connected layers
│   ├── conv.mojo               # Convolutional layers (1D, 2D, 3D)
│   ├── pooling.mojo            # Pooling layers (Max, Avg, Global)
│   ├── activation.mojo         # Activation functions (ReLU, Sigmoid, Tanh)
│   ├── normalization.mojo      # Batch/Layer/Instance normalization
│   ├── dropout.mojo            # Dropout layers
│   └── attention.mojo          # Attention mechanisms
├── ops/
│   ├── matmul.mojo             # Matrix multiplication
│   ├── conv_ops.mojo           # Convolution operations
│   ├── elementwise.mojo        # Element-wise operations
│   ├── reduction.mojo          # Reduction operations
│   └── broadcast.mojo          # Broadcasting utilities
├── types/
│   ├── tensor.mojo             # Tensor implementation
│   ├── dtype.mojo              # Data type definitions
│   ├── shape.mojo              # Shape utilities
│   └── device.mojo             # Device management (CPU/GPU)
└── utils/
    ├── init.mojo               # Weight initialization
    ├── memory.mojo             # Memory management
    ├── module.mojo             # Module base class
    └── parameter.mojo          # Parameter management
```

#### Training Components (`training/`)
```text
training/
├── optimizers/
│   ├── base.mojo               # Optimizer interface
│   ├── sgd.mojo                # SGD with momentum
│   ├── adam.mojo               # Adam optimizer
│   ├── adamw.mojo              # AdamW (Adam with weight decay)
│   ├── rmsprop.mojo            # RMSprop optimizer
│   └── utils.mojo              # Gradient clipping, etc.
├── schedulers/
│   ├── base.mojo               # Scheduler interface
│   ├── step.mojo               # Step decay
│   ├── cosine.mojo             # Cosine annealing
│   ├── exponential.mojo        # Exponential decay
│   └── warmup.mojo             # Learning rate warmup
├── metrics/
│   ├── base.mojo               # Metric interface
│   ├── accuracy.mojo           # Classification accuracy
│   ├── loss.mojo               # Loss tracking
│   ├── confusion.mojo          # Confusion matrix
│   └── regression.mojo         # MSE, MAE, R²
├── callbacks/
│   ├── base.mojo               # Callback interface
│   ├── checkpoint.mojo         # Model checkpointing
│   ├── early_stopping.mojo     # Early stopping
│   ├── logging.mojo            # Training logging
│   └── tensorboard.mojo        # TensorBoard integration
└── loops/
    ├── base.mojo               # Training loop interface
    ├── supervised.mojo          # Standard supervised training
    ├── validation.mojo          # Training with validation
    └── distributed.mojo         # Distributed training
```

---

## 3. API Contracts and Interfaces

### 3.1 Core Interfaces

#### Module Interface
```mojo
trait Module:
    """Base interface for all neural network modules."""
    
    fn forward(self, input: Tensor) -> Tensor:
        """Forward pass through the module."""
        ...
    
    fn parameters(self) -> List[Parameter]:
        """Return all trainable parameters."""
        ...
    
    fn train(inout self, mode: Bool = True):
        """Set training mode."""
        ...
    
    fn eval(inout self):
        """Set evaluation mode."""
        ...
```

#### Layer Interface
```mojo
trait Layer(Module):
    """Base interface for neural network layers."""
    
    fn reset_parameters(inout self):
        """Reset layer parameters."""
        ...
    
    fn extra_repr(self) -> String:
        """Extra representation string."""
        ...
```

### 3.2 Training Interfaces

#### Optimizer Interface
```mojo
trait Optimizer:
    """Base interface for optimizers."""
    
    fn step(inout self, parameters: List[Parameter], gradients: List[Tensor]):
        """Perform single optimization step."""
        ...
    
    fn zero_grad(inout self):
        """Zero all gradients."""
        ...
    
    fn state_dict(self) -> Dict[String, Any]:
        """Return optimizer state."""
        ...
```

#### Dataset Interface
```mojo
trait Dataset:
    """Base interface for datasets."""
    
    fn __len__(self) -> Int:
        """Return dataset size."""
        ...
    
    fn __getitem__(self, index: Int) -> Tuple[Tensor, Tensor]:
        """Get item at index."""
        ...
```

### 3.3 Data Flow Contracts

#### Tensor Shape Conventions
- **Images**: `[batch, channels, height, width]` (NCHW format)
- **Sequences**: `[batch, sequence_length, features]`
- **Tabular**: `[batch, features]`
- **Labels**: `[batch]` for classification, `[batch, targets]` for regression

#### Type Specifications
- **Default dtype**: `float32` for computations
- **Integer types**: `int64` for indices, `int32` for counts
- **Boolean types**: `bool` for masks and flags

---

## 4. Integration Patterns

### 4.1 Paper-Shared Integration

Papers import from shared library:
```mojo
# In papers/lenet5/src/model.mojo
from shared.core.layers import Conv2D, Linear, ReLU
from shared.training.optimizers import SGD
from shared.data import DataLoader, transforms

struct LeNet5:
    var conv1: Conv2D  # From shared
    var custom: CustomLayer  # Paper-specific
```

### 4.2 Dependency Flow

```text
Papers depend on → Shared Library
                   ├── Core (fundamental)
                   ├── Training (infrastructure)
                   ├── Data (processing)
                   └── Utils (helpers)

Shared components have minimal inter-dependencies:
Core ← Training, Data, Utils
Training → Core
Data → Core, Utils
Utils → Core
```

### 4.3 Extension Points

Papers can extend shared components:
1. **Custom Layers**: Inherit from `Layer` trait
2. **Custom Optimizers**: Implement `Optimizer` interface
3. **Custom Datasets**: Implement `Dataset` interface
4. **Custom Transforms**: Implement `Transform` interface

---

## 5. Migration and Refactoring Guidelines

### 5.1 When to Move Code to Shared

Code should be moved from papers to shared when:
- ✅ Used by 3+ paper implementations
- ✅ Represents a standard ML component
- ✅ Has a stable, well-defined interface
- ✅ Is properly tested and documented

### 5.2 Refactoring Process

1. **Identify Reusable Components**
   - Review multiple paper implementations
   - Find common patterns and duplicated code

2. **Design Generic Interface**
   - Remove paper-specific logic
   - Create flexible, composable API

3. **Implement in Shared**
   - Add comprehensive tests
   - Write detailed documentation
   - Include usage examples

4. **Migrate Papers**
   - Update papers to use shared component
   - Remove duplicated code
   - Test thoroughly

5. **Document Changes**
   - Update both READMEs
   - Add migration guide if breaking changes
   - Update examples

---

## 6. Performance Considerations

### 6.1 Optimization Guidelines

#### Mojo-Specific Optimizations
- Use `fn` for performance-critical functions
- Leverage SIMD for vectorizable operations
- Use `@always_inline` for hot paths
- Prefer stack allocation when possible
- Use `owned`/`borrowed` for memory safety

#### Data Loading Performance
- Implement lazy loading for large datasets
- Use prefetching to overlap I/O with computation
- Cache transformed data when possible
- Batch transforms for efficiency

### 6.2 Performance Targets

Shared components should achieve:
- Matrix operations: Within 2x of BLAS performance
- Data loading: No bottleneck on training
- Memory usage: Minimal allocations in hot paths
- Inference: Competitive with PyTorch/TensorFlow

---

## 7. Testing Requirements

### 7.1 Shared Library Testing

All shared components must have:
- ✅ Unit tests (≥90% coverage)
- ✅ Integration tests
- ✅ Performance benchmarks
- ✅ Numerical accuracy tests
- ✅ Edge case handling

### 7.2 Paper Implementation Testing

Each paper should include:
- ✅ Model architecture tests
- ✅ Training pipeline tests
- ✅ Accuracy verification
- ✅ Comparison with reference implementation
- ✅ End-to-end integration tests

---

## 8. Documentation Standards

### 8.1 Shared Library Documentation

Each component needs:
- **API Documentation**: Complete docstrings
- **Usage Examples**: Code snippets
- **Performance Notes**: Complexity, memory usage
- **Design Rationale**: Why decisions were made

### 8.2 Paper Documentation

Each paper requires:
- **Paper Summary**: Key contributions
- **Implementation Notes**: Deviations, design choices
- **Results**: Comparison with original
- **Usage Guide**: How to train and evaluate
- **Citations**: BibTeX entry

---

## 9. Quality Assurance Checklist

### Before Adding to Shared
- [ ] Used by 3+ papers or standard component?
- [ ] Has comprehensive tests?
- [ ] Well-documented with examples?
- [ ] Performance benchmarked?
- [ ] Reviewed by team?

### Before Implementing Paper
- [ ] Template structure copied?
- [ ] README updated with paper details?
- [ ] Dependencies identified?
- [ ] Test plan created?
- [ ] Integration points documented?

---

## 10. Future Enhancements

### Planned Improvements
1. **Distributed Training Support**
   - Data parallel training
   - Model parallel training
   - Pipeline parallelism

2. **Advanced Data Loading**
   - Multi-worker loading
   - Intelligent prefetching
   - Dynamic batching

3. **Model Zoo Integration**
   - Pre-trained weights management
   - Model hub integration
   - Transfer learning utilities

4. **Profiling and Debugging**
   - Performance profiler integration
   - Memory profiler
   - Gradient debugging tools

---

## Conclusion

This planning document establishes a comprehensive structure for the ML Odyssey repository's `papers/` and `shared/` directories. The design ensures:

1. **Consistency**: All papers follow the same structure
2. **Reusability**: Common components are shared
3. **Performance**: Optimized for ML workloads
4. **Maintainability**: Clear separation of concerns
5. **Extensibility**: Easy to add new papers and components

The established API contracts and integration patterns provide a solid foundation for implementing classic ML research papers in Mojo while maintaining code quality and performance standards.

## Next Steps

With this planning complete, the implementation phases can proceed:
1. **Test Phase**: Write tests for directory structure validation
2. **Implementation Phase**: Create the actual directories and files
3. **Package Phase**: Create distributable packages
4. **Cleanup Phase**: Refine and optimize based on usage

The foundation is now established for a well-organized, high-performance ML research repository.
