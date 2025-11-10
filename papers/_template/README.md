# Paper Implementation Template

This template provides a standardized structure for implementing research papers in Mojo. Copy this directory to start
a new paper implementation.

## Overview

This template is designed to ensure consistency across all paper implementations in the ML Odyssey project. It includes
all necessary directories, placeholder files, and documentation to help you quickly start implementing a research paper.

## Quick Start

To use this template for a new paper:

```bash
# Copy the template to a new directory
cp -r papers/_template papers/your-paper-name

# Navigate to the new directory
cd papers/your-paper-name

# Update this README with your paper details
# Start implementing in src/
```

## Directory Structure

```text
your-paper-name/
├── README.md                    # This file - update with paper details
├── src/                         # Mojo implementation code
│   ├── __init__.mojo           # Package initialization
│   └── .gitkeep                # Placeholder for empty directory
├── tests/                       # Test suite
│   ├── __init__.mojo           # Test package initialization
│   └── .gitkeep                # Placeholder for empty directory
├── data/                        # Data management
│   ├── raw/                    # Original, immutable datasets
│   ├── processed/              # Cleaned and transformed datasets
│   └── cache/                  # Cached computations
├── configs/                     # Configuration files
│   ├── config.yaml             # Example configuration
│   └── .gitkeep                # Placeholder for empty directory
├── notebooks/                   # Jupyter notebooks
│   └── .gitkeep                # Placeholder for empty directory
└── examples/                    # Demonstration scripts
    └── .gitkeep                # Placeholder for empty directory
```

## Directory Purposes

### src/

Contains the Mojo implementation of the paper's algorithms and models.

**Suggested structure**:

```text
src/
├── __init__.mojo               # Package exports
├── models/                      # Model architectures
│   ├── __init__.mojo
│   └── model.mojo              # Main model implementation
├── layers/                      # Custom layer implementations
│   ├── __init__.mojo
│   └── custom_layer.mojo       # Custom layers
├── utils/                       # Utility functions
│   ├── __init__.mojo
│   └── helpers.mojo            # Helper functions
└── data/                        # Data loading utilities
    ├── __init__.mojo
    └── loader.mojo             # Data loading logic
```

### tests/

Contains comprehensive tests for all implemented functionality.

**Test organization**:

```text
tests/
├── __init__.mojo               # Test package initialization
├── test_models.mojo            # Model tests
├── test_layers.mojo            # Layer tests
├── test_utils.mojo             # Utility function tests
└── test_data.mojo              # Data loading tests
```

**Testing principles**:

- Write unit tests for individual functions and classes
- Write integration tests for component interactions
- Use Test-Driven Development (TDD) when possible
- Aim for high code coverage (80%+)
- Include edge cases and error conditions

### data/

Manages datasets and processed data.

**Subdirectories**:

- `raw/` - Original, immutable datasets downloaded from source
- `processed/` - Cleaned, transformed, and ready-to-use datasets
- `cache/` - Cached computations and intermediate results

**Important notes**:

- Data files are NOT tracked in git (except .gitkeep files)
- Add data files to `.gitignore` in the paper directory
- Document data sources and preprocessing steps in this README
- Include download scripts or instructions in `examples/`

### configs/

Configuration files for experiments, training, and model hyperparameters.

**Example files**:

- `config.yaml` - Main configuration file
- `hyperparameters.yaml` - Model hyperparameters
- `experiment.yaml` - Experiment configurations

**Best practices**:

- Use YAML or JSON for configuration files
- Keep configurations separate from code
- Version control all configuration files
- Document all configuration options

### notebooks/

Jupyter notebooks for experimentation, visualization, and analysis.

**Suggested notebooks**:

- `exploration.ipynb` - Data exploration and visualization
- `training.ipynb` - Model training demonstrations
- `results.ipynb` - Results analysis and visualization
- `experiments.ipynb` - Experimental investigations

**Guidelines**:

- Keep notebooks focused and well-documented
- Clear all outputs before committing (optional: use pre-commit hook)
- Include markdown cells explaining the analysis
- Save final visualizations to files for inclusion in documentation

### examples/

Demonstration scripts showing how to use the implementation.

**Example scripts**:

- `train.mojo` - Training script
- `evaluate.mojo` - Evaluation script
- `inference.mojo` - Inference/prediction script
- `download_data.mojo` - Data download script

**Characteristics**:

- Self-contained and runnable
- Well-commented
- Demonstrate common use cases
- Include command-line argument parsing

## Implementation Guide

### Step 1: Update README

Replace this template README with paper-specific information:

1. Paper title, authors, and publication details
2. Brief description of the paper's contribution
3. Link to original paper (arXiv, conference, etc.)
4. Implementation notes and any deviations from the paper
5. Results comparison with original paper
6. Usage instructions specific to this implementation

### Step 2: Implement Models

Start with the core model architecture in `src/models/`:

1. Define model structure following the paper
2. Implement forward pass
3. Implement any custom layers in `src/layers/`
4. Add utility functions in `src/utils/`

### Step 3: Write Tests

Follow Test-Driven Development (TDD) principles:

1. Write tests before or alongside implementation
2. Test individual components (unit tests)
3. Test component interactions (integration tests)
4. Test edge cases and error conditions
5. Verify outputs match expected results

### Step 4: Data Handling

Implement data loading and preprocessing:

1. Create data download scripts in `examples/`
2. Implement preprocessing in `src/data/`
3. Document data sources and formats
4. Add data validation checks

### Step 5: Configuration

Set up configuration files:

1. Define model hyperparameters in `configs/`
2. Create experiment configurations
3. Document all configuration options
4. Provide sensible defaults

### Step 6: Examples

Create demonstration scripts:

1. Training script with argument parsing
2. Evaluation script for testing
3. Inference script for predictions
4. Data download and setup script

### Step 7: Documentation

Complete the documentation:

1. Update README with comprehensive information
2. Add docstrings to all functions and classes
3. Create notebooks demonstrating usage
4. Document results and comparisons

## Testing

Run tests using the project's test framework:

```bash
# Run all tests
mojo test tests/

# Run specific test file
mojo test tests/test_models.mojo

# Run with verbose output
mojo test tests/ --verbose
```

## Data Management

### Downloading Data

Create a download script in `examples/download_data.mojo`:

```mojo
# Example data download script
fn download_dataset() raises:
    """Download and prepare the dataset."""
    # Implementation here
    pass
```

### Data Preprocessing

Implement preprocessing in `src/data/loader.mojo`:

```mojo
# Example data loader
struct DataLoader:
    """Loads and preprocesses data for training."""

    fn __init__(inout self, config: Config):
        """Initialize data loader with configuration."""
        pass

    fn load_data(self) raises -> Data:
        """Load and preprocess data."""
        pass
```

## Configuration

Modify `configs/config.yaml` for your paper:

```yaml
# Paper-specific configuration
model:
  name: "your_model_name"
  architecture: "architecture_description"

training:
  batch_size: 32
  epochs: 10
  learning_rate: 0.001
```

## Common Patterns

### Model Implementation

```mojo
struct YourModel:
    """Model implementation following the paper."""

    var config: ModelConfig

    fn __init__(inout self, config: ModelConfig):
        """Initialize model with configuration."""
        self.config = config

    fn forward(self, input: Tensor) raises -> Tensor:
        """Forward pass through the model."""
        # Implementation
        pass
```

### Training Loop

```mojo
fn train(model: YourModel, data: DataLoader, config: TrainConfig) raises:
    """Train the model."""
    for epoch in range(config.epochs):
        for batch in data:
            # Training step
            pass
```

## Contributing

When extending this implementation:

1. Follow the existing code structure
2. Write tests for new functionality
3. Update documentation
4. Follow Mojo best practices
5. Keep code modular and reusable

## References

- Original Paper: [Add link to paper]
- Paper Authors: [Add author names]
- Publication: [Add publication details]
- Code Repository: [Add link if available]

## License

See the main repository LICENSE file.

## Paper-Specific Information

> NOTE: Replace this section with paper-specific details

### Paper Title

[Add paper title]

### Authors

[Add paper authors]

### Publication Details

[Add publication venue and date]

### Abstract

[Add paper abstract or brief description]

### Key Contributions

[List the paper's key contributions]

### Implementation Notes

[Add any notes about the implementation, deviations from the paper, or design decisions]

### Results

[Add results and comparison with original paper]

### Usage Example

[Add a simple usage example]

```mojo
# Example usage
from src.models.model import YourModel

fn main() raises:
    let model = YourModel(config)
    # Use the model
    pass
```

### Citation

If you use this implementation, please cite the original paper:

```bibtex
@article{...,
  title={...},
  author={...},
  journal={...},
  year={...}
}
```
