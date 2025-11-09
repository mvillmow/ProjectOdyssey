# papers/README.md - Content Templates and Examples

This document provides reusable templates and examples for the papers/README.md content.

## Template 1: Paper Directory Structure Section

Each paper is organized in its own directory with a standardized structure:

```text
papers/
├── README.md              # This file - guides for implementing papers
├── lenet-5/               # First paper implementation (proof of concept)
│   ├── README.md          # Paper-specific overview and usage
│   ├── src/               # Implementation source code (Mojo)
│   │   ├── main.mojo      # Primary implementation
│   │   ├── layers.mojo    # Layer implementations
│   │   └── utils.mojo     # Utility functions
│   ├── tests/             # Comprehensive test suite
│   │   ├── conftest.mojo  # Test configuration and fixtures
│   │   ├── test_main.mojo # Tests for main module
│   │   └── test_layers.mojo
│   ├── docs/              # Paper-specific documentation
│   │   ├── IMPLEMENTATION.md    # Implementation notes and decisions
│   │   ├── RESEARCH_PAPER.md    # Paper summary and background
│   │   └── ARCHITECTURE.md      # Design decisions and trade-offs
│   ├── data/              # Sample datasets (optional)
│   │   └── mnist/         # Example: MNIST data samples
│   ├── examples/          # Usage examples (optional)
│   │   └── train.mojo     # Example: training script
│   └── pyproject.toml     # Project metadata and dependencies
├── alexnet/               # Additional paper implementations
│   └── ...
└── .gitkeep              # Git placeholder for empty directory
```

### Directory Breakdown

#### Paper-Specific README

Paper-specific documentation should include:

- Paper title, authors, and citation
- Brief overview of what is implemented
- Quick start instructions
- Usage examples
- Results and benchmarks
- Known limitations
- Links to paper-specific docs

**Example**:

```markdown
# LeNet-5: Gradient-based Learning for Document Recognition

**Paper**: LeCun et al., 1998 - "Gradient-based learning applied to
document recognition"

## Overview

This is a complete implementation of LeNet-5 in Mojo...
```

#### Source Code Organization

All implementation code in Mojo, organized by functionality:

- Core algorithms and neural network layers
- Data loading and preprocessing
- Model training and inference
- Utilities and helper functions

#### Test Suite

Comprehensive test suite following TDD approach:

- `conftest.mojo` - Shared fixtures and test configuration
- `test_*.mojo` - Unit tests organized by module
- Test data and fixtures
- Performance benchmarks

#### Detailed Documentation

Detailed documentation for the implementation:

- **IMPLEMENTATION.md** - How we implemented it
- **RESEARCH_PAPER.md** - Summary of the original research
- **ARCHITECTURE.md** - Design decisions and alternatives considered
- **PERFORMANCE.md** - Benchmarks and optimization notes (optional)

#### Sample Data

Sample data for development and testing:

- Small datasets for quick testing
- Data loading scripts
- Data format documentation
- NOT large datasets (use external references)

#### Examples

Practical examples showing how to use the implementation:

- Training scripts
- Inference examples
- Visualization code
- Performance demonstrations

#### Project Metadata

Project metadata and dependency management:

```toml
[project]
name = "lenet-5"
version = "0.1.0"
description = "LeNet-5 implementation in Mojo"

[tool.mojo]
requires-mojo = ">=0.5.0"

[tool.pytest]
addopts = "--cov=src"
```

---

## Template 2: Adding a New Paper - Step-by-Step Guide

This step-by-step guide walks you through implementing a research paper in ML Odyssey.

### Phase 0: Pre-Implementation Planning

Before you start coding, invest time in understanding the research:

#### Research and Design (2-3 hours)

1. **Read and understand the paper**
   - Read the abstract and introduction
   - Understand the core algorithms
   - Review key equations and mathematical concepts
   - Check existing implementations in PyTorch, TensorFlow, etc.

2. **Plan your implementation**
   - Create a document outlining:
     - Main components and modules
     - Public API (functions and classes)
     - Expected inputs/outputs
     - Dependencies needed
     - Estimated scope and complexity

3. **Create a GitHub issue** (Plan Phase)
   - Title: `[Plan] Implement <PaperName>`
   - Include planning document
   - Identify 5-phase issues needed

#### Create Initial Structure

```bash
# Navigate to papers directory
cd papers/

# Create paper directory
mkdir my-paper
cd my-paper

# Create subdirectories
mkdir src tests docs examples data

# Create initial files
touch src/main.mojo
touch tests/conftest.mojo
touch tests/test_main.mojo
touch README.md
touch pyproject.toml
```

### Phase 1: Plan (GitHub Issue #N-1)

**Objective**: Design the implementation approach

**Deliverables**:

1. Implementation plan document
   - Module structure
   - Public APIs
   - Testing strategy
   - Documentation outline

2. Setup PR with:
   - Directory structure
   - Stub files (empty implementations)
   - Initial README.md template
   - Initial tests with fixtures

**Success Criteria**:

- ✅ Implementation plan documented
- ✅ Module structure defined
- ✅ Public APIs specified
- ✅ Tests outline created
- ✅ Directory structure in place

### Phase 2: Test (GitHub Issue #N-3)

**Objective**: Write comprehensive test suite before implementation

**Deliverables**:

1. Complete test suite
   - Unit tests for each module
   - Integration tests
   - Edge case handling
   - Error condition testing

2. Test data and fixtures
   - Sample inputs
   - Expected outputs
   - Benchmark datasets

3. Test documentation
   - How to run tests
   - Coverage report
   - Performance baselines

**Success Criteria**:

- ✅ All modules have test coverage
- ✅ Tests cover edge cases
- ✅ Test fixtures are reusable
- ✅ Coverage target: 80%+

### Phase 3: Implementation (GitHub Issue #N-4)

**Objective**: Implement modules to pass tests

**Deliverables**:

1. Complete implementation
   - All modules coded in Mojo
   - Pass all existing tests
   - Full docstrings
   - Type hints throughout

2. Code quality
   - Passes `mojo format`
   - Passes linting checks
   - Performance optimized
   - Clear variable names

3. Implementation notes
   - Document challenges overcome
   - Alternative approaches considered
   - Known limitations

**Success Criteria**:

- ✅ All tests pass
- ✅ Code formatted correctly
- ✅ 100% of public APIs documented
- ✅ Type hints complete

### Phase 4: Packaging (GitHub Issue #N-5)

**Objective**: Integrate, document, and optimize

**Deliverables**:

1. Integration
   - Works with build system
   - Dependencies resolved
   - CI/CD passes

2. Comprehensive documentation
   - IMPLEMENTATION.md - How it was built
   - RESEARCH_PAPER.md - Paper background
   - ARCHITECTURE.md - Design decisions
   - Usage examples

3. Performance optimization
   - Benchmark results
   - Memory profiling
   - Optimization notes

4. Examples and demos
   - Working examples
   - Training/inference scripts
   - Result demonstrations

**Success Criteria**:

- ✅ All documentation written
- ✅ Examples work correctly
- ✅ Performance benchmarked
- ✅ Ready for code review

### Phase 5: Cleanup (GitHub Issue #N-6)

**Objective**: Finalize and prepare for release

**Deliverables**:

1. Address review feedback
   - Code improvements
   - Documentation updates
   - Performance adjustments

2. Final refinements
   - Markdown formatting validated
   - All links checked
   - Examples updated

3. Prepare for release
   - Update main README
   - Add to examples section
   - Document lessons learned

**Success Criteria**:

- ✅ All review comments addressed
- ✅ Markdown validation passes
- ✅ All examples work
- ✅ Ready to merge

### Running the 5-Phase Workflow

All five phases follow this structure:

1. **Create GitHub issues** (one per phase)
   - Title format: `[Phase] Implement <PaperName>`
   - Labels: planning, testing, implementation, packaging, cleanup
   - Reference parent issue

2. **Work in dedicated branch**
   - Branch name: `<issue-number>-<paper-name>`
   - Create PR referencing issue

3. **Complete phase deliverables**
   - Make commits following conventional format
   - Run pre-commit hooks: `pre-commit run --all-files`
   - Submit for review

4. **Move to next phase**
   - Test, Implementation, and Packaging can run in parallel
   - Cleanup runs last, collecting feedback from all phases

**Timeline**: 2-4 weeks depending on paper complexity

---

## Template 3: Code Standards Section

All paper implementations follow these standards to ensure consistency and quality.

### Mojo Code Style

#### File Organization

```mojo
# 1. File header with description
"""Module for neural network layers.

This module implements common layer types used in neural networks.
"""

# 2. Imports (grouped: stdlib, third-party, local)
from sys import argv
from math import sqrt

from ml_odyssey.tensor import Tensor

# 3. Type definitions and globals
alias Dtype = DType.float32
var EPSILON: Dtype = 1e-5

# 4. Main implementations
fn create_layer(layer_type: String) -> Layer:
    """Create a neural network layer of specified type.

    Args:
        layer_type: Type of layer to create (e.g., "conv", "dense")

    Returns:
        Layer: A new layer of the specified type
    """
    ...
```

#### Naming Conventions

- **Functions**: `snake_case` - `create_tensor`, `forward_pass`
- **Classes/Types**: `PascalCase` - `ConvLayer`, `DenseNetwork`
- **Constants**: `UPPER_CASE` - `MAX_BATCH_SIZE`, `LEARNING_RATE`
- **Private functions**: Prefix with `_` - `_init_weights`

#### Documentation

Every public function and class must have docstrings:

```mojo
fn softmax(x: Tensor, axis: Int = -1) -> Tensor:
    """Compute softmax activation.

    Applies softmax normalization along specified axis.

    Args:
        x: Input tensor of any shape
        axis: Axis along which to compute softmax (default: -1)

    Returns:
        Tensor: Softmax probabilities with same shape as input

    Raises:
        ValueError: If axis is out of bounds
    """
    ...
```

#### Type Hints

Always use explicit type hints:

```mojo
# Good
fn multiply(a: Tensor, b: Tensor, factor: Float32) -> Tensor:
    return a * b * factor

# Bad
fn multiply(a, b, factor):
    return a * b * factor
```

### Testing Standards

#### Test File Organization

```mojo
# tests/test_layers.mojo

"""Tests for neural network layers."""

from mojo.testing import assert_equal
from src.layers import DenseLayer, ConvLayer

fn test_dense_layer_forward():
    """Test forward pass through dense layer."""
    let layer = DenseLayer(input_size=10, output_size=5)
    let input = Tensor(shape=[2, 10])
    let output = layer.forward(input)

    assert_equal(output.shape, [2, 5])

fn test_conv_layer_with_padding():
    """Test convolution with padding."""
    ...
```

#### Naming and Organization

- Test files: `test_<module>.mojo`
- Test functions: `test_<feature>_<condition>`
- Use descriptive names: `test_dense_layer_with_relu` not `test1`
- Group related tests with comments: `# Input validation tests`

#### Coverage Goals

- Minimum 80% code coverage
- All public APIs tested
- Edge cases covered
- Error conditions tested

### Documentation Standards

#### Markdown Requirements

All `.md` files must follow these rules:

1. **Line Length**: Maximum 120 characters
2. **Code Blocks**: Always specify language
3. **Blank Lines**: Around headings, lists, code blocks
4. **Heading Hierarchy**: Use proper nesting (H1 → H2 → H3)
5. **Lists**: Blank lines before and after

**Valid Code Block**:

Here is some implementation:

```mojo
fn hello():
    print("world")
```

More text here.

**Invalid Code Block**:

Here is some implementation:

```text
fn hello():
```

More text here.

### Performance Standards

- Profile critical paths
- Document algorithmic complexity
- Include benchmarks in PERFORMANCE.md
- Optimize for readability first, speed second
- Use Mojo's performance features when appropriate

### Documentation Requirements

Each paper implementation must include:

1. **README.md**
   - Paper citation and authors
   - Quick start guide
   - Usage examples
   - Results and performance

2. **IMPLEMENTATION.md**
   - Architecture overview
   - Module descriptions
   - Key algorithms explained
   - Known limitations

3. **RESEARCH_PAPER.md**
   - Paper abstract
   - Key contributions
   - Mathematical background
   - Links to resources

4. **ARCHITECTURE.md**
   - Design decisions
   - API documentation
   - Dependencies
   - Trade-offs considered

---

## Template 4: Paper Implementation Checklist

Before submitting your paper implementation for review:

### Code Quality

- [ ] All tests pass: `mojo test tests/`
- [ ] Code formatted: `mojo format src/`
- [ ] No linting errors: `pre-commit run --all-files`
- [ ] Type hints on all public APIs
- [ ] Docstrings on all public functions

### Testing

- [ ] Test coverage >= 80%: `coverage report`
- [ ] All edge cases tested
- [ ] Error conditions handled
- [ ] Integration tests included
- [ ] Fixtures in conftest.mojo

### Documentation

- [ ] README.md complete with examples
- [ ] IMPLEMENTATION.md explains design
- [ ] RESEARCH_PAPER.md summarizes paper
- [ ] ARCHITECTURE.md documents decisions
- [ ] All links valid and working
- [ ] Code examples run without errors

### Repository Standards

- [ ] Markdown validates: `npx markdownlint-cli2 docs/*.md`
- [ ] No lines exceed 120 characters
- [ ] Proper heading hierarchy
- [ ] Blank lines around sections
- [ ] Git history is clean

### Performance

- [ ] Benchmarks documented in PERFORMANCE.md
- [ ] Performance profile completed
- [ ] Memory usage measured
- [ ] Optimization notes included

### Final Review

- [ ] Code reviewed by team member
- [ ] All feedback addressed
- [ ] PR description complete
- [ ] Related issues linked
- [ ] Ready for merge!

---

## Example: LeNet-5 Paper Entry

This example shows how a completed paper would be referenced in the README.

The section should include:

- Paper name and title
- Implementation status and date
- Citation information
- What is implemented (bulleted list)
- Quick start instructions with code example
- Key features and characteristics
- Links to detailed documentation

Here's what this section looks like:

```text
### LeNet-5: Gradient-based Learning for Document Recognition

**Status**: ✅ Complete (2025-11-30)

**Paper**: LeCun et al., 1998 - "Gradient-based learning applied to
document recognition"

**What's Implemented**:
- Convolutional layers with learnable filters
- Pooling layers for dimensionality reduction
- Fully connected layers for classification
- Training on MNIST digit dataset
- Full test suite with 85%+ coverage

**Quick Start**:
cd papers/lenet-5
mojo src/train.mojo  # Train the model
mojo src/inference.mojo  # Run inference

**Key Features**:
- Pure Mojo implementation
- MNIST training and evaluation
- Achieves 98%+ accuracy
- Documented architecture decisions
- Complete test coverage

**See Also**:
- [Implementation Details](lenet-5/docs/IMPLEMENTATION.md)
- [Paper Summary](lenet-5/docs/RESEARCH_PAPER.md)
- [Architecture Notes](lenet-5/docs/ARCHITECTURE.md)
```

---

## Markdown Validation Checklist

Use this checklist when writing content for papers/README.md:

- [ ] No lines exceed 120 characters
- [ ] All code blocks have language specified
- [ ] Blank line before code blocks
- [ ] Blank line after code blocks
- [ ] Blank line before headings
- [ ] Blank line after headings
- [ ] Lists have blank lines before
- [ ] Lists have blank lines after
- [ ] Proper heading hierarchy (no skipped levels)
- [ ] No trailing whitespace
- [ ] File ends with newline
- [ ] All links are valid
- [ ] No orphaned sections

**Validation Command**:

```bash
npx markdownlint-cli2 papers/README.md
```
