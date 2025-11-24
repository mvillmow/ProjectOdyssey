# Issue #526: [Plan] Create Template - Design and Documentation

## Objective

Define the comprehensive specifications and design architecture for creating a standardized template directory structure that can be copied for each new paper implementation. The template will include standard subdirectories, placeholder files, and comprehensive documentation explaining how to use it effectively.

## Deliverables

- `papers/_template/` directory with complete structure
- Standard subdirectories: `src/`, `tests/`, `data/`, `configs/`, `notebooks/`, `examples/`, `scripts/`
- Template `README.md` explaining the structure, usage patterns, and implementation guide
- Placeholder files demonstrating expected organization (`.gitkeep`, `__init__.mojo`, example scripts)
- Configuration file templates (`config.yaml` with common settings)

## Success Criteria

- [ ] `_template/` directory exists in `papers/`
- [ ] All standard subdirectories are created with appropriate structure
- [ ] Template README explains usage clearly with comprehensive examples
- [ ] Template can be copied easily for new papers (simple `cp -r` command)
- [ ] Documentation includes implementation guide, testing strategy, and best practices
- [ ] Placeholder files demonstrate expected patterns for Mojo code organization
- [ ] Configuration templates show common settings and options

## Design Decisions

### 1. Directory Structure

**Decision**: Use a flat, well-organized structure with clear separation of concerns

### Rationale

- `src/` - Core implementation code (models, layers, utilities)
- `tests/` - Comprehensive test suite following TDD principles
- `data/` - Data management with `raw/`, `processed/`, and `cache/` subdirectories
- `configs/` - Configuration files separate from code (YAML/JSON)
- `notebooks/` - Jupyter notebooks for experimentation and visualization
- `examples/` - Demonstration scripts showing common use cases
- `scripts/` - Utility scripts for downloading papers, datasets, and reference implementations

### Alternatives Considered

- Nested structure (e.g., `src/models/`, `src/layers/`) - Decided to keep top-level flat for simplicity
- Combined data/configs directory - Separated for clarity and to avoid confusion

### 2. Placeholder Files

**Decision**: Include `.gitkeep` files for empty directories and `__init__.mojo` for package initialization

### Rationale

- `.gitkeep` ensures empty directories are tracked in git
- `__init__.mojo` demonstrates Mojo package structure
- Example scripts (`train.mojo`) show common patterns and best practices

### Alternatives Considered

- No placeholder files - Would lose directory structure in git
- More example files - Decided to keep minimal to avoid clutter

### 3. Documentation Strategy

**Decision**: Comprehensive README with multiple sections: Quick Start, Directory Structure, Implementation Guide, Testing, Common Patterns

### Rationale

- Quick Start section allows immediate usage
- Directory Structure explains purpose of each directory
- Implementation Guide provides step-by-step instructions
- Testing section emphasizes TDD principles
- Common Patterns show Mojo-specific code examples

### Key Sections

1. Overview - Brief introduction
1. Quick Start - Copy command and basic setup
1. Directory Structure - Visual tree and descriptions
1. Directory Purposes - Detailed explanation of each directory
1. Implementation Guide - 7-step process from README to documentation
1. Testing - Test organization and principles
1. Data Management - Download and preprocessing patterns
1. Configuration - YAML structure and best practices
1. Common Patterns - Code examples for models and training
1. Paper-Specific Information - Template section to be replaced

### 4. Configuration Template

**Decision**: Provide `config.yaml` with common sections: model, training, data, paths, experiment

### Rationale

- YAML is human-readable and widely used in ML projects
- Common sections cover typical paper implementation needs
- Clear structure makes it easy to extend for specific papers
- Sensible defaults reduce friction for new implementations

### Sections

- `model` - Model architecture configuration
- `training` - Training hyperparameters (batch size, epochs, learning rate)
- `data` - Dataset configuration and splits
- `paths` - Directory paths for data, checkpoints, results
- `experiment` - Experiment tracking and metadata

### 5. Mojo-First Approach

**Decision**: Use `.mojo` extensions for all code files and demonstrate Mojo patterns

### Rationale

- Project uses Mojo as primary language (see ADR-001)
- Template should reflect project standards
- Example scripts demonstrate Mojo idioms (structs, fn, error handling)

### Patterns Demonstrated

- Package initialization with `__init__.mojo`
- Struct-based models with `fn` methods
- Error handling with `raises`
- Type annotations and ownership

### 6. Data Directory Structure

**Decision**: Three-tier data organization: `raw/`, `processed/`, `cache/`

### Rationale

- `raw/` - Immutable original datasets (never modified)
- `processed/` - Cleaned and transformed datasets ready for training
- `cache/` - Cached computations and intermediate results for performance
- Clear separation prevents accidental data corruption
- Standard pattern familiar to ML practitioners

### 7. Testing Organization

**Decision**: Mirror source structure in tests directory with `test_*.mojo` files

### Rationale

- Easy to find tests corresponding to source files
- Follows common testing conventions
- Supports TDD workflow (write tests alongside implementation)
- `__init__.mojo` allows tests to be imported as package

### 8. Examples vs Scripts

**Decision**: Separate `examples/` (demonstration) from `scripts/` (automation)

### Rationale

- `examples/` - User-facing demonstration scripts (train, evaluate, inference)
- `scripts/` - Automation scripts for downloading dependencies (paper, dataset, reference)
- Clear distinction between "how to use" and "how to set up"
- Reduces confusion about script purposes

## References

### Source Plan

- [notes/plan/01-foundation/01-directory-structure/01-create-papers-dir/03-create-template/plan.md](../../../plan/01-foundation/01-directory-structure/01-create-papers-dir/03-create-template/plan.md)

### Parent Plan

- [notes/plan/01-foundation/01-directory-structure/01-create-papers-dir/plan.md](../../../plan/01-foundation/01-directory-structure/01-create-papers-dir/plan.md)

### Related Issues

- Issue #527 - [Test] Create Template - Testing Phase
- Issue #528 - [Impl] Create Template - Implementation Phase
- Issue #529 - [Package] Create Template - Packaging Phase
- Issue #530 - [Cleanup] Create Template - Cleanup Phase

### Architectural References

- [ADR-001: Language Selection for Tooling](../../review/adr/ADR-001-language-selection-tooling.md) - Mojo-first approach
- [5-Phase Development Workflow](../../review/README.md) - Plan → Test/Impl/Package → Cleanup

## Implementation Notes

### Current Status

The template has been fully implemented and is available at `papers/_template/`. This planning document retroactively captures the design decisions and architecture.

### Template Structure Created

```text
papers/_template/
├── README.md                   # Comprehensive template documentation (414 lines)
├── src/                        # Mojo implementation code
│   ├── __init__.mojo           # Package initialization
│   └── .gitkeep                # Placeholder for empty directory
├── scripts/                    # Automation scripts
│   ├── __init__.mojo           # Package initialization
│   └── .gitkeep                # Placeholder for empty directory
├── tests/                      # Test suite
│   ├── __init__.mojo           # Test package initialization
│   └── .gitkeep                # Placeholder for empty directory
├── data/                       # Data management
│   ├── raw/                    # Original, immutable datasets
│   │   └── .gitkeep
│   ├── processed/              # Cleaned and transformed datasets
│   │   └── .gitkeep
│   └── cache/                  # Cached computations
│       └── .gitkeep
├── configs/                    # Configuration files
│   ├── config.yaml             # Example configuration (46 lines)
│   └── .gitkeep
├── notebooks/                  # Jupyter notebooks
│   └── .gitkeep
└── examples/                   # Demonstration scripts
    ├── train.mojo              # Example training script
    └── .gitkeep
```text

### Key Files

1. **README.md** (414 lines)
   - Comprehensive documentation with 10 major sections
   - Quick Start guide with copy command
   - Detailed directory structure explanation
   - Implementation guide (7 steps)
   - Testing strategy and principles
   - Common patterns with Mojo code examples
   - Paper-specific information template

1. **configs/config.yaml** (46 lines)
   - Model configuration section
   - Training hyperparameters
   - Data configuration and splits
   - Path configuration
   - Experiment tracking settings

1. **examples/train.mojo**
   - Placeholder demonstration script
   - Shows expected pattern for training scripts

### Validation

The template successfully meets all success criteria:

- ✓ Directory exists at `papers/_template/`
- ✓ All standard subdirectories created with appropriate structure
- ✓ README provides comprehensive usage explanation with examples
- ✓ Simple copy command works: `cp -r papers/_template papers/new-paper`
- ✓ Documentation includes implementation guide, testing strategy, best practices
- ✓ Placeholder files demonstrate Mojo code organization patterns
- ✓ Configuration template shows common settings and extensible structure

### Design Strengths

1. **Comprehensive Documentation** - README covers all aspects from quick start to advanced patterns
1. **Mojo-First** - All examples use `.mojo` extensions and demonstrate Mojo idioms
1. **TDD Support** - Test directory structure and documentation emphasize test-driven development
1. **Data Management** - Three-tier data organization (raw/processed/cache) prevents corruption
1. **Configuration Separation** - YAML configs keep settings separate from code
1. **Extensibility** - Clear structure makes it easy to add paper-specific components
1. **Low Friction** - Single copy command and clear guide reduce setup time
1. **Best Practices** - Embedded examples show recommended patterns

### Notes for Subsequent Phases

#### For Test Phase (Issue #527)

- Verify template can be copied without errors
- Test that placeholder files maintain directory structure
- Validate README markdown formatting and links
- Check config.yaml is valid YAML
- Ensure .gitkeep files preserve empty directories in git

#### For Implementation Phase (Issue #528)

- Template is already implemented
- Focus on validation and documentation completeness
- Consider adding more example scripts if needed
- Verify Mojo package initialization works correctly

#### For Packaging Phase (Issue #529)

- Ensure template integrates with overall `papers/` directory structure
- Verify template README references are consistent with project documentation
- Check that template follows project conventions (per CLAUDE.md)

#### For Cleanup Phase (Issue #530)

- Review README for clarity and completeness
- Check for any unused or redundant files
- Ensure all links in documentation are valid
- Verify configuration file has sensible defaults
- Consider user feedback for improvements
