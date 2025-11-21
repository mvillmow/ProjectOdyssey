# Issue #551: [Plan] Create Utils - Design and Documentation

## Objective

Create the shared/utils/ directory for general utility functions and helper code that will be reused across paper implementations, including logging, configuration, file I/O, visualization, and other general-purpose utilities.

## Deliverables

- `shared/utils/` directory
- `shared/utils/README.md` explaining purpose and organization
- `shared/utils/__init__.py` for Python package structure

## Success Criteria

- [ ] `utils/` directory exists in `shared/`
- [ ] README clearly explains purpose and contents
- [ ] Directory is set up as a proper Python package
- [ ] Documentation guides what utility code is shared

## Design Decisions

### Architecture

**Directory Purpose**: The `shared/utils/` directory serves as a repository for general-purpose helper functions and utilities that don't fit into the more specialized directories (`core/`, `training/`, `data/`). This creates a clear separation of concerns:

- **Core**: Neural network primitives and fundamental ML operations
- **Training**: Training loops, optimization, and model training utilities
- **Data**: Dataset handling and data pipeline utilities
- **Utils**: Everything else - general-purpose helpers used across papers

**Package Structure**: The directory will be set up as a proper Python package with `__init__.py`, following standard Python conventions for reusable modules.

### Planned Utility Categories

Based on the plan notes, the utils directory will organize utilities into the following categories:

1. **Logging Utilities**
   - Structured logging for experiments
   - Progress tracking and reporting
   - Debug and info level logging helpers

1. **Configuration Management**
   - Config file parsing (YAML, JSON, etc.)
   - Environment variable handling
   - Hyperparameter management

1. **Visualization Utilities**
   - Plotting helpers for training metrics
   - Model architecture visualization
   - Data distribution visualization

1. **File I/O Utilities**
   - Safe file reading/writing
   - Directory creation and management
   - Path handling utilities

1. **General-Purpose Helpers**
   - Type checking and validation
   - Math utilities not specific to ML
   - String formatting and parsing
   - Time and date utilities

### Design Principles

**KISS (Keep It Simple, Stupid)**: Each utility function should do one thing well. Avoid over-engineering general-purpose helpers.

**YAGNI (You Aren't Gonna Need It)**: Start with a minimal set of utilities. Add more as actual needs arise from paper implementations, not based on speculation.

**DRY (Don't Repeat Yourself)**: If a helper function is used in 2+ paper implementations, it belongs in utils. If it's only used once, keep it local to that paper.

**Modularity**: Each utility category should be independently usable. Avoid circular dependencies between utility modules.

### API Design Guidelines

1. **Clear naming**: Function names should be self-documenting (e.g., `load_yaml_config()` not `load_config()`)
1. **Type hints**: All public functions must have type hints for parameters and return values
1. **Docstrings**: All public functions must have clear docstrings explaining purpose, parameters, and return values
1. **Error handling**: Utilities should raise informative exceptions with clear error messages
1. **Default behavior**: Sensible defaults that work for 80% of use cases

### Documentation Strategy

The `README.md` will serve as both:

1. **User Guide**: How to import and use utilities
1. **Developer Guide**: Guidelines for adding new utilities

It will include:

- Overview of the utils directory purpose
- Organization of utility categories
- Examples of common use cases
- Guidelines for when to add new utilities
- Links to related directories (core, training, data)

### Integration with Other Components

The utils directory is a foundational component that:

- **Is used by**: All paper implementations, training scripts, data pipelines
- **Depends on**: Standard library, minimal external dependencies
- **Does not depend on**: core, training, or data directories (to avoid circular dependencies)

This creates a clean dependency hierarchy: `papers → training/data/core → utils → stdlib`

## References

- **Source Plan**: [notes/plan/01-foundation/01-directory-structure/02-create-shared-dir/04-create-utils/plan.md](notes/plan/01-foundation/01-directory-structure/02-create-shared-dir/04-create-utils/plan.md)
- **Parent Plan**: [notes/plan/01-foundation/01-directory-structure/02-create-shared-dir/plan.md](notes/plan/01-foundation/01-directory-structure/02-create-shared-dir/plan.md)
- **Related Issues**:
  - Issue #552: [Test] Create Utils - Test Development
  - Issue #553: [Impl] Create Utils - Implementation
  - Issue #554: [Package] Create Utils - Integration and Packaging
  - Issue #555: [Cleanup] Create Utils - Cleanup and Finalization

## Implementation Notes

(This section will be filled in during the implementation phases as design decisions are validated and refined)
