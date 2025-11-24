# Issue #19: [Plan] Create Core

## Objective

Create the shared/core/ directory for fundamental building blocks and core functionality used across all paper
implementations. This establishes the foundation for reusable ML components written in Mojo.

## Deliverables

- `shared/core/` - Core library directory
- `shared/core/README.md` - Comprehensive documentation
- `shared/core/__init__.mojo` - Mojo package root
- `shared/core/layers/__init__.mojo` - Neural network layers
- `shared/core/ops/__init__.mojo` - Mathematical operations
- `shared/core/types/__init__.mojo` - Custom types and data structures
- `shared/core/utils/__init__.mojo` - Utility functions
- `notes/issues/19/README.md` - This documentation

## Success Criteria

- [x] core/ directory exists in shared/
- [x] README clearly explains purpose and contents
- [x] Directory is set up as proper Mojo package (uses __init__.mojo per project standards)
- [x] Documentation helps contributors know what belongs here
- [x] Subdirectories organized logically (layers, ops, types, utils)
- [x] All subdirectories have __init__.mojo files
- [x] Mojo-specific guidelines documented

## References

- Project Documentation: `CLAUDE.md`
- Agent Hierarchy: `agents/`
- GitHub Issue: [#19](https://github.com/mvillmow/ml-odyssey/issues/19)

## Implementation Notes

### Directory Structure Design

The core library is organized into four main subdirectories:

1. __layers/__ - Neural network layer implementations
   - Fundamental building blocks (Linear, Conv2D, etc.)
   - Activation functions (ReLU, Sigmoid, Tanh, etc.)
   - Normalization layers (BatchNorm, LayerNorm, etc.)
   - Pooling layers (MaxPool, AvgPool, etc.)

1. __ops/__ - Mathematical operations
   - Matrix operations (matmul, transpose, etc.)
   - Element-wise operations (add, multiply, etc.)
   - Reduction operations (sum, mean, max, etc.)
   - Broadcasting utilities

1. __types/__ - Custom types and data structures
   - Tensor types and utilities
   - Shape definitions
   - Data type definitions (DType extensions)
   - Container types

1. __utils/__ - Utility functions
   - Initialization utilities (Xavier, He, etc.)
   - Memory management helpers
   - Debugging and logging utilities
   - Performance profiling tools

### Key Decisions

1. __Mojo-First Design__: All package markers use `__init__.mojo` instead of `__init__.py` to establish Mojo
   as the primary language for this codebase.

1. __Separation of Concerns__: Clear separation between layers (high-level), ops (low-level), types (data
   structures), and utils (helpers) to maintain clean architecture.

1. __FUNDAMENTAL Building Blocks Only__: This directory contains only code that is shared across ALL or MOST
   paper implementations. Paper-specific code belongs in the respective paper directories.

1. __Performance-Critical__: All core components are implemented in Mojo for maximum performance, leveraging
   SIMD, type safety, and memory management features.

### Next Steps

After this planning phase is complete:

1. Issue #20: [Test] Create Core - Write tests for core directory structure
1. Issue #21: [Implementation] Create Core - Implement initial core components
1. Issue #22: [Packaging] Create Core - Package and integrate core library
1. Issue #23: [Cleanup] Create Core - Refactor and finalize

### Related Components

This component is part of the Shared Library section (02-shared-library):

- Issue #16: [Plan] Create Shared Directory
- Issue #17: [Plan] Shared Directory README
- Issue #18: [Plan] Create Shared __init__.mojo
- Issue #19: [Plan] Create Core (this issue)
- Issue #24: [Plan] Create Models
