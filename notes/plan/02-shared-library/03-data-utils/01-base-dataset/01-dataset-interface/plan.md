# Dataset Interface

## Overview
Define the base dataset interface that all dataset implementations must follow. This interface establishes the contract for data access, enabling interchangeable dataset implementations and consistent usage patterns throughout the codebase.

## Parent Plan
[../plan.md](../plan.md)

## Child Plans
None (leaf node)

## Inputs
- Dataset usage patterns and requirements
- Python sequence protocol specifications
- Access pattern needs (indexed, iterable)

## Outputs
- Abstract base class or trait for datasets
- Required method signatures (__len__, __getitem__)
- Optional method specifications (transform, collate)
- Clear documentation and usage examples

## Steps
1. Define abstract dataset interface
2. Specify required methods with type signatures
3. Document expected behavior and contracts
4. Provide usage examples and patterns

## Success Criteria
- [ ] Interface is minimal and clear
- [ ] Required methods cover essential operations
- [ ] Documentation explains usage patterns
- [ ] Interface works with standard Python idioms

## Notes
Follow Python sequence protocol (__len__, __getitem__). Keep interface minimal - only essential methods. Support both indexing and iteration. Make it easy to implement custom datasets.
