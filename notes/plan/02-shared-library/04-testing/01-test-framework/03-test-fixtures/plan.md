# Test Fixtures

## Overview

Implement test fixtures for reusable test data and setup logic. Fixtures provide consistent test environments, sample data, and common configurations that multiple tests can share. Good fixtures reduce duplication and make tests more maintainable.

## Parent Plan

[../plan.md](../plan.md)

## Child Plans

None (leaf node)

## Inputs

- Common test data requirements
- Setup and teardown patterns
- Shared configuration needs
- Resource management requirements

## Outputs

- Sample tensor fixtures (small, medium, large)
- Model fixtures (simple test models)
- Dataset fixtures (toy datasets)
- Configuration fixtures (training configs)
- Fixture scopes (function, module, session)

## Steps

1. Create sample tensor fixtures of various sizes
2. Build model fixtures for testing training
3. Implement dataset fixtures with known properties
4. Add configuration fixtures for common setups
5. Configure fixture scopes appropriately

## Success Criteria

- [ ] Fixtures provide consistent test data
- [ ] Setup and teardown work correctly
- [ ] Fixtures are reusable across tests
- [ ] Scoping minimizes overhead

## Notes

Use pytest-style fixtures if available in Mojo. Keep fixtures simple and focused. Document fixture contents and purpose. Use appropriate scopes to balance speed and isolation.
