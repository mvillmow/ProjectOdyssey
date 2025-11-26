# Issue #34 Track 4: Test Fixtures and Utilities

## Objective

Add missing test utilities and fixtures needed by test_training_loop.mojo. This track provides essential infrastructure for testing training loops and model implementations.

## Deliverables

- Float64 assertion overloads in `tests/shared/conftest.mojo`
- Linear layer stub implementation in `shared/core/layers/linear.mojo`
- Updated layer exports in `shared/core/layers/__init__.mojo`
- Verification that all fixtures are available and functional

## Success Criteria

- [x] Float64 assertion overloads added (assert_less for Float64)
- [x] create_simple_model() fixture available for SimpleMLP creation
- [x] create_mock_dataloader() fixture for generating test batches
- [x] Linear layer compiles with proper initialization
- [x] All fixtures integrated and ready for test use

## Implementation Notes

### Float64 Assertions

Added the missing `assert_less(Float64, Float64)` overload to conftest.mojo. The codebase already includes:
- `assert_greater(Float64, Float64)` - line 233
- `assert_greater_or_equal(Float64, Float64)` - line 313
- `assert_less_or_equal(Float64, Float64)` - line 345

Added after line 279 to complete the Float64 assertion family.

### Training Loop Fixtures

The following fixtures were already implemented in conftest.mojo:

1. **create_simple_model()** - lines 913-941
   - Creates SimpleMLP with input_dim=10, hidden_dim=5, output_dim=1
   - Uses constant initialization (init_value=0.1) for predictable testing
   - Returns a fully configured MLP for training tests

2. **create_simple_dataset()** - lines 944-999
   - Generates synthetic data with configurable dimensions
   - Returns list of (input, output) tuples
   - Uses deterministic seeding for reproducible tests

3. **create_mock_dataloader()** - lines 1040-1089
   - Wraps dataset in MockDataLoader structure
   - Provides batch_size and shuffle control
   - Returns loader with `__len__()` method for iteration

4. **MockDataLoader struct** - lines 1002-1037
   - Holds samples, batch_size, and shuffle flag
   - Provides `__len__()` to get number of batches

### Linear Layer Implementation

Created `/shared/core/layers/linear.mojo` with:
- **Initialization**: Uses `randn()` for weights, `zeros()` for bias
- **Forward method**: Stub that returns zeros (full implementation planned for later)
- **Parameters method**: Returns list of [weight, bias] for optimization
- **Proper typing**: Uses ExTensor for tensor operations

The Linear layer is intentionally a stub because:
1. Full matrix multiplication implementation requires GEMM operations (future work)
2. Testing infrastructure can work with stub behavior
3. Follows Issue #34 Track 4 requirements: "stub is fine"

### Layer Module Updates

Updated `shared/core/layers/__init__.mojo` to export Linear:
```mojo
from .linear import Linear
```

## References

- SimpleMLP implementation: `/tests/shared/fixtures/mock_models.mojo`
- ExTensor API: `/shared/core/extensor.mojo` (provides `randn()`, `zeros()`)
- Test fixture patterns: `/tests/shared/conftest.mojo` (lines 1000-1090)

## Testing Validation

All fixtures have been:
1. Verified to compile (existing fixtures)
2. Checked for proper function signatures
3. Confirmed to work with SimpleMLP and ExTensor APIs
4. Integrated into test infrastructure

The Linear layer compiles with proper shape handling:
- Single sample: (in_features,) → (out_features,)
- Batched: (batch_size, in_features) → (batch_size, out_features,)
