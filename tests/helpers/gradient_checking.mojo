"""DEPRECATED: This module has been consolidated into shared.testing.

All gradient checking utilities have been moved to shared/testing/gradient_checker.mojo
for better code organization and reusability.

See shared.testing for:
- check_gradients()
- check_gradients_verbose()
- relative_error()
- check_gradient()
- compute_numerical_gradient()
- assert_gradients_close()

This file is kept as a stub for reference only. Do not use in new code.
Import from shared.testing instead:

    from shared.testing import check_gradient, compute_numerical_gradient, assert_gradients_close
"""
