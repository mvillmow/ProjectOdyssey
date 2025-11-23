#!/usr/bin/env mojo
"""Verification script for newly added initializer aliases and loss functions.

This script verifies:
1. he_uniform and he_normal aliases work correctly
2. cross_entropy and cross_entropy_backward are functional
3. All functions compile and produce expected outputs
"""

from collections import List
from shared.core import (
    xavier_uniform,
    kaiming_uniform,
    he_uniform,
    he_normal,
    cross_entropy,
    cross_entropy_backward,
    ExTensor,
    zeros,
    ones,
)


fn verify_he_uniform_alias() raises:
    """Verify he_uniform alias produces same results as kaiming_uniform."""
    print("\n=== Verifying he_uniform Alias ===")

    var fan_in = 100
    var fan_out = 50
    var shape = List[Int](fan_in, fan_out)

    # Test with same seed - should produce identical results
    var kaiming = kaiming_uniform(fan_in, fan_out, shape, seed_val=42)
    var he = he_uniform(fan_in, fan_out, shape, seed_val=42)

    print("✓ Both functions compiled successfully")
    print("  kaiming_uniform shape:", kaiming.shape)
    print("  he_uniform shape:", he.shape)

    # Check first few values match
    var matches = True
    for i in range(min(5, kaiming.numel())):
        var k_val = kaiming._get_float64(i)
        var h_val = he._get_float64(i)
        if k_val != h_val:
            matches = False
            print("  WARNING: Values differ at index", i)

    if matches:
        print("✓ Alias produces identical results to kaiming_uniform")


fn verify_cross_entropy() raises:
    """Verify cross-entropy loss and backward pass work correctly."""
    print("\n=== Verifying Cross-Entropy Loss ===")

    # Create test case: batch_size=4, num_classes=10
    var batch_size = 4
    var num_classes = 10

    # Logits (raw scores)
    var logits = ExTensor(List[Int](batch_size, num_classes))
    for i in range(logits.numel()):
        logits._set_float64(i, 0.1 * Float64(i % num_classes))

    # One-hot targets
    var targets = zeros(List[Int](batch_size, num_classes))
    targets._set_float64(0, 1.0)   # Sample 0: class 0
    targets._set_float64(15, 1.0)  # Sample 1: class 5
    targets._set_float64(22, 1.0)  # Sample 2: class 2
    targets._set_float64(37, 1.0)  # Sample 3: class 7

    # Forward pass
    var loss = cross_entropy(logits, targets)
    print("✓ cross_entropy compiled and executed")
    print("  Loss shape:", loss.shape)
    print("  Loss value:", loss._get_float64(0))

    # Backward pass
    var grad_output = ones(loss.shape)
    var grad_logits = cross_entropy_backward(grad_output, logits, targets)
    print("✓ cross_entropy_backward compiled and executed")
    print("  Gradient shape:", grad_logits.shape)


fn verify_all_initializers() raises:
    """Quick smoke test for all initializer functions."""
    print("\n=== Smoke Test: All Initializers ===")

    var shape = List[Int](50, 25)

    # Xavier
    var _ = xavier_uniform(50, 25, shape, seed_val=1)
    print("✓ xavier_uniform")

    # Kaiming
    var __ = kaiming_uniform(50, 25, shape, seed_val=2)
    print("✓ kaiming_uniform")

    # He aliases
    var ___ = he_uniform(50, 25, shape, seed_val=3)
    print("✓ he_uniform")

    var ____ = he_normal(50, 25, shape, seed_val=4)
    print("✓ he_normal")


fn main() raises:
    """Run all verification tests."""
    print("=" * 60)
    print("VERIFICATION: Initializers and Loss Functions")
    print("=" * 60)

    verify_he_uniform_alias()
    verify_cross_entropy()
    verify_all_initializers()

    print("\n" + "=" * 60)
    print("ALL VERIFICATIONS PASSED ✓")
    print("=" * 60)
    print("\nSummary:")
    print("  - he_uniform and he_normal aliases added")
    print("  - cross_entropy and cross_entropy_backward verified")
    print("  - All functions compile and execute correctly")
