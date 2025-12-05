"""Mathematical constants for ML operations.

This module provides centralized mathematical constants used across
the codebase for activation functions, initializers, and elementwise ops.
"""

# Pi and related constants
alias PI: Float64 = 3.14159265358979323846

# Square roots
alias SQRT_2: Float64 = 1.4142135623730951
alias SQRT_2_OVER_PI: Float64 = 0.7978845608028654  # sqrt(2/pi) for GELU
alias INV_SQRT_2PI: Float64 = 0.3989422804014327  # 1/sqrt(2*pi) for normal distribution

# GELU activation constants
alias GELU_COEFF: Float64 = 0.044715  # Coefficient in GELU approximation

# Logarithms
alias LN2: Float64 = 0.6931471805599453  # Natural log of 2
alias LN10: Float64 = 2.302585092994046  # Natural log of 10
