"""Numerical stability constants for ML operations.

This module provides centralized epsilon and threshold values used
across the codebase for numerical stability in various operations.
"""

# Division safety - prevents division by zero
alias EPSILON_DIV: Float64 = 1e-10  # Small constant to prevent division by zero

# Loss function stability - for log operations in BCE, cross-entropy, etc.
alias EPSILON_LOSS: Float64 = 1e-7  # Small constant for numerical stability in log operations

# Normalization stability - for BatchNorm, LayerNorm, GroupNorm, InstanceNorm
alias EPSILON_NORM: Float64 = 1e-5  # Small constant for numerical stability in normalization

# Gradient safety thresholds
alias GRADIENT_MAX_NORM: Float64 = 1000.0  # Threshold for gradient explosion detection
alias GRADIENT_MIN_NORM: Float64 = 1e-7  # Threshold for gradient vanishing detection
