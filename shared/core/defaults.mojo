"""Default hyperparameters for ML operations.

These defaults are commonly used values that can be imported
to maintain consistency across the codebase.
"""

# Activation function defaults
alias DEFAULT_LEAKY_RELU_ALPHA: Float64 = 0.01
alias DEFAULT_ELU_ALPHA: Float64 = 1.0
alias DEFAULT_HARD_TANH_MIN: Float64 = -1.0
alias DEFAULT_HARD_TANH_MAX: Float64 = 1.0

# Regularization defaults
alias DEFAULT_DROPOUT_RATE: Float64 = 0.5
alias DEFAULT_BATCHNORM_MOMENTUM: Float64 = 0.1

# Initialization defaults
alias DEFAULT_UNIFORM_LOW: Float64 = -0.1
alias DEFAULT_UNIFORM_HIGH: Float64 = 0.1

# Augmentation defaults
alias DEFAULT_AUGMENTATION_PROB: Float64 = 0.5
alias DEFAULT_TEXT_AUGMENTATION_PROB: Float64 = 0.1

# Misc
alias DEFAULT_RANDOM_SEED: Int = 42
