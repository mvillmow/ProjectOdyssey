"""Default hyperparameters for ML operations.

These defaults are commonly used values that can be imported
to maintain consistency across the codebase.
"""

# Activation function defaults
alias DEFAULT_LEAKY_RELU_ALPHA: Float64 = 0.01  # Default negative slope for LeakyReLU
alias DEFAULT_ELU_ALPHA: Float64 = 1.0  # Default alpha parameter for ELU activation
alias DEFAULT_HARD_TANH_MIN: Float64 = -1.0  # Default minimum value for HardTanh
alias DEFAULT_HARD_TANH_MAX: Float64 = 1.0  # Default maximum value for HardTanh

# Regularization defaults
alias DEFAULT_DROPOUT_RATE: Float64 = 0.5  # Default dropout probability
alias DEFAULT_BATCHNORM_MOMENTUM: Float64 = 0.1  # Default momentum for batch normalization

# Initialization defaults
alias DEFAULT_UNIFORM_LOW: Float64 = -0.1  # Default lower bound for uniform initialization
alias DEFAULT_UNIFORM_HIGH: Float64 = 0.1  # Default upper bound for uniform initialization

# Augmentation defaults
alias DEFAULT_AUGMENTATION_PROB: Float64 = 0.5  # Default probability for data augmentation
alias DEFAULT_TEXT_AUGMENTATION_PROB: Float64 = 0.1  # Default probability for text augmentation

# Misc
alias DEFAULT_RANDOM_SEED: Int = 42  # Default random seed for reproducibility
