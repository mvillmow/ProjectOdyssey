"""
Utility Functions Library

Provides logging, visualization, configuration, and other helper utilities.

Modules:
    logging: Logging infrastructure for training and evaluation
    visualization: Plotting and visualization tools
    config: Configuration management utilities

Example:
    from shared.utils import Logger, plot_training_curves, load_config

    # Create logger
    logger = Logger("training.log")
    logger.info("Starting training...")

    # Plot results
    plot_training_curves(train_losses, val_losses)

    # Load configuration
    config = load_config("experiment.yaml")
"""

# Package version
alias VERSION = "0.1.0"

# ============================================================================
# Exports - Implemented modules
# ============================================================================

# Logging utilities
from .logging import (
    Logger,  # Main logger class
    LogLevel,  # Log level enum
    get_logger,  # Get or create logger
    StreamHandler,  # Console output handler
    FileHandler,  # File output handler
    LogRecord,  # Log record structure
    SimpleFormatter,  # Simple message formatter
    TimestampFormatter,  # Formatter with timestamps
    DetailedFormatter,  # Detailed formatter with location
    ColoredFormatter,  # Formatter with ANSI colors
)

# Configuration utilities
from .config import (
    Config,  # Configuration container
    load_config,  # Load from file (YAML/JSON)
    save_config,  # Save to file
    merge_configs,  # Merge multiple configs
    ConfigValidator,  # Validate configuration
)

# File I/O utilities
from .io import (
    Checkpoint,  # Checkpoint container
    save_checkpoint,  # Save model checkpoint
    load_checkpoint,  # Load model checkpoint
    serialize_tensor,  # Serialize tensor
    deserialize_tensor,  # Deserialize tensor
    safe_write_file,  # Atomic file write
    safe_read_file,  # Safe file read
    create_backup,  # Backup creation
    file_exists,  # Check file existence
    directory_exists,  # Check directory existence
    create_directory,  # Create directory
)

# Visualization utilities
from .visualization import (
    plot_training_curves,  # Plot loss/accuracy curves
    plot_loss_only,  # Plot single loss curve
    plot_accuracy_only,  # Plot single accuracy curve
    plot_confusion_matrix,  # Plot confusion matrix
    visualize_model_architecture,  # Model architecture diagram
    show_images,  # Display image grid
    visualize_feature_maps,  # Feature map visualization
    save_figure,  # Save matplotlib figure
)

# Random seed utilities
from .random import (
    set_seed,  # Set random seed globally
    get_global_seed,  # Get current seed
    get_random_state,  # Get current random state
    set_random_state,  # Restore random state
    RandomState,  # Random state container
    random_uniform,  # Generate uniform random
    random_normal,  # Generate normal random
    random_int,  # Generate random integer
    shuffle,  # Shuffle list in-place
)

# Profiling utilities
from .profiling import (
    Timer,  # Context manager for timing
    memory_usage,  # Get current memory usage
    profile_function,  # Profile function execution
    benchmark_function,  # Benchmark function
    MemoryStats,  # Memory statistics
    TimingStats,  # Timing statistics
    ProfilingReport,  # Profiling report
)

# ============================================================================
# Public API
# ============================================================================

# All symbols are exported via the imports above.
# Mojo does not support __all__ - exports are controlled by import statements.
