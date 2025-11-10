# Utilities Library

The `utils` library provides logging, visualization, configuration management, and other helper utilities for
ML Odyssey. These utilities support development, debugging, and experiment management.

## Purpose

This library provides:

- Structured logging for tracking experiments
- Visualization tools for analyzing results
- Configuration management for reproducible experiments
- Profiling and timing utilities for optimization
- Random seed management for reproducibility

**Key Principle**: Provide cross-cutting utilities that enhance productivity without adding complexity to core
ML functionality.

## Directory Organization

```text
utils/
├── __init__.mojo           # Package root - exports main utilities
├── README.md               # This file
├── logging.mojo            # Logging infrastructure
├── visualization.mojo      # Plotting and visualization
├── config.mojo             # Configuration management
├── random.mojo             # Random seed utilities
└── profiling.mojo          # Timing and profiling tools
```

## What Belongs in Utils?

### Include

- Logging infrastructure
- Visualization helpers (plotting, image display)
- Configuration loading/saving
- Random seed management
- Timing and profiling utilities
- Common helper functions used across modules

### Exclude

- Core ML functionality (belongs in core/)
- Training-specific utilities (belongs in training/)
- Data-specific utilities (belongs in data/)
- Paper-specific utilities (belongs with the paper)

## Components

### Logging (`logging.mojo`)

Structured logging for experiments and debugging.

#### Logger

Main logger class with multiple output handlers:

```mojo
struct Logger:
    """Structured logger with multiple handlers."""
    var name: String
    var level: LogLevel
    var handlers: List[Handler]

    fn __init__(inout self, name: String, level: LogLevel = LogLevel.INFO):
        """Create logger with name and level."""
        self.name = name
        self.level = level
        self.handlers = List[Handler]()

    fn add_handler(inout self, handler: Handler):
        """Add output handler."""
        self.handlers.append(handler)

    fn info(self, message: String):
        """Log info message."""
        if self.level <= LogLevel.INFO:
            self._log(LogLevel.INFO, message)

    fn warning(self, message: String):
        """Log warning message."""
        if self.level <= LogLevel.WARNING:
            self._log(LogLevel.WARNING, message)

    fn error(self, message: String):
        """Log error message."""
        if self.level <= LogLevel.ERROR:
            self._log(LogLevel.ERROR, message)

    fn debug(self, message: String):
        """Log debug message."""
        if self.level <= LogLevel.DEBUG:
            self._log(LogLevel.DEBUG, message)

    fn _log(self, level: LogLevel, message: String):
        """Internal logging method."""
        var record = LogRecord(self.name, level, message)
        for handler in self.handlers:
            handler.emit(record)
```

#### Log Levels

```mojo
@value
struct LogLevel:
    """Log level enumeration."""
    alias DEBUG = 10
    alias INFO = 20
    alias WARNING = 30
    alias ERROR = 40
    alias CRITICAL = 50
```

#### Handlers

Output handlers for different destinations:

```mojo
trait Handler:
    """Base handler interface."""
    fn emit(self, record: LogRecord)

struct StreamHandler(Handler):
    """Write logs to stdout/stderr."""
    fn emit(self, record: LogRecord):
        """Write to console."""
        print(f"[{record.level_name}] {record.name}: {record.message}")

struct FileHandler(Handler):
    """Write logs to file."""
    var filepath: String
    var file: FileDescriptor

    fn emit(self, record: LogRecord):
        """Write to file."""
        write_line(self.file, f"[{record.timestamp}] [{record.level_name}] {record.message}")
```

**Usage**:

```mojo
from shared.utils import Logger, StreamHandler, FileHandler

# Create logger
var logger = Logger("training")

# Add handlers
logger.add_handler(StreamHandler())
logger.add_handler(FileHandler("training.log"))

# Log messages
logger.info("Starting epoch 1")
logger.warning("Learning rate reduced")
logger.error("Validation loss increased")
```

### Visualization (`visualization.mojo`)

Plotting and visualization tools for analyzing training results.

#### plot_training_curves

Plot loss and accuracy curves:

```mojo
fn plot_training_curves(
    train_losses: List[Float32],
    val_losses: List[Float32],
    train_accs: Optional[List[Float32]] = None,
    val_accs: Optional[List[Float32]] = None,
    save_path: Optional[String] = None
):
    """
    Plot training curves.

    Args:
        train_losses: Training losses per epoch
        val_losses: Validation losses per epoch
        train_accs: Optional training accuracies
        val_accs: Optional validation accuracies
        save_path: Path to save figure

    Example:
        plot_training_curves(
            train_losses=train_losses,
            val_losses=val_losses,
            save_path="training_curves.png"
        )
    """
    # Create matplotlib figure
    # Plot loss curves
    # Plot accuracy curves if provided
    # Save or display
```

#### show_images

Display grid of images:

```mojo
fn show_images(
    images: Tensor,
    labels: Optional[List[String]] = None,
    nrow: Int = 8,
    normalize: Bool = True,
    save_path: Optional[String] = None
):
    """
    Display grid of images.

    Args:
        images: Batch of images (N, C, H, W)
        labels: Optional labels for each image
        nrow: Number of images per row
        normalize: Normalize image values to [0, 1]
        save_path: Path to save figure

    Example:
        # Show first batch of training data
        show_images(batch.inputs[:64], nrow=8)
    """
    # Create image grid
    # Add labels if provided
    # Display or save
```

#### plot_confusion_matrix

Plot confusion matrix heatmap:

```mojo
fn plot_confusion_matrix(
    y_true: List[Int],
    y_pred: List[Int],
    class_names: Optional[List[String]] = None,
    normalize: Bool = False,
    save_path: Optional[String] = None
):
    """
    Plot confusion matrix.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: Names of classes
        normalize: Normalize counts to percentages
        save_path: Path to save figure

    Example:
        plot_confusion_matrix(
            y_true=true_labels,
            y_pred=predictions,
            class_names=["cat", "dog"],
            normalize=True
        )
    """
    # Compute confusion matrix
    # Create heatmap
    # Add class names
    # Display or save
```

#### plot_lr_schedule

Visualize learning rate schedule:

```mojo
fn plot_lr_schedule(
    scheduler: Scheduler,
    num_epochs: Int,
    save_path: Optional[String] = None
):
    """
    Plot learning rate schedule.

    Args:
        scheduler: Learning rate scheduler
        num_epochs: Number of epochs to simulate
        save_path: Path to save figure

    Example:
        var scheduler = CosineAnnealingLR(optimizer, T_max=100)
        plot_lr_schedule(scheduler, num_epochs=100)
    """
    # Simulate scheduler for num_epochs
    # Plot learning rate curve
    # Display or save
```

### Configuration (`config.mojo`)

Configuration management for reproducible experiments.

#### Config

Configuration container:

```mojo
struct Config:
    """Configuration container with nested access."""
    var data: Dict[String, ConfigValue]

    fn __init__(inout self):
        """Create empty config."""
        self.data = Dict[String, ConfigValue]()

    fn get[T: ConfigValue](self, key: String) -> T:
        """Get config value by key."""
        return self.data[key].as[T]()

    fn set[T: ConfigValue](inout self, key: String, value: T):
        """Set config value."""
        self.data[key] = ConfigValue(value)

    fn has(self, key: String) -> Bool:
        """Check if key exists."""
        return key in self.data
```

#### load_config / save_config

Load and save configurations:

```mojo
fn load_config(filepath: String) -> Config:
    """
    Load configuration from YAML or JSON file.

    Args:
        filepath: Path to config file (.yaml or .json)

    Returns:
        Loaded configuration

    Example:
        var config = load_config("experiment.yaml")
        var lr = config.get[Float32]("learning_rate")
    """
    # Detect file format
    # Parse YAML or JSON
    # Convert to Config
    pass

fn save_config(config: Config, filepath: String):
    """
    Save configuration to file.

    Args:
        config: Configuration to save
        filepath: Output file path

    Example:
        var config = Config()
        config.set("learning_rate", 0.001)
        save_config(config, "experiment.yaml")
    """
    # Convert Config to YAML/JSON
    # Write to file
    pass
```

#### merge_configs

Merge multiple configurations:

```mojo
fn merge_configs(base: Config, override: Config) -> Config:
    """
    Merge two configs, with override taking precedence.

    Args:
        base: Base configuration
        override: Override configuration

    Returns:
        Merged configuration

    Example:
        var default_config = load_config("default.yaml")
        var experiment_config = load_config("experiment.yaml")
        var config = merge_configs(default_config, experiment_config)
    """
    # Deep merge configurations
    # Override takes precedence
    pass
```

**Usage**:

```mojo
from shared.utils import load_config, save_config

# Load config
var config = load_config("configs/lenet5.yaml")

# Access values
var learning_rate = config.get[Float32]("learning_rate")
var batch_size = config.get[Int]("batch_size")
var num_epochs = config.get[Int]("num_epochs")

# Modify and save
config.set("learning_rate", 0.005)
save_config(config, "configs/lenet5_tuned.yaml")
```

### Random Seed Management (`random.mojo`)

Utilities for reproducible experiments.

#### set_seed

Set global random seed:

```mojo
fn set_seed(seed: Int):
    """
    Set random seed for all random number generators.

    Ensures reproducible results by seeding:
    - Mojo standard library RNG
    - Custom RNGs in shared library

    Args:
        seed: Random seed value

    Example:
        set_seed(42)  # For reproducibility
    """
    # Set Mojo stdlib seed
    # Set custom RNG seeds
    pass
```

#### get_random_state / set_random_state

Save and restore random state:

```mojo
struct RandomState:
    """Container for random number generator state."""
    var state: List[UInt64]

fn get_random_state() -> RandomState:
    """Get current RNG state."""
    pass

fn set_random_state(state: RandomState):
    """Restore RNG state."""
    pass
```

**Usage**:

```mojo
from shared.utils import set_seed, get_random_state, set_random_state

# Set seed at start of experiment
set_seed(42)

# Save state before validation
var state = get_random_state()

# Restore state after validation
set_random_state(state)
```

### Profiling and Timing (`profiling.mojo`)

Performance measurement utilities.

#### Timer

Context manager for timing code blocks:

```mojo
struct Timer:
    """Context manager for timing code execution."""
    var name: String
    var start_time: Float64

    fn __init__(inout self, name: String = ""):
        """Create timer with optional name."""
        self.name = name
        self.start_time = 0.0

    fn __enter__(inout self):
        """Start timing."""
        self.start_time = now()

    fn __exit__(inout self):
        """Stop timing and print result."""
        var elapsed = now() - self.start_time
        var msg = f"{self.name}: " if self.name else ""
        print(f"{msg}{elapsed:.4f}s")
```

**Usage**:

```mojo
from shared.utils import Timer

with Timer("Forward pass"):
    var output = model.forward(batch.inputs)

with Timer("Backward pass"):
    var grads = compute_gradients(loss, model)
```

#### profile decorator

Profile function execution:

```mojo
fn profile[func: fn() -> None]():
    """
    Profile function execution.

    Measures:
    - Execution time
    - Memory usage
    - Number of calls

    Example:
        @profile
        fn train_epoch():
            # training code
            pass
    """
    # Measure time and memory
    # Print statistics
    pass
```

#### memory_usage

Get current memory usage:

```mojo
fn memory_usage() -> MemoryStats:
    """
    Get current memory usage statistics.

    Returns:
        Memory statistics (allocated, peak, available)

    Example:
        var mem = memory_usage()
        print(f"Memory: {mem.allocated_mb}MB / {mem.peak_mb}MB")
    """
    pass
```

## Usage Examples

### Complete Training Setup

```mojo
from shared.utils import (
    Logger, StreamHandler, FileHandler,
    load_config, set_seed,
    Timer, plot_training_curves
)

# Setup logging
var logger = Logger("lenet5")
logger.add_handler(StreamHandler())
logger.add_handler(FileHandler("logs/lenet5.log"))

# Load configuration
var config = load_config("configs/lenet5.yaml")
var lr = config.get[Float32]("learning_rate")
var epochs = config.get[Int]("epochs")

# Set seed for reproducibility
set_seed(config.get[Int]("seed"))

logger.info(f"Starting training with lr={lr}, epochs={epochs}")

# Training loop with timing
var train_losses = List[Float32]()
var val_losses = List[Float32]()

for epoch in range(epochs):
    with Timer(f"Epoch {epoch}"):
        var train_loss = train_epoch(model, train_loader, optimizer)
        var val_loss = validate_epoch(model, val_loader)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        logger.info(f"Epoch {epoch}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")

# Plot results
plot_training_curves(train_losses, val_losses, save_path="results/curves.png")
logger.info("Training complete!")
```

### Experiment Configuration

```yaml
# configs/lenet5.yaml
seed: 42
learning_rate: 0.001
batch_size: 128
epochs: 50
optimizer: sgd
momentum: 0.9
weight_decay: 0.0005

model:
  num_classes: 10

data:
  root: ./data
  augmentation: true
```

### Debugging with Logging

```mojo
from shared.utils import Logger, LogLevel

# Create debug logger
var logger = Logger("debug", level=LogLevel.DEBUG)

# Log detailed information
logger.debug(f"Batch shape: {batch.inputs.shape}")
logger.debug(f"Model parameters: {model.num_parameters()}")
logger.debug(f"Gradient norm: {gradient_norm(grads)}")

# Log warnings for potential issues
if val_loss > train_loss * 1.5:
    logger.warning("Validation loss significantly higher than training loss")

# Log errors for failures
if not os.path.exists(checkpoint_path):
    logger.error(f"Checkpoint not found: {checkpoint_path}")
```

## Best Practices

### Logging

1. **Use Appropriate Levels**: DEBUG for development, INFO for production
2. **Structured Messages**: Use consistent format for log messages
3. **Context**: Include relevant context (epoch, batch, etc.)
4. **Performance**: Avoid logging in tight loops

### Configuration

1. **Version Control**: Commit config files to track experiments
2. **Hierarchical Configs**: Use base configs + overrides for experiments
3. **Validation**: Validate config values at load time
4. **Documentation**: Document all config options

### Reproducibility

1. **Always Set Seed**: Set seed at start of every experiment
2. **Record Everything**: Log seed, config, environment info
3. **Version Dependencies**: Record package versions used
4. **Save Checkpoints**: Save model, optimizer, RNG state

### Profiling

1. **Profile Before Optimizing**: Measure to find bottlenecks
2. **Profile Realistic Workloads**: Use real data, not toy examples
3. **Compare Fairly**: Use same conditions for comparisons
4. **Focus on Hot Paths**: Optimize most time-consuming operations

## Testing

Utility functions should be tested for:

1. **Logging**: Messages written correctly to handlers
2. **Config**: Loading, saving, merging work correctly
3. **Reproducibility**: Same seed produces same results
4. **Timing**: Timer measurements are accurate
5. **Visualization**: Plots generated without errors

See `tests/shared/utils/` for comprehensive test suite.

## Integration with Other Modules

Utilities are used throughout the library:

```mojo
# In training loops
from shared.utils import Logger, Timer
var logger = Logger("training")
with Timer("Epoch"):
    loss = train_epoch(model, loader, optimizer)
    logger.info(f"Loss: {loss}")

# In data loading
from shared.utils import set_seed
set_seed(42)  # For reproducible data shuffling

# In model development
from shared.utils import plot_training_curves
plot_training_curves(history.train_loss, history.val_loss)
```

## Future Enhancements

Planned features for future releases:

1. **TensorBoard Integration**: Native TensorBoard logging
2. **W&B Integration**: Weights & Biases experiment tracking
3. **Advanced Profiling**: GPU profiling, kernel-level profiling
4. **Distributed Logging**: Aggregate logs from multiple workers
5. **Interactive Visualization**: Real-time training visualization
6. **Experiment Tracking**: Database for experiment management

## References

- [Python logging](https://docs.python.org/3/library/logging.html)
- [Matplotlib Documentation](https://matplotlib.org/stable/contents.html)
- [Mojo Performance Guide](https://docs.modular.com/mojo/manual/performance/)

## Contributing

When adding new utilities:

1. Keep utilities general and reusable
2. Add comprehensive tests
3. Document with examples
4. Follow existing patterns
5. Update this README
