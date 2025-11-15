#!/bin/bash
set -euo pipefail

VERSION="0.1.0"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PACKAGE_PATH="$SCRIPT_DIR/../dist/utils-${VERSION}.mojopkg"

# Verify package exists
if [[ ! -f "${PACKAGE_PATH}" ]]; then
    echo "❌ ERROR: Package not found at ${PACKAGE_PATH}"
    exit 1
fi

# Create clean test environment
TEMP_DIR=$(mktemp -d) || { echo "❌ Failed to create temp directory"; exit 1; }
trap 'rm -rf "$TEMP_DIR"' EXIT

cd "$TEMP_DIR"

# Install package
echo "Installing package from ${PACKAGE_PATH}..."
mojo install "${PACKAGE_PATH}"

# Test core exports (all 50 exports across 6 modules)
echo "Testing utils exports..."
cat > test_imports.mojo << 'EOF'
# Logging utilities (10 exports)
from utils import Logger, LogLevel, get_logger
from utils import StreamHandler, FileHandler, LogRecord
from utils import SimpleFormatter, TimestampFormatter, DetailedFormatter, ColoredFormatter

# Configuration (5 exports)
from utils import Config, load_config, save_config, merge_configs, ConfigValidator

# File I/O utilities (11 exports)
from utils import Checkpoint, save_checkpoint, load_checkpoint
from utils import serialize_tensor, deserialize_tensor
from utils import safe_write_file, safe_read_file, create_backup
from utils import file_exists, directory_exists, create_directory

# Visualization (8 exports)
from utils import plot_training_curves, plot_loss_only, plot_accuracy_only
from utils import plot_confusion_matrix, visualize_model_architecture
from utils import show_images, visualize_feature_maps, save_figure

# Random utilities (9 exports)
from utils import set_seed, get_global_seed
from utils import get_random_state, set_random_state, RandomState
from utils import random_uniform, random_normal, random_int, shuffle

# Profiling utilities (7 exports)
from utils import Timer, memory_usage
from utils import profile_function, benchmark_function
from utils import MemoryStats, TimingStats, ProfilingReport

fn main() raises:
    print("✅ All 50 critical imports successful!")
EOF

mojo run test_imports.mojo || { echo "❌ Import test failed"; exit 1; }

echo "✅ Installation verification complete! All 50 exports tested."
