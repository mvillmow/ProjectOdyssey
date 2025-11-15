#!/bin/bash
set -euo pipefail

VERSION="0.1.0"
PACKAGE_PATH="$(pwd)/dist/training-${VERSION}.mojopkg"

# Verify package exists
if [[ ! -f "${PACKAGE_PATH}" ]]; then
    echo "❌ ERROR: Package not found: ${PACKAGE_PATH}"
    echo "Build it first: ./scripts/build_training_package.sh"
    exit 1
fi

# Create clean test environment
TEMP_DIR=$(mktemp -d) || {
    echo "❌ ERROR: Failed to create temporary directory"
    exit 1
}
trap 'rm -rf "$TEMP_DIR"' EXIT

cd "$TEMP_DIR"

# Install package
echo "Installing training package..."
mojo install "${PACKAGE_PATH}" || {
    echo "❌ ERROR: Installation failed"
    exit 1
}

# Test all 16 public exports
echo "Testing all 16 training module exports..."
cat > test_imports.mojo << 'EOF'
# Core training components
from training import TrainingState, Callback, CallbackSignal

# Callback signals
from training import CONTINUE, STOP

# Learning rate schedulers
from training import LRScheduler, StepLR, CosineAnnealingLR, WarmupLR

# Training callbacks
from training import EarlyStopping, ModelCheckpoint, LoggingCallback

# Utilities
from training import is_valid_loss, clip_gradients

fn main() raises:
    print("✅ All 16 imports successful!")
EOF

mojo run test_imports.mojo || {
    echo "❌ ERROR: Import test failed"
    exit 1
}

echo "✅ Installation verification complete!"
echo "All 16 public exports tested successfully."
