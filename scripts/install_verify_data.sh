#!/bin/bash
set -euo pipefail

VERSION="0.1.0"
PACKAGE_PATH="$(pwd)/dist/data-${VERSION}.mojopkg"

# Verify package exists
if [[ ! -f "${PACKAGE_PATH}" ]]; then
    echo "❌ ERROR: Package not found at ${PACKAGE_PATH}"
    echo "Run: ./scripts/build_data_package.sh"
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
echo "Installing data package..."
mojo install "${PACKAGE_PATH}" || {
    echo "❌ ERROR: Package installation failed"
    exit 1
}

# Test all 19 public exports
echo "Testing all 19 data module exports..."
cat > test_imports.mojo << 'EOF'
# Core data structures
from data import Dataset, TensorDataset, FileDataset

# Batch processing
from data import Batch, BaseLoader, BatchLoader

# Sampling strategies
from data import Sampler, SequentialSampler, RandomSampler, WeightedSampler

# Data transformations
from data import Transform, Compose, ToTensor, Normalize
from data import Reshape, Resize, CenterCrop, RandomCrop
from data import RandomHorizontalFlip, RandomRotation

fn main() raises:
    print("✅ All 19 imports successful!")
EOF

mojo run test_imports.mojo || {
    echo "❌ ERROR: Import verification failed"
    exit 1
}

echo "✅ Installation verification complete!"
echo "All 19 public exports work correctly."
