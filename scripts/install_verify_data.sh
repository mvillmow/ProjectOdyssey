#!/bin/bash
# Installation verification script for Data module package
# Usage: ./scripts/install_verify_data.sh

set -e

echo "Testing data package installation..."

# Get absolute path to package
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
REPO_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"
PACKAGE_PATH="$REPO_ROOT/dist/data-0.1.0.mojopkg"

if [ ! -f "$PACKAGE_PATH" ]; then
    echo "Error: Package not found at $PACKAGE_PATH"
    exit 1
fi

# Create temporary directory
TEMP_DIR=$(mktemp -d)
echo "Testing in temporary directory: $TEMP_DIR"

# Cleanup function
cleanup() {
    cd "$REPO_ROOT"
    rm -rf "$TEMP_DIR"
    echo "Cleanup complete"
}
trap cleanup EXIT

cd "$TEMP_DIR"

# Install package
echo "Installing package from $PACKAGE_PATH..."
mojo install "$PACKAGE_PATH"

# Test imports
echo "Testing imports..."
mojo run -c "from data import Dataset; print('Dataset import OK')"
mojo run -c "from data import TensorDataset; print('TensorDataset import OK')"
mojo run -c "from data import BatchLoader; print('BatchLoader import OK')"
mojo run -c "from data import Transform; print('Transform import OK')"
mojo run -c "from data import Compose; print('Compose import OK')"

echo ""
echo "✅ Data package verification complete!"
echo "All imports successful"
