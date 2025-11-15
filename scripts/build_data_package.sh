#!/bin/bash
# Build script for Data module package
# Usage: ./scripts/build_data_package.sh

set -e

echo "Building Data module package..."

# Get script directory and repository root
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
REPO_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"

cd "$REPO_ROOT"

# Create dist directory if it doesn't exist
echo "Creating dist/ directory..."
mkdir -p dist

# Build package
echo "Building package: dist/data-0.1.0.mojopkg"
mojo package shared/data -o dist/data-0.1.0.mojopkg

# Verify package was created
if [ ! -f dist/data-0.1.0.mojopkg ]; then
    echo "Error: Package file was not created"
    exit 1
fi

# Show package information
echo ""
echo "Package created successfully:"
ls -lh dist/data-0.1.0.mojopkg
echo ""
file dist/data-0.1.0.mojopkg || echo "file command not available"

echo ""
echo "✅ Build complete!"
echo "Package: dist/data-0.1.0.mojopkg"
echo ""
echo "Next steps:"
echo "1. Make verification script executable: chmod +x scripts/install_verify_data.sh"
echo "2. Run verification: ./scripts/install_verify_data.sh"
