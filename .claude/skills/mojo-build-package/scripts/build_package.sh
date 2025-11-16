#!/usr/bin/env bash
#
# Build a Mojo package
#
# Usage:
#   ./build_package.sh <package-name> [--test]

set -euo pipefail

PACKAGE_NAME="${1:-}"
RUN_TESTS=false

if [[ -z "$PACKAGE_NAME" ]]; then
    echo "Error: Package name required"
    echo "Usage: $0 <package-name> [--test]"
    exit 1
fi

# Parse arguments
for arg in "$@"; do
    if [[ "$arg" == "--test" ]]; then
        RUN_TESTS=true
    fi
done

# Check if source directory exists
SRC_DIR="src/$PACKAGE_NAME"
if [[ ! -d "$SRC_DIR" ]]; then
    echo "Error: Source directory not found: $SRC_DIR"
    exit 1
fi

# Check for __init__.mojo
if [[ ! -f "$SRC_DIR/__init__.mojo" ]]; then
    echo "Error: Missing __init__.mojo in $SRC_DIR"
    echo "Every package must have an __init__.mojo file"
    exit 1
fi

# Create packages directory
PACKAGES_DIR="packages"
mkdir -p "$PACKAGES_DIR"

echo "Building package: $PACKAGE_NAME"
echo "Source: $SRC_DIR"
echo "Output: $PACKAGES_DIR/$PACKAGE_NAME.mojopkg"
echo ""

# Build package
if mojo package "$SRC_DIR" -o "$PACKAGES_DIR/$PACKAGE_NAME.mojopkg"; then
    echo ""
    echo "✅ Package built successfully: $PACKAGES_DIR/$PACKAGE_NAME.mojopkg"

    # Get file size
    SIZE=$(du -h "$PACKAGES_DIR/$PACKAGE_NAME.mojopkg" | cut -f1)
    echo "Size: $SIZE"

    # Run tests if requested
    if [[ "$RUN_TESTS" == true ]]; then
        echo ""
        echo "Running package tests..."
        if [[ -x "scripts/test_package.sh" ]]; then
            ./scripts/test_package.sh "$PACKAGE_NAME"
        else
            echo "⚠️  No test script found"
        fi
    fi
else
    echo ""
    echo "❌ Package build failed"
    exit 1
fi
