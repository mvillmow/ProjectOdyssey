#!/bin/bash
#
# Create Benchmark Distribution Package
#
# This script creates a distributable archive for the ML Odyssey
# benchmarking infrastructure.
#
# Usage:
#   ./scripts/create_benchmark_distribution.sh
#
# Outputs:
#   dist/benchmarks-0.1.0.tar.gz - Distribution archive
#
# Prerequisites:
#   - tar command available
#   - All benchmark files in place
#

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

echo "=========================================="
echo "Benchmark Distribution Package Creator"
echo "=========================================="
echo "Repository root: $REPO_ROOT"
echo ""

# 1. Create dist directory
echo "Step 1: Creating dist/ directory..."
mkdir -p dist
echo "✅ dist/ directory ready"
echo ""

# 2. Verify required files exist
echo "Step 2: Verifying required files..."
REQUIRED_FILES=(
    "benchmarks/README.md"
    "benchmarks/scripts/run_benchmarks.mojo"
    "benchmarks/scripts/compare_results.mojo"
    "benchmarks/baselines/baseline_results.json"
    "LICENSE"
)

MISSING=()
for file in "${REQUIRED_FILES[@]}"; do
    if [ ! -f "$file" ]; then
        MISSING+=("$file")
        echo "  ✗ $file (MISSING)"
    else
        echo "  ✓ $file"
    fi
done

if [ ${#MISSING[@]} -gt 0 ]; then
    echo ""
    echo "❌ ERROR: Missing required files"
    for file in "${MISSING[@]}"; do
        echo "  - $file"
    done
    exit 1
fi

echo "✅ All required files present"
echo ""

# 3. Create results directory with .gitkeep
echo "Step 3: Ensuring results/ directory exists..."
mkdir -p benchmarks/results
if [ ! -f "benchmarks/results/.gitkeep" ]; then
    cat > benchmarks/results/.gitkeep <<'EOF'
# This file preserves the results/ directory in version control
# Benchmark results will be stored here with timestamps
EOF
    echo "  Created benchmarks/results/.gitkeep"
fi
echo "✅ benchmarks/results/ directory ready"
echo ""

# 4. Create the distribution archive
echo "Step 4: Creating distribution archive..."
ARCHIVE_NAME="benchmarks-0.1.0.tar.gz"
ARCHIVE_PATH="dist/$ARCHIVE_NAME"

# Remove old archive if it exists
if [ -f "$ARCHIVE_PATH" ]; then
    echo "  Removing old archive: $ARCHIVE_PATH"
    rm "$ARCHIVE_PATH"
fi

# Create tar.gz archive
tar -czf "$ARCHIVE_PATH" \
    benchmarks/ \
    LICENSE

echo "✅ Created archive: $ARCHIVE_PATH"
echo ""

# 5. Verify archive
echo "Step 5: Verifying archive..."
if [ ! -f "$ARCHIVE_PATH" ]; then
    echo "❌ ERROR: Archive not found at $ARCHIVE_PATH"
    exit 1
fi

ARCHIVE_SIZE=$(stat -c%s "$ARCHIVE_PATH" 2>/dev/null || stat -f%z "$ARCHIVE_PATH" 2>/dev/null)
ARCHIVE_SIZE_KB=$((ARCHIVE_SIZE / 1024))

echo "  Archive: $ARCHIVE_PATH"
echo "  Size: $ARCHIVE_SIZE bytes ($ARCHIVE_SIZE_KB KB)"
echo ""

# 6. List archive contents
echo "Step 6: Archive contents (first 20 files):"
tar -tzf "$ARCHIVE_PATH" | head -20 | while read -r line; do
    echo "  $line"
done

TOTAL_FILES=$(tar -tzf "$ARCHIVE_PATH" | wc -l)
if [ "$TOTAL_FILES" -gt 20 ]; then
    echo "  ... and $((TOTAL_FILES - 20)) more files"
fi
echo "  Total files: $TOTAL_FILES"
echo ""

# 7. Test extraction
echo "Step 7: Testing archive extraction..."
TEST_DIR="dist/test_extract_$$"
mkdir -p "$TEST_DIR"

tar -xzf "$ARCHIVE_PATH" -C "$TEST_DIR"

# Verify key files
echo "  Verifying extracted files:"
TEST_FILES=(
    "benchmarks/README.md"
    "benchmarks/scripts/run_benchmarks.mojo"
    "benchmarks/scripts/compare_results.mojo"
    "benchmarks/baselines/baseline_results.json"
    "LICENSE"
)

for file in "${TEST_FILES[@]}"; do
    if [ -f "$TEST_DIR/$file" ]; then
        echo "    ✓ $file"
    else
        echo "    ✗ $file (MISSING)"
    fi
done

# Cleanup test extraction
rm -rf "$TEST_DIR"
echo "✅ Archive extracts successfully"
echo ""

# 8. Summary
echo "=========================================="
echo "PACKAGE CREATION COMPLETE"
echo "=========================================="
echo "Distribution archive: $ARCHIVE_PATH"
echo "Archive size: $ARCHIVE_SIZE bytes ($ARCHIVE_SIZE_KB KB)"
echo "Total files: $TOTAL_FILES"
echo ""
echo "To install:"
echo "  tar -xzf dist/$ARCHIVE_NAME"
echo "  cd benchmarks"
echo "  mojo scripts/run_benchmarks.mojo"
echo ""
echo "CI/CD workflow: .github/workflows/benchmark.yml"
echo "=========================================="
