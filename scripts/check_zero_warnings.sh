#!/usr/bin/env bash
# Verify zero-warnings policy for ML Odyssey builds

set -e

MODE="${1:-debug}"
NATIVE="${NATIVE:-0}"

if [ "$NATIVE" = "1" ]; then
    BUILD_TYPE="native"
else
    BUILD_TYPE="docker"
fi

echo "Building project ($BUILD_TYPE mode: $MODE) and checking for warnings..."

if [ "$NATIVE" = "1" ]; then
    build_output=$(NATIVE=1 just build "$MODE" 2>&1)
else
    build_output=$(just build "$MODE" 2>&1)
fi

warning_count=$(echo "$build_output" | grep -c "warning:" || echo "0")

if [ "$warning_count" -ne 0 ]; then
    echo "❌ FAILED: Found $warning_count warnings in $BUILD_TYPE build (mode: $MODE)"
    echo ""
    echo "First 20 warnings:"
    echo "$build_output" | grep "warning:" | head -20
    echo ""
    echo "Warning breakdown:"
    echo "$build_output" | grep "warning:" | sed 's/.*warning: //' | sort | uniq -c | sort -rn | head -10
    exit 1
fi

echo "✅ SUCCESS: Zero warnings achieved for $BUILD_TYPE build (mode: $MODE)!"
exit 0
