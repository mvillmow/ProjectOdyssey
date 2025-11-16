#!/usr/bin/env bash
#
# Run a specific test file
#
# Usage:
#   ./run_test.sh <test-file>
#
# Example:
#   ./run_test.sh "test_tensor_ops"

set -euo pipefail

TEST_FILE="${1:-}"

if [[ -z "$TEST_FILE" ]]; then
    echo "Error: Test file name required"
    echo "Usage: $0 <test-file>"
    exit 1
fi

# Find test file
FOUND_FILE=""
for dir in tests/unit tests/integration tests/performance tests; do
    if [[ -f "$dir/$TEST_FILE.mojo" ]]; then
        FOUND_FILE="$dir/$TEST_FILE.mojo"
        break
    elif [[ -f "$dir/$TEST_FILE.py" ]]; then
        FOUND_FILE="$dir/$TEST_FILE.py"
        break
    elif [[ -f "$dir/${TEST_FILE#test_}.mojo" ]]; then
        FOUND_FILE="$dir/${TEST_FILE#test_}.mojo"
        break
    elif [[ -f "$dir/${TEST_FILE#test_}.py" ]]; then
        FOUND_FILE="$dir/${TEST_FILE#test_}.py"
        break
    fi
done

if [[ -z "$FOUND_FILE" ]]; then
    echo "Error: Test file not found: $TEST_FILE"
    echo "Searched in: tests/unit, tests/integration, tests/performance"
    exit 1
fi

echo "Running: $FOUND_FILE"
echo ""

# Run appropriate test runner
if [[ "$FOUND_FILE" == *.mojo ]]; then
    mojo test "$FOUND_FILE"
else
    pytest -v "$FOUND_FILE"
fi
