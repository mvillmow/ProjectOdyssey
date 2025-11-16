#!/usr/bin/env bash
#
# Run Mojo tests with filtering options
#
# Usage:
#   ./run_tests.sh [test-name] [--unit|--integration|--performance]

set -euo pipefail

TEST_FILTER="${1:-}"
TEST_TYPE=""

# Parse arguments
for arg in "$@"; do
    case "$arg" in
        --unit)
            TEST_TYPE="unit"
            ;;
        --integration)
            TEST_TYPE="integration"
            ;;
        --performance)
            TEST_TYPE="performance"
            ;;
    esac
done

# Determine test directory
if [[ -n "$TEST_TYPE" ]]; then
    TEST_DIR="tests/$TEST_TYPE"
else
    TEST_DIR="tests"
fi

# Verify test directory exists
if [[ ! -d "$TEST_DIR" ]]; then
    echo "Error: Test directory not found: $TEST_DIR"
    exit 1
fi

echo "Running tests from: $TEST_DIR"
if [[ -n "$TEST_FILTER" ]] && [[ "$TEST_FILTER" != --* ]]; then
    echo "Filter: $TEST_FILTER"
fi
echo ""

# Run tests
if [[ -n "$TEST_FILTER" ]] && [[ "$TEST_FILTER" != --* ]]; then
    # Run specific test file
    if [[ -f "$TEST_DIR/$TEST_FILTER.mojo" ]]; then
        mojo test "$TEST_DIR/$TEST_FILTER.mojo"
    elif [[ -f "$TEST_DIR/test_$TEST_FILTER.mojo" ]]; then
        mojo test "$TEST_DIR/test_$TEST_FILTER.mojo"
    else
        # Search for matching files
        MATCHES=$(find "$TEST_DIR" -type f -name "*$TEST_FILTER*.mojo" 2>/dev/null || true)
        if [[ -z "$MATCHES" ]]; then
            echo "Error: No test files matching: $TEST_FILTER"
            exit 1
        fi

        echo "Found matching tests:"
        echo "$MATCHES"
        echo ""

        # Run all matches
        while IFS= read -r test_file; do
            echo "Running: $test_file"
            mojo test "$test_file"
            echo ""
        done <<< "$MATCHES"
    fi
else
    # Run all tests in directory
    mojo test "$TEST_DIR"
fi
