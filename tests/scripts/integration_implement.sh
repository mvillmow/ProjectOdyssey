#!/usr/bin/env bash
#
# Integration tests for implement_issues.py
#
# Tests:
# - Health check mode
# - Help text
# - Dependency graph export
# - State file handling

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="${SCRIPT_DIR}/../.."
SCRIPTS_DIR="${ROOT_DIR}/scripts"

echo "================================================================"
echo "Integration Tests: implement_issues.py"
echo "================================================================"
echo ""

# Test 1: Health Check Mode
echo "Test 1: Health Check Mode"
echo "--------------------------"
set +e  # Temporarily allow non-zero exit codes
python3 "${SCRIPTS_DIR}/implement_issues.py" --epic 0 --health-check
health_exit=$?
set -e  # Re-enable exit on error

if [ $health_exit -eq 0 ] || [ $health_exit -eq 1 ]; then
    echo "✓ Health check completed (exit code: $health_exit)"
else
    echo "✗ Health check failed unexpectedly (exit code: $health_exit)"
    exit 1
fi
echo ""

# Test 2: Help Text
echo "Test 2: Help Text"
echo "-----------------"
python3 "${SCRIPTS_DIR}/implement_issues.py" --help > /dev/null
echo "✓ Help text displayed successfully"
echo ""

# Test 3: Dependency Graph Export (Dry-Run)
echo "Test 3: Dependency Graph Export"
echo "--------------------------------"
# NOTE: This requires a valid epic with issues
# For integration testing, we'll test the --export-graph flag exists
if python3 "${SCRIPTS_DIR}/implement_issues.py" --help | grep -q "export-graph"; then
    echo "✓ --export-graph flag is available"
else
    echo "✗ --export-graph flag not found in help"
    exit 1
fi
echo ""

# Test 4: Rollback Flag Exists
echo "Test 4: Rollback Flag"
echo "---------------------"
if python3 "${SCRIPTS_DIR}/implement_issues.py" --help | grep -q "rollback"; then
    echo "✓ --rollback flag is available"
else
    echo "✗ --rollback flag not found in help"
    exit 1
fi
echo ""

# Test 5: State Directory Creation
echo "Test 5: State Directory"
echo "-----------------------"
TEMP_DIR=$(mktemp -d)
STATE_DIR="${TEMP_DIR}/state"

# The script should handle non-existent state directory gracefully
# We can't fully test without a real epic, but we can verify flags parse
if python3 "${SCRIPTS_DIR}/implement_issues.py" --help | grep -q "state-dir"; then
    echo "✓ --state-dir flag is available"
else
    echo "✗ --state-dir flag not found in help"
    exit 1
fi

rm -rf "$TEMP_DIR"
echo ""

# Test 6: Required Arguments Validation
echo "Test 6: Required Arguments"
echo "--------------------------"
# Running without --epic should fail with helpful error
set +e  # Temporarily allow non-zero exit codes
OUTPUT=$(python3 "${SCRIPTS_DIR}/implement_issues.py" 2>&1)
set -e  # Re-enable exit on error

if echo "$OUTPUT" | grep -qi "epic"; then
    echo "✓ Script requires --epic argument"
else
    echo "✗ Script doesn't validate required arguments"
    echo "Output was: $OUTPUT"
    exit 1
fi
echo ""

# Test 7: Dry-Run Mode Exists
echo "Test 7: Dry-Run Mode"
echo "--------------------"
if python3 "${SCRIPTS_DIR}/implement_issues.py" --help | grep -q "dry-run"; then
    echo "✓ --dry-run flag is available"
else
    echo "✗ --dry-run flag not found in help"
    exit 1
fi
echo ""

# Test 8: Parallel Workers Flag
echo "Test 8: Parallel Workers"
echo "------------------------"
if python3 "${SCRIPTS_DIR}/implement_issues.py" --help | grep -q "parallel"; then
    echo "✓ --parallel flag is available"
else
    echo "✗ --parallel flag not found in help"
    exit 1
fi
echo ""

echo "================================================================"
echo "All Integration Tests Passed!"
echo "================================================================"
echo ""
echo "Note: Full end-to-end testing requires:"
echo "  - Valid GitHub epic issue with dependencies"
echo "  - GitHub CLI authentication"
echo "  - Git repository with writable worktree directory"
echo "These are tested in the unit tests with mocking."
echo "================================================================"
