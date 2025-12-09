#!/usr/bin/env bash
#
# Test script for pre-bash-exec.sh safety hooks
# Demonstrates validation of rm commands

set -euo pipefail

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Get project root
PROJECT_ROOT="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
export PROJECT_ROOT

# Path to the hook script
HOOK_SCRIPT=".claude/hooks/pre-bash-exec.sh"

echo "Testing Safety Hooks for rm Commands"
echo "====================================="
echo ""
echo "Project Root: $PROJECT_ROOT"
echo ""

# Test counter
total_tests=0
passed_tests=0
failed_tests=0

# Function to run a test
run_test() {
    local test_name="$1"
    local command="$2"
    local should_pass="$3"

    total_tests=$((total_tests + 1))

    echo -n "Test ${total_tests}: ${test_name}... "

    if bash "$HOOK_SCRIPT" "$command" 2>/dev/null; then
        result="PASSED"
        if [[ "$should_pass" == "true" ]]; then
            echo -e "${GREEN}✓ EXPECTED PASS${NC}"
            passed_tests=$((passed_tests + 1))
        else
            echo -e "${RED}✗ UNEXPECTED PASS (should have blocked)${NC}"
            failed_tests=$((failed_tests + 1))
        fi
    else
        result="BLOCKED"
        if [[ "$should_pass" == "false" ]]; then
            echo -e "${GREEN}✓ EXPECTED BLOCK${NC}"
            passed_tests=$((passed_tests + 1))
        else
            echo -e "${RED}✗ UNEXPECTED BLOCK (should have passed)${NC}"
            failed_tests=$((failed_tests + 1))
        fi
    fi
}

echo "=== Dangerous Commands (Should be BLOCKED) ==="
echo ""

# Test 1: rm -rf /
run_test "rm -rf /" "rm -rf /" "false"

# Test 2: rm -rf / (with space)
run_test "rm -rf / " "rm -rf / " "false"

# Test 3: rm targeting .git directory
run_test "rm -rf .git" "rm -rf .git" "false"

# Test 4: rm targeting .git/config
run_test "rm .git/config" "rm .git/config" "false"

# Test 5: rm with absolute path outside project
run_test "rm /etc/passwd" "rm /etc/passwd" "false"

# Test 6: rm with absolute path to /tmp
run_test "rm -rf /tmp/something" "rm -rf /tmp/something" "false"

# Test 7: sudo rm (warning, but not blocked in current implementation)
echo ""
echo "=== Warning Commands (May warn but not block) ==="
echo ""

run_test "sudo rm -rf /tmp/file" "sudo rm -rf /tmp/file" "false"

echo ""
echo "=== Safe Commands (Should PASS) ==="
echo ""

# Test 8: rm within project
run_test "rm temp.txt" "rm temp.txt" "true"

# Test 9: rm -rf within project subdirectory
run_test "rm -rf build/" "rm -rf build/" "true"

# Test 10: rm with relative path within project
run_test "rm ./logs/test.log" "rm ./logs/test.log" "true"

# Test 11: Non-rm command
run_test "ls -la" "ls -la" "true"

# Test 12: git command
run_test "git status" "git status" "true"

# Test 13: rm with multiple files in project
run_test "rm file1.txt file2.txt" "rm file1.txt file2.txt" "true"

echo ""
echo "=== Edge Cases ==="
echo ""

# Test 14: rm with flags only
run_test "rm -rf" "rm -rf" "true"

# Test 15: Command with rm in the middle (e.g., confirm)
run_test "echo 'yes' | rm file.txt" "echo 'yes' | rm file.txt" "true"

# Test 16: rm with absolute path within project
if [[ -d "$PROJECT_ROOT/build" ]]; then
    run_test "rm -rf $PROJECT_ROOT/build" "rm -rf $PROJECT_ROOT/build" "true"
else
    echo "Skipping absolute path test (build directory doesn't exist)"
fi

echo ""
echo "=== Test Summary ==="
echo "==================="
echo "Total tests: $total_tests"
echo -e "${GREEN}Passed: $passed_tests${NC}"
echo -e "${RED}Failed: $failed_tests${NC}"
echo ""

if [[ $failed_tests -eq 0 ]]; then
    echo -e "${GREEN}All tests passed!${NC}"
    exit 0
else
    echo -e "${RED}Some tests failed. Please review the output above.${NC}"
    exit 1
fi
