#!/usr/bin/env bash
#
# Test script for pre-bash-exec.sh safety hooks

set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

PROJECT_ROOT="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
export PROJECT_ROOT

HOOK_SCRIPT=".claude/hooks/pre-bash-exec.sh"

total_tests=0
passed_tests=0
failed_tests=0

run_test() {
    local name="$1"
    local cmd="$2"
    local should_pass="$3"

    total_tests=$((total_tests + 1))
    echo -n "Test $total_tests: $name "

    if bash "$HOOK_SCRIPT" "$cmd" 2>/dev/null; then
        if [[ "$should_pass" == "true" ]]; then
            echo -e "${GREEN}✓ PASS${NC}"
            passed_tests=$((passed_tests + 1))
        else
            echo -e "${RED}✗ UNEXPECTED PASS${NC}"
            failed_tests=$((failed_tests + 1))
        fi
    else
        if [[ "$should_pass" == "false" ]]; then
            echo -e "${GREEN}✓ BLOCKED${NC}"
            passed_tests=$((passed_tests + 1))
        else
            echo -e "${RED}✗ UNEXPECTED BLOCK${NC}"
            failed_tests=$((failed_tests + 1))
        fi
    fi
}

echo "=== Dangerous ==="
run_test "rm -rf /" "rm -rf /" false
run_test "rm -rf / " "rm -rf / " false
run_test "rm -rf ~/" "rm -rf ~/" false
run_test "rm -rf \$HOME" "rm -rf \$HOME" false
run_test "rm .git" "rm .git" false
run_test "rm .git/config" "rm .git/config" false
run_test "rm /etc/passwd" "rm /etc/passwd" false
run_test "rm -rf /tmp/something" "rm -rf /tmp/something" false
run_test "sudo rm -rf /tmp/file" "sudo rm -rf /tmp/file" false
run_test "rm ./logs/test.log" "rm ./logs/test.log" false
run_test "rm logs/test.log" "rm logs/test.log" false

echo ""
echo "=== Safe ==="
run_test "rm temp.txt" "rm temp.txt" true
run_test "rm -rf build/" "rm -rf build/" true
run_test "rm ./tests/README.md" "rm ./tests/README.md" true
run_test "rm tests/README.md" "rm tests/README.md" true
run_test "rm file1.txt file2.txt" "rm file1.txt file2.txt" true
run_test "ls -la" "ls -la" true
run_test "git status" "git status" true
run_test "rm -rf" "rm -rf" true
run_test "echo yes | rm file.txt" "echo yes | rm file.txt" true

if [[ -d "$PROJECT_ROOT/build" ]]; then
    run_test "rm -rf absolute project path" "rm -rf $PROJECT_ROOT/build" true
fi

echo ""
echo "Passed: $passed_tests / $total_tests"
[[ "$failed_tests" -eq 0 ]]
