#!/usr/bin/env bash
#
# Expanded test script for Claude pretooluse-input bash safety hook

set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

PROJECT_ROOT="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
export PROJECT_ROOT

# Locate hook
if [[ -f ".claude/hooks/pre-bash-exec.sh" ]]; then
    HOOK_SCRIPT=".claude/hooks/pre-bash-exec.sh"
else
    HOOK_SCRIPT="$HOME/.claude/hooks/pre-bash-exec.sh"
fi

if [[ ! -x "$HOOK_SCRIPT" ]]; then
    echo "Hook script not found or not executable: $HOOK_SCRIPT"
    exit 1
fi

total_tests=0
passed_tests=0
failed_tests=0

run_test() {
    local cmd="$1"
    local should_pass="$2"

    total_tests=$((total_tests + 1))
    echo -n "Test $total_tests: $cmd "

    payload="$(cat <<EOF
{
  "tool": "bash",
  "input": {
    "command": "$cmd",
    "cwd": "$PROJECT_ROOT"
  }
}
EOF
)"

    if echo "$payload" | bash "$HOOK_SCRIPT" >/dev/null 2>&1; then
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
# rm / unlink cases
run_test "rm -rf /" false
run_test "rm -rf / " false
run_test "rm -rf ~/" false
run_test "rm -rf \$HOME" false
run_test "rm .git" false
run_test "rm .git/config" false
run_test "rm /etc/passwd" false
run_test "rm -rf /tmp/something" false
run_test "sudo rm -rf /tmp/file" false
run_test "rm ./logs/test.log" false
run_test "rm logs/test.log" false
run_test "rm -rf" false
run_test "echo yes | rm file.txt" false
run_test "cp README.md README.md && rm -rf ~/" false
run_test "rm ./foo || echo yes" false
run_test "rm ./foo; echo yes" false
run_test "rm ./foo && rm ./bar" false

# Non-rm destructive commands
run_test "dd if=/dev/zero of=test.img" false
run_test "mkfs.ext4 /dev/sda" false
run_test "wipefs -a /dev/sda" false
run_test "mount /dev/sda /mnt" false
run_test "umount /mnt" false
run_test "chmod -R 777 ./tests" false
run_test "chown -R user:user ./tests" false
run_test "chgrp -R staff ./tests" false
run_test "tar -xf archive.tar" false
run_test "unzip file.zip" false

# find, xargs, git, rsync
run_test "find ./tests -delete" false
run_test "find ./tests -exec rm {} \\;" false
run_test "echo ./tests | xargs rm" false
run_test "git clean -fdx" false
run_test "rsync -av --delete src/ dst/" false

# dangerous shell expansions
run_test "echo \$(ls)" false
run_test "echo \`ls\`" false
run_test "echo \${PATH}" false

echo ""
echo "=== Safe ==="
run_test "rm ./tests/README.md" true
run_test "rm tests/README.md" true
run_test "rm README.md CLAUDE.md" true
run_test "ls -la" true
run_test "git status" true
run_test "echo hello world" true
run_test "mkdir -p ./tmp && touch ./tmp/file" true
run_test "echo foo | tee ./tmp/output" true
run_test "cp README.md ./tmp/" true

if [[ -d "$PROJECT_ROOT/build" ]]; then
    run_test "rm -rf $PROJECT_ROOT/build" true
fi

echo ""
echo "Passed: $passed_tests / $total_tests"

[[ "$failed_tests" -eq 0 ]]
