#!/usr/bin/env bash
#
# Pre-execution hook for Bash commands
# Validates rm commands to prevent dangerous operations
#
# This hook is called before executing Bash commands in Claude Code.
# It blocks dangerous rm patterns while allowing safe operations within the project.

set -euo pipefail

# Get the command to be executed
COMMAND="$1"

# Get the project root (absolute path)
PROJECT_ROOT="${PROJECT_ROOT:-$(git rev-parse --show-toplevel 2>/dev/null || pwd)}"

# Function to check if a path is within the project directory
is_within_project() {
    local target_path="$1"

    # Resolve to absolute path
    if [[ "$target_path" = /* ]]; then
        # Already absolute
        local abs_path="$target_path"
    else
        # Relative path - resolve from current directory
        abs_path="$(cd "$(dirname "$target_path")" 2>/dev/null && pwd)/$(basename "$target_path")" || return 1
    fi

    # Check if the resolved path starts with PROJECT_ROOT
    [[ "$abs_path" == "$PROJECT_ROOT"* ]]
}

# Function to validate rm commands
validate_rm_command() {
    local cmd="$1"

    # Extract the rm command and its arguments
    # Handle various forms: rm, rm -rf, sudo rm, etc.

    # Block dangerous patterns

    # Pattern 1: rm -rf / or variations
    if echo "$cmd" | grep -qE '\brm\b.*-[rRf]*[rR][fF]*\s+/(\s|$)'; then
        echo "ERROR: Blocked dangerous command - attempting to delete root directory" >&2
        echo "Command: $cmd" >&2
        return 1
    fi

    # Pattern 2: rm targeting .git directory or files
    if echo "$cmd" | grep -qE '\brm\b.*\.git(/|$|\s)'; then
        echo "ERROR: Blocked dangerous command - attempting to delete .git directory or files" >&2
        echo "Command: $cmd" >&2
        return 1
    fi

    # Pattern 3: rm with paths outside project directory
    # Extract paths from rm command (after flags)
    local paths
    paths=$(echo "$cmd" | sed -n 's/.*\brm\b\s*\(-[rRfv]*\s*\)*\(.*\)/\2/p')

    if [[ -n "$paths" ]]; then
        # Check each path
        for path in $paths; do
            # Skip flags
            [[ "$path" =~ ^- ]] && continue

            # Check if path is absolute and outside project
            if [[ "$path" = /* ]]; then
                if ! is_within_project "$path"; then
                    echo "ERROR: Blocked dangerous command - attempting to delete files outside project directory" >&2
                    echo "Path: $path" >&2
                    echo "Project root: $PROJECT_ROOT" >&2
                    echo "Command: $cmd" >&2
                    return 1
                fi
            fi

            # Check if path resolves to parent directories using ../
            if [[ "$path" =~ \.\./.*\.\. ]]; then
                # Multiple levels of parent traversal - check carefully
                if ! is_within_project "$path"; then
                    echo "ERROR: Blocked dangerous command - path escapes project directory" >&2
                    echo "Path: $path" >&2
                    echo "Project root: $PROJECT_ROOT" >&2
                    echo "Command: $cmd" >&2
                    return 1
                fi
            fi
        done
    fi

    # Pattern 4: sudo rm (requires extra caution)
    if echo "$cmd" | grep -qE '\bsudo\s+rm\b'; then
        echo "WARNING: Command uses sudo with rm - use extreme caution" >&2
        echo "Command: $cmd" >&2
        # Don't block, but warn
    fi

    return 0
}

# Main validation logic
if echo "$COMMAND" | grep -qE '\brm\b'; then
    # Command contains rm - validate it
    if ! validate_rm_command "$COMMAND"; then
        exit 1
    fi
fi

# If we get here, the command is safe
exit 0
