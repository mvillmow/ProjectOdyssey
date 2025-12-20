#!/usr/bin/env bash
#
# Pre-execution hook for Bash commands
# Validates rm commands to prevent dangerous operations

set -euo pipefail

COMMAND="$1"

PROJECT_ROOT="${PROJECT_ROOT:-$(git rev-parse --show-toplevel 2>/dev/null || pwd)}"
HOME_DIR="$(cd ~ && pwd)"

# Resolve a path to an absolute canonical path (best-effort, no eval)
# Returns empty string if parent directory doesn't exist (indicates unsafe path)
resolve_path() {
    local p="$1"

    # Expand ~ manually
    if [[ "$p" == "~" ]]; then
        printf "%s\n" "$HOME_DIR"
        return
    elif [[ "$p" == "~/"* ]]; then
        printf "%s\n" "$HOME_DIR/${p#~/}"
        return
    fi

    # Absolute path
    if [[ "$p" == /* ]]; then
        printf "%s\n" "$p"
        return
    fi

    # Relative path - check if parent directory exists
    local dir
    dir="$(dirname "$p")"

    # If directory doesn't exist, return special marker to block
    if [[ "$dir" != "." && ! -d "$dir" ]]; then
        printf "__NONEXISTENT__\n"
        return
    fi

    # Handle ./ prefix
    if [[ "$p" == "./"* ]]; then
        p="${p#./}"
    fi

    # Use pwd to get absolute path
    printf "%s\n" "$(pwd)/$p"
}

is_within_project() {
    local abs
    abs="$(resolve_path "$1")" || return 1
    [[ "$abs" == "$PROJECT_ROOT"* ]]
}

validate_rm_command() {
    local cmd="$1"

    # Normalize sudo rm → rm
    cmd="${cmd#sudo }"

    # --- HARD BLOCKS ---

    # Root deletion
    if echo "$cmd" | grep -Eq '\brm\b.*\s+/\s*$'; then
        echo "ERROR: Blocked rm of /" >&2
        return 1
    fi

    # Home directory deletion
    if echo "$cmd" | grep -Eq '\brm\b.*\s+(~/?|\$HOME(/|$))'; then
        echo "ERROR: Blocked rm targeting home directory" >&2
        return 1
    fi

    # .git deletion
    if echo "$cmd" | grep -Eq '\brm\b.*\s+\.git(/|$|\s)'; then
        echo "ERROR: Blocked rm targeting .git" >&2
        return 1
    fi

    # Extract arguments after rm (use [[:space:]] for macOS compatibility)
    local args
    args="$(echo "$cmd" | sed -n 's/.*rm[[:space:]]*\(-[A-Za-z]*[[:space:]]*\)*\(.*\)/\2/p')"

    for token in $args; do
        [[ "$token" =~ ^- ]] && continue

        local abs
        abs="$(resolve_path "$token")"

        # Block paths with non-existent parent directories (likely typos)
        if [[ "$abs" == "__NONEXISTENT__" ]]; then
            echo "ERROR: Blocked rm targeting non-existent directory" >&2
            return 1
        fi

        # Block root traversal
        if [[ "$abs" == "/" ]]; then
            echo "ERROR: Blocked rm targeting /" >&2
            return 1
        fi

        # Allow if within project root (check this FIRST)
        if [[ "$abs" == "$PROJECT_ROOT"/* ]]; then
            continue
        fi

        # Block home directory itself or paths within home (but outside project)
        if [[ "$abs" == "$HOME_DIR" || "$abs" == "$HOME_DIR/"* ]]; then
            echo "ERROR: Blocked rm targeting home directory ($abs)" >&2
            return 1
        fi

        # Block any absolute path outside project
        if [[ "$abs" == /* ]]; then
            echo "ERROR: Blocked rm outside project root" >&2
            echo "Target: $abs" >&2
            return 1
        fi
    done

    return 0
}

if echo "$COMMAND" | grep -qE '\brm\b'; then
    validate_rm_command "$COMMAND" || exit 1
fi

exit 0
