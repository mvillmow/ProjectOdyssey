#!/usr/bin/env bash
#
# Pre-execution hook for Bash commands
# Validates rm commands to prevent dangerous operations

set -euo pipefail

COMMAND="$1"

PROJECT_ROOT="${PROJECT_ROOT:-$(git rev-parse --show-toplevel 2>/dev/null || pwd)}"
HOME_DIR="$(cd ~ && pwd)"

# Resolve a path to an absolute canonical path (best-effort, no eval)
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

    # Relative path
    printf "%s\n" "$(cd "$(dirname "$p")" 2>/dev/null && pwd)/$(basename "$p")"
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

    # Extract arguments after rm
    local args
    args="$(echo "$cmd" | sed -n 's/.*\brm\b\s*\(-[A-Za-z]*\s*\)*\(.*\)/\2/p')"

    for token in $args; do
        [[ "$token" =~ ^- ]] && continue

        local abs
        abs="$(resolve_path "$token")"

        # Block home even if expanded
        if [[ "$abs" == "$HOME_DIR" || "$abs" == "$HOME_DIR/"* ]]; then
            echo "ERROR: Blocked rm targeting home directory ($abs)" >&2
            return 1
        fi

        # Block root traversal
        if [[ "$abs" == "/" ]]; then
            echo "ERROR: Blocked rm targeting /" >&2
            return 1
        fi

        # Absolute path outside project
        if [[ "$abs" == /* && "$abs" != "$PROJECT_ROOT"* ]]; then
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
