#!/usr/bin/env python3
"""
Claude pretooluse-input hook in Python
Blocks destructive bash commands.

Contract:
 - exit 0  => allow
 - exit !=0 => block (must emit JSON)

Features:
 - Handles multi-command bash input separated by ;, &&, ||, or |, respecting quotes.
 - Detects destructive commands inside pipelines and compound commands.
 - Blocks dangerous system-level commands, recursive permission/ownership changes,
   destructive `rm` usage, and destructive `find`, `xargs`, `git clean`, `rsync` usage.
"""

import json
import os
import sys
import shlex
from pathlib import Path

EXIT_ALLOW = 0
EXIT_BLOCK = 1


def block(msg: str):
    """Emit JSON reason and exit with block code."""
    print(json.dumps({"action": "block", "reason": msg}))
    sys.exit(EXIT_BLOCK)


def validate_path(p: str, project_root: Path, home_dir: Path):
    """
    Validate file paths for destructive commands.

    Blocks:
     - Filesystem root "/"
     - Any .git directories
     - Home directory outside project root
     - Paths outside project root
     - Any rm target that does not exist
    """
    p = os.path.expandvars(os.path.expanduser(p))
    p = Path(p).resolve()

    if p == Path("/"):
        block("targeting filesystem root")
    if ".git" in p.parts:
        block("targeting .git directory")

    try:
        p.relative_to(project_root)
    except ValueError:
        if home_dir in p.parents or p == home_dir:
            block("targeting home directory outside project root")
        block(f"outside project root: {p}")

    # New: rm target must exist
    if not p.exists():
        block(f"rm target does not exist: {p}")


def split_commands(cmd: str):
    """
    Split a bash command string into individual subcommands.

    Handles compound commands separated by:
      - ';'  : sequential commands
      - '&&' : logical AND
      - '||' : logical OR
      - '|'  : pipeline
    Respects quoted arguments.

    Example:
        "echo hi | rm file.txt && echo bye"
    Returns:
        ["echo hi", "rm file.txt", "echo bye"]
    """
    tokens = shlex.split(cmd, posix=True)
    parts = []
    current = []
    # Iterate over each token and split on separators
    for t in tokens:
        if t in (";", "&&", "||", "|"):
            if current:
                parts.append(" ".join(current))
                current = []
        else:
            current.append(t)
    if current:
        parts.append(" ".join(current))
    return parts


def main():
    payload = json.load(sys.stdin)

    tool = payload.get("tool")
    if tool != "bash":
        sys.exit(EXIT_ALLOW)

    cmd = payload.get("input", {}).get("command", "")
    cwd = payload.get("input", {}).get("cwd", os.getcwd())

    if not cmd:
        sys.exit(EXIT_ALLOW)

    project_root = Path(os.environ.get("PROJECT_ROOT", Path(cwd).resolve()))
    home_dir = Path.home()

    # Split multi-command input correctly
    try:
        parts = split_commands(cmd)
    except Exception:
        block("failed to parse command")

    for subcmd in parts:
        # Remove sudo prefix
        if subcmd.startswith("sudo "):
            subcmd = subcmd[5:].strip()

        # Dangerous shell expansions
        if any(x in subcmd for x in ("$(", "`", "${")):
            block("dangerous shell expansion detected")

        # Tokenize arguments
        try:
            tokens = shlex.split(subcmd)
        except Exception:
            block("failed to parse subcommand tokens")

        if not tokens:
            continue

        cmdword = tokens[0]

        # ---- destructive checks ----
        if cmdword in ("rm", "unlink"):
            if "--no-preserve-root" in tokens:
                block("rm uses --no-preserve-root")
            args = [t for t in tokens[1:] if not t.startswith("-")]
            if not args:
                block("rm with no paths")
            for p in args:
                validate_path(p, project_root, home_dir)

        elif cmdword == "find":
            # Detect destructive find usage
            if "-delete" in tokens or ("-exec" in tokens and "rm" in tokens):
                block("destructive find usage")

        elif cmdword == "xargs":
            # Detect xargs with rm
            if "rm" in tokens:
                block("xargs rm is blocked")

        elif cmdword == "git":
            # Block git clean with dangerous flags
            if "clean" in tokens and any(f in tokens for f in ("-f", "-d", "-x", "-fdx")):
                block("git clean with force flags")

        elif cmdword == "rsync":
            if "--delete" in tokens:
                block("rsync --delete is blocked")

        elif cmdword in ("chmod", "chown", "chgrp"):
            if "-R" in tokens:
                block("recursive permission or ownership change blocked")

        elif cmdword in ("dd", "wipefs", "mount", "umount") or cmdword.startswith("mkfs"):
            block("dangerous system-level command blocked")

        elif cmdword == "tar":
            for t in tokens[1:]:
                # Block extraction flags like -xf
                if t.startswith("-") and "x" in t:
                    block("tar extraction blocked")

        elif cmdword == "unzip":
            block("unzip blocked")

    # All checks passed
    sys.exit(EXIT_ALLOW)


if __name__ == "__main__":
    main()
