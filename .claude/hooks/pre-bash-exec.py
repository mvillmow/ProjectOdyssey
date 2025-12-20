#!/usr/bin/env python3
"""
Claude pretooluse-input hook in Python
Blocks destructive bash commands.
Contract:
 - exit 0  => allow
 - exit !=0 => block (must emit JSON)
"""

import json
import os
import sys
import shlex
from pathlib import Path

EXIT_ALLOW = 0
EXIT_BLOCK = 1

def block(msg: str):
    print(json.dumps({"action": "block", "reason": msg}))
    sys.exit(EXIT_BLOCK)

def validate_path(p: str, project_root: Path, home_dir: Path):
    # Expand ~ and $HOME
    p = os.path.expandvars(os.path.expanduser(p))
    p = Path(p).resolve()

    # Block dangerous locations
    if p == Path("/"):
        block("targeting filesystem root")
    if ".git" in str(p.parts):
        block("targeting .git directory")

    # Block everything in HOME except project root
    try:
        p.relative_to(project_root)
    except ValueError:
        if home_dir in p.parents or p == home_dir:
            block("targeting home directory outside project root")
        # Block anything outside project root
        block(f"outside project root: {p}")

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

    # Split multi-command input on ;, &&, ||, | while respecting quotes
    try:
        # shlex.split does not handle ;, &&, ||, | directly, so split manually first
        import re
        split_pattern = r'(;|&&|\|\|?\|)'
        parts = [p.strip() for p in re.split(split_pattern, cmd) if p.strip() and not re.match(split_pattern, p)]
    except Exception:
        block("failed to parse command")

    for subcmd in parts:
        # Remove sudo
        if subcmd.startswith("sudo "):
            subcmd = subcmd[5:].strip()

        # Dangerous shell expansions
        if "$(" in subcmd or "`" in subcmd or "${" in subcmd:
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
            if "-delete" in tokens or "-exec" in tokens and "rm" in tokens:
                block("destructive find usage")
        elif cmdword == "xargs":
            if "rm" in tokens:
                block("xargs rm is blocked")
        elif cmdword == "git":
            if "clean" in tokens and any(f in tokens for f in ("-f", "-d", "-x", "-fdx")):
                block("git clean with force flags")
        elif cmdword == "rsync":
            if "--delete" in tokens:
                block("rsync --delete is blocked")
        elif cmdword in ("chmod", "chown", "chgrp"):
            if "-R" in tokens:
                block("recursive permission or ownership change blocked")
        elif cmdword in ("dd", "mkfs", "wipefs", "mount", "umount"):
            block("dangerous system-level command blocked")
        elif cmdword == "tar":
            for t in tokens[1:]:
                if "x" in t and t.startswith("-"):
                    block("tar extraction blocked")
        elif cmdword == "unzip":
            block("unzip blocked")

    # All checks passed
    sys.exit(EXIT_ALLOW)

if __name__ == "__main__":
    main()
