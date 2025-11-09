#!/usr/bin/env bash

# System Information Collection Script
# Collects system information for bug reports and issue creation

set -euo pipefail

echo "=== System Information ==="
echo ""

# Operating System
echo "OS Information:"
if [[ -f /etc/os-release ]]; then
    # Linux
    . /etc/os-release
    echo "  OS: $NAME $VERSION"
elif [[ "$OSTYPE" == "darwin"* ]]; then
    # macOS
    echo "  OS: macOS $(sw_vers -productVersion)"
else
    # Windows or other
    echo "  OS: $OSTYPE"
fi
echo ""

# Kernel/Architecture
echo "Architecture:"
echo "  Kernel: $(uname -s)"
echo "  Machine: $(uname -m)"
echo ""

# Python version
echo "Python:"
if command -v python3 &> /dev/null; then
    echo "  Version: $(python3 --version 2>&1 | cut -d' ' -f2)"
    echo "  Path: $(which python3)"
else
    echo "  Not found"
fi
echo ""

# Mojo version
echo "Mojo:"
if command -v mojo &> /dev/null; then
    echo "  Version: $(mojo --version 2>&1 || echo 'Unable to determine')"
    echo "  Path: $(which mojo)"
else
    echo "  Not found"
fi
echo ""

# Pixi version
echo "Pixi:"
if command -v pixi &> /dev/null; then
    echo "  Version: $(pixi --version 2>&1 | cut -d' ' -f2)"
    echo "  Path: $(which pixi)"
else
    echo "  Not found"
fi
echo ""

# Git version
echo "Git:"
if command -v git &> /dev/null; then
    echo "  Version: $(git --version 2>&1 | cut -d' ' -f3)"
    echo "  Path: $(which git)"
else
    echo "  Not found"
fi
echo ""

# Current directory and git status (if in a git repo)
echo "Current Directory:"
echo "  Path: $(pwd)"
if git rev-parse --git-dir &> /dev/null; then
    echo "  Git Repository: Yes"
    echo "  Branch: $(git branch --show-current)"
    echo "  Commit: $(git rev-parse --short HEAD)"
else
    echo "  Git Repository: No"
fi
echo ""

# Environment variables (selected)
echo "Environment:"
echo "  SHELL: ${SHELL:-Not set}"
echo "  LANG: ${LANG:-Not set}"
echo ""

echo "=== End System Information ==="
