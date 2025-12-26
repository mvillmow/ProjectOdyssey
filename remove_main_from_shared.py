#!/usr/bin/env python3
"""Remove main() functions from shared package files.

Mojo package compilation doesn't allow main() functions within packages.
This script removes all main() functions and their bodies from files in shared/.
"""

import re
import subprocess
from pathlib import Path

def find_files_with_main():
    """Find all .mojo files in shared/ that contain main() functions."""
    result = subprocess.run(
        ['grep', '-r', '-l', '^fn main():\\|^def main():', 'shared/', '--include=*.mojo'],
        capture_output=True,
        text=True
    )
    if result.returncode == 0:
        return [line.strip() for line in result.stdout.split('\n') if line.strip()]
    return []

def remove_main_function(filepath):
    """Remove main() function and its body from a file.

    Handles both:
    - fn main():
    - def main():

    Removes the function definition and all indented lines that follow it.
    """
    with open(filepath, 'r') as f:
        lines = f.readlines()

    output_lines = []
    skip_mode = False
    main_indent = None

    for i, line in enumerate(lines):
        # Check if this is a main() function definition
        if re.match(r'^(fn|def) main\(\):', line):
            skip_mode = True
            main_indent = len(line) - len(line.lstrip())
            # Also remove the blank line before main() if present
            if output_lines and output_lines[-1].strip() == '':
                output_lines.pop()
            continue

        # If we're in skip mode, check if we're still in the function body
        if skip_mode:
            # Empty lines and comments might be part of main()
            if line.strip() == '' or line.strip().startswith('#'):
                continue

            # Get current line's indentation
            current_indent = len(line) - len(line.lstrip())

            # If indentation is less than or equal to main's indent, we've exited the function
            if current_indent <= main_indent:
                skip_mode = False
                main_indent = None
            else:
                # Still inside main() body, skip this line
                continue

        if not skip_mode:
            output_lines.append(line)

    # Write back
    with open(filepath, 'w') as f:
        f.writelines(output_lines)

    print(f"✓ Removed main() from {filepath}")

def main():
    files_with_main = find_files_with_main()

    if not files_with_main:
        print("No files with main() found in shared/")
        return

    print(f"Found {len(files_with_main)} files with main() functions")
    print()

    for filepath in files_with_main:
        remove_main_function(filepath)

    print()
    print(f"✅ Removed main() from {len(files_with_main)} files")

if __name__ == '__main__':
    main()
