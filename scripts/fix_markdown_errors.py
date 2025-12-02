#!/usr/bin/env python3

"""
Script to identify and categorize markdown linting errors.

Usage:
    python3 fix_markdown_errors.py
"""

import json
import subprocess
import sys
from collections import defaultdict


def run_markdownlint():
    """Run markdownlint and capture all errors."""
    try:
        result = subprocess.run(
            [
                "npx",
                "markdownlint-cli2",
                "--config",
                ".markdownlint.json",
                "**/*.md",
                "--ignore",
                "notes/plan/**",
                "--ignore",
                "notes/issues/**",
                "--ignore",
                "notes/review/**",
            ],
            cwd="/home/mvillmow/ml-odyssey",
            capture_output=True,
            text=True,
            timeout=120,
        )
        return result.stdout, result.stderr, result.returncode
    except Exception as e:
        print(f"Error running markdownlint: {e}")
        return "", str(e), 1


def parse_errors(output):
    """Parse markdownlint output into structured errors."""
    errors_by_type = defaultdict(list)
    errors_by_file = defaultdict(list)

    for line in output.split("\n"):
        line = line.strip()
        if not line:
            continue
        # Format: filename:line:col: MD### message
        if ":" in line and "MD" in line:
            parts = line.split(":")
            if len(parts) >= 4:
                filename = parts[0]
                line_num = parts[1]
                col_num = parts[2]
                message = ":".join(parts[3:]).strip()

                # Extract error code
                error_code = None
                for token in message.split():
                    if token.startswith("MD") and token[2:].isdigit():
                        error_code = token
                        break

                if error_code:
                    error_info = {
                        "file": filename,
                        "line": line_num,
                        "col": col_num,
                        "message": message,
                        "code": error_code,
                    }
                    errors_by_type[error_code].append(error_info)
                    errors_by_file[filename].append(error_info)

    return errors_by_type, errors_by_file


def main():
    print("Running markdownlint to identify all errors...")
    stdout, stderr, returncode = run_markdownlint()

    if stderr:
        print(f"stderr: {stderr}")

    print(f"\nFull output:\n{stdout}")

    errors_by_type, errors_by_file = parse_errors(stdout)

    print("\n=== ERRORS BY TYPE ===")
    for error_type in sorted(errors_by_type.keys()):
        count = len(errors_by_type[error_type])
        print(f"{error_type}: {count} errors")
        files = set(e["file"] for e in errors_by_type[error_type])
        print(f"  Files: {len(files)}")
        for error in errors_by_type[error_type][:3]:
            print(f"    {error['file']}:{error['line']} - {error['message']}")
        if count > 3:
            print(f"    ... and {count - 3} more")

    print("\n=== SUMMARY ===")
    total_errors = sum(len(v) for v in errors_by_type.values())
    print(f"Total errors: {total_errors}")
    print(f"Total files with errors: {len(errors_by_file)}")

    # Save for processing
    with open("/home/mvillmow/ml-odyssey/markdown_errors.json", "w") as f:
        json.dump(
            {
                "by_type": {k: v for k, v in errors_by_type.items()},
                "by_file": {k: v for k, v in errors_by_file.items()},
            },
            f,
            indent=2,
        )

    return 0 if total_errors == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
