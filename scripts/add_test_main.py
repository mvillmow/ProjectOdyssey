#!/usr/bin/env python3
"""
Auto-generate main() functions for Mojo test files.

This script parses Mojo test files to find test functions (those starting with 'test_')
and generates a standardized main() function that:
1. Executes each test with try/except handling
2. Tracks pass/fail counts
3. Prints formatted results
4. Raises an error if any tests fail

Usage:
    python scripts/add_test_main.py <test_file.mojo>
    python scripts/add_test_main.py tests/shared/core/test_*.mojo
"""

import argparse
import re
import sys
from pathlib import Path
from typing import List


def find_test_functions(file_content: str) -> List[str]:
    """Find all test function names in a Mojo file.

    Args:
        file_content: Content of the Mojo file

    Returns:
        List of test function names (e.g., ['test_addition', 'test_multiplication'])
    """
    # Pattern to match function definitions starting with 'test_'
    # Matches: fn test_name(...) raises:
    pattern = r"^fn\s+(test_\w+)\s*\([^)]*\)\s*(?:raises\s*)?:"

    test_functions = []
    for line in file_content.splitlines():
        match = re.match(pattern, line.strip())
        if match:
            test_functions.append(match.group(1))

    return test_functions


def has_main_function(file_content: str) -> bool:
    """Check if file already has a main() function.

    Args:
        file_content: Content of the Mojo file

    Returns:
        True if main() function exists, False otherwise
    """
    pattern = r"^fn\s+main\s*\([^)]*\)\s*(?:raises\s*)?:"

    for line in file_content.splitlines():
        if re.match(pattern, line.strip()):
            return True

    return False


def generate_main_function(test_functions: List[str], file_path: str) -> str:
    """Generate a main() function that runs all test functions.

    Args:
        test_functions: List of test function names
        file_path: Path to the test file (for display in output)

    Returns:
        String containing the complete main() function
    """
    if not test_functions:
        return ""

    # Start the main function
    main_code = [
        "",
        "",
        "# ============================================================================",
        "# Main Test Runner",
        "# ============================================================================",
        "",
        "",
        "fn main() raises:",
        '    """Run all tests in this file."""',
        "    var total = 0",
        "    var passed = 0",
        "    var failed = 0",
        "",
        '    print("\\n" + "=" * 70)',
        f'    print("Running tests from: {Path(file_path).name}")',
        '    print("=" * 70 + "\\n")',
        "",
    ]

    # Add each test
    for test_func in test_functions:
        main_code.extend(
            [
                "    # " + test_func,
                "    total += 1",
                "    try:",
                f"        {test_func}()",
                "        passed += 1",
                f'        print("  ✓ {test_func}")',
                "    except e:",
                "        failed += 1",
                f'        print("  ✗ {test_func}:", e)',
                "",
            ]
        )

    # Add summary
    main_code.extend(
        [
            "    # Summary",
            '    print("\\n" + "=" * 70)',
            '    print("Results:", passed, "/", total, "passed,", failed, "failed")',
            '    print("=" * 70)',
            "",
            "    if failed > 0:",
            '        raise Error("Tests failed")',
        ]
    )

    return "\n".join(main_code)


def add_main_to_file(file_path: Path, dry_run: bool = False) -> bool:
    """Add main() function to a Mojo test file.

    Args:
        file_path: Path to the Mojo test file
        dry_run: If True, don't actually modify the file

    Returns:
        True if main() was added, False if skipped or error
    """
    try:
        # Read the file
        content = file_path.read_text()

        # Check if main() already exists
        if has_main_function(content):
            print(f"⏭️  SKIP: {file_path} (already has main())")
            return False

        # Find test functions
        test_functions = find_test_functions(content)

        if not test_functions:
            print(f"⏭️  SKIP: {file_path} (no test functions found)")
            return False

        # Generate main() function
        main_func = generate_main_function(test_functions, str(file_path))

        # Add main() to file
        new_content = content.rstrip() + "\n" + main_func + "\n"

        if dry_run:
            print(f"🔍 DRY-RUN: {file_path} (would add main() for {len(test_functions)} tests)")
            return True

        # Write the file
        file_path.write_text(new_content)
        print(f"✅ ADDED: {file_path} (added main() for {len(test_functions)} tests)")
        return True

    except Exception as e:
        print(f"❌ ERROR: {file_path}: {e}", file=sys.stderr)
        return False


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Auto-generate main() functions for Mojo test files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Process a single file
    python scripts/add_test_main.py tests/shared/core/test_arithmetic.mojo

    # Process multiple files
    python scripts/add_test_main.py tests/shared/core/test_*.mojo

    # Dry-run mode (don't modify files)
    python scripts/add_test_main.py --dry-run tests/shared/core/test_*.mojo

    # Process all test files in a directory
    python scripts/add_test_main.py tests/shared/core/test_*.mojo
        """,
    )

    parser.add_argument("files", nargs="+", type=Path, help="Mojo test files to process")

    parser.add_argument("--dry-run", action="store_true", help="Show what would be done without modifying files")

    args = parser.parse_args()

    # Process each file
    processed = 0
    added = 0
    skipped = 0
    errors = 0

    for file_path in args.files:
        if not file_path.exists():
            print(f"⚠️  WARNING: {file_path} does not exist", file=sys.stderr)
            continue

        if not file_path.is_file():
            print(f"⚠️  WARNING: {file_path} is not a file", file=sys.stderr)
            continue

        if not file_path.suffix == ".mojo":
            print(f"⏭️  SKIP: {file_path} (not a .mojo file)")
            skipped += 1
            continue

        processed += 1
        result = add_main_to_file(file_path, dry_run=args.dry_run)

        if result:
            added += 1
        else:
            skipped += 1

    # Summary
    print("\n" + "=" * 70)
    print(f"Summary: {processed} files processed")
    print(f"  ✅ Added:   {added}")
    print(f"  ⏭️  Skipped: {skipped}")
    if errors > 0:
        print(f"  ❌ Errors:  {errors}")
    print("=" * 70)

    if args.dry_run:
        print("\n🔍 DRY-RUN MODE: No files were modified")

    return 0 if errors == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
