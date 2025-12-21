#!/usr/bin/env python3
"""Fix List mutation patterns and missing dtype arguments in model files."""

import re
from pathlib import Path


def fix_list_patterns(filepath):
    """Fix List[Int]().append() patterns in a Mojo file."""
    with open(filepath, "r") as f:
        content = f.read()

    original = content

    # Pattern 1: Four-element lists (must be first to avoid partial matches)
    # Handles multi-line patterns
    content = re.sub(
        r"List\[Int\]\(\)\s*\.append\((\w+)\)\s*\.append\((\w+)\)\s*\.append\((\w+)\)\s*\.append\((\w+)\)",
        r"[\1, \2, \3, \4]",
        content,
        flags=re.MULTILINE,
    )

    # Pattern 2: Three-element lists
    content = re.sub(
        r"List\[Int\]\(\)\.append\((\w+)\)\.append\((\w+)\)\.append\((\w+)\)",
        r"[\1, \2, \3]",
        content,
    )

    # Pattern 3: Two-element lists
    content = re.sub(r"List\[Int\]\(\)\.append\((\w+)\)\.append\((\w+)\)", r"[\1, \2]", content)

    # Pattern 4: Single-element lists
    content = re.sub(r"List\[Int\]\(\)\.append\((\w+)\)", r"[\1]", content)

    # Fix zeros() calls missing dtype
    # Pattern: zeros([...])  where not followed by a comma (meaning no dtype)
    # We need to add , DType.float32 before the closing paren
    content = re.sub(r"zeros\((\[[^\]]+\])\)(?!\s*,)", r"zeros(\1, DType.float32)", content)

    # Fix constant() calls - these already have dtype default so less critical
    # But for consistency, we can leave them as-is since they default to float32

    if content != original:
        with open(filepath, "w") as f:
            f.write(content)
        return True
    return False


def main():
    files_to_fix = [
        "examples/googlenet-cifar10/model.mojo",
        "examples/googlenet-cifar10/test_model.mojo",
        "examples/mobilenetv1-cifar10/model.mojo",
        "examples/mobilenetv1-cifar10/test_model.mojo",
    ]

    for filepath in files_to_fix:
        path = Path(filepath)
        if not path.exists():
            print(f"⚠️  File not found: {filepath}")
            continue

        print(f"Processing: {filepath}")
        if fix_list_patterns(filepath):
            print(f"✓ Fixed List patterns in {filepath}")
        else:
            print(f"  No changes needed in {filepath}")

    print("\n✓ Bulk replacements complete!")
    print("⚠️  Still need to manually fix:")
    print("  - kaiming_normal() duplicate fan_in/fan_out keyword args")
    print("  - xavier_normal() duplicate fan_in/fan_out keyword args")
    print("  - Constructor signatures (mut self → out self)")


if __name__ == "__main__":
    main()
