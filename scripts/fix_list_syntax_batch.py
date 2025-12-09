#!/usr/bin/env python3
"""Batch processor to fix List[Int](args) → [args] syntax across all remaining files.

This script systematically processes all files in phases:
- Phase 2: Test files (~240 occurrences)
- Phase 3: Shared library files (~48 occurrences)
- Phase 4: Examples (~6 occurrences)
"""

import re
import sys
from pathlib import Path
from typing import List, Tuple

# Phase 2: Test files
PHASE_2_FILES = [
    "tests/shared/core/test_extensor_operators.mojo",
    "tests/shared/testing/test_tensor_factory.mojo",
    "tests/shared/testing/test_assertions.mojo",
    "tests/shared/core/test_validation.mojo",
    "tests/shared/core/test_extensor_new_methods.mojo",
    "tests/shared/training/test_optimizer_utils.mojo",
    "tests/shared/training/test_training_loop.mojo",
    "tests/core/types/test_fp4_tensor.mojo",
    "tests/shared/core/layers/test_relu.mojo",
    "tests/shared/core/layers/test_dropout.mojo",
    "tests/shared/testing/test_fixtures.mojo",
    "tests/shared/training/test_mixed_precision.mojo",
    "tests/shared/core/test_utils.mojo",
    "tests/shared/test_data_generators.mojo",
    "tests/test_core_operations.mojo",
    "tests/shared/core/layers/test_linear_struct.mojo",
    "tests/shared/core/test_module.mojo",
]

# Phase 3: Shared library files
PHASE_3_FILES = [
    "shared/training/optimizers/optimizer_utils.mojo",
    "shared/testing/tensor_factory.mojo",
    "shared/autograd/grad_utils.mojo",
    "shared/testing/fixtures.mojo",
    "shared/testing/data_generators.mojo",
    "shared/testing/models.mojo",
    "shared/training/metrics/results_printer.mojo",
]

# Phase 4: Examples
PHASE_4_FILES = [
    "examples/integer_example.mojo",
]


def fix_list_int_syntax(content: str) -> Tuple[str, int]:
    """Fix List[Int](args) → [args] syntax, preserving List[Int]() (empty).

    Returns:
        Tuple of (modified_content, replacement_count)
    """
    # Pattern: List[Int]( followed by digit(s), capturing everything until matching )
    # This pattern matches List[Int](1, 2, 3) but NOT List[Int]()
    pattern = r"List\[Int\]\(([0-9][^)]*)\)"
    replacement = r"[\1]"

    modified, count = re.subn(pattern, replacement, content)
    return modified, count


def process_file(file_path: Path) -> Tuple[bool, int]:
    """Process a single file, fixing List[Int] syntax.

    Returns:
        Tuple of (success, replacement_count)
    """
    try:
        # Read original content
        content = file_path.read_text()

        # Apply fixes
        modified_content, count = fix_list_int_syntax(content)

        if count > 0:
            # Write modified content
            file_path.write_text(modified_content)
            print(f"✅ {file_path}: {count} occurrences fixed")
            return True, count
        else:
            print(f"⏭️  {file_path}: No changes needed")
            return True, 0

    except Exception as e:
        print(f"❌ {file_path}: Error - {e}")
        return False, 0


def process_phase(phase_name: str, file_list: List[str], repo_root: Path) -> Tuple[int, int, int]:
    """Process all files in a phase.

    Returns:
        Tuple of (total_files, successful_files, total_fixes)
    """
    print(f"\n{'=' * 80}")
    print(f"{phase_name}")
    print(f"{'=' * 80}\n")

    total_files = len(file_list)
    successful_files = 0
    total_fixes = 0

    for file_rel_path in file_list:
        file_path = repo_root / file_rel_path

        if not file_path.exists():
            print(f"⚠️  {file_path}: File not found, skipping")
            continue

        success, count = process_file(file_path)
        if success:
            successful_files += 1
            total_fixes += count

    print(f"\n{phase_name} Summary: {successful_files}/{total_files} files processed, {total_fixes} total fixes")
    return total_files, successful_files, total_fixes


def main():
    """Main entry point."""
    # Get repository root (assume script is in scripts/ directory)
    repo_root = Path(__file__).parent.parent.resolve()
    print(f"Repository root: {repo_root}")

    # Process all phases
    phase_results = []

    # Phase 2: Test files
    phase_results.append(process_phase("PHASE 2: Test Files", PHASE_2_FILES, repo_root))

    # Phase 3: Shared library files
    phase_results.append(process_phase("PHASE 3: Shared Library Files", PHASE_3_FILES, repo_root))

    # Phase 4: Examples
    phase_results.append(process_phase("PHASE 4: Examples", PHASE_4_FILES, repo_root))

    # Overall summary
    print(f"\n{'=' * 80}")
    print("OVERALL SUMMARY")
    print(f"{'=' * 80}\n")

    total_files_all = sum(r[0] for r in phase_results)
    total_success_all = sum(r[1] for r in phase_results)
    total_fixes_all = sum(r[2] for r in phase_results)

    print(f"Total files processed: {total_success_all}/{total_files_all}")
    print(f"Total occurrences fixed: {total_fixes_all}")

    if total_success_all == total_files_all:
        print("\n✅ All files processed successfully!")
        return 0
    else:
        print(f"\n⚠️  {total_files_all - total_success_all} files had errors")
        return 1


if __name__ == "__main__":
    sys.exit(main())
