#!/usr/bin/env python3
"""Configuration linting tool for ML Odyssey.

This script validates YAML configuration files for:
- Proper formatting (2-space indent)
- Valid YAML syntax
- Schema compliance
- Unused parameters
- Duplicate values
- Deprecated keys
- Performance suggestions

Usage:
    python scripts/lint_configs.py [options] [config_files...]

Examples:
    # Lint all configs
    python scripts/lint_configs.py configs/

    # Lint specific file
    python scripts/lint_configs.py configs/defaults/training.yaml

    # Remove unused parameters
    python scripts/lint_configs.py --remove-unused configs/

    # Verbose output
    python scripts/lint_configs.py -v configs/
"""

import argparse
import sys
import re
from pathlib import Path
from typing import Dict, List, Optional


class ConfigLinter:
    """Lints YAML configuration files."""

    def __init__(self, verbose: bool = False):
        """Initialize the linter.

        Args:
            verbose: Enable verbose output
        """
        self.verbose = verbose
        self.errors: List[str] = []
        self.warnings: List[str] = []
        self.suggestions: List[str] = []

        # Define deprecated keys and their replacements
        self.deprecated_keys = {
            "optimizer.type": "optimizer.name",
            "model.num_layers": "model.layers",
            "lr": "learning_rate",
            "val_split": "validation_split",
        }

        # Define required keys for different config types
        self.required_keys = {
            "training": ["epochs", "batch_size"],
            "model": ["architecture"],
            "optimizer": ["name", "learning_rate"],
        }

        # Performance thresholds
        self.perf_thresholds = {
            "batch_size": (8, 512),  # min, max
            "learning_rate": (0.00001, 1.0),
            "epochs": (1, 10000),
        }

    def lint_file(self, filepath: Path) -> bool:
        """Lint a single configuration file.

        Args:
            filepath: Path to YAML file

        Returns:
            True if file passes linting, False otherwise
        """
        self.errors = []
        self.warnings = []
        self.suggestions = []

        if not filepath.exists():
            self.errors.append(f"File not found: {filepath}")
            return False

        if self.verbose:
            print(f"Linting: {filepath}")

        try:
            with open(filepath, "r") as f:
                content = f.read()
        except Exception as e:
            self.errors.append(f"Failed to read file: {e}")
            return False

        # Check YAML syntax
        if not self._check_yaml_syntax(content, filepath):
            return False

        # Check formatting
        self._check_formatting(content, filepath)

        # Parse configuration
        config = self._parse_yaml(content)
        if config is None:
            return False

        # Check for issues
        self._check_deprecated_keys(config, filepath)
        self._check_required_keys(config, filepath)
        self._check_duplicate_values(config, filepath)
        self._check_performance(config, filepath)
        self._check_unused_parameters(config, filepath)

        return len(self.errors) == 0

    def _check_yaml_syntax(self, content: str, filepath: Path) -> bool:
        """Check if YAML syntax is valid.

        Args:
            content: File content
            filepath: Path to file

        Returns:
            True if syntax is valid
        """
        try:
            # Simple YAML validation
            lines = content.split("\n")
            brace_count = 0
            bracket_count = 0

            for i, line in enumerate(lines):
                # Skip comments
                stripped = line.split("#")[0]

                # Count braces and brackets
                brace_count += stripped.count("{") - stripped.count("}")
                bracket_count += stripped.count("[") - stripped.count("]")

                # Check for common issues
                if ":" in stripped and not re.match(r"^\s*[\w\-]+:", stripped):
                    if "://" not in stripped:  # Not a URL
                        self.warnings.append(f"{filepath}:{i + 1} - Possible malformed key")

            if brace_count != 0:
                self.errors.append(f"{filepath} - Unmatched braces")
                return False

            if bracket_count != 0:
                self.errors.append(f"{filepath} - Unmatched brackets")
                return False

            return True

        except Exception as e:
            self.errors.append(f"{filepath} - Syntax check failed: {e}")
            return False

    def _check_formatting(self, content: str, filepath: Path):
        """Check formatting standards.

        Args:
            content: File content
            filepath: Path to file
        """
        lines = content.split("\n")

        for i, line in enumerate(lines):
            # Skip empty lines and comments
            if not line.strip() or line.strip().startswith("#"):
                continue

            # Check indentation (should be 2 spaces)
            indent = len(line) - len(line.lstrip())
            if indent > 0 and indent % 2 != 0:
                self.warnings.append(f"{filepath}:{i + 1} - Odd indentation ({indent} spaces)")

            # Check for tabs
            if "\t" in line:
                self.errors.append(f"{filepath}:{i + 1} - Tab found (use 2 spaces)")

            # Check for trailing whitespace
            if line.rstrip() != line:
                self.warnings.append(f"{filepath}:{i + 1} - Trailing whitespace")

    def _parse_yaml(self, content: str) -> Optional[Dict]:
        """Parse YAML content into dictionary.

        Args:
            content: YAML content

        Returns:
            Parsed dictionary or None if parsing fails
        """
        try:
            # Simple YAML parser
            result = {}
            current_section = None
            lines = content.split("\n")

            for line in lines:
                stripped = line.strip()

                # Skip comments and empty lines
                if not stripped or stripped.startswith("#"):
                    continue

                # Check for section header
                if not line.startswith(" ") and ":" in stripped:
                    key = stripped.split(":")[0].strip()
                    value = stripped.split(":", 1)[1].strip()

                    if not value:
                        current_section = key
                        result[key] = {}
                    else:
                        result[key] = self._parse_value(value)
                        current_section = None

                # Check for nested key
                elif current_section and ":" in stripped:
                    key = stripped.split(":")[0].strip()
                    value = stripped.split(":", 1)[1].strip()
                    if isinstance(result[current_section], dict):
                        result[current_section][key] = self._parse_value(value)

            return result

        except Exception as e:
            self.errors.append(f"Failed to parse YAML: {e}")
            return None

    def _parse_value(self, value: str):
        """Parse a YAML value string.

        Args:
            value: Value string

        Returns:
            Parsed value
        """
        value = value.strip()

        # Boolean
        if value.lower() in ["true", "yes"]:
            return True
        if value.lower() in ["false", "no"]:
            return False

        # List
        if value.startswith("[") and value.endswith("]"):
            items = value[1:-1].split(",")
            return [self._parse_value(item.strip()) for item in items]

        # Number
        try:
            if "." in value:
                return float(value)
            return int(value)
        except ValueError:
            pass

        # String (remove quotes if present)
        if (value.startswith('"') and value.endswith('"')) or (value.startswith("'") and value.endswith("'")):
            return value[1:-1]

        return value

    def _check_deprecated_keys(self, config: Dict, filepath: Path):
        """Check for deprecated configuration keys.

        Args:
            config: Parsed configuration
            filepath: Path to file
        """
        for old_key, new_key in self.deprecated_keys.items():
            if self._has_nested_key(config, old_key):
                self.warnings.append(f"{filepath} - Deprecated key '{old_key}' (use '{new_key}' instead)")

    def _check_required_keys(self, config: Dict, filepath: Path):
        """Check for required keys based on config type.

        Args:
            config: Parsed configuration
            filepath: Path to file
        """
        # Determine config type from content
        for section, required in self.required_keys.items():
            if section in config:
                for key in required:
                    if key not in config.get(section, {}):
                        self.warnings.append(f"{filepath} - Missing required key '{section}.{key}'")

    def _check_duplicate_values(self, config: Dict, filepath: Path):
        """Check for duplicate values that might be errors.

        Args:
            config: Parsed configuration
            filepath: Path to file
        """
        seen_values: Dict[str, List[str]] = {}

        def collect_values(obj, prefix=""):
            if isinstance(obj, dict):
                for key, value in obj.items():
                    new_prefix = f"{prefix}.{key}" if prefix else key
                    collect_values(value, new_prefix)
            elif isinstance(obj, (str, int, float, bool)):
                value_str = str(obj)
                if value_str not in seen_values:
                    seen_values[value_str] = []
                seen_values[value_str].append(prefix)

        collect_values(config)

        # Report duplicates (excluding common values)
        common_values = {"true", "false", "0", "1", ""}
        for value, keys in seen_values.items():
            if len(keys) > 2 and value not in common_values:
                if len(value) > 3:  # Skip short values
                    self.suggestions.append(f"{filepath} - Value '{value}' appears in: {', '.join(keys[:3])}...")

    def _check_performance(self, config: Dict, filepath: Path):
        """Check for performance-related configuration issues.

        Args:
            config: Parsed configuration
            filepath: Path to file
        """
        # Check batch size
        batch_size = self._get_nested_value(config, "training.batch_size")
        if batch_size is not None:
            min_bs, max_bs = self.perf_thresholds["batch_size"]
            if batch_size < min_bs:
                self.suggestions.append(f"{filepath} - Small batch_size ({batch_size}) may be inefficient")
            elif batch_size > max_bs:
                self.warnings.append(f"{filepath} - Large batch_size ({batch_size}) may cause OOM")

        # Check learning rate
        lr = self._get_nested_value(config, "optimizer.learning_rate")
        if lr is not None:
            min_lr, max_lr = self.perf_thresholds["learning_rate"]
            if lr < min_lr:
                self.warnings.append(f"{filepath} - Very small learning_rate ({lr})")
            elif lr > max_lr:
                self.warnings.append(f"{filepath} - Very large learning_rate ({lr})")

    def _check_unused_parameters(self, config: Dict, filepath: Path):
        """Check for potentially unused parameters.

        Args:
            config: Parsed configuration
            filepath: Path to file
        """
        # Known unused patterns
        if "debug" in config and config.get("debug"):
            if "production" in str(filepath).lower():
                self.warnings.append(f"{filepath} - Debug mode enabled in production config")

    def _has_nested_key(self, config: Dict, key_path: str) -> bool:
        """Check if nested key exists in config.

        Args:
            config: Configuration dictionary
            key_path: Dot-separated key path

        Returns:
            True if key exists
        """
        keys = key_path.split(".")
        current = config
        for key in keys:
            if isinstance(current, dict) and key in current:
                current = current[key]
            else:
                return False
        return True

    def _get_nested_value(self, config: Dict, key_path: str):
        """Get value of nested key.

        Args:
            config: Configuration dictionary
            key_path: Dot-separated key path

        Returns:
            Value if found, None otherwise
        """
        keys = key_path.split(".")
        current = config
        for key in keys:
            if isinstance(current, dict) and key in current:
                current = current[key]
            else:
                return None
        return current

    def print_results(self):
        """Print linting results."""
        if self.errors:
            print("\n❌ ERRORS:")
            for error in self.errors:
                print(f"  - {error}")

        if self.warnings:
            print("\n⚠️  WARNINGS:")
            for warning in self.warnings:
                print(f"  - {warning}")

        if self.suggestions:
            print("\n💡 SUGGESTIONS:")
            for suggestion in self.suggestions:
                print(f"  - {suggestion}")

        if not (self.errors or self.warnings or self.suggestions):
            print("✅ All checks passed!")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Lint configuration files for ML Odyssey")
    parser.add_argument("paths", nargs="+", help="Configuration files or directories to lint")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose output")
    parser.add_argument(
        "--remove-unused",
        action="store_true",
        help="Remove unused parameters (not implemented)",
    )

    args = parser.parse_args()

    # Collect files to lint
    files_to_lint = []
    for path_str in args.paths:
        path = Path(path_str)
        if path.is_file():
            if path.suffix in [".yaml", ".yml"]:
                files_to_lint.append(path)
        elif path.is_dir():
            files_to_lint.extend(path.rglob("*.yaml"))
            files_to_lint.extend(path.rglob("*.yml"))
        else:
            print(f"Warning: Path not found: {path}")

    if not files_to_lint:
        print("No configuration files found to lint")
        return 1

    # Lint files
    linter = ConfigLinter(verbose=args.verbose)
    failed_files = []
    total_errors = 0
    total_warnings = 0

    print(f"Linting {len(files_to_lint)} configuration file(s)...\n")

    for filepath in sorted(files_to_lint):
        if not linter.lint_file(filepath):
            failed_files.append(filepath)

        total_errors += len(linter.errors)
        total_warnings += len(linter.warnings)

        if linter.errors or linter.warnings or linter.suggestions:
            print(f"\n📄 {filepath}")
            linter.print_results()

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Files checked: {len(files_to_lint)}")
    print(f"Files passed: {len(files_to_lint) - len(failed_files)}")
    print(f"Files failed: {len(failed_files)}")
    print(f"Total errors: {total_errors}")
    print(f"Total warnings: {total_warnings}")

    if failed_files:
        print("\nFailed files:")
        for filepath in failed_files:
            print(f"  - {filepath}")
        return 1

    print("\n✅ All configuration files passed linting!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
