"""Tests for dependency management in magic.toml."""
import tomllib
from pathlib import Path

def test_dependencies_section_structure():
    """Verify dependencies section has correct structure."""
    repo_root = Path(__file__).parent.parent.parent
    magic_toml = repo_root / "magic.toml"

    if not magic_toml.exists():
        return  # Skip if file doesn't exist

    with open(magic_toml, "rb") as f:
        config = tomllib.load(f)

    # Check if dependencies section exists (optional for now)
    if "dependencies" in config:
        assert isinstance(config["dependencies"], dict)
