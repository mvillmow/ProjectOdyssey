"""Tests for magic.toml configuration file."""
import tomllib
from pathlib import Path

def test_magic_toml_exists():
    """Verify magic.toml exists at repository root."""
    repo_root = Path(__file__).parent.parent.parent
    magic_toml = repo_root / "magic.toml"
    assert magic_toml.exists(), "magic.toml should exist at repository root"

def test_magic_toml_valid_syntax():
    """Verify magic.toml has valid TOML syntax."""
    repo_root = Path(__file__).parent.parent.parent
    magic_toml = repo_root / "magic.toml"
    with open(magic_toml, "rb") as f:
        config = tomllib.load(f)
    assert isinstance(config, dict), "magic.toml should parse to a dictionary"

def test_magic_toml_has_project_metadata():
    """Verify magic.toml contains required project metadata."""
    repo_root = Path(__file__).parent.parent.parent
    magic_toml = repo_root / "magic.toml"
    with open(magic_toml, "rb") as f:
        config = tomllib.load(f)

    assert "project" in config, "magic.toml should have [project] section"
    project = config["project"]
    assert "name" in project, "Project should have name"
    assert "version" in project, "Project should have version"
    assert "description" in project, "Project should have description"
