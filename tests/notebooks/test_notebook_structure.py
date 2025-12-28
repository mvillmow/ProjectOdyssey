"""Tests for notebook structure and validity."""

import json
from pathlib import Path


NOTEBOOK_DIR = Path(__file__).parent.parent.parent / "notebooks"


def test_notebooks_directory_exists():
    """Verify notebooks directory exists."""
    assert NOTEBOOK_DIR.exists(), f"Notebooks directory not found: {NOTEBOOK_DIR}"
    assert NOTEBOOK_DIR.is_dir(), f"{NOTEBOOK_DIR} is not a directory"


def test_readme_exists():
    """Verify README.md exists in notebooks directory."""
    readme = NOTEBOOK_DIR / "README.md"
    assert readme.exists(), f"README.md not found in {NOTEBOOK_DIR}"
    assert readme.is_file(), f"{readme} is not a file"


def test_expected_notebooks_exist():
    """Verify all expected notebooks are present."""
    expected_notebooks = [
        "01_introduction.ipynb",
        "02_tensor_operations.ipynb",
        "03_building_models.ipynb",
        "04_training_mnist.ipynb",
        "05_visualization.ipynb",
        "06_advanced_techniques.ipynb",
    ]

    for notebook_name in expected_notebooks:
        notebook_path = NOTEBOOK_DIR / notebook_name
        assert notebook_path.exists(), f"Missing notebook: {notebook_name}"
        assert notebook_path.is_file(), f"{notebook_name} is not a file"


def test_notebooks_have_valid_json():
    """Verify all notebooks have valid JSON format."""
    for notebook_path in NOTEBOOK_DIR.glob("*.ipynb"):
        try:
            with open(notebook_path) as f:
                notebook = json.load(f)
        except json.JSONDecodeError as e:
            raise AssertionError(f"Invalid JSON in {notebook_path.name}: {e}")

        # Verify basic notebook structure
        assert "cells" in notebook, f"{notebook_path.name} missing 'cells' key"
        assert "metadata" in notebook, f"{notebook_path.name} missing 'metadata' key"
        assert isinstance(notebook["cells"], list), f"{notebook_path.name} cells is not a list"


def test_utils_package_exists():
    """Verify notebooks/utils package is properly set up."""
    utils_dir = NOTEBOOK_DIR / "utils"
    assert utils_dir.exists(), f"utils directory not found in {NOTEBOOK_DIR}"
    assert utils_dir.is_dir(), f"{utils_dir} is not a directory"

    init_file = utils_dir / "__init__.py"
    assert init_file.exists(), f"__init__.py not found in {utils_dir}"


def test_required_util_modules_exist():
    """Verify all required utility modules are present."""
    utils_dir = NOTEBOOK_DIR / "utils"
    required_modules = [
        "mojo_bridge.py",
        "tensor_utils.py",
        "visualization.py",
        "progress.py",
    ]

    for module_name in required_modules:
        module_path = utils_dir / module_name
        assert module_path.exists(), f"Missing module: {module_name}"
        assert module_path.is_file(), f"{module_name} is not a file"


def test_notebook_sizes():
    """Verify notebooks don't exceed size limits."""
    max_size = 500 * 1024  # 500KB

    for notebook_path in NOTEBOOK_DIR.glob("*.ipynb"):
        size = notebook_path.stat().st_size
        assert size < max_size, (
            f"{notebook_path.name} exceeds {max_size} bytes (actual: {size} bytes). "
            "Clear outputs with: just jupyter-clear"
        )


def test_notebook_cell_structure():
    """Verify notebooks have proper cell structure."""
    for notebook_path in NOTEBOOK_DIR.glob("*.ipynb"):
        with open(notebook_path) as f:
            notebook = json.load(f)

        cells = notebook.get("cells", [])
        assert len(cells) > 0, f"{notebook_path.name} has no cells"

        for i, cell in enumerate(cells):
            assert "cell_type" in cell, f"Cell {i} in {notebook_path.name} missing cell_type"
            assert "source" in cell, f"Cell {i} in {notebook_path.name} missing source"

            cell_type = cell["cell_type"]
            assert cell_type in ["code", "markdown"], (
                f"Cell {i} in {notebook_path.name} has invalid type: {cell_type}"
            )


def test_notebook_metadata():
    """Verify notebooks have proper metadata."""
    for notebook_path in NOTEBOOK_DIR.glob("*.ipynb"):
        with open(notebook_path) as f:
            notebook = json.load(f)

        metadata = notebook.get("metadata", {})
        assert isinstance(metadata, dict), f"{notebook_path.name} metadata is not a dict"

        # Check for kernel info (optional but good to have)
        if "kernelspec" in metadata:
            kernelspec = metadata["kernelspec"]
            assert "display_name" in kernelspec
            assert "name" in kernelspec
