"""
Test fixtures for tools/ directory infrastructure tests.

This module provides reusable fixtures for future tool development and testing.
These fixtures can be used by tool developers when creating new tools in the
tools/ directory.

Available Fixtures:
- Sample tool structure templates
- Example Mojo template files
- Example Python tool with ADR-001 justification header
- Mock tool configurations

Usage:
    from tests.tooling.tools.fixtures import (
        create_sample_tool_structure,
        get_mojo_template_example,
        get_python_tool_example,
    )
"""

from pathlib import Path
from typing import Dict


def get_mojo_template_example() -> str:
    """
    Get example Mojo template file content.

    This provides a sample template that could be used by paper-scaffold
    or other code generation tools.

    Returns:
        String containing example Mojo template content
    """
    return """
# {{model_name}} - Mojo Implementation

fn forward(input: Tensor) -> Tensor:
    \"\"\"
    Forward pass for {{model_name}}.

    Args:
        input: Input tensor

    Returns:
        Output tensor
    \"\"\"
    # TODO: Implement forward pass
    return input
""".strip()


def get_python_tool_example() -> str:
    """
    Get example Python tool with ADR-001 justification header.

    This provides a sample Python tool that follows the project's
    language selection documentation requirements.

    Returns:
        String containing example Python tool with proper header
    """
    return """
#!/usr/bin/env python3

\"\"\"
Tool: sample-tool/generator.py
Purpose: Generate boilerplate code for new models

Language: Python
Justification:
  - Heavy regex usage for template substitution
  - String manipulation for file generation
  - No performance requirements (one-time generation)
  - Integration with filesystem operations

Reference: ADR-001
\"\"\"

from pathlib import Path
from typing import Dict


def generate_code(template: str, variables: Dict[str, str]) -> str:
    \"\"\"
    Generate code from template with variable substitution.

    Args:
        template: Template string
        variables: Dictionary of variable substitutions

    Returns:
        Generated code with variables substituted
    \"\"\"
    result = template
    for key, value in variables.items():
        result = result.replace(f"{{{{{key}}}}}", value)
    return result


if __name__ == "__main__":
    print("Example Python tool with ADR-001 justification")
""".strip()


def create_sample_tool_structure(base_path: Path, tool_name: str) -> Dict[str, Path]:
    """
    Create a sample tool directory structure for testing.

    This creates a complete tool structure that can be used for testing
    tool creation workflows.

    Args:
        base_path: Base directory where tool should be created
        tool_name: Name of the tool

    Returns:
        Dictionary mapping structure element names to their paths

    Example:
        >>> tool_paths = create_sample_tool_structure(
        ...     Path("/tmp/test"),
        ...     "my-tool"
        ... )
        >>> tool_paths["root"]
        Path("/tmp/test/my-tool")
        >>> tool_paths["readme"]
        Path("/tmp/test/my-tool/README.md")
    """
    # Create tool root directory
    tool_root = base_path / tool_name
    tool_root.mkdir(parents=True, exist_ok=True)

    # Create README.md
    readme_path = tool_root / "README.md"
    readme_content = f"""
# {tool_name}

Tool description goes here.

## Usage

```bash
python {tool_name}/main.py
```

## Purpose

Describe what this tool does.
""".strip()
    readme_path.write_text(readme_content)

    # Create main tool file
    main_path = tool_root / "main.py"
    main_content = get_python_tool_example()
    main_path.write_text(main_content)

    # Create tests directory
    tests_dir = tool_root / "tests"
    tests_dir.mkdir(exist_ok=True)

    # Create test file
    test_path = tests_dir / f"test_{tool_name.replace('-', '_')}.py"
    test_content = f"""
\"\"\"
Test suite for {tool_name}.
\"\"\"

def test_{tool_name.replace('-', '_')}_placeholder():
    \"\"\"Placeholder test for {tool_name}.\"\"\"
    assert True
""".strip()
    test_path.write_text(test_content)

    # Create examples directory
    examples_dir = tool_root / "examples"
    examples_dir.mkdir(exist_ok=True)

    # Return paths dictionary
    return {
        "root": tool_root,
        "readme": readme_path,
        "main": main_path,
        "tests": tests_dir,
        "test_file": test_path,
        "examples": examples_dir,
    }


def get_tool_readme_template() -> str:
    """
    Get template for tool README.md files.

    Returns:
        String containing README.md template
    """
    return """
# {{tool_name}}

{{tool_description}}

## Purpose

{{purpose_statement}}

## Installation

```bash
# Installation instructions
```

## Usage

```bash
# Usage examples
{{usage_example}}
```

## Language Choice

**Language**: {{language}}

**Justification**:
{{language_justification}}

**Reference**: [ADR-001](../../notes/review/adr/ADR-001-language-selection-tooling.md)

## Examples

See `examples/` directory for usage examples.

## Testing

```bash
pytest tests/
```

## Contributing

See main [tools/README.md](../README.md) for contribution guidelines.
""".strip()


def get_mojo_tool_template() -> str:
    """
    Get template for Mojo tool implementation.

    Returns:
        String containing Mojo tool template
    """
    return """
\"\"\"
{{tool_name}} - Mojo Implementation

Purpose: {{purpose}}
\"\"\"

fn main():
    \"\"\"Main entry point for {{tool_name}}.\"\"\"
    print("{{tool_name}} - Mojo tool")

# TODO: Implement tool functionality
""".strip()


# Export public API
__all__ = [
    "get_mojo_template_example",
    "get_python_tool_example",
    "create_sample_tool_structure",
    "get_tool_readme_template",
    "get_mojo_tool_template",
]
