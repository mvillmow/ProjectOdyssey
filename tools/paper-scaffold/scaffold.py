#!/usr/bin/env python3

"""
Tool: paper-scaffold/scaffold.py
Purpose: Generate complete paper implementation directory structure from templates

Language: Python
Justification:
  - Heavy string substitution for template processing
  - File and directory manipulation
  - No performance requirements (one-time generation)
  - No ML/AI computation involved

Reference: ADR-001
Last Review: 2025-11-16
"""

import argparse
import sys
from pathlib import Path
from typing import Dict


def render_template(template_content: str, variables: Dict[str, str]) -> str:
    """
    Render template by replacing {{VARIABLE}} placeholders.

    Args:
        template_content: Template string with {{VAR}} placeholders
        variables: Dictionary of variable replacements

    Returns:
        Rendered template string
    """
    result = template_content
    for key, value in variables.items():
        placeholder = f"{{{{{key}}}}}"
        result = result.replace(placeholder, value)
    return result


def load_template(template_path: Path) -> str:
    """
    Load template from file.

    Args:
        template_path: Path to template file

    Returns:
        Template content
    """
    if not template_path.exists():
        raise FileNotFoundError(f"Template not found: {template_path}")

    return template_path.read_text()


def create_paper_structure(
    paper_name: str,
    paper_title: str,
    authors: str,
    year: str,
    paper_url: str,
    description: str,
    output_dir: Path,
    templates_dir: Path,
) -> None:
    """
    Create complete paper implementation directory structure.

    Args:
        paper_name: Short name for directory (e.g., 'lenet5')
        paper_title: Full paper title
        authors: Paper authors
        year: Publication year
        paper_url: URL to original paper
        description: Brief description
        output_dir: Output directory path
        templates_dir: Templates directory path
    """
    # Prepare template variables
    model_name = "".join(word.capitalize() for word in paper_name.split("_"))
    variables = {
        "PAPER_NAME": paper_name,
        "PAPER_TITLE": paper_title,
        "MODEL_NAME": model_name,
        "AUTHORS": authors,
        "YEAR": year,
        "PAPER_URL": paper_url,
        "DESCRIPTION": description,
    }

    # Create base directory
    base_dir = output_dir / paper_name
    base_dir.mkdir(parents=True, exist_ok=True)
    print(f"Created directory: {base_dir}")

    # Create subdirectories
    (base_dir / "tests").mkdir(exist_ok=True)
    (base_dir / "notes").mkdir(exist_ok=True)
    print("  Created subdirectories: tests/, notes/")

    # Template files to generate
    template_files = {
        "README.md.tmpl": "README.md",
        "model.mojo.tmpl": "model.mojo",
        "train.mojo.tmpl": "train.mojo",
        "test_model.mojo.tmpl": "tests/test_model.mojo",
    }

    # Generate files from templates
    for template_name, output_name in template_files.items():
        template_path = templates_dir / template_name
        output_path = base_dir / output_name

        try:
            # Load and render template
            template_content = load_template(template_path)
            rendered_content = render_template(template_content, variables)

            # Write output file
            output_path.write_text(rendered_content)
            print(f"  Generated: {output_name}")

        except Exception as e:
            print(f"Warning: Could not generate {output_name}: {e}", file=sys.stderr)

    # Create placeholder files for remaining structure
    placeholder_files = [
        ("data.mojo", "# TODO: Data loading and preprocessing"),
        ("metrics.mojo", "# TODO: Evaluation metrics"),
        (
            "notes/architecture.md",
            "# Architecture Notes\n\nTODO: Add architecture details",
        ),
        ("notes/results.md", "# Reproduction Results\n\nTODO: Document results"),
    ]

    for file_name, content in placeholder_files:
        file_path = base_dir / file_name
        file_path.write_text(content + "\n")
        print(f"  Created placeholder: {file_name}")

    print(f"\n✓ Successfully created paper structure at: {base_dir}")
    print("\nNext steps:")
    print(f"  1. cd {base_dir}")
    print("  2. Review and update README.md")
    print("  3. Implement model in model.mojo")
    print("  4. Run tests: mojo tests/test_model.mojo")


def main() -> int:
    """Main CLI interface."""
    parser = argparse.ArgumentParser(
        description="Generate paper implementation directory structure",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  python scaffold.py \\
    --paper lenet5 \\
    --title "LeNet-5: Gradient-Based Learning Applied to Document Recognition" \\
    --authors "LeCun et al." \\
    --year 1998 \\
    --url "http://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf" \\
    --description "Convolutional neural network for handwritten digit recognition" \\
    --output papers/
        """,
    )

    parser.add_argument("--paper", required=True, help="Short paper name (e.g., 'lenet5', 'alexnet')")
    parser.add_argument("--title", required=True, help="Full paper title")
    parser.add_argument("--authors", required=True, help="Paper authors")
    parser.add_argument("--year", required=True, help="Publication year")
    parser.add_argument("--url", required=True, help="URL to original paper")
    parser.add_argument(
        "--description",
        default="TODO: Add paper description",
        help="Brief paper description",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("papers"),
        help="Output directory (default: papers/)",
    )
    parser.add_argument(
        "--templates",
        type=Path,
        default=None,
        help="Templates directory (default: auto-detect)",
    )

    args = parser.parse_args()

    try:
        # Auto-detect templates directory if not specified
        if args.templates is None:
            script_dir = Path(__file__).parent
            args.templates = script_dir / "templates"

        # Validate templates directory
        if not args.templates.exists():
            print(
                f"Error: Templates directory not found: {args.templates}",
                file=sys.stderr,
            )
            print(
                "Please specify --templates or ensure templates/ exists.",
                file=sys.stderr,
            )
            return 1

        # Create paper structure
        create_paper_structure(
            paper_name=args.paper,
            paper_title=args.title,
            authors=args.authors,
            year=args.year,
            paper_url=args.url,
            description=args.description,
            output_dir=args.output,
            templates_dir=args.templates,
        )

        return 0

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
