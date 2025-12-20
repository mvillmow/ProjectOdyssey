#!/usr/bin/env python3
"""
Tests for interactive user prompts.

Tests Issue #770 (User Prompts - Write Tests):
- Interactive input collection
- Input validation
- Default value handling
- Error message display
"""

import sys
from pathlib import Path
from unittest.mock import patch
from io import StringIO

# Add tools directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "tools" / "paper-scaffold"))


class TestInteractivePrompter:
    """Test interactive prompt functionality (Issue #770)."""

    def test_prompt_for_paper_name(self):
        """Test prompting for paper name."""
        from prompts import InteractivePrompter

        prompter = InteractivePrompter()

        # Mock user input
        with patch("builtins.input", return_value="LeNet-5"):
            result = prompter.prompt_for_paper_name()

        assert result == "LeNet-5"

    def test_prompt_for_title(self):
        """Test prompting for paper title."""
        from prompts import InteractivePrompter

        prompter = InteractivePrompter()

        with patch("builtins.input", return_value="LeNet-5: Gradient-Based Learning"):
            result = prompter.prompt_for_title()

        assert result == "LeNet-5: Gradient-Based Learning"

    def test_prompt_for_authors(self):
        """Test prompting for authors."""
        from prompts import InteractivePrompter

        prompter = InteractivePrompter()

        with patch("builtins.input", return_value="LeCun et al."):
            result = prompter.prompt_for_authors()

        assert result == "LeCun et al."

    def test_prompt_with_default_value(self):
        """Test prompting accepts default values."""
        from prompts import InteractivePrompter
        import datetime

        prompter = InteractivePrompter()

        # User presses enter (empty input) to accept default
        with patch("builtins.input", return_value=""):
            result = prompter.prompt_for_year()

        # Should return current year as default
        current_year = str(datetime.datetime.now().year)
        assert result == current_year

    def test_prompt_with_custom_year(self):
        """Test user can override default year."""
        from prompts import InteractivePrompter

        prompter = InteractivePrompter()

        with patch("builtins.input", return_value="1998"):
            result = prompter.prompt_for_year()

        assert result == "1998"

    def test_prompt_validation_empty_required_field(self):
        """Test validation prevents empty required fields."""
        from prompts import InteractivePrompter

        prompter = InteractivePrompter()

        # Simulate user entering empty string, then valid value
        with patch("builtins.input", side_effect=["", "LeNet-5"]):
            with patch("sys.stdout", new_callable=StringIO) as mock_stdout:
                result = prompter.prompt_for_paper_name()

        assert result == "LeNet-5"
        # Check that error message was displayed
        output = mock_stdout.getvalue()
        assert "required" in output.lower() or "cannot be empty" in output.lower()

    def test_prompt_validation_invalid_year(self):
        """Test validation rejects invalid years."""
        from prompts import InteractivePrompter

        prompter = InteractivePrompter()

        # Try invalid year, then valid year
        with patch("builtins.input", side_effect=["abcd", "1998"]):
            with patch("sys.stdout", new_callable=StringIO) as mock_stdout:
                result = prompter.prompt_for_year()

        assert result == "1998"
        output = mock_stdout.getvalue()
        assert "invalid" in output.lower() or "must be" in output.lower()

    def test_prompt_validation_year_range(self):
        """Test year validation checks reasonable range."""
        from prompts import InteractivePrompter
        import datetime

        prompter = InteractivePrompter()

        # Try year in future, then valid year
        future_year = str(datetime.datetime.now().year + 10)
        with patch("builtins.input", side_effect=[future_year, "1998"]):
            with patch("sys.stdout", new_callable=StringIO) as mock_stdout:
                result = prompter.prompt_for_year()

        assert result == "1998"
        output = mock_stdout.getvalue()
        assert "future" in output.lower() or "range" in output.lower()

    def test_prompt_optional_field(self):
        """Test optional fields can be skipped."""
        from prompts import InteractivePrompter

        prompter = InteractivePrompter()

        # User presses enter without providing value
        with patch("builtins.input", return_value=""):
            result = prompter.prompt_for_url()

        # Should return default placeholder
        assert "TODO" in result or result == ""

    def test_prompt_url_validation(self):
        """Test URL validation accepts valid URLs."""
        from prompts import InteractivePrompter

        prompter = InteractivePrompter()

        with patch("builtins.input", return_value="https://arxiv.org/abs/1234.5678"):
            result = prompter.prompt_for_url()

        assert result == "https://arxiv.org/abs/1234.5678"

    def test_collect_all_metadata(self):
        """Test collecting complete metadata."""
        from prompts import InteractivePrompter

        prompter = InteractivePrompter()

        # Mock all user inputs
        inputs = [
            "LeNet-5",  # paper name
            "LeNet-5: Gradient-Based Learning",  # title
            "LeCun et al.",  # authors
            "1998",  # year
            "http://yann.lecun.com/exdb/lenet/",  # url
            "CNN for handwritten digit recognition",  # description
        ]

        with patch("builtins.input", side_effect=inputs):
            metadata = prompter.collect_metadata()

        assert metadata["PAPER_NAME"] == "LeNet-5"
        assert metadata["PAPER_TITLE"] == "LeNet-5: Gradient-Based Learning"
        assert metadata["AUTHORS"] == "LeCun et al."
        assert metadata["YEAR"] == "1998"
        assert metadata["PAPER_URL"] == "http://yann.lecun.com/exdb/lenet/"
        assert metadata["DESCRIPTION"] == "CNN for handwritten digit recognition"

    def test_collect_metadata_with_defaults(self):
        """Test collecting metadata using defaults."""
        from prompts import InteractivePrompter
        import datetime

        prompter = InteractivePrompter()

        # Provide required fields, accept defaults for optional
        inputs = [
            "LeNet-5",  # paper name
            "LeNet-5 Paper",  # title
            "LeCun et al.",  # authors
            "",  # year (use default)
            "",  # url (use default)
            "",  # description (use default)
        ]

        with patch("builtins.input", side_effect=inputs):
            metadata = prompter.collect_metadata()

        assert metadata["PAPER_NAME"] == "LeNet-5"
        assert metadata["PAPER_TITLE"] == "LeNet-5 Paper"
        assert metadata["AUTHORS"] == "LeCun et al."
        assert metadata["YEAR"] == str(datetime.datetime.now().year)

    def test_merge_with_existing_args(self):
        """Test merging prompts with existing CLI arguments."""
        from prompts import InteractivePrompter

        prompter = InteractivePrompter(quiet=True)  # Quiet mode to avoid extra output

        # Existing args from CLI
        existing = {
            "paper": "LeNet-5",
            "title": "LeNet-5 Paper",
            "authors": None,  # Missing
            "year": None,  # Missing
            "url": None,  # Missing
            "description": None,  # Missing
        }

        # Only prompt for missing fields
        inputs = [
            "LeCun et al.",  # authors (missing)
            "1998",  # year (missing)
            "",  # url (optional, use default)
            "",  # description (optional, use default)
        ]

        with patch("builtins.input", side_effect=inputs):
            metadata = prompter.collect_metadata(existing=existing)

        # Should use existing values where provided
        assert metadata["PAPER_NAME"] == "LeNet-5"
        assert metadata["PAPER_TITLE"] == "LeNet-5 Paper"
        # Should prompt for missing values
        assert metadata["AUTHORS"] == "LeCun et al."
        assert metadata["YEAR"] == "1998"

    def test_help_text_display(self):
        """Test that helpful prompts are displayed."""
        from prompts import InteractivePrompter

        prompter = InteractivePrompter()

        # Mock input to capture the prompt message passed to input()
        with patch("builtins.input", return_value="LeNet-5") as mock_input:
            prompter.prompt_for_paper_name()

        # Check that input() was called with a descriptive prompt
        call_args = mock_input.call_args[0][0]
        assert "paper" in call_args.lower() or "name" in call_args.lower()

    def test_example_text_display(self):
        """Test that examples are shown in prompts."""
        from prompts import InteractivePrompter

        prompter = InteractivePrompter()

        # Mock input to capture the prompt message
        with patch("builtins.input", return_value="http://example.com") as mock_input:
            prompter.prompt_for_url()

        # Check that input() was called with example text
        call_args = mock_input.call_args[0][0]
        assert "example" in call_args.lower() or "http" in call_args.lower()


class TestInteractiveMode:
    """Test interactive mode integration (Issue #771)."""

    def test_interactive_mode_enabled_when_args_missing(self):
        """Test interactive mode activates when args missing."""
        # This will be tested after integration
        pass

    def test_non_interactive_mode_with_all_args(self):
        """Test CLI args bypass interactive mode."""
        # This will be tested after integration
        pass


if __name__ == "__main__":
    # Run tests manually without pytest
    print("Running user prompt tests...")

    import traceback

    test_class = TestInteractivePrompter()
    test_methods = [m for m in dir(test_class) if m.startswith("test_")]

    passed = 0
    failed = 0

    for method_name in test_methods:
        try:
            method = getattr(test_class, method_name)
            method()
            print(f"✓ {method_name}")
            passed += 1
        except Exception as e:
            print(f"✗ {method_name}: {e}")
            traceback.print_exc()
            failed += 1

    print(f"\n{passed} passed, {failed} failed")
    sys.exit(0 if failed == 0 else 1)
