#!/usr/bin/env python3
"""Tests for GitHub issue and PR templates."""

import pytest
import yaml
from pathlib import Path


ISSUE_TEMPLATE_DIR = Path(".github/ISSUE_TEMPLATE")
PR_TEMPLATE_DIR = Path(".github/PULL_REQUEST_TEMPLATE")
PR_TEMPLATE_FILE = PR_TEMPLATE_DIR / "pull_request_template.md"


class TestIssueTemplates:
    """Test GitHub issue templates."""

    def test_issue_template_directory_exists(self):
        """Test that issue template directory exists."""
        assert ISSUE_TEMPLATE_DIR.exists()
        assert ISSUE_TEMPLATE_DIR.is_dir()

    def test_all_templates_are_valid_yaml(self):
        """Test that all .yml issue templates are valid YAML."""
        template_files = list(ISSUE_TEMPLATE_DIR.glob("*.yml"))
        assert len(template_files) > 0, "No .yml templates found"

        for template_file in template_files:
            with open(template_file, "r") as f:
                try:
                    data = yaml.safe_load(f)
                    assert data is not None, f"{template_file.name} is empty"
                except yaml.YAMLError as e:
                    pytest.fail(f"{template_file.name} has invalid YAML: {e}")

    def test_template_config_exists(self):
        """Test that config.yml exists."""
        config_file = ISSUE_TEMPLATE_DIR / "config.yml"
        assert config_file.exists()

        with open(config_file, "r") as f:
            config = yaml.safe_load(f)
            assert config is not None


class TestBugReportTemplate:
    """Test bug report issue template."""

    @pytest.fixture
    def template_data(self):
        """Load bug report template."""
        template_file = ISSUE_TEMPLATE_DIR / "01-bug-report.yml"
        with open(template_file, "r") as f:
            return yaml.safe_load(f)

    def test_template_exists(self):
        """Test that bug report template exists."""
        template_file = ISSUE_TEMPLATE_DIR / "01-bug-report.yml"
        assert template_file.exists()

    def test_has_required_metadata(self, template_data):
        """Test template has required metadata fields."""
        assert "name" in template_data
        assert "description" in template_data
        assert "title" in template_data
        assert "labels" in template_data

    def test_has_bug_label(self, template_data):
        """Test template auto-assigns bug label."""
        labels = template_data.get("labels", [])
        assert "bug" in labels or "type:bug" in labels or any("bug" in label.lower() for label in labels)

    def test_has_body_sections(self, template_data):
        """Test template has body sections."""
        assert "body" in template_data
        assert len(template_data["body"]) > 0

    def test_has_description_field(self, template_data):
        """Test template has bug description field."""
        body = template_data.get("body", [])
        description_fields = [field for field in body if field.get("id") == "description"]
        assert len(description_fields) > 0

    def test_has_environment_field(self, template_data):
        """Test template has environment/system info field."""
        body = template_data.get("body", [])
        env_fields = [
            field
            for field in body
            if "environment" in field.get("id", "").lower()
            or "system" in field.get("attributes", {}).get("label", "").lower()
        ]
        assert len(env_fields) > 0


class TestFeatureRequestTemplate:
    """Test feature request issue template."""

    @pytest.fixture
    def template_data(self):
        """Load feature request template."""
        template_file = ISSUE_TEMPLATE_DIR / "02-feature-request.yml"
        with open(template_file, "r") as f:
            return yaml.safe_load(f)

    def test_template_exists(self):
        """Test that feature request template exists."""
        template_file = ISSUE_TEMPLATE_DIR / "02-feature-request.yml"
        assert template_file.exists()

    def test_has_enhancement_label(self, template_data):
        """Test template auto-assigns enhancement label."""
        labels = template_data.get("labels", [])
        assert any("enhancement" in label.lower() or "feature" in label.lower() for label in labels)

    def test_has_description_field(self, template_data):
        """Test template has feature description field."""
        body = template_data.get("body", [])
        assert len(body) > 0


class TestPaperImplementationTemplate:
    """Test paper implementation issue template."""

    @pytest.fixture
    def template_data(self):
        """Load paper implementation template."""
        template_file = ISSUE_TEMPLATE_DIR / "03-paper-implementation.yml"
        with open(template_file, "r") as f:
            return yaml.safe_load(f)

    def test_template_exists(self):
        """Test that paper implementation template exists."""
        template_file = ISSUE_TEMPLATE_DIR / "03-paper-implementation.yml"
        assert template_file.exists()

    def test_has_paper_label(self, template_data):
        """Test template auto-assigns paper label."""
        labels = template_data.get("labels", [])
        assert any("paper" in label.lower() for label in labels)

    def test_has_description_field(self, template_data):
        """Test template has paper information field."""
        body = template_data.get("body", [])
        description_fields = [field for field in body if field.get("id") == "description"]
        assert len(description_fields) > 0


class TestDocumentationTemplate:
    """Test documentation issue template."""

    @pytest.fixture
    def template_data(self):
        """Load documentation template."""
        template_file = ISSUE_TEMPLATE_DIR / "04-documentation.yml"
        with open(template_file, "r") as f:
            return yaml.safe_load(f)

    def test_template_exists(self):
        """Test that documentation template exists."""
        template_file = ISSUE_TEMPLATE_DIR / "04-documentation.yml"
        assert template_file.exists()

    def test_has_documentation_label(self, template_data):
        """Test template auto-assigns documentation label."""
        labels = template_data.get("labels", [])
        assert any("doc" in label.lower() for label in labels)


class TestInfrastructureTemplate:
    """Test infrastructure issue template."""

    @pytest.fixture
    def template_data(self):
        """Load infrastructure template."""
        template_file = ISSUE_TEMPLATE_DIR / "05-infrastructure.yml"
        with open(template_file, "r") as f:
            return yaml.safe_load(f)

    def test_template_exists(self):
        """Test that infrastructure template exists."""
        template_file = ISSUE_TEMPLATE_DIR / "05-infrastructure.yml"
        assert template_file.exists()

    def test_has_infrastructure_label(self, template_data):
        """Test template auto-assigns infrastructure label."""
        labels = template_data.get("labels", [])
        assert any("infrastructure" in label.lower() for label in labels)


class TestQuestionTemplate:
    """Test question/support issue template."""

    @pytest.fixture
    def template_data(self):
        """Load question template."""
        template_file = ISSUE_TEMPLATE_DIR / "06-question.yml"
        with open(template_file, "r") as f:
            return yaml.safe_load(f)

    def test_template_exists(self):
        """Test that question template exists."""
        template_file = ISSUE_TEMPLATE_DIR / "06-question.yml"
        assert template_file.exists()

    def test_has_question_label(self, template_data):
        """Test template auto-assigns question label."""
        labels = template_data.get("labels", [])
        assert any("question" in label.lower() for label in labels)


class TestPerformanceIssueTemplate:
    """Test performance issue template."""

    @pytest.fixture
    def template_data(self):
        """Load performance issue template."""
        template_file = ISSUE_TEMPLATE_DIR / "07-performance-issue.yml"
        with open(template_file, "r") as f:
            return yaml.safe_load(f)

    def test_template_exists(self):
        """Test that performance issue template exists."""
        template_file = ISSUE_TEMPLATE_DIR / "07-performance-issue.yml"
        assert template_file.exists()

    def test_has_performance_label(self, template_data):
        """Test template auto-assigns performance label."""
        labels = template_data.get("labels", [])
        assert any("performance" in label.lower() for label in labels)

    def test_has_metrics_fields(self, template_data):
        """Test template has performance metrics fields."""
        body = template_data.get("body", [])
        # Should have current and expected metrics fields
        metric_fields = [
            field
            for field in body
            if "metric" in field.get("id", "").lower()
            or "metric" in field.get("attributes", {}).get("label", "").lower()
        ]
        assert len(metric_fields) >= 2  # At least current and expected metrics

    def test_has_environment_field(self, template_data):
        """Test template has hardware/environment field."""
        body = template_data.get("body", [])
        env_fields = [
            field
            for field in body
            if "environment" in field.get("id", "").lower()
            or "hardware" in field.get("attributes", {}).get("label", "").lower()
        ]
        assert len(env_fields) > 0

    def test_has_baseline_field(self, template_data):
        """Test template has baseline comparison field."""
        body = template_data.get("body", [])
        baseline_fields = [field for field in body if "baseline" in field.get("id", "").lower()]
        assert len(baseline_fields) > 0


class TestAllTemplatesConsistency:
    """Test consistency across all templates."""

    def test_all_templates_have_labels(self):
        """Test that all templates auto-assign labels."""
        template_files = [f for f in ISSUE_TEMPLATE_DIR.glob("*.yml") if f.name != "config.yml"]

        for template_file in template_files:
            with open(template_file, "r") as f:
                data = yaml.safe_load(f)
                assert "labels" in data, f"{template_file.name} has no labels"
                assert len(data["labels"]) > 0, f"{template_file.name} has empty labels"

    def test_all_templates_have_title(self):
        """Test that all templates have title prefix."""
        template_files = [f for f in ISSUE_TEMPLATE_DIR.glob("*.yml") if f.name != "config.yml"]

        for template_file in template_files:
            with open(template_file, "r") as f:
                data = yaml.safe_load(f)
                assert "title" in data, f"{template_file.name} has no title"

    def test_all_templates_have_description(self):
        """Test that all templates have description."""
        template_files = [f for f in ISSUE_TEMPLATE_DIR.glob("*.yml") if f.name != "config.yml"]

        for template_file in template_files:
            with open(template_file, "r") as f:
                data = yaml.safe_load(f)
                assert "description" in data, f"{template_file.name} has no description"
                assert len(data["description"]) > 0, f"{template_file.name} has empty description"


class TestPRTemplate:
    """Test Pull Request template."""

    def test_pr_template_exists(self):
        """Test that PR template file exists."""
        assert PR_TEMPLATE_FILE.exists()

    def test_pr_template_not_empty(self):
        """Test that PR template has content."""
        content = PR_TEMPLATE_FILE.read_text()
        assert len(content) > 0

    def test_has_description_section(self):
        """Test PR template has Description section."""
        content = PR_TEMPLATE_FILE.read_text()
        assert "## Description" in content or "# Description" in content

    def test_has_related_issues_section(self):
        """Test PR template has Related Issues section."""
        content = PR_TEMPLATE_FILE.read_text()
        assert "Related Issues" in content or "Closes #" in content

    def test_has_changes_section(self):
        """Test PR template has Changes section."""
        content = PR_TEMPLATE_FILE.read_text()
        assert "## Changes" in content or "# Changes" in content

    def test_has_testing_section(self):
        """Test PR template has Testing section."""
        content = PR_TEMPLATE_FILE.read_text()
        assert "## Testing" in content or "# Testing" in content or "Test" in content

    def test_has_checklist(self):
        """Test PR template has checklist items."""
        content = PR_TEMPLATE_FILE.read_text()
        # Check for checkbox markdown syntax
        assert "- [ ]" in content or "- []" in content

    def test_checklist_comprehensive(self):
        """Test PR template has comprehensive checklist."""
        content = PR_TEMPLATE_FILE.read_text()
        # Should have multiple checklist categories
        assert content.count("- [ ]") >= 10 or content.count("- []") >= 10

    def test_has_documentation_section(self):
        """Test PR template has Documentation section."""
        content = PR_TEMPLATE_FILE.read_text()
        assert "Documentation" in content

    def test_has_quality_checklist(self):
        """Test PR template has quality checklist."""
        content = PR_TEMPLATE_FILE.read_text()
        assert "Quality" in content or "quality" in content.lower()

    def test_has_security_section(self):
        """Test PR template has Security section."""
        content = PR_TEMPLATE_FILE.read_text()
        assert "Security" in content or "security" in content.lower()

    def test_has_performance_section(self):
        """Test PR template has Performance section."""
        content = PR_TEMPLATE_FILE.read_text()
        assert "Performance" in content or "performance" in content.lower()

    def test_valid_markdown(self):
        """Test PR template is valid markdown."""
        content = PR_TEMPLATE_FILE.read_text()
        # Basic markdown validation - should have headers
        assert "#" in content
        # Should not have obvious syntax errors
        assert not content.count("```") % 2  # Balanced code blocks


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
