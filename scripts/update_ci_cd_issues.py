#!/usr/bin/env python3
"""
Script to update all github_issue.md files in the 05-ci-cd section with detailed bodies.
"""

import os
import re
from pathlib import Path
from typing import Dict, List, Tuple


def parse_plan_md(plan_path: Path) -> Dict[str, str]:
    """Parse a plan.md file and extract all sections."""
    if not plan_path.exists():
        return {}

    content = plan_path.read_text()

    sections = {
        'title': '',
        'overview': '',
        'inputs': '',
        'outputs': '',
        'steps': '',
        'success_criteria': '',
        'notes': ''
    }

    # Extract title (first # heading)
    title_match = re.search(r'^#\s+(.+)$', content, re.MULTILINE)
    if title_match:
        sections['title'] = title_match.group(1).strip()

    # Extract sections
    section_patterns = {
        'overview': r'##\s+Overview\s*\n(.*?)(?=\n##|\Z)',
        'inputs': r'##\s+Inputs\s*\n(.*?)(?=\n##|\Z)',
        'outputs': r'##\s+Outputs\s*\n(.*?)(?=\n##|\Z)',
        'steps': r'##\s+(?:Implementation )?Steps\s*\n(.*?)(?=\n##|\Z)',
        'success_criteria': r'##\s+Success Criteria\s*\n(.*?)(?=\n##|\Z)',
        'notes': r'##\s+(?:Notes|Additional Notes)\s*\n(.*?)(?=\n##|\Z)'
    }

    for key, pattern in section_patterns.items():
        match = re.search(pattern, content, re.MULTILINE | re.DOTALL)
        if match:
            sections[key] = match.group(1).strip()

    return sections


def generate_issue_body(issue_type: str, plan_sections: Dict[str, str], task_name: str) -> str:
    """Generate a detailed issue body based on the issue type and plan sections."""

    title = plan_sections.get('title', task_name)
    overview = plan_sections.get('overview', '')
    inputs = plan_sections.get('inputs', '')
    outputs = plan_sections.get('outputs', '')
    steps = plan_sections.get('steps', '')
    success_criteria = plan_sections.get('success_criteria', '')
    notes = plan_sections.get('notes', '')

    if issue_type == 'Plan':
        body = f"""## Overview
{overview}

## Planning Tasks

### Design Decisions
- Review the requirements and constraints for {title}
- Document architectural decisions and design patterns
- Identify dependencies and integration points
- Define interfaces and contracts

### Documentation
- Create detailed technical specifications
- Document API designs and data structures
- Outline configuration requirements
- Plan testing strategies

## Expected Inputs
{inputs if inputs else 'N/A'}

## Expected Outputs
{outputs if outputs else 'N/A'}

## Success Criteria
{success_criteria if success_criteria else '- Planning documentation is complete and reviewed\n- All design decisions are documented\n- Dependencies and interfaces are clearly defined'}

## Notes
{notes if notes else 'This is the planning phase - focus on design and documentation before implementation.'}"""

    elif issue_type == 'Test':
        body = f"""## Overview
{overview}

## Test Development Tasks

### Test Planning
- Identify test scenarios for {title}
- Define test fixtures and mock data
- Plan integration test requirements
- Document edge cases and error conditions

### Test Implementation
- Write unit tests for core functionality
- Create integration tests for workflows
- Implement property-based tests where applicable
- Set up test fixtures and utilities

### Test Steps
{steps if steps else 'See plan.md for detailed implementation steps'}

## Expected Inputs
{inputs if inputs else 'N/A'}

## Expected Outputs
- Comprehensive test suite with high coverage
- Test documentation and examples
- CI-ready test configurations

## Success Criteria
{success_criteria if success_criteria else '- All test cases pass\n- Code coverage meets requirements\n- Tests are maintainable and well-documented'}

## Notes
{notes if notes else 'Follow TDD principles - write tests before implementation.'}"""

    elif issue_type == 'Implementation':
        body = f"""## Overview
{overview}

## Implementation Tasks

### Core Implementation
{steps if steps else '- Implement the functionality as specified in plan.md\n- Follow the design decisions from the planning phase\n- Ensure code quality and maintainability'}

### Requirements
- All tests from the Test issue must be passing
- Code must follow project style guidelines
- Implementation must match the design specifications

## Expected Inputs
{inputs if inputs else 'N/A'}

## Expected Outputs
{outputs if outputs else 'N/A'}

## Success Criteria
{success_criteria if success_criteria else '- All functionality is implemented\n- All tests pass\n- Code is reviewed and approved'}

## Notes
{notes if notes else 'Focus on clean, maintainable code that follows best practices.'}"""

    elif issue_type == 'Packaging':
        body = f"""## Overview
Integration and packaging tasks for {title}.

{overview}

## Packaging Tasks

### Integration
- Integrate with existing codebase
- Verify compatibility with dependencies
- Test integration points and interfaces
- Update configuration files as needed

### Documentation
- Update API documentation
- Add usage examples and tutorials
- Document configuration options
- Update changelog and release notes

### Validation
- Run full test suite
- Verify CI/CD pipeline passes
- Check code coverage and quality metrics
- Perform integration testing

## Expected Outputs
{outputs if outputs else 'N/A'}

## Success Criteria
{success_criteria if success_criteria else '- Code is fully integrated\n- All tests pass in CI/CD\n- Documentation is complete and accurate'}

## Notes
{notes if notes else 'Ensure all components work together seamlessly before closing this issue.'}"""

    elif issue_type == 'Cleanup':
        body = f"""## Overview
Refactoring and finalization tasks for {title}.

{overview}

## Cleanup Tasks

### Code Refinement
- Refactor code for clarity and maintainability
- Remove any temporary or debug code
- Optimize performance where applicable
- Apply consistent code style and formatting

### Documentation Review
- Review and update all documentation
- Ensure comments are clear and accurate
- Update README and guides as needed
- Document any known limitations

### Final Validation
- Run complete test suite
- Verify all success criteria are met
- Check for code smells and technical debt
- Ensure CI/CD pipeline is green

## Success Criteria
{success_criteria if success_criteria else '- Code is clean and well-documented\n- All tests pass\n- No outstanding TODOs or technical debt'}

## Final Checks
- [ ] All code is refactored and optimized
- [ ] Documentation is complete and accurate
- [ ] Test suite passes completely
- [ ] CI/CD pipeline is successful
- [ ] Code review is approved

## Notes
{notes if notes else 'This is the final phase - ensure everything is polished and production-ready.'}"""

    else:
        body = f"See plan: [plan.md](plan.md)"

    return body


def update_github_issue_file(github_issue_path: Path, plan_path: Path) -> Tuple[bool, str]:
    """Update a github_issue.md file with detailed bodies."""

    try:
        # Parse the plan.md file
        plan_sections = parse_plan_md(plan_path)

        if not plan_sections.get('title'):
            return False, f"Could not extract title from {plan_path}"

        task_name = plan_sections['title']

        # Generate bodies for each issue type
        issue_types = {
            'Plan': ('planning, documentation', 'Design and Documentation'),
            'Test': ('testing, tdd', 'Write Tests'),
            'Implementation': ('implementation', 'Implementation'),
            'Packaging': ('packaging, integration', 'Integration and Packaging'),
            'Cleanup': ('cleanup, documentation', 'Refactor and Finalize')
        }

        content_lines = ["# GitHub Issues\n"]

        for issue_prefix, (labels, suffix) in issue_types.items():
            body = generate_issue_body(issue_prefix, plan_sections, task_name)

            content_lines.append(f"**{issue_prefix} Issue**:")
            content_lines.append(f"- Title: [{issue_prefix}] {task_name} - {suffix}")
            content_lines.append(f"- Body: \n```\n{body}\n```")
            content_lines.append(f"- Labels: {labels}")
            content_lines.append(f"- URL: [to be filled]\n")

        # Write the updated content
        github_issue_path.write_text('\n'.join(content_lines))

        return True, f"Successfully updated {github_issue_path}"

    except Exception as e:
        return False, f"Error processing {github_issue_path}: {str(e)}"


def main():
    """Main function to process all github_issue.md files in 05-ci-cd."""

    base_path = Path("/home/mvillmow/ml-odyssey/notes/plan/05-ci-cd")

    if not base_path.exists():
        print(f"Error: Base path {base_path} does not exist")
        return

    # Find all github_issue.md files
    github_issue_files = list(base_path.rglob("github_issue.md"))

    print(f"Found {len(github_issue_files)} github_issue.md files to process\n")

    results = {
        'success': [],
        'errors': []
    }

    for github_issue_path in sorted(github_issue_files):
        # Find the corresponding plan.md file
        plan_path = github_issue_path.parent / "plan.md"

        if not plan_path.exists():
            results['errors'].append(f"No plan.md found for {github_issue_path}")
            continue

        success, message = update_github_issue_file(github_issue_path, plan_path)

        if success:
            results['success'].append(message)
            print(f"✓ {github_issue_path.relative_to(base_path)}")
        else:
            results['errors'].append(message)
            print(f"✗ {github_issue_path.relative_to(base_path)}: {message}")

    # Print summary
    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    print(f"Successfully updated: {len(results['success'])} files")
    print(f"Errors: {len(results['errors'])} files")

    if results['errors']:
        print("\nErrors:")
        for error in results['errors']:
            print(f"  - {error}")

    return len(results['errors']) == 0


if __name__ == "__main__":
    main()
