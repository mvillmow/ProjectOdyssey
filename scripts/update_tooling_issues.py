#!/usr/bin/env python3
"""
Update github_issue.md files in 03-tooling with detailed bodies from plan.md files.
"""

import os
from pathlib import Path
import re


def read_plan_file(plan_path):
    """Read and parse a plan.md file."""
    with open(plan_path, 'r') as f:
        content = f.read()

    sections = {}

    # Extract title (first h1)
    title_match = re.search(r'^# (.+)$', content, re.MULTILINE)
    sections['title'] = title_match.group(1) if title_match else "Unknown"

    # Extract overview
    overview_match = re.search(r'## Overview\n(.+?)(?=\n## |$)', content, re.DOTALL)
    sections['overview'] = overview_match.group(1).strip() if overview_match else ""

    # Extract inputs
    inputs_match = re.search(r'## Inputs\n(.+?)(?=\n## |$)', content, re.DOTALL)
    sections['inputs'] = inputs_match.group(1).strip() if inputs_match else ""

    # Extract outputs
    outputs_match = re.search(r'## Outputs\n(.+?)(?=\n## |$)', content, re.DOTALL)
    sections['outputs'] = outputs_match.group(1).strip() if outputs_match else ""

    # Extract steps
    steps_match = re.search(r'## Steps\n(.+?)(?=\n## |$)', content, re.DOTALL)
    sections['steps'] = steps_match.group(1).strip() if steps_match else ""

    # Extract success criteria
    criteria_match = re.search(r'## Success Criteria\n(.+?)(?=\n## |$)', content, re.DOTALL)
    sections['success_criteria'] = criteria_match.group(1).strip() if criteria_match else ""

    # Extract notes
    notes_match = re.search(r'## Notes\n(.+?)(?=\n## |$)', content, re.DOTALL)
    sections['notes'] = notes_match.group(1).strip() if notes_match else ""

    return sections


def generate_plan_body(sections):
    """Generate the Plan issue body."""
    body = f"""## Overview
{sections['overview']}

## Objectives
This planning phase will:
- Define detailed specifications and requirements
- Design the architecture and approach
- Document API contracts and interfaces
- Create comprehensive design documentation

## Inputs
{sections['inputs']}

## Expected Outputs
{sections['outputs']}

## Success Criteria
{sections['success_criteria']}

## Additional Notes
{sections['notes']}"""
    return body


def generate_test_body(sections):
    """Generate the Test issue body."""
    body = f"""## Overview
{sections['overview']}

## Testing Objectives
This phase focuses on:
- Writing comprehensive test cases following TDD principles
- Creating test fixtures and mock data
- Defining test scenarios for edge cases
- Setting up test infrastructure

## What to Test
Based on the expected outputs:
{sections['outputs']}

## Test Success Criteria
{sections['success_criteria']}

## Implementation Steps
{sections['steps']}

## Notes
{sections['notes']}"""
    return body


def generate_implementation_body(sections):
    """Generate the Implementation issue body."""
    body = f"""## Overview
{sections['overview']}

## Implementation Goals
- Implement the functionality to pass all tests
- Follow Mojo best practices and coding standards
- Ensure code is clean, documented, and maintainable
- Meet all requirements specified in the plan

## Required Inputs
{sections['inputs']}

## Expected Outputs
{sections['outputs']}

## Implementation Steps
{sections['steps']}

## Success Criteria
{sections['success_criteria']}

## Notes
{sections['notes']}"""
    return body


def generate_packaging_body(sections):
    """Generate the Packaging issue body."""
    body = f"""## Overview
{sections['overview']}

## Packaging Objectives
- Integrate the implementation with existing codebase
- Ensure all dependencies are properly configured
- Verify compatibility with other components
- Package for deployment/distribution

## Integration Requirements
Based on outputs:
{sections['outputs']}

## Integration Steps
{sections['steps']}

## Success Criteria
{sections['success_criteria']}

## Notes
{sections['notes']}"""
    return body


def generate_cleanup_body(sections):
    """Generate the Cleanup issue body."""
    body = f"""## Overview
{sections['overview']}

## Cleanup Objectives
- Refactor code for optimal quality and maintainability
- Remove technical debt and temporary workarounds
- Ensure comprehensive documentation
- Perform final validation and optimization

## Cleanup Tasks
- Code review and refactoring
- Documentation finalization
- Performance optimization
- Final testing and validation

## Success Criteria
{sections['success_criteria']}

## Notes
{sections['notes']}"""
    return body


def update_github_issue_file(issue_path, plan_sections):
    """Update a github_issue.md file with detailed bodies."""
    title = plan_sections['title']

    # Generate all issue bodies
    plan_body = generate_plan_body(plan_sections)
    test_body = generate_test_body(plan_sections)
    impl_body = generate_implementation_body(plan_sections)
    package_body = generate_packaging_body(plan_sections)
    cleanup_body = generate_cleanup_body(plan_sections)

    # Create the complete github_issue.md content
    content = f"""# GitHub Issues

**Plan Issue**:
- Title: [Plan] {title} - Design and Documentation
- Body:
```
{plan_body}
```
- Labels: planning, documentation
- URL: [to be filled]

**Test Issue**:
- Title: [Test] {title} - Write Tests
- Body:
```
{test_body}
```
- Labels: testing, tdd
- URL: [to be filled]

**Implementation Issue**:
- Title: [Impl] {title} - Implementation
- Body:
```
{impl_body}
```
- Labels: implementation
- URL: [to be filled]

**Packaging Issue**:
- Title: [Package] {title} - Integration and Packaging
- Body:
```
{package_body}
```
- Labels: packaging, integration
- URL: [to be filled]

**Cleanup Issue**:
- Title: [Cleanup] {title} - Refactor and Finalize
- Body:
```
{cleanup_body}
```
- Labels: cleanup, documentation
- URL: [to be filled]
"""

    with open(issue_path, 'w') as f:
        f.write(content)


def process_directory(base_dir):
    """Process all github_issue.md files in the directory tree."""
    base_path = Path(base_dir)
    updated_files = []
    errors = []

    # Find all github_issue.md files
    for issue_file in base_path.rglob('github_issue.md'):
        # Find corresponding plan.md file
        plan_file = issue_file.parent / 'plan.md'

        if not plan_file.exists():
            errors.append(f"Missing plan.md for {issue_file}")
            continue

        try:
            # Read and parse plan
            plan_sections = read_plan_file(plan_file)

            # Update github_issue.md
            update_github_issue_file(issue_file, plan_sections)

            updated_files.append(str(issue_file))
            print(f"Updated: {issue_file}")

        except Exception as e:
            errors.append(f"Error processing {issue_file}: {str(e)}")
            print(f"ERROR: {issue_file} - {str(e)}")

    return updated_files, errors


def main():
    """Main execution function."""
    base_dir = '/home/mvillmow/ml-odyssey/notes/plan/03-tooling'

    print(f"Processing github_issue.md files in {base_dir}...")
    print("=" * 80)

    updated_files, errors = process_directory(base_dir)

    print("\n" + "=" * 80)
    print(f"\nSummary:")
    print(f"  Updated: {len(updated_files)} files")
    print(f"  Errors: {len(errors)} files")

    if errors:
        print("\nErrors:")
        for error in errors:
            print(f"  - {error}")

    print(f"\nAll updates completed!")


if __name__ == '__main__':
    main()
