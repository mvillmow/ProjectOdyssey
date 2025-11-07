#!/usr/bin/env python3
"""
Script to update GitHub issue files in 06-agentic-workflows with detailed, specific bodies.
"""

import os
import re
from pathlib import Path


def read_plan_file(plan_path):
    """Read and parse a plan.md file."""
    if not os.path.exists(plan_path):
        return None

    with open(plan_path, 'r') as f:
        content = f.read()

    # Extract sections
    sections = {
        'title': '',
        'overview': '',
        'inputs': '',
        'outputs': '',
        'steps': '',
        'success_criteria': '',
        'notes': ''
    }

    # Extract title (first heading)
    title_match = re.search(r'^#\s+(.+)$', content, re.MULTILINE)
    if title_match:
        sections['title'] = title_match.group(1).strip()

    # Extract overview
    overview_match = re.search(r'##\s+Overview\s*\n(.*?)(?=\n##|\Z)', content, re.DOTALL)
    if overview_match:
        sections['overview'] = overview_match.group(1).strip()

    # Extract inputs
    inputs_match = re.search(r'##\s+Inputs\s*\n(.*?)(?=\n##|\Z)', content, re.DOTALL)
    if inputs_match:
        sections['inputs'] = inputs_match.group(1).strip()

    # Extract outputs
    outputs_match = re.search(r'##\s+Outputs\s*\n(.*?)(?=\n##|\Z)', content, re.DOTALL)
    if outputs_match:
        sections['outputs'] = outputs_match.group(1).strip()

    # Extract steps
    steps_match = re.search(r'##\s+Steps\s*\n(.*?)(?=\n##|\Z)', content, re.DOTALL)
    if steps_match:
        sections['steps'] = steps_match.group(1).strip()

    # Extract success criteria
    criteria_match = re.search(r'##\s+Success Criteria\s*\n(.*?)(?=\n##|\Z)', content, re.DOTALL)
    if criteria_match:
        sections['success_criteria'] = criteria_match.group(1).strip()

    # Extract notes
    notes_match = re.search(r'##\s+Notes\s*\n(.*?)(?=\Z)', content, re.DOTALL)
    if notes_match:
        sections['notes'] = notes_match.group(1).strip()

    return sections


def generate_plan_issue_body(plan_sections):
    """Generate detailed body for Plan issue."""
    title = plan_sections['title']
    overview = plan_sections['overview']
    inputs = plan_sections['inputs']
    outputs = plan_sections['outputs']

    body = f"""## Planning Phase: {title}

### Objective
{overview}

### Required Inputs
{inputs}

### Expected Outputs
{outputs}

### Planning Tasks
1. Review and validate the requirements and constraints
2. Design the architecture and component structure
3. Identify dependencies and integration points
4. Document design decisions and rationale
5. Create detailed technical specifications
6. Review plan with stakeholders

### Deliverables
- Detailed design document
- Architecture diagrams (if applicable)
- Technical specifications
- Updated plan.md with any refinements

### Success Criteria
- All design decisions documented and justified
- Architecture is clear and well-defined
- Dependencies identified and documented
- Plan reviewed and approved
"""
    return body


def generate_test_issue_body(plan_sections):
    """Generate detailed body for Test issue."""
    title = plan_sections['title']
    overview = plan_sections['overview']
    success_criteria = plan_sections['success_criteria']

    body = f"""## Testing Phase: {title}

### Objective
Write comprehensive tests following TDD principles for {title.lower()}.

### Overview
{overview}

### Testing Tasks
1. Review success criteria and acceptance requirements
2. Design test cases covering all requirements
3. Write unit tests for individual components
4. Write integration tests for component interactions
5. Create test fixtures and mock data as needed
6. Document test coverage and test scenarios
7. Set up test automation in CI/CD pipeline

### Test Coverage Requirements
{success_criteria}

### Deliverables
- Complete test suite (unit and integration tests)
- Test documentation
- Test fixtures and test data
- CI/CD test automation configuration

### Success Criteria
- All tests written before implementation (TDD)
- Test coverage meets project standards
- Tests are automated and reproducible
- Test documentation is clear and complete
"""
    return body


def generate_implementation_issue_body(plan_sections):
    """Generate detailed body for Implementation issue."""
    title = plan_sections['title']
    overview = plan_sections['overview']
    steps = plan_sections['steps']
    outputs = plan_sections['outputs']

    body = f"""## Implementation Phase: {title}

### Objective
Implement {title.lower()} according to the design specifications and passing all tests.

### Overview
{overview}

### Implementation Steps
{steps}

### Expected Outputs
{outputs}

### Implementation Tasks
1. Review design specifications and test requirements
2. Set up development environment and dependencies
3. Implement core functionality following TDD approach
4. Ensure all tests pass
5. Add error handling and edge case management
6. Add logging and monitoring capabilities
7. Write inline documentation and docstrings
8. Conduct code self-review

### Deliverables
- Complete implementation code
- All tests passing
- Inline code documentation
- Error handling and logging

### Success Criteria
- Implementation matches design specifications
- All tests pass successfully
- Code follows project style guidelines
- Error handling is comprehensive
- Code is well-documented
"""
    return body


def generate_packaging_issue_body(plan_sections):
    """Generate detailed body for Packaging issue."""
    title = plan_sections['title']
    overview = plan_sections['overview']

    body = f"""## Packaging Phase: {title}

### Objective
Package and integrate {title.lower()} into the broader system.

### Overview
{overview}

### Packaging Tasks
1. Review integration points and dependencies
2. Create or update module/package structure
3. Configure build and packaging scripts
4. Update project configuration files (pyproject.toml, magic.toml, etc.)
5. Create integration tests
6. Update import statements and module exports
7. Verify compatibility with existing components
8. Update version numbers and changelog

### Integration Checklist
- [ ] Module properly structured and organized
- [ ] Dependencies documented and configured
- [ ] Integration tests pass
- [ ] No breaking changes to existing code
- [ ] Imports and exports properly configured
- [ ] Build/packaging scripts updated

### Deliverables
- Packaged and integrated component
- Updated configuration files
- Integration test results
- Updated changelog

### Success Criteria
- Component successfully integrates with existing system
- All integration tests pass
- No regression in existing functionality
- Package builds successfully
- Documentation updated
"""
    return body


def generate_cleanup_issue_body(plan_sections):
    """Generate detailed body for Cleanup issue."""
    title = plan_sections['title']
    overview = plan_sections['overview']
    notes = plan_sections['notes']

    body = f"""## Cleanup Phase: {title}

### Objective
Refactor, optimize, and finalize {title.lower()} for production readiness.

### Overview
{overview}

### Cleanup Tasks
1. Code review and refactoring
   - Remove dead code and debug statements
   - Improve code clarity and maintainability
   - Apply DRY principles
   - Optimize performance bottlenecks

2. Documentation finalization
   - Complete API documentation
   - Update README and usage guides
   - Add examples and tutorials
   - Document known issues and limitations

3. Testing and validation
   - Run full test suite
   - Perform manual testing
   - Check edge cases
   - Validate performance benchmarks

4. Final polish
   - Code formatting and linting
   - Consistent naming conventions
   - Remove TODOs and FIXMEs
   - Update comments and docstrings

### Additional Notes
{notes if notes else 'No additional notes.'}

### Deliverables
- Refactored and optimized code
- Complete documentation
- Final test results
- Performance benchmarks (if applicable)

### Success Criteria
- Code is clean, maintainable, and well-documented
- All tests pass with good coverage
- Documentation is complete and accurate
- No critical issues remain
- Code meets all quality standards
- Ready for production use
"""
    return body


def update_github_issue_file(github_issue_path, plan_path):
    """Update a github_issue.md file with detailed bodies."""
    # Read the plan
    plan_sections = read_plan_file(plan_path)
    if not plan_sections:
        return False, "Could not read plan.md"

    # Read current github_issue.md
    with open(github_issue_path, 'r') as f:
        content = f.read()

    # Generate detailed bodies
    plan_body = generate_plan_issue_body(plan_sections)
    test_body = generate_test_issue_body(plan_sections)
    impl_body = generate_implementation_issue_body(plan_sections)
    package_body = generate_packaging_issue_body(plan_sections)
    cleanup_body = generate_cleanup_issue_body(plan_sections)

    # Extract title from plan
    title = plan_sections['title']

    # Create new content
    new_content = f"""# GitHub Issues

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

    # Write updated content
    with open(github_issue_path, 'w') as f:
        f.write(new_content)

    return True, "Successfully updated"


def main():
    base_path = '/home/mvillmow/ml-odyssey/notes/plan/06-agentic-workflows'

    # Find all github_issue.md files
    github_issue_files = []
    for root, dirs, files in os.walk(base_path):
        if 'github_issue.md' in files:
            github_issue_path = os.path.join(root, 'github_issue.md')
            plan_path = os.path.join(root, 'plan.md')
            if os.path.exists(plan_path):
                github_issue_files.append((github_issue_path, plan_path))

    github_issue_files.sort()

    print(f"Found {len(github_issue_files)} github_issue.md files with corresponding plan.md")
    print("=" * 80)

    success_count = 0
    errors = []

    for github_issue_path, plan_path in github_issue_files:
        rel_path = github_issue_path.replace(base_path + '/', '')
        print(f"\nProcessing: {rel_path}")

        try:
            success, message = update_github_issue_file(github_issue_path, plan_path)
            if success:
                print(f"  ✓ {message}")
                success_count += 1
            else:
                print(f"  ✗ {message}")
                errors.append((rel_path, message))
        except Exception as e:
            error_msg = str(e)
            print(f"  ✗ Error: {error_msg}")
            errors.append((rel_path, error_msg))

    # Print summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total files processed: {len(github_issue_files)}")
    print(f"Successfully updated: {success_count}")
    print(f"Errors: {len(errors)}")

    if errors:
        print("\nErrors encountered:")
        for file_path, error_msg in errors:
            print(f"  - {file_path}: {error_msg}")

    print("\nDone!")


if __name__ == '__main__':
    main()
