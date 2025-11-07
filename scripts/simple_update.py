#!/usr/bin/env python3
"""Simple batch updater for github_issue.md files"""
import os
import re
from pathlib import Path
import sys

def extract_component_name(file_path):
    """Extract a human-readable component name from the file path."""
    parts = file_path.replace('/home/mvillmow/ml-odyssey/notes/plan/', '').replace('/github_issue.md', '').split('/')
    readable_parts = []
    for part in parts:
        clean = part.split('-', 1)[1] if part and part[0].isdigit() and '-' in part else part
        readable = clean.replace('-', ' ').title()
        readable_parts.append(readable)
    if len(readable_parts) == 1:
        return readable_parts[0]
    else:
        return ' - '.join(readable_parts)

def get_plan_link(github_issue_path):
    """Generate the relative link to plan.md from the same directory."""
    dir_path = os.path.dirname(github_issue_path)
    plan_path = os.path.join(dir_path, 'plan.md')
    rel_path = plan_path.replace('/home/mvillmow/ml-odyssey/', '')
    return rel_path

def read_plan_overview(github_issue_path):
    """Read the overview section from the corresponding plan.md file."""
    dir_path = os.path.dirname(github_issue_path)
    plan_path = os.path.join(dir_path, 'plan.md')

    if not os.path.exists(plan_path):
        return "This component is part of the ml-odyssey project."

    try:
        with open(plan_path, 'r') as f:
            content = f.read()
        overview_match = re.search(r'## Overview\n(.+?)(?=\n##|\Z)', content, re.DOTALL)
        overview = overview_match.group(1).strip() if overview_match else "This component is part of the ml-odyssey project."
        return overview
    except Exception as e:
        return "This component is part of the ml-odyssey project."

def generate_github_issue_content(github_issue_path):
    """Generate the new content for a github_issue.md file."""
    component_name = extract_component_name(github_issue_path)
    plan_link = get_plan_link(github_issue_path)
    overview = read_plan_overview(github_issue_path)

    content = f"""# GitHub Issues

## Plan Issue

**Title**: [Plan] {component_name} - Design and Documentation

**Body**:
```
## Overview
{overview}

## Planning Tasks
- [ ] Review parent plan and understand context
- [ ] Research best practices and approaches
- [ ] Define detailed implementation strategy
- [ ] Document design decisions and rationale
- [ ] Identify dependencies and prerequisites
- [ ] Create architectural diagrams if needed
- [ ] Define success criteria and acceptance tests
- [ ] Document edge cases and considerations

## Deliverables
- Detailed implementation plan
- Architecture documentation
- Design decisions documented in plan.md
- Success criteria defined

## Reference
See detailed plan: {plan_link}
```

**Labels**: planning, documentation

**URL**: [to be filled]

---

## Test Issue

**Title**: [Test] {component_name} - Write Tests

**Body**:
```
## Overview
Write comprehensive tests for {component_name} following TDD principles.

## Testing Tasks
- [ ] Review plan and identify testable components
- [ ] Write unit test structure and fixtures
- [ ] Implement test cases for happy paths
- [ ] Implement test cases for edge cases
- [ ] Implement test cases for error conditions
- [ ] Add integration tests if needed
- [ ] Ensure test coverage meets threshold (>80%)
- [ ] Document test strategy and approach

## Test Types Required
- Unit tests for core functionality
- Edge case tests
- Error handling tests
- Integration tests (if applicable)

## Acceptance Criteria
- [ ] All test files created in tests/ directory
- [ ] Test coverage >80%
- [ ] All tests pass
- [ ] Tests are well-documented

## Reference
See detailed plan: {plan_link}
```

**Labels**: testing, tdd

**URL**: [to be filled]

---

## Implementation Issue

**Title**: [Impl] {component_name} - Implementation

**Body**:
```
## Overview
Implement {component_name} according to the plan and passing all tests.

## Implementation Tasks
- [ ] Review plan.md and understand requirements
- [ ] Review and run existing tests (should fail - TDD)
- [ ] Implement core functionality
- [ ] Implement error handling
- [ ] Add logging and debugging support
- [ ] Optimize for readability (not performance)
- [ ] Add inline documentation and comments
- [ ] Ensure all tests pass
- [ ] Manual testing and validation

## Implementation Guidelines
- Keep it simple - no premature optimization
- Follow existing code style and patterns
- Add comments for complex logic
- Use descriptive variable names
- Handle errors gracefully

## Acceptance Criteria
- [ ] All functionality implemented
- [ ] All tests passing
- [ ] Code is readable and well-commented
- [ ] No unnecessary complexity

## Reference
See detailed plan: {plan_link}
```

**Labels**: implementation

**URL**: [to be filled]

---

## Packaging Issue

**Title**: [Package] {component_name} - Integration and Packaging

**Body**:
```
## Overview
Integrate {component_name} with existing codebase and ensure proper packaging.

## Packaging Tasks
- [ ] Ensure proper module/package structure
- [ ] Add to appropriate __init__.mojo files
- [ ] Update import statements and dependencies
- [ ] Verify integration with existing components
- [ ] Update configuration files if needed
- [ ] Add to build system if applicable
- [ ] Test imports and module loading
- [ ] Update documentation with usage examples

## Integration Checks
- [ ] Works with existing shared library
- [ ] No circular dependencies
- [ ] Proper error propagation
- [ ] Compatible with existing interfaces

## Acceptance Criteria
- [ ] Component properly integrated
- [ ] All imports work correctly
- [ ] No breaking changes to existing code
- [ ] Integration tests pass

## Reference
See detailed plan: {plan_link}
```

**Labels**: packaging, integration

**URL**: [to be filled]

---

## Cleanup Issue

**Title**: [Cleanup] {component_name} - Refactor and Finalize

**Body**:
```
## Overview
Final cleanup, refactoring, and documentation for {component_name}.

## Cleanup Tasks
- [ ] Review code for simplification opportunities
- [ ] Remove dead code and unused imports
- [ ] Ensure consistent code style
- [ ] Add/update docstrings for all public functions
- [ ] Update README.md with usage examples
- [ ] Add inline comments for complex logic
- [ ] Verify all tests still pass
- [ ] Run linter and fix issues
- [ ] Update CHANGELOG if applicable
- [ ] Final code review

## Documentation Tasks
- [ ] Complete API documentation
- [ ] Add usage examples
- [ ] Document any gotchas or limitations
- [ ] Update parent documentation if needed

## Acceptance Criteria
- [ ] Code is clean and well-organized
- [ ] All documentation complete
- [ ] Linter passes with no warnings
- [ ] Ready for code review

## Reference
See detailed plan: {plan_link}
```

**Labels**: cleanup, documentation

**URL**: [to be filled]
"""

    return content

if __name__ == '__main__':
    plan_dir = Path('/home/mvillmow/ml-odyssey/notes/plan')
    github_issue_files = sorted(plan_dir.rglob('github_issue.md'))

    total = len(github_issue_files)
    print(f"Found {total} github_issue.md files", file=sys.stderr)

    updated_count = 0
    failed = []

    for i, file_path in enumerate(github_issue_files, 1):
        try:
            new_content = generate_github_issue_content(str(file_path))
            with open(file_path, 'w') as f:
                f.write(new_content)
            updated_count += 1
            if updated_count <= 5 or i % 50 == 0:
                print(f"[{i}/{total}] Updated: {file_path}", file=sys.stderr)
        except Exception as e:
            print(f"[{i}/{total}] FAILED {file_path}: {e}", file=sys.stderr)
            failed.append(str(file_path))

    print(f"\n{'='*80}", file=sys.stderr)
    print(f"Update Summary:", file=sys.stderr)
    print(f"  Total files found: {total}", file=sys.stderr)
    print(f"  Successfully updated: {updated_count}", file=sys.stderr)
    print(f"  Failed: {len(failed)}", file=sys.stderr)

    if failed:
        print(f"\nFailed files:", file=sys.stderr)
        for f in failed[:10]:  # Show first 10 failures
            print(f"  - {f}", file=sys.stderr)
        if len(failed) > 10:
            print(f"  ... and {len(failed) - 10} more", file=sys.stderr)

    # Exit with status
    sys.exit(0 if len(failed) == 0 else 1)
