#!/bin/bash

# Script to update all github_issue.md files with detailed bodies
# Run this from the repository root: bash update_github_issues.sh

set -e

PLAN_DIR="notes/plan"
COUNT=0
TOTAL=$(find "$PLAN_DIR" -name "github_issue.md" | wc -l)

echo "Found $TOTAL github_issue.md files to update"
echo "Starting update process..."
echo ""

# Function to extract component name from path
get_component_name() {
    local path=$1
    # Remove notes/plan/ prefix and /github_issue.md suffix
    local clean_path=$(echo "$path" | sed 's|notes/plan/||' | sed 's|/github_issue.md||')

    # Convert to human-readable name
    # Example: 01-foundation/02-configuration-files/01-magic-toml
    # -> Foundation - Configuration Files - Magic TOML
    echo "$clean_path" | sed 's|/| - |g' | sed 's|-| |g' | sed 's|  *| |g' | \
        awk '{for(i=1;i<=NF;i++){if($i !~ /^[0-9]+$/){printf "%s%s", (i==1?"":" "), toupper(substr($i,1,1)) tolower(substr($i,2))}}} END {print ""}'
}

# Function to get overview from plan.md
get_overview() {
    local dir=$(dirname "$1")
    local plan_file="$dir/plan.md"

    if [ -f "$plan_file" ]; then
        # Extract Overview section content
        sed -n '/## Overview/,/##/{/##/!p;}' "$plan_file" | sed '/^$/d' | head -3
    else
        echo "Implementation component for the project"
    fi
}

# Process each github_issue.md file
find "$PLAN_DIR" -name "github_issue.md" | while read -r file; do
    COUNT=$((COUNT + 1))

    if [ $((COUNT % 50)) -eq 0 ]; then
        echo "Processed $COUNT/$TOTAL files..."
    fi

    # Get component info
    COMPONENT_NAME=$(get_component_name "$file")
    OVERVIEW=$(get_overview "$file")

    # Get relative path to plan.md from repo root
    DIR=$(dirname "$file")
    PLAN_LINK="$DIR/plan.md"

    # Create new content
    cat > "$file" << 'EOF'
# GitHub Issues

## Plan Issue

**Title**: [Plan] COMPONENT_NAME - Design and Documentation

**Body**:
```
## Overview
OVERVIEW_TEXT

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
See detailed plan: PLAN_LINK
```

**Labels**: planning, documentation

**URL**: [to be filled]

---

## Test Issue

**Title**: [Test] COMPONENT_NAME - Write Tests

**Body**:
```
## Overview
Write comprehensive tests for COMPONENT_NAME following TDD principles.

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
See detailed plan: PLAN_LINK
```

**Labels**: testing, tdd

**URL**: [to be filled]

---

## Implementation Issue

**Title**: [Impl] COMPONENT_NAME - Implementation

**Body**:
```
## Overview
Implement COMPONENT_NAME according to the plan and passing all tests.

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
See detailed plan: PLAN_LINK
```

**Labels**: implementation

**URL**: [to be filled]

---

## Packaging Issue

**Title**: [Package] COMPONENT_NAME - Integration and Packaging

**Body**:
```
## Overview
Integrate COMPONENT_NAME with existing codebase and ensure proper packaging.

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
See detailed plan: PLAN_LINK
```

**Labels**: packaging, integration

**URL**: [to be filled]

---

## Cleanup Issue

**Title**: [Cleanup] COMPONENT_NAME - Refactor and Finalize

**Body**:
```
## Overview
Final cleanup, refactoring, and documentation for COMPONENT_NAME.

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
See detailed plan: PLAN_LINK
```

**Labels**: cleanup, documentation

**URL**: [to be filled]
EOF

    # Replace placeholders
    sed -i "s|COMPONENT_NAME|$COMPONENT_NAME|g" "$file"
    sed -i "s|OVERVIEW_TEXT|$OVERVIEW|g" "$file"
    sed -i "s|PLAN_LINK|$PLAN_LINK|g" "$file"

done

echo ""
echo "Update complete! Processed $TOTAL files."
echo "All github_issue.md files now have detailed bodies for each issue type."
