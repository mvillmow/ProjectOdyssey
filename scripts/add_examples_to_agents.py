#!/usr/bin/env python3
"""
Add Examples sections to agent files that are missing them.

This script reads agent files and adds appropriate Examples sections
based on the agent's role type.
"""

import re
from pathlib import Path
from typing import Dict

# Agent-specific examples based on role patterns
EXAMPLES_BY_PATTERN = {
    'orchestrator': """## Examples

### Example 1: Coordinating Multi-Phase Workflow

**Scenario**: Implementing a new component across multiple subsections

**Actions**:
1. Break down component into design, implementation, and testing phases
2. Delegate design work to design agents
3. Delegate implementation to implementation specialists
4. Coordinate parallel work streams
5. Monitor progress and resolve blockers

**Outcome**: Component delivered with all phases complete and integrated

### Example 2: Resolving Cross-Component Dependencies

**Scenario**: Two subsections have conflicting approaches to shared interface

**Actions**:
1. Identify dependency conflict between subsections
2. Escalate to design agents for interface specification
3. Coordinate implementation updates across both subsections
4. Validate integration through testing phase

**Outcome**: Unified interface with both components working correctly
""",
    'design': """## Examples

### Example 1: Module Architecture Design

**Scenario**: Designing architecture for neural network training module

**Actions**:
1. Analyze requirements and define module boundaries
2. Design component interfaces and data flow
3. Create architectural diagrams and specifications
4. Define integration points with existing modules
5. Document design decisions and trade-offs

**Outcome**: Clear architectural specification ready for implementation

### Example 2: Interface Refactoring

**Scenario**: Simplifying complex API with too many parameters

**Actions**:
1. Analyze current interface usage patterns
2. Identify common parameter combinations
3. Design simplified API with sensible defaults
4. Plan backward compatibility strategy
5. Document migration path

**Outcome**: Cleaner API with improved developer experience
""",
    'specialist': """## Examples

### Example 1: Component Implementation Planning

**Scenario**: Breaking down backpropagation algorithm into implementable functions

**Actions**:
1. Analyze algorithm requirements from design spec
2. Break down into functions: forward pass, backward pass, parameter update
3. Define function signatures and data structures
4. Create implementation plan with dependencies
5. Delegate functions to engineers

**Outcome**: Clear implementation plan with well-defined function boundaries

### Example 2: Code Quality Improvement

**Scenario**: Refactoring complex function with multiple responsibilities

**Actions**:
1. Analyze function complexity and identify separate concerns
2. Extract sub-functions with single responsibilities
3. Improve naming and add type hints
4. Add documentation and usage examples
5. Coordinate with test engineer for test updates

**Outcome**: Maintainable code following single responsibility principle
""",
    'engineer': """## Examples

### Example 1: Implementing Convolution Layer

**Scenario**: Writing Mojo implementation of 2D convolution

**Actions**:
1. Review function specification and interface design
2. Implement forward pass with proper tensor operations
3. Add error handling and input validation
4. Optimize with SIMD where applicable
5. Write inline documentation

**Outcome**: Working convolution implementation ready for testing

### Example 2: Fixing Bug in Gradient Computation

**Scenario**: Gradient shape mismatch causing training failures

**Actions**:
1. Reproduce bug with minimal test case
2. Trace tensor dimensions through backward pass
3. Fix dimension handling in gradient computation
4. Verify fix with unit tests
5. Update documentation if needed

**Outcome**: Correct gradient computation with all tests passing
""",
    'review-specialist': """## Examples

### Example 1: Code Review for Numerical Stability

**Scenario**: Reviewing implementation with potential overflow issues

**Actions**:
1. Identify operations that could overflow (exp, large multiplications)
2. Check for numerical stability patterns (log-sum-exp, epsilon values)
3. Provide specific fixes with mathematical justification
4. Reference best practices and paper specifications
5. Categorize findings by severity

**Outcome**: Numerically stable implementation preventing runtime errors

### Example 2: Architecture Review Feedback

**Scenario**: Implementation tightly coupling unrelated components

**Actions**:
1. Analyze component dependencies and coupling
2. Identify violations of separation of concerns
3. Suggest refactoring with interface-based design
4. Provide concrete code examples of improvements
5. Group similar issues into single review comment

**Outcome**: Actionable feedback leading to better architecture
"""
}

def get_agent_type(filename: str) -> str:
    """Determine agent type from filename."""
    if 'orchestrator' in filename:
        return 'orchestrator'
    elif 'design' in filename:
        return 'design'
    elif 'review-specialist' in filename:
        return 'review-specialist'
    elif 'specialist' in filename:
        return 'specialist'
    elif 'engineer' in filename:
        return 'engineer'
    else:
        return 'specialist'  # default

def add_examples_section(file_path: Path, dry_run: bool = False) -> bool:
    """Add Examples section to agent file if missing."""
    content = file_path.read_text()

    # Check if Examples section already exists
    if re.search(r'^## Examples', content, re.MULTILINE):
        print(f"  ✓ {file_path.name} already has Examples section")
        return False

    # Determine agent type and get appropriate examples
    agent_type = get_agent_type(file_path.name)
    examples = EXAMPLES_BY_PATTERN.get(agent_type, EXAMPLES_BY_PATTERN['specialist'])

    # Find insertion point (before final separator or end)
    # Look for final --- separator
    final_separator_match = re.search(r'\n---\n\*[^\*]+\*\s*$', content)
    if final_separator_match:
        insertion_point = final_separator_match.start()
        new_content = content[:insertion_point] + '\n' + examples + content[insertion_point:]
    else:
        # Look for **Configuration File**: pattern
        config_match = re.search(r'\n---\n\n\*\*Configuration File\*\*:', content)
        if config_match:
            insertion_point = config_match.start()
            new_content = content[:insertion_point] + '\n' + examples + content[insertion_point:]
        else:
            # Insert at end
            new_content = content.rstrip() + '\n\n' + examples + '\n'

    if dry_run:
        print(f"  Would add Examples to {file_path.name} (type: {agent_type})")
        return True
    else:
        file_path.write_text(new_content)
        print(f"  ✅ Added Examples to {file_path.name} (type: {agent_type})")
        return True

def main():
    import sys

    agents_dir = Path(".claude/agents")
    dry_run = "--dry-run" in sys.argv

    if dry_run:
        print("DRY RUN MODE - No files will be modified\n")

    agent_files = sorted(agents_dir.glob("*.md"))
    modified_count = 0

    for agent_file in agent_files:
        if add_examples_section(agent_file, dry_run):
            modified_count += 1

    print(f"\n{'Would modify' if dry_run else 'Modified'} {modified_count} files")

if __name__ == "__main__":
    main()
