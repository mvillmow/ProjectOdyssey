# Test Correctness Review

## Overview
Create tests for the correctness review capabilities to ensure the agent accurately identifies bugs, logic errors, and edge case issues in code.

## Parent Plan
[../plan.md](../plan.md)

## Child Plans
None (implementation level)

## Inputs
- Correctness reviewer prompt template
- Code samples with known bugs
- Expected bug findings
- Testing framework

## Outputs
- Unit tests for correctness review
- Test cases with various bug types
- Assertions for issue detection
- False positive/negative tracking
- Test documentation

## Steps
1. Create test cases with buggy code
2. Define expected findings for each test
3. Write tests for correctness review
4. Validate detection accuracy

## Success Criteria
- [ ] Tests cover common bug types
- [ ] Detection rate is high for known bugs
- [ ] False positive rate is low
- [ ] Edge cases are properly tested
- [ ] Severity ratings are accurate
- [ ] All tests pass consistently

## Notes
Test with various bug types: null errors, boundary issues, logic bugs, type errors, resource leaks. Include both obvious and subtle bugs. Verify that edge cases are caught. Check that false positives are minimal.
