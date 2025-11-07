# GitHub Issues

**Plan Issue**:
- Title: [Plan] Style Reviewer - Design and Documentation
- Body:
```
## Planning Phase: Style Reviewer

### Objective
Create a prompt template for reviewing code style, focusing on readability, naming conventions, documentation, and adherence to coding standards. The template ensures consistent code quality.

### Required Inputs
- Style review criteria
- Project coding standards
- Understanding of readability principles
- Examples of good style reviews

### Expected Outputs
- Style reviewer prompt template
- Style checklist with conventions
- Structured output format
- Documentation quality guidelines
- Refactoring suggestion patterns

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

```
- Labels: planning, documentation
- URL: [to be filled]

**Test Issue**:
- Title: [Test] Style Reviewer - Write Tests
- Body:
```
## Testing Phase: Style Reviewer

### Objective
Write comprehensive tests following TDD principles for style reviewer.

### Overview
Create a prompt template for reviewing code style, focusing on readability, naming conventions, documentation, and adherence to coding standards. The template ensures consistent code quality.

### Testing Tasks
1. Review success criteria and acceptance requirements
2. Design test cases covering all requirements
3. Write unit tests for individual components
4. Write integration tests for component interactions
5. Create test fixtures and mock data as needed
6. Document test coverage and test scenarios
7. Set up test automation in CI/CD pipeline

### Test Coverage Requirements
- [ ] Template checks naming conventions
- [ ] Documentation quality is evaluated
- [ ] Code readability is assessed
- [ ] Style guidelines are enforced
- [ ] Suggestions improve code clarity
- [ ] Template produces actionable feedback

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

```
- Labels: testing, tdd
- URL: [to be filled]

**Implementation Issue**:
- Title: [Impl] Style Reviewer - Implementation
- Body:
```
## Implementation Phase: Style Reviewer

### Objective
Implement style reviewer according to the design specifications and passing all tests.

### Overview
Create a prompt template for reviewing code style, focusing on readability, naming conventions, documentation, and adherence to coding standards. The template ensures consistent code quality.

### Implementation Steps
1. Design output format for style issues
2. Create style checklist
3. Define naming convention rules
4. Add documentation standards

### Expected Outputs
- Style reviewer prompt template
- Style checklist with conventions
- Structured output format
- Documentation quality guidelines
- Refactoring suggestion patterns

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

```
- Labels: implementation
- URL: [to be filled]

**Packaging Issue**:
- Title: [Package] Style Reviewer - Integration and Packaging
- Body:
```
## Packaging Phase: Style Reviewer

### Objective
Package and integrate style reviewer into the broader system.

### Overview
Create a prompt template for reviewing code style, focusing on readability, naming conventions, documentation, and adherence to coding standards. The template ensures consistent code quality.

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

```
- Labels: packaging, integration
- URL: [to be filled]

**Cleanup Issue**:
- Title: [Cleanup] Style Reviewer - Refactor and Finalize
- Body:
```
## Cleanup Phase: Style Reviewer

### Objective
Refactor, optimize, and finalize style reviewer for production readiness.

### Overview
Create a prompt template for reviewing code style, focusing on readability, naming conventions, documentation, and adherence to coding standards. The template ensures consistent code quality.

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
Focus on: naming (variables, functions, classes), documentation (docstrings, comments), formatting, code organization, magic numbers, code duplication. Use XML tags like <style_issue>, <category>, <suggestion>. Keep feedback constructive.

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

```
- Labels: cleanup, documentation
- URL: [to be filled]
