# GitHub Issues

**Plan Issue**:
- Title: [Plan] Prompt Templates - Design and Documentation
- Body:
```
## Planning Phase: Prompt Templates

### Objective
Create structured prompt templates for the code review agent's different review types: correctness review, performance review, and style review. Each template uses Claude best practices with clear criteria and structured output.

### Required Inputs
- Review criteria definitions
- Claude best practices for code review prompts
- Examples of good code reviews
- Knowledge of XML tag patterns

### Expected Outputs
- Correctness reviewer prompt template
- Performance reviewer prompt template
- Style reviewer prompt template
- Structured feedback formats
- Documentation for using templates

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
- Title: [Test] Prompt Templates - Write Tests
- Body:
```
## Testing Phase: Prompt Templates

### Objective
Write comprehensive tests following TDD principles for prompt templates.

### Overview
Create structured prompt templates for the code review agent's different review types: correctness review, performance review, and style review. Each template uses Claude best practices with clear criteria and structured output.

### Testing Tasks
1. Review success criteria and acceptance requirements
2. Design test cases covering all requirements
3. Write unit tests for individual components
4. Write integration tests for component interactions
5. Create test fixtures and mock data as needed
6. Document test coverage and test scenarios
7. Set up test automation in CI/CD pipeline

### Test Coverage Requirements
- [ ] All templates use structured output formats
- [ ] Templates follow review criteria
- [ ] Feedback is specific and actionable
- [ ] Templates include severity ratings
- [ ] Output format is consistent across templates

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
- Title: [Impl] Prompt Templates - Implementation
- Body:
```
## Implementation Phase: Prompt Templates

### Objective
Implement prompt templates according to the design specifications and passing all tests.

### Overview
Create structured prompt templates for the code review agent's different review types: correctness review, performance review, and style review. Each template uses Claude best practices with clear criteria and structured output.

### Implementation Steps
1. Create correctness reviewer template with bug detection focus
2. Create performance reviewer template with efficiency focus
3. Create style reviewer template with readability focus

### Expected Outputs
- Correctness reviewer prompt template
- Performance reviewer prompt template
- Style reviewer prompt template
- Structured feedback formats
- Documentation for using templates

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
- Title: [Package] Prompt Templates - Integration and Packaging
- Body:
```
## Packaging Phase: Prompt Templates

### Objective
Package and integrate prompt templates into the broader system.

### Overview
Create structured prompt templates for the code review agent's different review types: correctness review, performance review, and style review. Each template uses Claude best practices with clear criteria and structured output.

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
- Title: [Cleanup] Prompt Templates - Refactor and Finalize
- Body:
```
## Cleanup Phase: Prompt Templates

### Objective
Refactor, optimize, and finalize prompt templates for production readiness.

### Overview
Create structured prompt templates for the code review agent's different review types: correctness review, performance review, and style review. Each template uses Claude best practices with clear criteria and structured output.

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
Use XML tags to structure findings: <issue>, <location>, <severity>, <suggestion>. Include examples of the type of issues to look for. Guide the agent through systematic review processes. Keep feedback constructive.

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
