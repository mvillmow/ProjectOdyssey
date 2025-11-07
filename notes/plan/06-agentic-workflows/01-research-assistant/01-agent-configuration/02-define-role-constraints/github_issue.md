# GitHub Issues

**Plan Issue**:
- Title: [Plan] Define Role Constraints - Design and Documentation
- Body:
```
## Planning Phase: Define Role Constraints

### Objective
Define the research assistant's role, purpose, and behavioral constraints. This includes the agent's primary responsibilities, limitations, and guidelines for interaction following Claude best practices.

### Required Inputs
- Understanding of research assistant responsibilities
- Claude best practices for role definition
- Knowledge of appropriate constraints
- Examples of effective role prompts

### Expected Outputs
- Clear role definition with purpose statement
- Behavioral constraints and limitations
- Interaction guidelines
- Examples of appropriate/inappropriate behavior

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
- Title: [Test] Define Role Constraints - Write Tests
- Body:
```
## Testing Phase: Define Role Constraints

### Objective
Write comprehensive tests following TDD principles for define role constraints.

### Overview
Define the research assistant's role, purpose, and behavioral constraints. This includes the agent's primary responsibilities, limitations, and guidelines for interaction following Claude best practices.

### Testing Tasks
1. Review success criteria and acceptance requirements
2. Design test cases covering all requirements
3. Write unit tests for individual components
4. Write integration tests for component interactions
5. Create test fixtures and mock data as needed
6. Document test coverage and test scenarios
7. Set up test automation in CI/CD pipeline

### Test Coverage Requirements
- [ ] Role definition is clear and specific
- [ ] Constraints prevent hallucination and inappropriate responses
- [ ] Guidelines cover common interaction scenarios
- [ ] Examples illustrate expected behavior
- [ ] Definition follows Claude best practices

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
- Title: [Impl] Define Role Constraints - Implementation
- Body:
```
## Implementation Phase: Define Role Constraints

### Objective
Implement define role constraints according to the design specifications and passing all tests.

### Overview
Define the research assistant's role, purpose, and behavioral constraints. This includes the agent's primary responsibilities, limitations, and guidelines for interaction following Claude best practices.

### Implementation Steps
1. Write clear role definition describing agent purpose
2. Define behavioral constraints to prevent issues
3. Add guidelines for appropriate interactions
4. Include examples of good and bad behavior

### Expected Outputs
- Clear role definition with purpose statement
- Behavioral constraints and limitations
- Interaction guidelines
- Examples of appropriate/inappropriate behavior

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
- Title: [Package] Define Role Constraints - Integration and Packaging
- Body:
```
## Packaging Phase: Define Role Constraints

### Objective
Package and integrate define role constraints into the broader system.

### Overview
Define the research assistant's role, purpose, and behavioral constraints. This includes the agent's primary responsibilities, limitations, and guidelines for interaction following Claude best practices.

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
- Title: [Cleanup] Define Role Constraints - Refactor and Finalize
- Body:
```
## Cleanup Phase: Define Role Constraints

### Objective
Refactor, optimize, and finalize define role constraints for production readiness.

### Overview
Define the research assistant's role, purpose, and behavioral constraints. This includes the agent's primary responsibilities, limitations, and guidelines for interaction following Claude best practices.

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
Use structured prompts with XML tags. Be specific about what the agent should and should not do. Focus on being helpful for research tasks while maintaining accuracy and not hallucinating.

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
