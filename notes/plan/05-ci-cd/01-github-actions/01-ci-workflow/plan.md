# CI Workflow

## Overview

Create a GitHub Actions workflow that runs on every pull request to execute tests, verify code quality, and ensure the build succeeds. This is the primary quality gate for all code changes.

## Parent Plan
[Parent](../plan.md)

## Child Plans
- [01-setup-environment](./01-setup-environment/plan.md)
- [02-run-tests](./02-run-tests/plan.md)
- [03-report-status](./03-report-status/plan.md)

## Inputs
- Set up GitHub Actions environment for Mojo
- Run all tests automatically on pull requests
- Report test results and status clearly
- Cache dependencies for faster builds
- Provide actionable feedback on failures

## Outputs
- Completed ci workflow
- Set up GitHub Actions environment for Mojo (completed)

## Steps
1. Setup Environment
2. Run Tests
3. Report Status

## Success Criteria
- [ ] Workflow triggers on pull requests and pushes to main
- [ ] Mojo environment configured correctly
- [ ] All tests execute successfully
- [ ] Test results displayed clearly
- [ ] Failed tests provide helpful error messages
- [ ] Workflow completes within 5 minutes

## Notes
- Use official Mojo installation steps
- Cache Mojo installation and dependencies
- Run tests in parallel when possible
- Include code coverage reporting
- Keep workflow YAML simple and readable