# User Prompts

## Overview

Implement interactive prompts to collect paper information from users when not provided via command-line arguments. Prompts guide users through entering required metadata like paper title, author, and description.

## Parent Plan

[../plan.md](../plan.md)

## Child Plans

None (Level 4 - implementation level)

## Inputs

- Missing argument values
- Prompt definitions and defaults
- Validation rules for each field

## Outputs

- Collected user input
- Validated paper metadata
- Error messages for invalid input

## Steps

1. Determine which values need to be collected
2. Display prompts with helpful descriptions
3. Collect and validate user input
4. Re-prompt on invalid input with error explanation

## Success Criteria

- [ ] Prompts are clear and informative
- [ ] Input is validated in real-time
- [ ] Default values are offered where appropriate
- [ ] Error messages help users fix mistakes

## Notes

Make prompts conversational and helpful. Show examples of valid input. Allow users to go back and change previous answers. Support both required and optional fields.
