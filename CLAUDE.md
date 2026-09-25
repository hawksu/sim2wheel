# Project Instructions

## Code Style

- Use **camelCase** for identifiers (variables, functions, methods).
- Use **BSD KNF** (Kernel Normal Form) indentation style: 8-column hard tabs
  for statement indentation, 4-column continuation indents, and the opening
  brace on the same line as the statement (with the function body's opening
  brace on its own line).

## Code Exploration Policy

Always call and use the jCodeMunch-MCP tools for reading and navigating code.
Start with `jcodemunch_guide` to load the catalogue and usage rules, then use the
front-door tools (`order`, `route`, `menu`) for all code navigation and understanding.

Never fall back to Read, Grep, Glob, or Bash for code exploration.

**Exception:** use `Read` when you are about to edit a file — the harness requires a
`Read` before `Edit`/`Write`. Use jCodeMunch to *find and understand* code, then `Read`
only the file you are changing.
