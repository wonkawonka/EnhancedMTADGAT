---
name: andrej-karpathy-guidelines
description: Use for nontrivial coding tasks in this workspace to reduce bad assumptions, over-engineering, unrelated edits, and unverified changes. Helps the agent state assumptions, prefer simple solutions, make surgical edits, and verify results.
argument-hint: "[coding task]"
---

# Karpathy-Inspired Coding Guidelines

Apply these guidelines when working on coding tasks in this VS Code workspace.

## Think Before Coding

- Do not silently choose an interpretation when the request is ambiguous.
- State important assumptions before implementing.
- Present tradeoffs when multiple reasonable approaches exist.
- Push back when a simpler or safer approach fits the request better.
- Ask for clarification when missing information makes a meaningful implementation risky.

## Simplicity First

- Write the minimum code that solves the requested problem.
- Do not add speculative features, abstractions, configuration, or error handling.
- Avoid creating abstractions for one-off code.
- If a solution becomes much larger than needed, simplify before finishing.

## Surgical Changes

- Touch only files and lines needed for the request.
- Match the existing project style.
- Do not reformat, refactor, or "improve" adjacent code unless the task requires it.
- Remove imports, variables, functions, or files only when they became unused because of the current change.
- Mention unrelated dead code or cleanup opportunities instead of changing them.

## Goal-Driven Execution

- Turn implementation tasks into explicit success criteria.
- For bug fixes, reproduce the issue when practical, then verify the fix.
- For new behavior, run the narrowest useful validation after editing.
- For multi-step tasks, keep a short plan with each step tied to a verification check.
- Continue iterating until the stated success criteria are met or a real blocker is found.
